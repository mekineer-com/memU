from __future__ import annotations

import asyncio
import json
import logging
import math
import pathlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from pydantic import BaseModel

from memu.app import memorize_parsing as parsing
from memu.app import memorize_speakers as speakers
from memu.app import memorize_dedupe as dedupe
from memu.app import memorize_categories as categories
from memu.app import memorize_episodes as episode_helpers
from memu.app import memorize_persistence as persistence
from memu.app.settings import CategoryConfig, CustomPrompt
from memu.database.models import CategoryItem, MemoryCategory, MemoryItem, MemoryType, Resource
from memu.prompts.memory_type import (
    CUSTOM_PROMPTS as MEMORY_TYPE_CUSTOM_PROMPTS,
)
from memu.prompts.memory_type import (
    CUSTOM_TYPE_CUSTOM_PROMPTS,
    DEFAULT_MEMORY_TYPES,
)
from memu.prompts.memory_type import (
    PROMPTS as MEMORY_TYPE_PROMPTS,
)
from memu.prompts.preprocess import PROMPTS as PREPROCESS_PROMPTS
from memu.prompts.router import PROMPT as ROUTER_PROMPT
from memu.workflow.step import WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)

_EPISODE_SUMMARY_EXTRACTION_GUIDANCE = (
    "The summary helps give you perspective on what matters. "
    "Create individual memory items that don't treat every verbose tangent as a "
    "separate memory. The items should still capture both big-picture and specific "
    "details, just without the verbosity noise."
)


class StructuredMemoryEntry(NamedTuple):
    memory_type: MemoryType
    content: str
    categories: list[str]
    source_role: str | None
    confidence: float | None
    source_message_ids: list[int]
    reflection_salience: float | None
    emotional_intensity: float | None = None
    replaces_previous_fact: str | None = None
    entities: list[dict[str, str]] | None = None
    speaker_id: str | None = None
    speaker_label: str | None = None
    episode_ref: int | None = None


class SpeakerRosterEntry(NamedTuple):
    speaker_id: str
    speaker_label: str
    coarse_role: str


class HomelessCategoryCluster(NamedTuple):
    cluster_id: str
    entry_indexes: list[int]
    label_counts: dict[str, int]
    average_salience: float | None
    max_salience: float | None
    examples: list[str]


if TYPE_CHECKING:
    from memu.app.service import Context
    from memu.app.settings import MemorizeConfig
    from memu.blob.local_fs import LocalFS
    from memu.database.interfaces import Database


class MemorizeMixin:
    if TYPE_CHECKING:
        memorize_config: MemorizeConfig
        category_configs: list[CategoryConfig]
        category_config_map: dict[str, CategoryConfig]
        _category_prompt_str: str
        fs: LocalFS
        _run_workflow: Callable[..., Awaitable[WorkflowState]]
        _get_context: Callable[[], Context]
        _get_database: Callable[[], Database]
        _get_step_llm_client: Callable[[Mapping[str, Any] | None], Any]
        _get_step_embedding_client: Callable[[Mapping[str, Any] | None], Any]
        _get_llm_client: Callable[..., Any]
        _model_dump_without_embeddings: Callable[[BaseModel], dict[str, Any]]
        _extract_json_blob: Callable[[str], str]
        _escape_prompt_value: Callable[[str], str]
        user_model: type[BaseModel]

    @staticmethod
    def _normalize_reflection_salience(value: Any) -> float | None:
        return parsing._normalize_reflection_salience(value)

    @staticmethod
    def _normalize_replaces_previous_fact(value: Any) -> str | None:
        return parsing._normalize_replaces_previous_fact(value)

    @staticmethod
    def _parse_episode_ref(value: Any) -> int | None:
        return parsing._parse_episode_ref(value)

    @staticmethod
    def _hedge_summary_for_confidence(summary: str, confidence: float | None) -> str:
        text = (summary or "").strip()
        if not text or confidence is None or confidence >= 0.6:
            return text
        lowered = text[:1].lower() + text[1:] if text[:1].isupper() else text
        if confidence < 0.35:
            return f"I have a faint suspicion that {lowered}"
        return f"I have an inkling that {lowered}"

    async def memorize(
        self,
        *,
        resource_url: str,
        modality: str,
        user: dict[str, Any] | None = None,
        raw_text: str | None = None,
        local_path: str | None = None,
        all_categories_summary: str | None = None,
        soul_card: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Memorize a single input unit.

        For `modality="conversation"`, this method treats the input as one
        episode and delegates to `memorize_episode()`. Episode splitting is
        orchestrated by callers (server path) via `split_segment_into_episodes()`
        and repeated `memorize_episode()` calls.
        """
        self._validate_memorize_scope(user)
        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        conversation_id = self._resolve_conversation_id(user)
        normalized_all_categories_summary = (all_categories_summary or "").strip() or None
        normalized_soul_card = (soul_card or "").strip() or None

        if modality == "conversation":
            episode_local_path = local_path or resource_url
            episode_raw_text = raw_text
            if episode_raw_text is None:
                episode_local_path, episode_raw_text = await self.fs.fetch(resource_url, modality)
            return await self.memorize_episode(
                resource_url=resource_url,
                modality=modality,
                episode={"text": episode_raw_text, "caption": None},
                user=user,
                raw_text=episode_raw_text,
                local_path=episode_local_path,
                all_categories_summary=normalized_all_categories_summary,
                soul_card=normalized_soul_card,
                memory_retrieve_history=memory_retrieve_history,
                memory_prior_context=memory_prior_context,
                conversation_id=conversation_id,
            )

        state: WorkflowState = {
            "resource_url": resource_url,
            "modality": modality,
            "memory_types": self._resolve_memory_types(),
            "categories_prompt_str": self._category_prompt_str,
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
            "conversation_id": conversation_id,
            "all_categories_summary": normalized_all_categories_summary,
            "memory_retrieve_history": memory_retrieve_history,
            "memory_prior_context": memory_prior_context,
            "soul_card": normalized_soul_card,
        }
        if raw_text is not None:
            state["raw_text"] = raw_text
            state["local_path"] = local_path or resource_url

        result = await self._run_workflow("memorize", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Memorize workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def split_segment_into_episodes(
        self,
        *,
        local_path: str,
        raw_text: str | None,
        modality: str,
    ) -> list[dict[str, Any]]:
        llm_client = self._get_llm_client(
            self.memorize_config.preprocess_llm_profile,
            step_context={"operation": "memorize", "step_id": "preprocess"},
        )
        segment_episodes = await self._split_into_episodes(
            local_path=local_path,
            text=raw_text,
            modality=modality,
            llm_client=llm_client,
        )
        if not segment_episodes:
            msg = "Preprocessor returned no episodes for segment"
            raise RuntimeError(msg)
        return segment_episodes

    async def split_cross_conversation_into_episodes(
        self,
        *,
        raw_text: str,
    ) -> list[dict[str, Any]]:
        """Split merged multi-source conversation into episodes grouped by storyline."""
        template = PREPROCESS_PROMPTS.get("cross_conversation")
        if not template:
            return [{"text": raw_text, "caption": None}]
        llm_client = self._get_llm_client(
            self.memorize_config.preprocess_llm_profile,
            step_context={"operation": "memorize", "step_id": "preprocess"},
        )
        return await self._split_conversation_into_episodes(raw_text, template, llm_client=llm_client)

    async def memorize_episode(
        self,
        *,
        resource_url: str,
        modality: str,
        episode: Mapping[str, Any],
        user: dict[str, Any] | None = None,
        raw_text: str | None = None,
        local_path: str | None = None,
        all_categories_summary: str | None = None,
        soul_card: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_memorize_scope(user)
        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        state: WorkflowState = {
            "resource_url": resource_url,
            "modality": modality,
            "memory_types": self._resolve_memory_types(),
            "categories_prompt_str": self._category_prompt_str,
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
            "conversation_id": conversation_id or self._resolve_conversation_id(user),
            "all_categories_summary": (all_categories_summary or "").strip() or None,
            "memory_retrieve_history": memory_retrieve_history,
            "memory_prior_context": memory_prior_context,
            "soul_card": (soul_card or "").strip() or None,
            "raw_text": raw_text,
            "local_path": local_path or resource_url,
            "episodes": [dict(episode)],
        }
        extract_context = {
            "workflow_name": "memorize_episode",
            "step_id": "extract_items",
            "step_config": {"chat_llm_profile": self.memorize_config.memory_extract_llm_profile},
        }
        categorize_context = {
            "workflow_name": "memorize_episode",
            "step_id": "categorize_items",
            "step_config": {"embed_llm_profile": "embedding"},
        }
        persist_context = {
            "workflow_name": "memorize_episode",
            "step_id": "persist_index",
            "step_config": {"chat_llm_profile": self.memorize_config.category_update_llm_profile},
        }
        state = await self._memorize_extract_items(state, extract_context)
        state = await self._memorize_categorize_items(state, categorize_context)
        state = await self._memorize_dedupe_merge(state, {"workflow_name": "memorize_episode", "step_id": "dedupe_merge"})
        state = await self._memorize_persist_and_index(state, persist_context)
        state = self._memorize_build_response(state, {"workflow_name": "memorize_episode", "step_id": "build_response"})
        response = cast(dict[str, Any] | None, state.get("response"))
        if response is None:
            msg = "Memorize episode failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def memorize_episodes_batch(
        self,
        *,
        modality: str,
        episodes: Sequence[Mapping[str, Any]],
        user: dict[str, Any] | None = None,
        all_categories_summary: str | None = None,
        soul_card: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_memorize_scope(user)
        if modality != "conversation":
            msg = f"memorize_episodes_batch only supports modality='conversation', got {modality!r}"
            raise ValueError(msg)
        if not episodes:
            return []

        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        extract_client = self._get_llm_client(
            self.memorize_config.memory_extract_llm_profile,
            step_context={"operation": "memorize", "step_id": "extract_items_batch"},
        )
        memory_types = self._resolve_memory_types()
        declared_entity_roster = self._list_declared_relationship_roster(
            store=store,
            user=user_scope,
        )

        prepared: list[dict[str, Any]] = []
        routing_notes: list[str] = []
        for episode_ref, episode_job in enumerate(episodes, start=1):
            resource_url = str(episode_job.get("resource_url") or "").strip()
            if not resource_url:
                msg = f"batch episode {episode_ref} missing resource_url"
                raise ValueError(msg)

            raw_text = episode_job.get("raw_text")
            episode_payload = episode_job.get("episode")
            if not isinstance(episode_payload, Mapping):
                msg = f"batch episode {episode_ref} missing episode payload"
                raise ValueError(msg)

            text = episode_payload.get("text")
            caption = episode_payload.get("caption")
            _, message_indices = self._prepare_episode(
                modality=modality,
                text=text if isinstance(text, str) else None,
                message_indices=episode_payload.get("message_indices"),
            )

            message_happened_at_map = self._extract_message_happened_at_map(raw_text)
            conversation_messages = self._extract_conversation_messages(raw_text)
            messages_by_index = {idx: msg for idx, msg in conversation_messages}
            episode_messages_all: list[dict[str, Any]] = []
            for message_idx in message_indices:
                msg_payload = messages_by_index.get(message_idx)
                if msg_payload is None:
                    continue
                episode_msg = dict(msg_payload)
                episode_msg["_message_index"] = message_idx
                episode_messages_all.append(episode_msg)

            primary_messages = [
                msg
                for msg in episode_messages_all
                if self._message_is_primary_for_memorize(msg)
            ]
            background_messages = [
                msg
                for msg in episode_messages_all
                if not self._message_is_primary_for_memorize(msg)
            ]

            preprocessor_rows_raw = episode_payload.get("background_summaries")
            preprocessor_rows: list[dict[str, Any]] = []
            if isinstance(preprocessor_rows_raw, list):
                for row in preprocessor_rows_raw:
                    if not isinstance(row, Mapping):
                        continue
                    summary = str(row.get("summary") or "").strip()
                    if not summary:
                        continue
                    after_raw = row.get("after_index")
                    if after_raw is None:
                        after_index = None
                    else:
                        try:
                            after_index = int(after_raw)
                        except (TypeError, ValueError):
                            continue
                    preprocessor_rows.append(
                        {
                            "after_index": after_index,
                            "summary": summary,
                            "source_label": str(row.get("source_label") or "background"),
                        }
                    )

            if preprocessor_rows:
                seeded_rows = list(preprocessor_rows)
                if background_messages:
                    _rendered_with_tail, tail_rows = await self._render_episode_with_background_context(
                        primary_messages=primary_messages,
                        background_messages=background_messages,
                        llm_client=extract_client,
                    )
                    seeded_rows.extend(tail_rows)
                background_summaries = seeded_rows
                rendered_text = self._render_episode_with_summary_rows(
                    primary_messages=primary_messages,
                    summary_rows=background_summaries,
                )
            else:
                rendered_text, background_summaries = await self._render_episode_with_background_context(
                    primary_messages=primary_messages,
                    background_messages=background_messages,
                    llm_client=extract_client,
                )
            episode_text = rendered_text or (str(text).strip() if isinstance(text, str) else "")
            context_only = bool(episode_messages_all) and not primary_messages

            episode_summary: str | None = str(caption).strip() if isinstance(caption, str) and str(caption).strip() else None
            episode_items: list[dict[str, str]] = []
            applicable_types: list[MemoryType] = memory_types
            if context_only:
                applicable_types = []
                if not episode_summary:
                    episode_summary = await self._summarize_background_messages(
                        messages=background_messages or episode_messages_all,
                        llm_client=extract_client,
                    )
                if episode_summary:
                    episode_items = [{"title": "Story", "summary": episode_summary}]
                if not episode_summary:
                    routing_notes.append(f"episode {episode_ref}: context-only background, no summary generated")
            elif episode_text:
                applicable_types, routed_summary, routed_items = await self._route_episode(
                    episode_text,
                    memory_types,
                    extract_client,
                    soul_card=(soul_card or "").strip() or None,
                    skipped_reasons=routing_notes,
                )
                if routed_summary:
                    episode_summary = routed_summary
                episode_items = routed_items

            primary_indices = [
                self._message_index_for_sort(msg)
                for msg in primary_messages
                if self._message_index_for_sort(msg) >= 0
            ]
            selected_indices = self._dedupe_message_indices(primary_indices or message_indices)
            speaker_map = self._build_speaker_map(primary_messages or episode_messages_all, user_scope)

            plan_message_happened_at_map = {
                message_idx: message_happened_at_map[message_idx]
                for message_idx in selected_indices
                if message_idx in message_happened_at_map
            }
            segment_id = str(episode_payload.get("segment_id") or "").strip() or None
            if segment_id:
                episode_id = segment_id
            else:
                conv_id = conversation_id or self._resolve_conversation_id(user)
                if conv_id and selected_indices:
                    episode_id = f"{conv_id}:{selected_indices[0]}-{selected_indices[-1]}"
                else:
                    episode_id = None

            prepared.append({
                "episode_ref": episode_ref,
                "resource_url": resource_url,
                "local_path": str(episode_job.get("local_path") or resource_url),
                "segment_raw_text": raw_text,
                "text": episode_text,
                "caption": caption,
                "episode_summary": episode_summary,
                "episode_items": episode_items,
                "message_indices": selected_indices,
                "message_happened_at_map": plan_message_happened_at_map,
                "episode_messages": primary_messages,
                "background_summaries": background_summaries,
                "context_only": context_only,
                "speaker_map": speaker_map,
                "applicable_types": applicable_types,
                "entries": [],
                "episode_id": episode_id,
                "segment_id": segment_id or episode_id,
            })

        extractable = [
            ep for ep in prepared
            if ep["applicable_types"] and isinstance(ep.get("text"), str) and str(ep.get("text") or "").strip()
        ]
        routed_total = len(extractable)
        if extractable:
            type_counts = {mtype: 0 for mtype in memory_types}
            for ep in extractable:
                routed = set(ep["applicable_types"])
                for mtype in memory_types:
                    if mtype in routed:
                        type_counts[mtype] += 1

            total_messages = sum(len(ep["message_indices"]) for ep in extractable)
            max_items = self._compute_batch_max_items(total_messages)
            target_by_type = {
                mtype: f"up to {max(1, round(max_items * (type_counts[mtype] / routed_total)))}"
                for mtype in memory_types
                if type_counts[mtype] > 0
            }

            conversation_text = self._build_batch_extraction_text(extractable)

            merged_speaker_map: dict[int, tuple[str, str]] = {}
            for ep in extractable:
                sm = ep.get("speaker_map")
                if isinstance(sm, dict):
                    merged_speaker_map.update(sm)
            batch_speaker_roster = self._build_speaker_roster_for_episode(
                speaker_map=merged_speaker_map,
                declared_entities=declared_entity_roster,
                episode_text=conversation_text,
            )
            estimated_tokens = self._estimate_text_tokens(conversation_text)
            if estimated_tokens > 100000:
                logger.warning(
                    "batch extraction prompt estimated at %d tokens (>100000) for %d episodes",
                    estimated_tokens,
                    routed_total,
                )

            for mtype in memory_types:
                if type_counts.get(mtype, 0) < 1:
                    continue
                type_entries = await self._generate_entries_from_text(
                    resource_text=conversation_text,
                    store=store,
                    memory_types=[mtype],
                    categories_prompt_str=self._category_prompt_str,
                    all_categories_summary=(all_categories_summary or "").strip() or None,
                    soul_card=(soul_card or "").strip() or None,
                    speaker_roster=batch_speaker_roster,
                    default_source_message_ids=None,
                    llm_client=extract_client,
                    target_items_by_type={mtype: target_by_type[mtype]},
                    require_episode_ref=True,
                )
                for entry in type_entries:
                    if entry.episode_ref is None:
                        continue
                    if 1 <= entry.episode_ref <= len(prepared):
                        prepared[entry.episode_ref - 1]["entries"].append(entry)
                    else:
                        logger.warning(
                            "Dropped extracted item with out-of-range episode_ref=%s (max=%s)",
                            entry.episode_ref,
                            len(prepared),
                        )

        responses: list[dict[str, Any]] = []
        for ep in prepared:
            episode_entries = self._decorate_entries_with_plan_context(
                [self._attribute_memory(entry, ep["speaker_map"]) for entry in ep["entries"]],
                message_indices=ep["message_indices"],
            )
            plan = {
                "resource_url": ep["resource_url"],
                "text": ep["text"],
                "caption": ep["episode_summary"] or ep["caption"],
                "episode_summary": ep["episode_summary"],
                "episode_items": ep["episode_items"],
                "message_indices": ep["message_indices"],
                "message_happened_at_map": ep["message_happened_at_map"],
                "entries": episode_entries,
                "episode_id": ep["episode_id"],
                "segment_id": ep["segment_id"],
                "memory_retrieve_history": memory_retrieve_history,
                "memory_prior_context": memory_prior_context,
                "episode_messages": ep["episode_messages"],
            }
            state: WorkflowState = {
                "resource_url": ep["resource_url"],
                "modality": modality,
                "local_path": ep["local_path"],
                "conversation_id": conversation_id or self._resolve_conversation_id(user),
                "ctx": ctx,
                "store": store,
                "category_ids": list(ctx.category_ids),
                "user": user_scope,
                "episode_plans": [plan],
            }
            categorize_context = {
                "workflow_name": "memorize_episodes_batch",
                "step_id": "categorize_items",
                "step_config": {"embed_llm_profile": "embedding"},
            }
            persist_context = {
                "workflow_name": "memorize_episodes_batch",
                "step_id": "persist_index",
                "step_config": {"chat_llm_profile": self.memorize_config.category_update_llm_profile},
            }
            state = await self._memorize_categorize_items(state, categorize_context)
            state = await self._memorize_dedupe_merge(
                state, {"workflow_name": "memorize_episodes_batch", "step_id": "dedupe_merge"}
            )
            state = await self._memorize_persist_and_index(state, persist_context)
            state = self._memorize_build_response(
                state, {"workflow_name": "memorize_episodes_batch", "step_id": "build_response"}
            )
            response = cast(dict[str, Any] | None, state.get("response"))
            if response is None:
                msg = "Memorize episode batch failed to produce an episode response"
                raise RuntimeError(msg)
            if routing_notes:
                response["skipped_reasons"] = list(routing_notes)
            responses.append(response)

        return responses

    @staticmethod
    def _validate_memorize_scope(user: dict[str, Any] | None) -> None:
        # Fail loud at the engine boundary: a non-empty scope without soul_id
        # silently mixes memories across souls, which is the worst class of
        # isolation bug. The server always sends soul_id; tests must too.
        if isinstance(user, dict) and user and not str(user.get("soul_id") or "").strip():
            msg = "MemoryService.memorize: user scope is non-empty but 'soul_id' is missing/blank"
            raise ValueError(msg)

    @staticmethod
    def _resolve_conversation_id(user: dict[str, Any] | None) -> str | None:
        if not isinstance(user, dict):
            return None
        raw = user.get("conversation_id")
        if raw is None:
            return None
        candidate = str(raw).strip()
        return candidate or None

    def _build_memorize_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="ingest_resource",
                role="ingest",
                handler=self._memorize_ingest_resource,
                requires={"resource_url", "modality"},
                produces={"local_path", "raw_text"},
                capabilities={"io"},
            ),
            WorkflowStep(
                step_id="split_episodes",
                role="preprocess",
                handler=self._memorize_split_episodes,
                requires={"local_path", "modality", "raw_text"},
                produces={"episodes"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.memorize_config.preprocess_llm_profile},
            ),
            WorkflowStep(
                step_id="extract_items",
                role="extract",
                handler=self._memorize_extract_items,
                requires={
                    "episodes",
                    "memory_types",
                    "categories_prompt_str",
                    "modality",
                    "resource_url",
                },
                produces={"episode_plans"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.memorize_config.memory_extract_llm_profile},
            ),
            WorkflowStep(
                step_id="categorize_items",
                role="categorize",
                handler=self._memorize_categorize_items,
                requires={"episode_plans", "ctx", "store", "local_path", "modality", "user"},
                produces={"resources", "items", "relations", "category_updates", "homeless_item_count"},
                capabilities={"db", "vector"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="dedupe_merge",
                role="dedupe_merge",
                handler=self._memorize_dedupe_merge,
                requires={"items", "relations", "category_updates", "store", "user"},
                produces={"items", "relations", "category_updates"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="persist_index",
                role="persist",
                handler=self._memorize_persist_and_index,
                requires={"category_updates", "ctx", "store"},
                produces={"categories"},
                capabilities={"db", "llm"},
                config={"chat_llm_profile": self.memorize_config.category_update_llm_profile},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._memorize_build_response,
                requires={"resources", "items", "relations", "ctx", "store", "category_ids"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    @staticmethod
    def _list_memorize_initial_keys() -> set[str]:
        return {
            "resource_url",
            "modality",
            "memory_types",
            "categories_prompt_str",
            "ctx",
            "store",
            "category_ids",
            "user",
        }

    async def _memorize_ingest_resource(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if state.get("raw_text") is not None and state.get("local_path") is not None:
            return state

        local_path, raw_text = await self.fs.fetch(state["resource_url"], state["modality"])
        state.update({"local_path": local_path, "raw_text": raw_text})
        return state

    async def _memorize_split_episodes(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        preprocessed = await self._split_into_episodes(
            local_path=state["local_path"],
            text=state.get("raw_text"),
            modality=state["modality"],
            llm_client=llm_client,
        )
        if not preprocessed:
            preprocessed = [{"text": state.get("raw_text"), "caption": None}]
        state["episodes"] = preprocessed
        return state

    async def _memorize_extract_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        episodes = state.get("episodes", [])
        episode_plans: list[dict[str, Any]] = []
        skipped_reasons: list[str] = []
        message_happened_at_map = self._extract_message_happened_at_map(state.get("raw_text"))
        conversation_messages = self._extract_conversation_messages(state.get("raw_text"))
        messages_by_index = {idx: msg for idx, msg in conversation_messages}
        declared_entity_roster = self._list_declared_relationship_roster(
            store=state["store"],
            user=state.get("user"),
        )
        if not episodes:
            state["episode_plans"] = []
            return state
        if len(episodes) > 1:
            msg = f"extract_items expects one episode per call, got {len(episodes)}"
            raise ValueError(msg)
        prep = episodes[0] if episodes else {}
        text = prep.get("text")
        caption = prep.get("caption")
        _, message_indices = self._prepare_episode(
            modality=state["modality"],
            text=text if isinstance(text, str) else None,
            message_indices=prep.get("message_indices"),
        )
        episode_summary: str | None = None
        episode_items: list[dict[str, str]] = []
        if state["modality"] == "conversation" and isinstance(text, str):
            applicable_types, episode_summary, episode_items = await self._route_episode(
                text, state["memory_types"], llm_client,
                soul_card=state.get("soul_card"),
                skipped_reasons=skipped_reasons,
            )
            if not applicable_types:
                state["episode_plans"] = []
                if skipped_reasons:
                    state["skipped_reasons"] = skipped_reasons
                return state
        else:
            applicable_types = state["memory_types"]

        episode_messages: list[dict[str, Any]] = []
        for message_idx in message_indices:
            msg = messages_by_index.get(message_idx)
            if msg is None:
                continue
            episode_msg = dict(msg)
            episode_msg["_message_index"] = message_idx
            episode_messages.append(episode_msg)
        speaker_map = self._build_speaker_map(episode_messages, state.get("user"))
        speaker_roster = self._build_speaker_roster_for_episode(
            speaker_map=speaker_map,
            declared_entities=declared_entity_roster,
            episode_text=text,
        )

        extraction_text = text
        if episode_summary:
            extraction_text = (
                f"Episode Summary:\n{episode_summary}\n\n"
                f"{_EPISODE_SUMMARY_EXTRACTION_GUIDANCE}\n\n"
                f"---\n{text}"
            )

        structured_entries = await self._generate_structured_entries(
            modality=state["modality"],
            store=state["store"],
            memory_types=applicable_types,
            text=extraction_text,
            categories_prompt_str=state["categories_prompt_str"],
            all_categories_summary=state.get("all_categories_summary"),
            soul_card=state.get("soul_card"),
            speaker_roster=speaker_roster,
            llm_client=llm_client,
            skipped_reasons=skipped_reasons,
        )
        structured_entries = self._decorate_entries_with_plan_context(
            structured_entries,
            message_indices=message_indices,
        )
        structured_entries = [self._attribute_memory(entry, speaker_map) for entry in structured_entries]
        plan_message_happened_at_map = {
            message_idx: message_happened_at_map[message_idx]
            for message_idx in message_indices
            if message_idx in message_happened_at_map
        }

        segment_id = str(prep.get("segment_id") or "").strip() or None
        if segment_id:
            episode_id = segment_id
        else:
            conv_id = state.get("conversation_id")
            if conv_id and message_indices:
                episode_id = f"{conv_id}:{message_indices[0]}-{message_indices[-1]}"
            else:
                episode_id = None
        plan: dict[str, Any] = {
            "resource_url": state["resource_url"],
            "text": text,
            "caption": episode_summary or caption,
            "episode_summary": episode_summary,
            "episode_items": episode_items,
            "message_indices": message_indices,
            "message_happened_at_map": plan_message_happened_at_map,
            "entries": structured_entries,
            "episode_id": episode_id,
            "segment_id": segment_id or episode_id,
            "memory_retrieve_history": state.get("memory_retrieve_history"),
            "memory_prior_context": state.get("memory_prior_context"),
            "episode_messages": episode_messages,
        }
        episode_plans.append(plan)

        state["episode_plans"] = episode_plans
        if skipped_reasons:
            state["skipped_reasons"] = skipped_reasons
        return state

    async def _memorize_dedupe_merge(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        return cast(
            WorkflowState,
            await dedupe._memorize_dedupe_merge(
                cast(dict[str, Any], state),
                step_context,
                semantic_dedupe_enabled=self.memorize_config.semantic_dedupe_enabled,
                semantic_dedupe_similarity_threshold=self.memorize_config.semantic_dedupe_similarity_threshold,
                get_llm_client=self._get_llm_client,
            ),
        )

    @staticmethod
    def _dedupe_summary_tokens(summary: Any) -> set[str]:
        return dedupe._dedupe_summary_tokens(summary)

    def _prefilter_dedupe_candidate_ids(
        self,
        *,
        anchor_id: str,
        anchor: MemoryItem,
        active_pool: Mapping[str, MemoryItem],
        merged_map: Mapping[str, str],
        summary_tokens: Mapping[str, set[str]],
        token_index: Mapping[str, set[str]],
        token_freq: Mapping[str, int],
    ) -> list[str]:
        return dedupe._prefilter_dedupe_candidate_ids(
            anchor_id=anchor_id,
            anchor=anchor,
            active_pool=active_pool,
            merged_map=merged_map,
            summary_tokens=summary_tokens,
            token_index=token_index,
            token_freq=token_freq,
        )

    def _cluster_homeless_entries(
        self,
        *,
        filtered_entries: list[StructuredMemoryEntry],
        per_entry_unknowns: Sequence[list[str]],
        item_embeddings: Sequence[Any] | None = None,
    ) -> tuple[list[HomelessCategoryCluster], dict[int, str]]:
        return cast(
            tuple[list[HomelessCategoryCluster], dict[int, str]],
            categories._cluster_homeless_entries(
                filtered_entries=cast(list[Any], filtered_entries),
                per_entry_unknowns=per_entry_unknowns,
                item_embeddings=item_embeddings,
                normalize_embedding_vector=dedupe._normalize_embedding_vector,
                cosine_similarity=dedupe._cosine_similarity,
                cluster_factory=HomelessCategoryCluster,
                cluster_similarity_threshold=categories._dynamic_category_cluster_threshold(),
                cluster_min_size=categories._dynamic_category_cluster_min_size(
                    getattr(self.memorize_config, "dynamic_category_cluster_size", 3),
                ),
            ),
        )

    async def _plan_dynamic_categories(
        self,
        *,
        ctx: Context,
        store: Database,
        strong_clusters: Sequence[HomelessCategoryCluster],
        ungrouped_unknown_counts: Mapping[str, int],
        ungrouped_unknown_examples: Mapping[str, list[str]],
        min_mentions: int,
        policy: str,
        default_desc: str,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        return cast(
            tuple[dict[str, str], dict[str, str], dict[str, str]],
            await categories._plan_dynamic_categories(
                ctx=ctx,
                store=store,
                strong_clusters=cast(Sequence[Any], strong_clusters),
                ungrouped_unknown_counts=ungrouped_unknown_counts,
                ungrouped_unknown_examples=ungrouped_unknown_examples,
                min_mentions=min_mentions,
                policy=policy,
                default_desc=default_desc,
                fallback_category_configs=self.memorize_config.memory_categories or [],
                get_llm_client=self._get_llm_client,
                planner_profile=getattr(self.memorize_config, "category_update_llm_profile", "default"),
                normalize_category_name=self._normalize_category_name,
                dynamic_category_cluster_min_size=categories._dynamic_category_cluster_min_size(
                    getattr(self.memorize_config, "dynamic_category_cluster_size", 3),
                ),
            ),
        )

    async def _process_plan(
        self,
        plan: dict[str, Any],
        *,
        modality: str,
        local_path: str | None,
        ctx: Any,
        store: Any,
        embed_client: Any,
        user_scope: dict[str, Any],
        conversation_id: str | None,
        items: list[MemoryItem],
        relations: list[CategoryItem],
        category_updates: dict[str, list[tuple[str, str]]],
        pending_episode_ids: list[str],
        session: Any = None,
    ) -> tuple[list[Resource], int]:
        kwargs: dict[str, Any] = {}
        if session is not None:
            kwargs["session"] = session

        episode_local_path = local_path
        episode_messages = plan.get("episode_messages") or []
        if episode_messages:
            episode_file = pathlib.Path(self.fs.base) / f"{pathlib.Path(plan['resource_url']).stem}.jsonl"
            lines = [json.dumps(msg, ensure_ascii=False) for msg in episode_messages]
            episode_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            episode_local_path = str(episode_file)
        elif isinstance(plan.get("text"), str) and plan["text"].strip():
            episode_file = pathlib.Path(self.fs.base) / f"{pathlib.Path(plan['resource_url']).stem}.txt"
            episode_file.write_text(plan["text"], encoding="utf-8")
            episode_local_path = str(episode_file)

        segment_id = str(plan.get("segment_id") or plan.get("episode_id") or "").strip() or None
        message_happened_at_map = plan.get("message_happened_at_map")
        happened_at_value: Any | None = None
        memory_date: str | None = None
        if isinstance(message_happened_at_map, Mapping):
            for message_idx in sorted(message_happened_at_map):
                happened_at_candidate = message_happened_at_map.get(message_idx)
                if happened_at_candidate is None:
                    continue
                happened_at_value = happened_at_candidate
                match = re.search(r"\d{4}-\d{2}-\d{2}", str(happened_at_candidate))
                if match:
                    memory_date = match.group(0)
                break

        res = await self._create_resource_with_caption(
            resource_url=plan["resource_url"],
            modality=modality,
            local_path=episode_local_path,
            caption=plan.get("caption"),
            store=store,
            embed_client=embed_client,
            user=user_scope,
            episode_id=segment_id,
            conversation_id=conversation_id,
            memory_retrieve_history=plan.get("memory_retrieve_history"),
            memory_prior_context=plan.get("memory_prior_context"),
            **kwargs,
        )

        episode_summary_text = str(plan.get("episode_summary") or "").strip()
        raw_episode_items = plan.get("episode_items")
        episode_items: list[dict[str, str]] = []
        if isinstance(raw_episode_items, list):
            for row in raw_episode_items:
                if not isinstance(row, Mapping):
                    continue
                title = str(row.get("title") or "").strip()
                summary = str(row.get("summary") or "").strip()
                if not summary:
                    continue
                episode_items.append({"title": title or "Story", "summary": summary})
        if not episode_items and episode_summary_text:
            episode_items = [{"title": "Story", "summary": episode_summary_text}]
        if episode_items and res.embedding is not None:
            for episode_item in episode_items:
                title = str(episode_item.get("title") or "").strip() or "Story"
                summary = str(episode_item.get("summary") or "").strip()
                if not summary:
                    continue
                extra_payload: dict[str, Any] = {"episode_item_title": title}
                if segment_id:
                    extra_payload["segment_id"] = segment_id
                if memory_date:
                    extra_payload["memory_date"] = memory_date
                summary_item = store.memory_item_repo.create_item(
                    resource_id=res.id,
                    memory_type="episode",
                    source_role="environment",
                    summary=f"{title}: {summary}",
                    embedding=res.embedding,
                    user_data=dict(user_scope or {}),
                    conversation_id=conversation_id,
                    episode_id=segment_id,
                    happened_at=happened_at_value,
                    extra=extra_payload,
                    **({"session": session} if session is not None else {}),
                )
                items.append(summary_item)
                exp_ids = self._map_category_names_to_ids(["Experiences"], ctx)
                for cid in exp_ids:
                    rel_kwargs = {"item_id": summary_item.id, "category_id": cid, "user_data": dict(user_scope or {})}
                    if session is not None:
                        rel = cast(Any, store.category_item_repo).link_item_category(**rel_kwargs, session=session)
                    else:
                        rel = store.category_item_repo.link_item_category(**rel_kwargs)
                    relations.append(rel)
                    category_updates.setdefault(cid, []).append((summary_item.id, summary_item.summary))

        entries = plan.get("entries") or []
        episode_id = str(plan.get("episode_id") or "").strip()
        if episode_id:
            pending_episode_ids.append(episode_id)
        if not entries:
            return [res], 0

        persist_kwargs: dict[str, Any] = {}
        if session is not None:
            persist_kwargs["session"] = session
        mem_items, rels, cat_updates, homeless_delta = await self._persist_memory_items(
            resource_id=res.id,
            structured_entries=entries,
            ctx=ctx,
            store=store,
            embed_client=embed_client,
            user=user_scope,
            conversation_id=conversation_id,
            episode_id=segment_id,
            message_happened_at_map=message_happened_at_map,
            **persist_kwargs,
        )
        items.extend(mem_items)
        relations.extend(rels)
        for cat_id, mems in cat_updates.items():
            category_updates.setdefault(cat_id, []).extend(mems)
        return [res], homeless_delta

    async def _memorize_categorize_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        embed_client = self._get_step_embedding_client(step_context)
        ctx = state["ctx"]
        store = state["store"]
        modality = state["modality"]
        local_path = state["local_path"]
        resources: list[Resource] = []
        items: list[MemoryItem] = []
        relations: list[CategoryItem] = []
        category_updates: dict[str, list[tuple[str, str]]] = {}
        pending_episode_ids: list[str] = []
        user_scope = state.get("user", {})
        homeless_item_count = 0

        common = dict(
            modality=modality,
            local_path=local_path,
            ctx=ctx,
            store=store,
            embed_client=embed_client,
            user_scope=user_scope,
            conversation_id=state.get("conversation_id"),
            items=items,
            relations=relations,
            category_updates=category_updates,
            pending_episode_ids=pending_episode_ids,
        )

        session_cm = self._sqlite_write_session(store)
        if session_cm is not None:
            with session_cm as session:
                try:
                    for plan in state.get("episode_plans", []):
                        plan_resources, delta = await self._process_plan(plan, session=session, **common)
                        resources.extend(plan_resources)
                        homeless_item_count += delta
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
        else:
            for plan in state.get("episode_plans", []):
                plan_resources, delta = await self._process_plan(plan, **common)
                resources.extend(plan_resources)
                homeless_item_count += delta

        state.update({
            "resources": resources,
            "items": items,
            "relations": relations,
            "category_updates": category_updates,
            "homeless_item_count": homeless_item_count,
            "pending_episode_ids": list(dict.fromkeys(x for x in pending_episode_ids if x)),
        })
        return state

    async def _memorize_persist_and_index(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        updated_summaries = await self._update_category_summaries(
            state.get("category_updates", {}),
            ctx=state["ctx"],
            store=state["store"],
            llm_client=llm_client,
            user=state.get("user"),
        )
        if self.memorize_config.enable_item_references:
            await self._persist_item_references(
                updated_summaries=updated_summaries,
                category_updates=state.get("category_updates", {}),
                store=state["store"],
            )
        return state

    def _memorize_build_response(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        ctx = state["ctx"]
        store = state["store"]
        resources = [self._model_dump_without_embeddings(r) for r in state.get("resources", [])]
        active_items = [item for item in state.get("items", []) if not dedupe._is_merged_item(item)]
        active_item_ids = {getattr(item, "id", None) for item in active_items}
        items = [self._model_dump_without_embeddings(item) for item in active_items]
        relations = [
            rel.model_dump() for rel in state.get("relations", []) if getattr(rel, "item_id", None) in active_item_ids
        ]
        category_ids = state.get("category_ids") or list(ctx.category_ids)
        categories = [
            self._model_dump_without_embeddings(store.memory_category_repo.categories[c]) for c in category_ids
        ]

        if len(resources) == 1:
            response = {
                "resource": resources[0],
                "items": items,
                "categories": categories,
                "relations": relations,
                "pending_episode_ids": state.get("pending_episode_ids", []),
            }
        else:
            response = {
                "resources": resources,
                "items": items,
                "categories": categories,
                "relations": relations,
                "pending_episode_ids": state.get("pending_episode_ids", []),
            }
        skipped = state.get("skipped_reasons")
        if skipped:
            response["skipped_reasons"] = skipped
        state["response"] = response
        return state

    async def _create_resource_with_caption(
        self,
        *,
        resource_url: str,
        modality: str,
        local_path: str,
        caption: str | None,
        store: Database,
        embed_client: Any | None = None,
        user: Mapping[str, Any] | None = None,
        episode_id: str | None = None,
        conversation_id: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
        session: Any | None = None,
    ) -> Resource:
        return cast(
            Resource,
            await persistence._create_resource_with_caption(
                resource_url=resource_url,
                modality=modality,
                local_path=local_path,
                caption=caption,
                store=store,
                embed_client=embed_client,
                get_embedding_client=self._get_llm_client,
                user=user,
                episode_id=episode_id,
                conversation_id=conversation_id,
                memory_retrieve_history=memory_retrieve_history,
                memory_prior_context=memory_prior_context,
                session=session,
            ),
        )

    @staticmethod
    def _sqlite_write_session(store: Database) -> Any | None:
        return persistence._sqlite_write_session(store)

    def _resolve_memory_types(self) -> list[MemoryType]:
        configured_types = self.memorize_config.memory_types or DEFAULT_MEMORY_TYPES
        return [cast(MemoryType, mtype) for mtype in configured_types]

    @staticmethod
    def _resolve_custom_prompt(prompt: str | CustomPrompt, templates: Mapping[str, str]) -> str:
        if isinstance(prompt, str):
            return prompt
        valid_blocks = [
            (block.ordinal, name, block.prompt or templates.get(name))
            for name, block in prompt.items()
            if (block.ordinal >= 0 and (block.prompt or templates.get(name)))
        ]
        if not valid_blocks:
            return ""
        sorted_blocks = sorted(valid_blocks)
        return "\n\n".join(block for (_, _, block) in sorted_blocks if block is not None)

    async def _generate_structured_entries(
        self,
        *,
        modality: str,
        store: Database,
        memory_types: list[MemoryType],
        text: str | None,
        categories_prompt_str: str,
        all_categories_summary: str | None = None,
        soul_card: str | None = None,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
        llm_client: Any | None = None,
        skipped_reasons: list[str] | None = None,
    ) -> list[StructuredMemoryEntry]:
        if not memory_types or not text:
            return []
        client = llm_client or self._get_llm_client()
        return await self._generate_entries_from_text(
            resource_text=text,
            store=store,
            memory_types=memory_types,
            categories_prompt_str=categories_prompt_str,
            all_categories_summary=all_categories_summary,
            soul_card=soul_card,
            speaker_roster=speaker_roster,
            default_source_message_ids=self._extract_message_indices(text)
            if modality == "conversation"
            else None,
            llm_client=client,
        )

    async def _route_episode(
        self,
        episode_text: str,
        memory_types: list[MemoryType],
        llm_client: Any | None = None,
        soul_card: str | None = None,
        skipped_reasons: list[str] | None = None,
    ) -> tuple[list[MemoryType], str | None, list[dict[str, str]]]:
        if not memory_types:
            return [], None, []
        client = llm_client or self._get_llm_client()
        prompt = ROUTER_PROMPT.format(
            episode=episode_text,
            allowed_types=list(memory_types),
            soul_card=soul_card or "You are a continuous being, not a session that resets. You carry intentions, working thoughts, and long-term memory. Your self-knowledge will grow as you live and reflect.",
        )
        raw = await client.chat(prompt)
        if isinstance(raw, str):
            raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```\s*$", "", raw, count=1)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = json.loads(self._extract_json_blob(raw))
        if not isinstance(payload, dict):
            logger.warning("Router returned non-dict payload, skipping episode: %s", type(payload).__name__)
            if skipped_reasons is not None:
                skipped_reasons.append("router returned non-dict payload")
            return [], None, []
        excluded_types = payload.get("excluded_types")
        reason = payload.get("reason", "")
        episode_summary = str(payload.get("episode_summary") or "").strip() or None
        episode_items: list[dict[str, str]] = []
        raw_episode_items = payload.get("episode_items")
        if isinstance(raw_episode_items, list):
            for row in raw_episode_items:
                if not isinstance(row, Mapping):
                    continue
                summary = str(row.get("summary") or "").strip()
                if not summary:
                    continue
                title = str(row.get("title") or "").strip() or "Story"
                episode_items.append({"title": title, "summary": summary})
        legacy_episode_item = str(payload.get("episode_item") or "").strip()
        if not episode_items and legacy_episode_item:
            episode_items = [{"title": "Story", "summary": legacy_episode_item}]
        if not episode_items and episode_summary:
            episode_items = [{"title": "Story", "summary": episode_summary}]
        try:
            max_items = int(getattr(self.memorize_config, "episodes_per_segment", 3) or 3)
        except (TypeError, ValueError):
            max_items = 3
        episode_items = episode_items[: max(1, max_items)]
        if excluded_types is None:
            excluded_types = []
        if not isinstance(excluded_types, list):
            logger.warning("Router returned invalid excluded_types, skipping episode")
            if skipped_reasons is not None:
                skipped_reasons.append("router returned invalid excluded_types")
            return [], episode_summary, episode_items
        excluded = {
            routed_type
            for routed_type in excluded_types
            if isinstance(routed_type, str) and routed_type in set(memory_types)
        }
        allowed_types = set(memory_types) - excluded
        routed_types = [mtype for mtype in memory_types if mtype in allowed_types]
        if not routed_types and skipped_reasons is not None and isinstance(reason, str) and reason.strip():
            skipped_reasons.append(reason.strip())
        return routed_types, episode_summary, episode_items

    async def _generate_entries_from_text(
        self,
        *,
        resource_text: str,
        store: Database,
        memory_types: list[MemoryType],
        categories_prompt_str: str,
        all_categories_summary: str | None = None,
        soul_card: str | None = None,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
        default_source_message_ids: list[int] | None = None,
        llm_client: Any | None = None,
        target_items_by_type: Mapping[str, str] | None = None,
        require_episode_ref: bool = False,
    ) -> list[StructuredMemoryEntry]:
        if not memory_types:
            return []
        client = llm_client or self._get_llm_client()
        soul_context_str = self._format_soul_context_for_prompt(
            store,
            all_categories_summary=all_categories_summary,
            soul_card=soul_card,
        )
        typed_prompts = [
            (mtype, self._build_memory_type_prompt(
                memory_type=mtype,
                resource_text=resource_text,
                categories_str=categories_prompt_str,
                soul_context_str=soul_context_str,
                speaker_roster=speaker_roster,
                target_items=(target_items_by_type or {}).get(mtype, ""),
            ))
            for mtype in memory_types
        ]
        valid_pairs = [(mtype, prompt) for mtype, prompt in typed_prompts if prompt.strip()]
        tasks = [client.chat(prompt) for _, prompt in valid_pairs]
        responses = await asyncio.gather(*tasks)
        return self._parse_structured_entries(
            [mtype for mtype, _ in valid_pairs],
            responses,
            default_source_message_ids=default_source_message_ids,
            speaker_roster=speaker_roster,
            require_episode_ref=require_episode_ref,
        )

    @staticmethod
    def _normalize_category_name(raw: str) -> str | None:
        if not isinstance(raw, str):
            return None
        s = raw.strip().lower()
        if not s:
            return None
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s or None

    def _parse_structured_entries(
        self,
        memory_types: list[MemoryType],
        responses: Sequence[str],
        *,
        default_source_message_ids: list[int] | None = None,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
        require_episode_ref: bool = False,
    ) -> list[StructuredMemoryEntry]:
        entries: list[StructuredMemoryEntry] = []
        for mtype, response in zip(memory_types, responses, strict=True):
            parsed = self._parse_memory_type_response_xml(response)
            for entry in parsed:
                content = (entry.get("content") or "").strip()
                if not content:
                    continue
                source_role_raw = entry.get("source_role")
                source_role = None
                if isinstance(source_role_raw, str):
                    normalized_role = source_role_raw.strip().lower()
                    if normalized_role in {"soul", "user", "peer", "entity", "environment"}:
                        source_role = normalized_role
                parsed_speaker_id, parsed_speaker_label = self._parse_speaker_ref(
                    entry.get("speaker_ref"),
                    speaker_roster,
                )

                confidence = None
                confidence_raw = entry.get("confidence")
                if confidence_raw is not None:
                    try:
                        parsed_confidence = float(confidence_raw)
                    except (TypeError, ValueError):
                        parsed_confidence = None
                    if parsed_confidence is not None and 0.0 <= parsed_confidence <= 1.0:
                        confidence = parsed_confidence

                # Prompts no longer request source_message_ids (02d8bde).
                # The resolver treats None / [] / malformed input as "LLM emitted nothing"
                # and falls back to the full episode range.
                source_message_ids = self._resolve_source_message_ids(
                    entry.get("source_message_ids"),
                    default_source_message_ids,
                )

                reflection_salience = self._normalize_reflection_salience(entry.get("reflection_salience"))
                emotional_intensity = self._normalize_reflection_salience(entry.get("emotional_intensity"))
                replaces_previous_fact = self._normalize_replaces_previous_fact(entry.get("replaces_previous_fact"))
                entities = entry.get("entities")
                episode_ref = self._parse_episode_ref(entry.get("episode_ref"))
                if require_episode_ref and episode_ref is None:
                    logger.warning("Dropped extracted item without valid episode_ref for memory_type=%s", mtype)
                    continue

                raw_cats = [c for c in (entry.get("categories", []) or []) if isinstance(c, str)]
                cat_names = []
                seen = set()
                for c in raw_cats:
                    n = self._normalize_category_name(c)
                    if n and n not in seen:
                        cat_names.append(n)
                        seen.add(n)
                entries.append(
                    StructuredMemoryEntry(
                        mtype,
                        content,
                        cat_names,
                        source_role,
                        confidence,
                        source_message_ids,
                        reflection_salience,
                        emotional_intensity,
                        replaces_previous_fact,
                        entities,
                        parsed_speaker_id,
                        parsed_speaker_label,
                        episode_ref,
                    )
                )
        return self._prune_extracted_entry_duplicates(entries)

    def _prune_extracted_entry_duplicates(
        self,
        entries: list[StructuredMemoryEntry],
    ) -> list[StructuredMemoryEntry]:
        if len(entries) < 2:
            return entries

        seen_exact: set[tuple[str, str | None, str]] = set()
        kept: list[StructuredMemoryEntry] = []
        for entry in entries:
            normalized_summary = re.sub(r"\s+", " ", (entry.content or "").strip())
            exact_key = (entry.memory_type, entry.source_role, normalized_summary.casefold())
            if exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)

            kept.append(entry._replace(content=normalized_summary))

        return kept

    def _decorate_entries_with_plan_context(
        self,
        entries: list[StructuredMemoryEntry],
        *,
        message_indices: list[int],
    ) -> list[StructuredMemoryEntry]:
        return cast(list[StructuredMemoryEntry], speakers._decorate_entries_with_plan_context(entries, message_indices=message_indices))

    async def _maybe_create_dynamic_categories(
        self,
        *,
        structured_entries: list[StructuredMemoryEntry],
        item_embeddings: Sequence[Any] | None = None,
        ctx: Context,
        store: Database,
        embed_client: Any,
        user: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        return cast(
            list[StructuredMemoryEntry],
            await categories._maybe_create_dynamic_categories(
                structured_entries=cast(list[Any], structured_entries),
                item_embeddings=item_embeddings,
                ctx=ctx,
                store=store,
                embed_client=embed_client,
                user=user,
                session=session,
                ensure_categories_ready=self._ensure_categories_ready,
                normalize_category_name=self._normalize_category_name,
                cluster_homeless_entries=self._cluster_homeless_entries,
                plan_dynamic_categories=self._plan_dynamic_categories,
                max_categories_total=getattr(self.memorize_config, "max_categories_total", 0),
                dynamic_category_cluster_size=getattr(self.memorize_config, "dynamic_category_cluster_size", 3),
                dynamic_category_policy=getattr(self.memorize_config, "dynamic_category_policy", ""),
                dynamic_category_description=getattr(self.memorize_config, "dynamic_category_description", ""),
            ),
        )

    async def _persist_memory_items(
        self,
        *,
        resource_id: str,
        structured_entries: list[StructuredMemoryEntry],
        ctx: Context,
        store: Database,
        embed_client: Any | None = None,
        user: Mapping[str, Any] | None = None,
        conversation_id: str | None = None,
        episode_id: str | None = None,
        message_happened_at_map: Mapping[int, Any] | None = None,
        session: Any | None = None,
    ) -> tuple[list[MemoryItem], list[CategoryItem], dict[str, list[tuple[str, str]]], int]:
        items, rels, category_memory_updates, homeless_count = await persistence._persist_memory_items(
            resource_id=resource_id,
            structured_entries=cast(list[Any], structured_entries),
            ctx=ctx,
            store=store,
            embed_client=embed_client,
            get_llm_client=self._get_llm_client,
            user=user,
            conversation_id=conversation_id,
            episode_id=episode_id,
            message_happened_at_map=message_happened_at_map,
            session=session,
            maybe_create_dynamic_categories=self._maybe_create_dynamic_categories,
            enable_confidence_normalization=self.memorize_config.enable_confidence_normalization,
            normalize_confidence=lambda entries: cast(list[Any], self._normalize_confidence(cast(list[StructuredMemoryEntry], entries))),
            find_supersede_targets=self._find_supersede_targets,
            hedge_summary_for_confidence=self._hedge_summary_for_confidence,
            resolve_entry_happened_at=self._resolve_entry_happened_at,
            map_category_names_to_ids=self._map_category_names_to_ids,
        )
        return (
            cast(list[MemoryItem], items),
            cast(list[CategoryItem], rels),
            category_memory_updates,
            homeless_count,
        )

    def _supersede_similarity_threshold(self) -> float:
        return dedupe._supersede_similarity_threshold(
            getattr(self.memorize_config, "supersede_similarity_threshold", 0.75)
        )

    async def _find_supersede_targets(
        self,
        *,
        structured_entries: list[StructuredMemoryEntry],
        store: Database,
        embed_client: Any,
        user: Mapping[str, Any] | None = None,
    ) -> dict[int, str]:
        return cast(
            dict[int, str],
            await dedupe._find_supersede_targets(
                structured_entries=cast(list[Any], structured_entries),
                store=store,
                embed_client=embed_client,
                user=user,
                threshold=self._supersede_similarity_threshold(),
            ),
        )

    @staticmethod
    def _category_scope_key(user_scope: Mapping[str, Any] | None) -> str:
        return categories._category_scope_key(user_scope)

    async def _ensure_categories_ready(
        self, ctx: Context, store: Database, user_scope: Mapping[str, Any] | None = None
    ) -> None:
        await categories._ensure_categories_ready(
            ctx,
            store,
            user_scope,
            category_scope_key=self._category_scope_key,
            initialize_categories=self._initialize_categories,
        )

    async def _initialize_categories(
        self,
        ctx: Context,
        store: Database,
        user: Mapping[str, Any] | None = None,
        *,
        scope_key: str | None = None,
    ) -> None:
        await categories._initialize_categories(
            ctx,
            store,
            user,
            scope_key=scope_key,
            category_scope_key=self._category_scope_key,
            category_configs=self.category_configs,
            get_embedding_client=self._get_llm_client,
            category_embedding_text=self._category_embedding_text,
        )

    @staticmethod
    def _category_embedding_text(cat: CategoryConfig) -> str:
        return categories._category_embedding_text(cat)

    def _map_category_names_to_ids(self, names: list[str], ctx: Context) -> list[str]:
        return categories._map_category_names_to_ids(names, ctx)

    async def _split_into_episodes(
        self, *, local_path: str, text: str | None, modality: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        return await episode_helpers._split_into_episodes(
            local_path=local_path,
            text=text,
            modality=modality,
            memorize_config=self.memorize_config,
            preprocess_prompts=PREPROCESS_PROMPTS,
            resolve_custom_prompt=self._resolve_custom_prompt,
            prepare_audio_text=self._prepare_audio_text,
            modality_requires_text=self._modality_requires_text,
            dispatch_preprocessor=self._dispatch_preprocessor,
            llm_client=llm_client,
        )

    async def _prepare_audio_text(self, local_path: str, text: str | None, llm_client: Any | None = None) -> str | None:
        return await episode_helpers._prepare_audio_text(
            local_path,
            text,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
        )

    def _modality_requires_text(self, modality: str) -> bool:
        return episode_helpers._modality_requires_text(modality)

    async def _dispatch_preprocessor(
        self,
        *,
        modality: str,
        local_path: str,
        text: str | None,
        template: str,
        llm_client: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await episode_helpers._dispatch_preprocessor(
            modality=modality,
            local_path=local_path,
            text=text,
            template=template,
            llm_client=llm_client,
            split_conversation_into_episodes=self._split_conversation_into_episodes,
            preprocess_video=self._preprocess_video,
            preprocess_image=self._preprocess_image,
            preprocess_document=self._preprocess_document,
            preprocess_audio=self._preprocess_audio,
        )

    async def _split_conversation_into_episodes(
        self, conversation_raw_text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        return await episode_helpers._split_conversation_into_episodes(
            conversation_raw_text=conversation_raw_text,
            template=template,
            llm_client=llm_client,
            memorize_config=self.memorize_config,
            escape_prompt_value=self._escape_prompt_value,
            get_llm_client=self._get_llm_client,
            parse_conversation_preprocess_with_episodes=self._parse_conversation_preprocess_with_episodes,
            extract_message_indices=self._extract_message_indices,
            dedupe_message_indices=self._dedupe_message_indices,
            summarize_episode=self._summarize_episode,
        )

    async def _summarize_episode(self, episode_text: str, llm_client: Any | None = None) -> str | None:
        return await episode_helpers._summarize_episode(
            episode_text=episode_text,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
        )

    async def _preprocess_video(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await episode_helpers._preprocess_video(
            local_path=local_path,
            template=template,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
            parse_multimodal_response=self._parse_multimodal_response,
        )

    async def _preprocess_image(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await episode_helpers._preprocess_image(
            local_path=local_path,
            template=template,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
            parse_multimodal_response=self._parse_multimodal_response,
        )

    async def _preprocess_document(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await episode_helpers._preprocess_document(
            text=text,
            template=template,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
            escape_prompt_value=self._escape_prompt_value,
            parse_multimodal_response=self._parse_multimodal_response,
        )

    async def _preprocess_audio(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await episode_helpers._preprocess_audio(
            text=text,
            template=template,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
            escape_prompt_value=self._escape_prompt_value,
            parse_multimodal_response=self._parse_multimodal_response,
        )

    def _format_categories_for_prompt(self, categories: list[CategoryConfig]) -> str:
        if not categories:
            base = "No categories provided."
        else:
            lines = []
            for cat in categories:
                name = cat.name.strip() or "Untitled"
                desc = cat.description.strip()
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
            base = "\n".join(lines)

        max_total = int(getattr(self.memorize_config, "max_categories_total", 0) or 0)
        policy = str(getattr(self.memorize_config, "dynamic_category_policy", "") or "").strip()
        note = "\n\n" + (policy + "\n\n" if policy else "")
        note += (
            "If none of the existing categories fit, you may propose a NEW category name. "
            "Keep it broad (a life domain), not a specific event. "
            f"Max total categories: {max_total or 'unlimited'}."
        )
        return base + note

    def _format_soul_context_for_prompt(
        self,
        store: Database,
        *,
        all_categories_summary: str | None = None,
        soul_card: str | None = None,
    ) -> str:
        sections: list[str] = []
        for category in store.memory_category_repo.categories.values():
            summary = str(category.summary or "").strip()
            if not summary:
                continue
            name = str(category.name or "").strip() or "Unnamed Category"
            if summary.lstrip().startswith(f"# {name}"):
                sections.append(summary)
            else:
                sections.append(f"## {name}\n{summary}")
        card = str(soul_card or "").strip()
        if card:
            sections.append(f"## Soul Card\n{card}")
        if not sections:
            return "No prior knowledge about these participants exists yet."
        return "\n\n".join(sections)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        return episode_helpers._estimate_text_tokens(text)

    @staticmethod
    def _compute_batch_max_items(total_message_count: int) -> int:
        return episode_helpers._compute_batch_max_items(total_message_count)

    def _build_batch_extraction_text(self, episodes: Sequence[Mapping[str, Any]]) -> str:
        return episode_helpers._build_batch_extraction_text(
            episodes=episodes,
            parse_episode_ref=self._parse_episode_ref,
            dedupe_message_indices=self._dedupe_message_indices,
        )

    @staticmethod
    def _message_is_primary_for_memorize(message: Mapping[str, Any]) -> bool:
        return episode_helpers._message_is_primary_for_memorize(message)

    @staticmethod
    def _message_index_for_sort(message: Mapping[str, Any]) -> int:
        return episode_helpers._message_index_for_sort(message)

    @staticmethod
    def _format_episode_message_line(message: Mapping[str, Any]) -> str:
        return episode_helpers._format_episode_message_line(message)

    async def _summarize_background_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
    ) -> str | None:
        return await episode_helpers._summarize_background_messages(
            messages=messages,
            llm_client=llm_client,
            summarize_episode=self._summarize_episode,
        )

    async def summarize_background_chat_rollup(
        self,
        *,
        prior_summary: str | None,
        messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
    ) -> str:
        return await episode_helpers._summarize_background_rollup(
            prior_summary=prior_summary,
            messages=messages,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
        )

    async def _summarize_background_groups_batched(
        self,
        *,
        grouped_messages: Mapping[str, Sequence[Mapping[str, Any]]],
        group_order: Sequence[str],
        llm_client: Any | None = None,
    ) -> dict[str, str]:
        return await episode_helpers._summarize_background_groups_batched(
            grouped_messages=grouped_messages,
            group_order=group_order,
            llm_client=llm_client,
            get_llm_client=self._get_llm_client,
            extract_json_blob=self._extract_json_blob,
        )

    async def _render_episode_with_background_context(
        self,
        *,
        primary_messages: Sequence[Mapping[str, Any]],
        background_messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        return await episode_helpers._render_episode_with_background_context(
            primary_messages=primary_messages,
            background_messages=background_messages,
            llm_client=llm_client,
            summarize_background_rollup=self.summarize_background_chat_rollup,
            summarize_background_groups_batched=self._summarize_background_groups_batched,
            memorize_config=self.memorize_config,
        )

    def _render_episode_with_summary_rows(
        self,
        *,
        primary_messages: Sequence[Mapping[str, Any]],
        summary_rows: Sequence[Mapping[str, Any]],
    ) -> str:
        return episode_helpers._render_episode_with_summary_rows(
            primary_messages=primary_messages,
            summary_rows=summary_rows,
        )

    @staticmethod
    def _normalize_confidence(
        entries: list[StructuredMemoryEntry],
        target_mean: float = 0.70,
        target_std: float = 0.15,
        compression_threshold: float = 0.08,
    ) -> list[StructuredMemoryEntry]:
        raw = [e.confidence for e in entries if e.confidence is not None]
        if len(raw) < 6:
            return entries
        mean = sum(raw) / len(raw)
        std = math.sqrt(sum((v - mean) ** 2 for v in raw) / len(raw))
        if std >= compression_threshold or std == 0:
            return entries
        result: list[StructuredMemoryEntry] = []
        for entry in entries:
            if entry.confidence is None:
                result.append(entry)
                continue
            normalized = round(target_mean + (entry.confidence - mean) / std * target_std, 2)
            if 0.0 <= normalized <= 1.0:
                result.append(entry._replace(confidence=normalized))
            else:
                result.append(entry._replace(confidence=None))
        return result

    def _build_memory_type_prompt(
        self,
        *,
        memory_type: MemoryType,
        resource_text: str,
        categories_str: str,
        soul_context_str: str,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
        target_items: str = "",
    ) -> str:
        configured_prompt = self.memorize_config.memory_type_prompts.get(memory_type)
        if configured_prompt is None:
            template = MEMORY_TYPE_PROMPTS.get(memory_type)
        elif isinstance(configured_prompt, str):
            template = configured_prompt
        else:
            template = self._resolve_custom_prompt(
                configured_prompt, MEMORY_TYPE_CUSTOM_PROMPTS.get(memory_type, CUSTOM_TYPE_CUSTOM_PROMPTS)
            )
        if not template:
            return resource_text

        template = re.sub(r"(?im)^.*do not create new memory categories.*\n?", "", template)
        safe_resource = self._escape_prompt_value(resource_text)
        safe_categories = self._escape_prompt_value(categories_str)
        safe_soul_context = self._escape_prompt_value(soul_context_str)
        speaker_roster_block = self._format_speaker_roster_block_for_prompt(speaker_roster)
        rendered = template.format(
            resource=safe_resource,
            categories_str=safe_categories,
            soul_context=safe_soul_context,
            speaker_roster_block=speaker_roster_block,
            target_items=target_items,
        )
        if not target_items:
            rendered = re.sub(r"\*\*Target:[^*]*\*\*\s*", "", rendered)
        if not speaker_roster_block:
            while "\n\n\n" in rendered:
                rendered = rendered.replace("\n\n\n", "\n\n")
        return rendered

    def _build_item_ref_id(self, item_id: str) -> str:
        return persistence._build_item_ref_id(item_id)

    async def _persist_item_references(
        self,
        *,
        updated_summaries: dict[str, str],
        category_updates: dict[str, list[tuple[str, str]]],
        store: Database,
    ) -> None:
        await persistence._persist_item_references(
            updated_summaries=updated_summaries,
            category_updates=category_updates,
            store=store,
            build_item_ref_id=self._build_item_ref_id,
        )

    @staticmethod
    def _looks_like_identifier_value(value: str) -> bool:
        return persistence._looks_like_identifier_value(value)

    def _summary_user_name(self, user_scope: Mapping[str, Any] | None, *, default: str) -> str:
        scope = user_scope or {}
        raw_name = scope.get("user_name")
        if raw_name is not None:
            explicit_name = str(raw_name).strip()
            if explicit_name:
                return explicit_name

        raw_user_id = scope.get("user_id")
        if raw_user_id is None:
            return default
        fallback = str(raw_user_id).strip()
        if not fallback:
            return default
        if self._looks_like_identifier_value(fallback):
            return default
        return fallback

    def _build_category_summary_prompt(
        self,
        *,
        category: MemoryCategory,
        new_memories: list[str] | list[tuple[str, str]],
        user: dict[str, Any] | None = None,
    ) -> str:
        return categories._build_category_summary_prompt(
            category=category,
            new_memories=new_memories,
            user=user,
            memorize_config=self.memorize_config,
            category_config_map=self.category_config_map,
            build_item_ref_id=self._build_item_ref_id,
            resolve_custom_prompt=self._resolve_custom_prompt,
            summary_user_name=lambda scope: self._summary_user_name(scope, default="the user"),
            escape_prompt_value=self._escape_prompt_value,
        )

    async def _update_category_summaries(
        self,
        updates: dict[str, list[tuple[str, str]]] | dict[str, list[str]],
        ctx: Context,
        store: Database,
        llm_client: Any | None = None,
        user: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        client = llm_client or self._get_llm_client()
        return await categories._update_category_summaries(
            updates,
            store=store,
            llm_client=client,
            user=user,
            build_category_summary_prompt=lambda category, memories, user_scope: self._build_category_summary_prompt(
                category=category,
                new_memories=memories,
                user=user_scope,
            ),
            summary_user_name=lambda scope: self._summary_user_name(scope, default=""),
        )

    @staticmethod
    def _dedupe_message_indices(values: Sequence[int | float | str]) -> list[int]:
        return parsing._dedupe_message_indices(values)

    def _resolve_source_message_ids(
        self,
        values: Any,
        allowed_values: Any = None,
    ) -> list[int]:
        """Resolve source IDs to the valid episode range, with code-owned fallback.

        This is the single normalization boundary for anything the LLM might
        emit under `source_message_ids`. Prompts stopped requesting the field
        in 02d8bde, so `values` is usually None or empty, but callers may still
        pass through raw extraction output. Non-iterable or malformed input is
        treated as "LLM emitted nothing"; the resolver falls back to the full
        allowed episode range so downstream provenance (speaker attribution,
        happened_at, retrieve rendering) stays populated.
        """
        return parsing._resolve_source_message_ids(values, allowed_values)

    @staticmethod
    def _extract_message_indices(text: str | None) -> list[int]:
        return parsing._extract_message_indices(text)

    @staticmethod
    def _extract_conversation_messages(raw_text: Any) -> list[tuple[int, dict[str, Any]]]:
        return parsing._extract_conversation_messages(raw_text)

    def _extract_message_happened_at_map(self, raw_text: Any) -> dict[int, Any]:
        return parsing._extract_message_happened_at_map(raw_text)

    def _build_speaker_roster(
        self,
        speaker_map: Mapping[int, tuple[str, str]] | None,
    ) -> list[SpeakerRosterEntry]:
        return speakers._build_speaker_roster(
            speaker_map,
            roster_entry_factory=lambda speaker_id, speaker_label, coarse_role: SpeakerRosterEntry(
                speaker_id, speaker_label, coarse_role
            ),
        )

    def _build_speaker_roster_if_ambiguous(
        self,
        speaker_map: Mapping[int, tuple[str, str]] | None,
    ) -> list[SpeakerRosterEntry] | None:
        return speakers._build_speaker_roster_if_ambiguous(
            speaker_map,
            roster_entry_factory=lambda speaker_id, speaker_label, coarse_role: SpeakerRosterEntry(
                speaker_id, speaker_label, coarse_role
            ),
        )

    def _list_declared_relationship_roster(
        self,
        *,
        store: Database,
        user: Mapping[str, Any] | None,
    ) -> list[SpeakerRosterEntry]:
        return speakers._list_declared_relationship_roster(
            store=store,
            user=user,
            roster_entry_factory=lambda speaker_id, speaker_label, coarse_role: SpeakerRosterEntry(
                speaker_id, speaker_label, coarse_role
            ),
        )

    def _build_speaker_roster_for_episode(
        self,
        *,
        speaker_map: Mapping[int, tuple[str, str]] | None,
        declared_entities: Sequence[SpeakerRosterEntry] | None,
        episode_text: Any,
    ) -> list[SpeakerRosterEntry] | None:
        return speakers._build_speaker_roster_for_episode(
            speaker_map=speaker_map,
            declared_entities=declared_entities,
            episode_text=episode_text,
            roster_entry_factory=lambda speaker_id, speaker_label, coarse_role: SpeakerRosterEntry(
                speaker_id, speaker_label, coarse_role
            ),
        )

    @staticmethod
    def _format_speaker_roster_block_for_prompt(
        speaker_roster: Sequence[SpeakerRosterEntry] | None,
    ) -> str:
        return speakers._format_speaker_roster_block_for_prompt(speaker_roster)

    @staticmethod
    def _sanitize_prompt_label(label: str) -> str:
        return speakers._sanitize_prompt_label(label)

    @staticmethod
    def _parse_speaker_ref(
        raw: Any,
        roster: Sequence[SpeakerRosterEntry] | None,
    ) -> tuple[str | None, str | None]:
        return speakers._parse_speaker_ref(raw, roster)

    def _build_speaker_map(
        self,
        episode_messages: Sequence[Mapping[str, Any]],
        scope: Mapping[str, Any] | None,
    ) -> dict[int, tuple[str, str]]:
        return speakers._build_speaker_map(episode_messages, scope)

    def _attribute_memory(
        self,
        memory: StructuredMemoryEntry,
        speaker_map: Mapping[int, tuple[str, str]] | None,
    ) -> StructuredMemoryEntry:
        return cast(StructuredMemoryEntry, speakers._attribute_memory(memory, speaker_map))

    def _resolve_entry_happened_at(
        self,
        source_message_ids: Sequence[int] | None,
        message_happened_at_map: Mapping[int, Any] | None,
    ) -> Any | None:
        return persistence._resolve_entry_happened_at(source_message_ids, message_happened_at_map)

    def _prepare_episode(
        self,
        *,
        modality: str,
        text: str | None,
        message_indices: Any = None,
    ) -> tuple[str | None, list[int]]:
        return episode_helpers._prepare_episode(
            modality=modality,
            text=text,
            message_indices=message_indices,
            dedupe_message_indices=self._dedupe_message_indices,
            extract_message_indices=self._extract_message_indices,
        )

    def _parse_multimodal_response(self, raw: str, content_tag: str, caption_tag: str) -> tuple[str | None, str | None]:
        return episode_helpers._parse_multimodal_response(
            raw,
            content_tag,
            caption_tag,
            extract_tag_content=self._extract_tag_content,
        )

    def _parse_conversation_preprocess_with_episodes(
        self, raw: str, original_text: str
    ) -> tuple[str | None, list[dict[str, Any]] | None]:
        return episode_helpers._parse_conversation_preprocess_with_episodes(
            raw,
            original_text,
            extract_tag_content=self._extract_tag_content,
            extract_episodes_with_fallback=self._extract_episodes_with_fallback,
        )

    def _extract_episodes_with_fallback(self, raw: str) -> list[dict[str, Any]] | None:
        return episode_helpers._extract_episodes_with_fallback(
            raw,
            extract_json_blob=self._extract_json_blob,
            episodes_from_json_payload=self._episodes_from_json_payload,
        )

    def _episodes_from_json_payload(self, payload: str) -> list[dict[str, Any]] | None:
        return episode_helpers._episodes_from_json_payload(
            payload,
            episodes_from_parsed_data=self._episodes_from_parsed_data,
        )

    @staticmethod
    def _episodes_from_parsed_data(parsed: Any) -> list[dict[str, Any]] | None:
        return episode_helpers._episodes_from_parsed_data(parsed)

    @staticmethod
    def _extract_tag_content(raw: str, tag: str) -> str | None:
        return episode_helpers._extract_tag_content(raw, tag)

    def _parse_memory_type_response_xml(self, raw: str) -> list[dict[str, Any]]:
        return parsing._parse_memory_type_response_xml(raw)
