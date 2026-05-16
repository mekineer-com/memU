from __future__ import annotations

import asyncio
import hashlib
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
from memu.app.settings import CategoryConfig, CustomPrompt
from memu.database.models import CategoryItem, MemoryCategory, MemoryItem, MemoryType, Resource, Triple
from memu.prompts.category_summary import (
    CUSTOM_PROMPT as CATEGORY_SUMMARY_CUSTOM_PROMPT,
)
from memu.prompts.category_summary import (
    PROMPT as CATEGORY_SUMMARY_PROMPT,
)
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
from memu.utils.conversation import format_conversation_for_preprocess
from memu.utils.video import VideoFrameExtractor
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

    def _episode_entry_sort_key(self, entry: StructuredMemoryEntry) -> tuple[float, int]:
        confidence = entry[4] if entry[4] is not None else 0.0
        tie_payload = "\x1f".join([
            entry[0],
            entry[1],
            "\x1e".join(entry[2]),
            entry[3] or "",
        ])
        tie_break = int.from_bytes(
            hashlib.blake2s(tie_payload.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        return (confidence, tie_break)

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
            segment_episodes = [{"text": raw_text, "caption": None}]
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

            if preprocessor_rows and primary_messages:
                background_summaries = preprocessor_rows
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
            episode_item: str | None = None
            applicable_types: list[MemoryType] = memory_types
            if context_only:
                applicable_types = []
                if not episode_summary:
                    episode_summary = await self._summarize_background_messages(
                        messages=background_messages or episode_messages_all,
                        llm_client=extract_client,
                    )
                episode_item = episode_summary
                if not episode_summary:
                    routing_notes.append(f"episode {episode_ref}: context-only background, no summary generated")
            elif episode_text:
                applicable_types, routed_summary, routed_item = await self._route_episode(
                    episode_text,
                    memory_types,
                    extract_client,
                    soul_card=(soul_card or "").strip() or None,
                    skipped_reasons=routing_notes,
                )
                if routed_summary:
                    episode_summary = routed_summary
                episode_item = routed_item

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
                "episode_item": episode_item,
                "message_indices": selected_indices,
                "message_happened_at_map": plan_message_happened_at_map,
                "episode_messages": primary_messages,
                "background_summaries": background_summaries,
                "context_only": context_only,
                "speaker_map": speaker_map,
                "applicable_types": applicable_types,
                "entries": [],
                "episode_id": episode_id,
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
                "episode_item": ep["episode_item"],
                "message_indices": ep["message_indices"],
                "message_happened_at_map": ep["message_happened_at_map"],
                "entries": episode_entries,
                "episode_id": ep["episode_id"],
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

    @staticmethod
    def _normalize_text_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for value in raw:
            text = str(value or "").strip()
            if not text:
                continue
            out.append(text)
        return out

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
        episode_item: str | None = None
        if state["modality"] == "conversation" and isinstance(text, str):
            applicable_types, episode_summary, episode_item = await self._route_episode(
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
            "episode_item": episode_item,
            "message_indices": message_indices,
            "message_happened_at_map": plan_message_happened_at_map,
            "entries": structured_entries,
            "episode_id": episode_id,
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
    def _extract_scope_field(
        scope: Mapping[str, Any],
        *,
        keys: Sequence[str],
    ) -> tuple[str, str] | None:
        return dedupe._extract_scope_field(scope, keys=keys)

    def _build_semantic_dedupe_scope(self, scope: Mapping[str, Any] | None) -> dict[str, str] | None:
        return dedupe._build_semantic_dedupe_scope(scope)

    @staticmethod
    def _normalize_embedding_vector(embedding: Any) -> list[float] | None:
        return dedupe._normalize_embedding_vector(embedding)

    @staticmethod
    def _is_merged_item(item: Any) -> bool:
        return dedupe._is_merged_item(item)

    @staticmethod
    def _item_embedding(item: Any) -> list[float] | None:
        return dedupe._item_embedding(item)

    async def _dedupe_reembed_for_similarity(
        self,
        *,
        item: Any,
        embed_client: Any,
        cache: dict[str, list[float] | None],
    ) -> list[float] | None:
        return await dedupe._dedupe_reembed_for_similarity(
            item=item,
            embed_client=embed_client,
            cache=cache,
        )

    @staticmethod
    def _summary_len(item: Any) -> int:
        return dedupe._summary_len(item)

    def _choose_survivor_and_redundant(self, left: MemoryItem, right: MemoryItem) -> tuple[MemoryItem, MemoryItem]:
        return cast(tuple[MemoryItem, MemoryItem], dedupe._choose_survivor_and_redundant(left, right))

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        return dedupe._cosine_similarity(a, b)

    @staticmethod
    def _filter_merged_from_category_updates(
        updates: Any,
        merged_ids: set[str],
    ) -> dict[str, list[tuple[str, str]]]:
        return dedupe._filter_merged_from_category_updates(updates, merged_ids)

    @staticmethod
    def _dedupe_summary_tokens(summary: Any) -> set[str]:
        return dedupe._dedupe_summary_tokens(summary)

    @staticmethod
    def _dedupe_source_role(item: Any) -> str | None:
        return dedupe._dedupe_source_role(item)

    @staticmethod
    def _dedupe_speaker_id(item: Any) -> str | None:
        return dedupe._dedupe_speaker_id(item)

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

    def _dynamic_category_cluster_threshold(self) -> float:
        return 0.75

    def _dynamic_category_cluster_min_size(self) -> int:
        cluster_size = int(getattr(self.memorize_config, "dynamic_category_cluster_size", 3) or 3)
        return max(2, cluster_size)

    def _cluster_homeless_entries(
        self,
        *,
        filtered_entries: list[StructuredMemoryEntry],
        per_entry_unknowns: Sequence[list[str]],
        item_embeddings: Sequence[Any] | None = None,
    ) -> tuple[list[HomelessCategoryCluster], dict[int, str]]:
        if not filtered_entries or not item_embeddings:
            return [], {}

        normalized_embeddings: list[list[float] | None] = [
            self._normalize_embedding_vector(raw) for raw in item_embeddings[: len(filtered_entries)]
        ]
        if len(normalized_embeddings) < len(filtered_entries):
            normalized_embeddings.extend([None] * (len(filtered_entries) - len(normalized_embeddings)))

        candidate_indexes = [
            idx
            for idx, (entry, unknowns, embedding) in enumerate(
                zip(filtered_entries, per_entry_unknowns, normalized_embeddings, strict=True)
            )
            if not entry.categories and unknowns and embedding is not None
        ]
        if len(candidate_indexes) < 2:
            return [], {}

        threshold = self._dynamic_category_cluster_threshold()
        min_size = self._dynamic_category_cluster_min_size()
        adjacency: dict[int, set[int]] = {idx: set() for idx in candidate_indexes}

        for pos, left_idx in enumerate(candidate_indexes):
            left_embedding = normalized_embeddings[left_idx]
            if left_embedding is None:
                continue
            for right_idx in candidate_indexes[pos + 1 :]:
                right_embedding = normalized_embeddings[right_idx]
                if right_embedding is None:
                    continue
                if self._cosine_similarity(left_embedding, right_embedding) < threshold:
                    continue
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)

        visited: set[int] = set()
        clusters: list[HomelessCategoryCluster] = []
        entry_cluster_ids: dict[int, str] = {}

        for idx in candidate_indexes:
            if idx in visited:
                continue
            stack = [idx]
            component: list[int] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in sorted(adjacency.get(current, ())):
                    if neighbor not in visited:
                        stack.append(neighbor)

            component.sort()
            if len(component) < min_size:
                continue

            label_counts: dict[str, int] = {}
            saliences: list[float] = []
            examples: list[str] = []
            for entry_idx in component:
                for label in per_entry_unknowns[entry_idx]:
                    label_counts[label] = label_counts.get(label, 0) + 1
                salience = filtered_entries[entry_idx].reflection_salience
                if salience is not None:
                    saliences.append(salience)
                content = filtered_entries[entry_idx].content.strip()
                if content and len(examples) < 4:
                    examples.append(content)

            cluster_id = f"cluster_{len(clusters) + 1}"
            cluster = HomelessCategoryCluster(
                cluster_id=cluster_id,
                entry_indexes=component,
                label_counts=label_counts,
                average_salience=(sum(saliences) / len(saliences)) if saliences else None,
                max_salience=max(saliences) if saliences else None,
                examples=examples,
            )
            clusters.append(cluster)
            for entry_idx in component:
                entry_cluster_ids[entry_idx] = cluster_id

        return clusters, entry_cluster_ids

    def _build_existing_category_block(self, *, ctx: Context, store: Database) -> str:
        existing_lines: list[str] = []
        seen_existing: set[str] = set()
        for category_id in getattr(ctx, "category_ids", []) or []:
            category = store.memory_category_repo.categories.get(category_id)
            if category is None:
                continue
            nm = str(category.name or "").strip()
            if not nm:
                continue
            key = nm.casefold()
            if key in seen_existing:
                continue
            seen_existing.add(key)
            desc = str(getattr(category, "description", "") or "").strip()
            existing_lines.append(f"- {nm}: {desc}" if desc else f"- {nm}")
        if not existing_lines:
            for cfg in self.memorize_config.memory_categories or []:
                nm = (cfg.name or "").strip()
                if not nm:
                    continue
                desc = (cfg.description or "").strip()
                existing_lines.append(f"- {nm}: {desc}" if desc else f"- {nm}")
        return "\n".join(existing_lines) if existing_lines else "(none)"

    @staticmethod
    def _build_cluster_prompt_block(strong_clusters: Sequence[HomelessCategoryCluster]) -> str:
        cluster_lines: list[str] = []
        for cluster in strong_clusters[:12]:
            label_block = ", ".join(
                f"{label}({count})"
                for label, count in sorted(cluster.label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            )
            salience_bits: list[str] = []
            if cluster.average_salience is not None:
                salience_bits.append(f"avg_salience={cluster.average_salience:.2f}")
            if cluster.max_salience is not None:
                salience_bits.append(f"max_salience={cluster.max_salience:.2f}")
            salience_block = f" {' '.join(salience_bits)}" if salience_bits else ""
            examples_block = "\n".join(f"    - {example}" for example in cluster.examples)
            header = (
                f"- {cluster.cluster_id} size={len(cluster.entry_indexes)} labels={label_block or '(none)'}"
                f"{salience_block}"
            )
            cluster_lines.append(f"{header}\n{examples_block}" if examples_block else header)
        return "\n".join(cluster_lines) if cluster_lines else "(none)"

    @staticmethod
    def _build_ungrouped_candidate_block(
        *,
        ungrouped_unknown_counts: Mapping[str, int],
        ungrouped_unknown_examples: Mapping[str, list[str]],
    ) -> str:
        candidates_sorted = sorted(ungrouped_unknown_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        cand_lines: list[str] = []
        for cand, cnt in candidates_sorted[:20]:
            examples = ungrouped_unknown_examples.get(cand, [])
            ex_block = "\n".join(f"    - {e}" for e in examples) if examples else ""
            cand_lines.append(f"- {cand} (count={cnt})\n{ex_block}" if ex_block else f"- {cand} (count={cnt})")
        return "\n".join(cand_lines) if cand_lines else "(none)"

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
        existing_block = self._build_existing_category_block(ctx=ctx, store=store)
        clusters_block = self._build_cluster_prompt_block(strong_clusters)
        candidates_block = self._build_ungrouped_candidate_block(
            ungrouped_unknown_counts=ungrouped_unknown_counts,
            ungrouped_unknown_examples=ungrouped_unknown_examples,
        )
        desc_line = default_desc or (
            "Categories are life domains and are thus broad by nature. "
            "Life domains are the core, interconnected areas of a being's existence—such as health, relationships, work, and finances."
        )
        policy_block = (
            f"""

Extra guidance (optional):
{policy}
""".strip()
            if policy
            else ""
        )
        system_prompt = f"""You are organizing memory categories.

{desc_line}

{policy_block}

Rules:
- Prefer mapping candidates into EXISTING categories.
- Homeless clusters are stronger evidence than raw labels. Use cluster meaning first and treat labels as hints.
- Only propose NEW categories if the topic is a broad life domain AND it is mentioned at least {min_mentions} times across the extracted memories OR it is clearly important.
- If multiple close candidates (e.g., health/medical) should be merged, merge them and accumulate the count.
- New category names must be 1-3 words, letters/spaces only (no underscores); we will normalize to snake_case later.
- Do not recreate an existing category under a new name.

Output ONLY valid JSON with this shape:
{{"create": [{{"cluster": "cluster_1", "name": "...", "description": "...", "important": true|false}}, {{"name": "...", "description": "...", "from": ["candidate1", "candidate2"], "important": true|false}}], "map": [{{"cluster": "cluster_2", "to": "existing_category"}}, {{"from": "candidate", "to": "existing_category"}}]}}
""".strip()
        user_prompt = f"""EXISTING CATEGORIES:
{existing_block}

HOMELESS CLUSTERS (embedding-similar memories that do not fit existing categories):
{clusters_block}

UNGROUPED UNKNOWN LABELS (homeless items not in a strong cluster):
{candidates_block}

Decide which clusters/candidates should map into existing categories, and which (if any) justify creating a NEW life-domain category.""".strip()

        cluster_mapping: dict[str, str] = {}
        label_mapping: dict[str, str] = {}
        new_defs: dict[str, str] = {}
        cluster_by_id = {cluster.cluster_id: cluster for cluster in strong_clusters}

        def _valid_new_name(raw: str) -> bool:
            raw = (raw or "").strip()
            if not raw:
                return False
            if len(raw.split()) > 3:
                return False
            return bool(re.fullmatch(r"[A-Za-z ]+", raw))

        try:
            planner_profile = getattr(self.memorize_config, "category_update_llm_profile", "default")
            planner = self._get_llm_client(planner_profile)
            resp = await planner.chat(user_prompt, system_prompt=system_prompt)
        except Exception:
            logger.warning("dynamic-category planner LLM call failed", exc_info=True)
            return cluster_mapping, label_mapping, new_defs
        match = re.search(r"\{[\s\S]*\}", resp or "")
        if match is None:
            return cluster_mapping, label_mapping, new_defs
        try:
            plan = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("dynamic-category planner returned unparseable JSON: %.200s", resp)
            return cluster_mapping, label_mapping, new_defs

        if not isinstance(plan, dict):
            return cluster_mapping, label_mapping, new_defs

        for entry in plan.get("create", []) or []:
            if not isinstance(entry, dict):
                continue
            raw_name = str(entry.get("name", "") or "").strip()
            if not _valid_new_name(raw_name):
                continue
            desc = str(entry.get("description", "") or "").strip() or default_desc
            important = bool(entry.get("important", False))
            cluster_id = str(entry.get("cluster", "") or "").strip()
            norm_name = self._normalize_category_name(raw_name)
            if not norm_name:
                continue

            if cluster_id and cluster_id in cluster_by_id:
                cluster = cluster_by_id[cluster_id]
                if len(cluster.entry_indexes) < self._dynamic_category_cluster_min_size() and not important:
                    continue
                if norm_name not in ctx.category_name_to_id:
                    new_defs.setdefault(norm_name, desc)
                cluster_mapping[cluster_id] = norm_name
                continue

            src = entry.get("from", []) or []
            if not isinstance(src, list):
                src = [src]
            src_norm = [
                sn
                for s in src
                if isinstance(s, str)
                for sn in [self._normalize_category_name(s)]
                if sn and sn in ungrouped_unknown_counts
            ]
            if not src_norm:
                continue
            total = sum(ungrouped_unknown_counts.get(s, 0) for s in src_norm)
            if total < min_mentions and not important:
                continue
            if norm_name not in ctx.category_name_to_id:
                new_defs.setdefault(norm_name, desc)
            for source_name in src_norm:
                label_mapping[source_name] = norm_name

        for entry in plan.get("map", []) or []:
            if not isinstance(entry, dict):
                continue
            cluster_id = str(entry.get("cluster", "") or "").strip()
            if cluster_id and cluster_id in cluster_by_id:
                tgt = self._normalize_category_name(str(entry.get("to", "") or ""))
                if tgt and (tgt in ctx.category_name_to_id or tgt in new_defs):
                    cluster_mapping[cluster_id] = tgt
                continue
            src = self._normalize_category_name(str(entry.get("from", "") or ""))
            tgt = self._normalize_category_name(str(entry.get("to", "") or ""))
            if not src or src not in ungrouped_unknown_counts:
                continue
            if tgt and (tgt in ctx.category_name_to_id or tgt in new_defs):
                label_mapping[src] = tgt

        return cluster_mapping, label_mapping, new_defs

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

        res = await self._create_resource_with_caption(
            resource_url=plan["resource_url"],
            modality=modality,
            local_path=episode_local_path,
            caption=plan.get("caption"),
            store=store,
            embed_client=embed_client,
            user=user_scope,
            episode_id=plan.get("episode_id"),
            conversation_id=conversation_id,
            memory_retrieve_history=plan.get("memory_retrieve_history"),
            memory_prior_context=plan.get("memory_prior_context"),
            **kwargs,
        )

        episode_summary_text = str(plan.get("episode_summary") or "").strip()
        episode_item_text = str(plan.get("episode_item") or "").strip() or episode_summary_text
        if episode_item_text and res.embedding is not None:
            summary_item = store.memory_item_repo.create_item(
                resource_id=res.id,
                memory_type="episode",
                source_role="environment",
                summary=episode_item_text,
                embedding=res.embedding,
                user_data=dict(user_scope or {}),
                conversation_id=conversation_id,
                episode_id=plan.get("episode_id"),
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
                category_updates.setdefault(cid, []).append((summary_item.id, episode_item_text))

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
            episode_id=plan.get("episode_id"),
            message_happened_at_map=plan.get("message_happened_at_map"),
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
        active_items = [item for item in state.get("items", []) if not self._is_merged_item(item)]
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
        caption_text = caption.strip() if caption else None
        if caption_text:
            client = embed_client or self._get_llm_client("embedding")
            caption_embedding = (await client.embed([caption_text]))[0]
        else:
            caption_embedding = None

        resource_kwargs: dict[str, Any] = {
            "url": resource_url,
            "modality": modality,
            "local_path": local_path,
            "caption": caption_text,
            "embedding": caption_embedding,
            "user_data": dict(user or {}),
        }
        if episode_id:
            resource_kwargs["episode_id"] = episode_id
        if conversation_id:
            resource_kwargs["conversation_id"] = conversation_id
        if memory_retrieve_history:
            resource_kwargs["memory_retrieve_history"] = memory_retrieve_history
        if memory_prior_context:
            resource_kwargs["memory_prior_context"] = memory_prior_context
        if session is not None:
            res = cast(Any, store.resource_repo).create_resource(**resource_kwargs, session=session)
        else:
            res = store.resource_repo.create_resource(**resource_kwargs)
        return cast(Resource, res)

    @staticmethod
    def _sqlite_write_session(store: Database) -> Any | None:
        try:
            from memu.database.sqlite.sqlite import SQLiteStore
        except ImportError:
            return None
        if isinstance(store, SQLiteStore):
            return store._sessions.session()
        return None

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
    ) -> tuple[list[MemoryType], str | None, str | None]:
        if not memory_types:
            return [], None, None
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
            try:
                payload = json.loads(self._extract_json_blob(raw))
            except Exception:
                logger.warning("Router returned unparseable response, skipping episode: %.120s", raw)
                if skipped_reasons is not None:
                    skipped_reasons.append("router returned unparseable JSON")
                return [], None, None
        if not isinstance(payload, dict):
            logger.warning("Router returned non-dict payload, skipping episode: %s", type(payload).__name__)
            if skipped_reasons is not None:
                skipped_reasons.append("router returned non-dict payload")
            return [], None, None
        memorable = payload.get("memorable")
        excluded_types = payload.get("excluded_types")
        reason = payload.get("reason", "")
        episode_summary = str(payload.get("episode_summary") or "").strip() or None
        episode_item = str(payload.get("episode_item") or "").strip() or None
        if memorable is False:
            if skipped_reasons is not None and reason:
                skipped_reasons.append(reason)
            return [], episode_summary, episode_item
        if excluded_types is None:
            excluded_types = []
        if not isinstance(excluded_types, list):
            logger.warning("Router returned invalid excluded_types, skipping episode")
            if skipped_reasons is not None:
                skipped_reasons.append("router returned invalid excluded_types")
            return [], episode_summary, episode_item
        excluded = {
            routed_type
            for routed_type in excluded_types
            if isinstance(routed_type, str) and routed_type in set(memory_types)
        }
        allowed_types = set(memory_types) - excluded
        return [mtype for mtype in memory_types if mtype in allowed_types], episode_summary, episode_item

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

    def _extract_episode_text(self, lines: list[str], start_idx: int, end_idx: int) -> str | None:
        episode_lines = []
        for line in lines:
            match = re.match(r"\[(\d+)\]", line)
            if not match:
                continue
            idx = int(match.group(1))
            if start_idx <= idx <= end_idx:
                episode_lines.append(line)
        return "\n".join(episode_lines) if episode_lines else None

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
        await self._ensure_categories_ready(ctx, store, user)

        max_total = int(getattr(self.memorize_config, "max_categories_total", 0) or 0)
        min_mentions = int(getattr(self.memorize_config, "dynamic_category_cluster_size", 3) or 3)
        policy = str(getattr(self.memorize_config, "dynamic_category_policy", "") or "").strip()
        default_desc = str(getattr(self.memorize_config, "dynamic_category_description", "") or "").strip()

        cur_total = len(getattr(ctx, "category_ids", []) or [])
        remaining = (max_total - cur_total) if max_total else None
        if remaining is not None and remaining <= 0:
            return [
                entry._replace(categories=[c for c in (entry.categories or []) if c in ctx.category_name_to_id])
                for entry in structured_entries
            ]

        unknown_counts: dict[str, int] = {}
        per_entry_unknowns: list[list[str]] = []
        filtered_entries: list[StructuredMemoryEntry] = []

        for entry in structured_entries:
            known: list[str] = []
            unknown: list[str] = []
            for c in entry.categories or []:
                n = self._normalize_category_name(c)
                if not n:
                    continue
                if n in ctx.category_name_to_id:
                    known.append(n)
                else:
                    unknown.append(n)

            seen: set[str] = set()
            known_dedup: list[str] = []
            for k in known:
                if k not in seen:
                    known_dedup.append(k)
                    seen.add(k)
            homeless_unknowns = unknown if not known_dedup else []
            for n in homeless_unknowns:
                unknown_counts[n] = unknown_counts.get(n, 0) + 1
            filtered_entries.append(entry._replace(categories=known_dedup))
            per_entry_unknowns.append(homeless_unknowns)

        if not unknown_counts:
            return filtered_entries

        homeless_clusters, entry_cluster_ids = self._cluster_homeless_entries(
            filtered_entries=filtered_entries,
            per_entry_unknowns=per_entry_unknowns,
            item_embeddings=item_embeddings,
        )
        strong_clusters = [cluster for cluster in homeless_clusters if cluster.label_counts]
        clustered_indexes = set(entry_cluster_ids)

        ungrouped_unknown_counts: dict[str, int] = {}
        ungrouped_unknown_examples: dict[str, list[str]] = {}
        for idx, unknowns in enumerate(per_entry_unknowns):
            if idx in clustered_indexes:
                continue
            for label in unknowns:
                ungrouped_unknown_counts[label] = ungrouped_unknown_counts.get(label, 0) + 1
                ex_list = ungrouped_unknown_examples.setdefault(label, [])
                if len(ex_list) < 3:
                    ex_list.append(filtered_entries[idx].content)

        cluster_mapping, label_mapping, new_defs = await self._plan_dynamic_categories(
            ctx=ctx,
            store=store,
            strong_clusters=strong_clusters,
            ungrouped_unknown_counts=ungrouped_unknown_counts,
            ungrouped_unknown_examples=ungrouped_unknown_examples,
            min_mentions=min_mentions,
            policy=policy,
            default_desc=default_desc,
        )

        to_create = [name for name in new_defs if name not in ctx.category_name_to_id]
        if remaining is not None:
            to_create = to_create[:remaining]

        if to_create:
            texts = [f"{name}: {new_defs.get(name, default_desc)}" for name in to_create]
            vecs = await embed_client.embed(texts)
            for name, vec in zip(to_create, vecs, strict=True):
                desc = new_defs.get(name, default_desc)
                cat = store.memory_category_repo.get_or_create_category(
                    name=name,
                    description=desc or "",
                    embedding=vec,
                    user_data=dict(user or {}),
                    session=session,
                )
                ctx.category_ids.append(cat.id)
                ctx.category_name_to_id[name.lower()] = cat.id

        updated: list[StructuredMemoryEntry] = []
        for idx, (entry, unk) in enumerate(zip(filtered_entries, per_entry_unknowns, strict=True)):
            cats = list(entry.categories)
            cluster_target = cluster_mapping.get(entry_cluster_ids.get(idx, ""))
            mapped_cluster_name = self._normalize_category_name(cluster_target) if cluster_target else None
            if (
                mapped_cluster_name
                and mapped_cluster_name in ctx.category_name_to_id
                and mapped_cluster_name not in cats
            ):
                cats.append(mapped_cluster_name)
            for unknown_name in unk or []:
                target_name = label_mapping.get(unknown_name)
                if not target_name:
                    continue
                mapped_name = self._normalize_category_name(target_name)
                if mapped_name and mapped_name in ctx.category_name_to_id and mapped_name not in cats:
                    cats.append(mapped_name)
            updated.append(entry._replace(categories=cats))

        return updated

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
        summary_payloads = [entry.content for entry in structured_entries]
        client = embed_client or self._get_llm_client()
        item_embeddings = await client.embed(summary_payloads) if summary_payloads else []
        items: list[MemoryItem] = []
        rels: list[CategoryItem] = []
        category_memory_updates: dict[str, list[tuple[str, str]]] = {}
        superseded_targets: set[str] = set()

        structured_entries = await self._maybe_create_dynamic_categories(
            structured_entries=structured_entries,
            item_embeddings=item_embeddings,
            ctx=ctx,
            store=store,
            embed_client=client,
            user=user,
            session=session,
        )
        if self.memorize_config.enable_confidence_normalization:
            structured_entries = self._normalize_confidence(structured_entries)
        homeless_count = sum(1 for entry in structured_entries if not entry.categories)
        supersede_targets = await self._find_supersede_targets(
            structured_entries=structured_entries,
            store=store,
            embed_client=client,
            user=user,
        )
        for idx, (entry, emb) in enumerate(zip(structured_entries, item_embeddings, strict=True)):
            resolved_summary = self._hedge_summary_for_confidence(entry.content, entry.confidence)
            item_kwargs = {
                "resource_id": resource_id,
                "memory_type": entry.memory_type,
                "summary": resolved_summary,
                "embedding": emb,
                "user_data": dict(user or {}),
                "source_role": entry.source_role,
                "speaker_id": entry.speaker_id,
                "speaker_label": entry.speaker_label,
                "confidence": entry.confidence,
                "source_message_ids": entry.source_message_ids,
                "happened_at": self._resolve_entry_happened_at(entry.source_message_ids, message_happened_at_map),
                "reflection_salience": entry.reflection_salience,
                "emotional_intensity": entry.emotional_intensity,
                "conversation_id": conversation_id,
                "episode_id": episode_id,
            }
            if session is not None:
                item = cast(Any, store.memory_item_repo).create_item(**item_kwargs, session=session)
            else:
                item = store.memory_item_repo.create_item(**item_kwargs)
            items.append(item)
            if entry.entities:
                for ent_data in entry.entities:
                    ent_name = str(ent_data.get("name") or "").strip()
                    ent_type = str(ent_data.get("type") or "").strip()
                    if not ent_name or not ent_type:
                        continue
                    entity_record = store.entity_repo.get_or_create(
                        ent_name,
                        ent_type,
                        user_data=dict(user or {}),
                        session=session,
                    )
                    store.triple_repo.add(Triple(
                        subject_id=item.id,
                        subject_kind="memory",
                        predicate="mentions",
                        object_id=entity_record.id,
                        object_kind="entity",
                        source_memory_id=item.id,
                    ), user_data=dict(user or {}), session=session)
            target_item_id = supersede_targets.get(idx)
            if target_item_id and target_item_id != item.id and target_item_id not in superseded_targets:
                superseded_targets.add(target_item_id)
                store.triple_repo.add(Triple(
                    subject_id=target_item_id,
                    subject_kind="memory",
                    predicate="evolved_into",
                    object_id=item.id,
                    object_kind="memory",
                    source_memory_id=item.id,
                ), user_data=dict(user or {}), session=session)
            mapped_cat_ids = self._map_category_names_to_ids(entry.categories, ctx)
            if resolved_summary.strip():
                for cid in mapped_cat_ids:
                    category_memory_updates.setdefault(cid, []).append((item.id, resolved_summary))
                    rel_kwargs = {"item_id": item.id, "category_id": cid, "user_data": dict(user or {})}
                    if session is not None:
                        rel = cast(Any, store.category_item_repo).link_item_category(**rel_kwargs, session=session)
                    else:
                        rel = store.category_item_repo.link_item_category(**rel_kwargs)
                    rels.append(rel)

        return items, rels, category_memory_updates, homeless_count

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
        if not isinstance(user_scope, Mapping):
            return "__global__"
        user_id = str(user_scope.get("user_id") or "").strip()
        soul_id = str(user_scope.get("soul_id") or "").strip()
        if not user_id and not soul_id:
            return "__global__"
        return f"user={user_id}|soul={soul_id}"

    async def _ensure_categories_ready(
        self, ctx: Context, store: Database, user_scope: Mapping[str, Any] | None = None
    ) -> None:
        scope_key = self._category_scope_key(user_scope)
        if ctx.categories_ready and ctx.category_scope_key == scope_key:
            return
        async with ctx._init_lock:
            if ctx.categories_ready and ctx.category_scope_key == scope_key:
                return
            await self._initialize_categories(ctx, store, user_scope, scope_key=scope_key)

    async def _initialize_categories(
        self,
        ctx: Context,
        store: Database,
        user: Mapping[str, Any] | None = None,
        *,
        scope_key: str | None = None,
    ) -> None:
        resolved_scope_key = scope_key or self._category_scope_key(user)
        if ctx.categories_ready and ctx.category_scope_key == resolved_scope_key:
            return
        if not self.category_configs:
            ctx.categories_ready = True
            ctx.category_scope_key = resolved_scope_key
            ctx.category_init_scope_key = None
            return
        cat_texts = [self._category_embedding_text(cfg) for cfg in self.category_configs]
        cat_vecs = await self._get_llm_client("embedding").embed(cat_texts)
        ctx.category_ids = []
        ctx.category_name_to_id = {}
        for cfg, vec in zip(self.category_configs, cat_vecs, strict=True):
            name = cfg.name.strip() or "Untitled"
            description = cfg.description.strip()
            cat = store.memory_category_repo.get_or_create_category(
                name=name, description=description, embedding=vec, user_data=dict(user or {})
            )
            ctx.category_ids.append(cat.id)
            ctx.category_name_to_id[name.lower()] = cat.id
        ctx.categories_ready = True
        ctx.category_scope_key = resolved_scope_key
        ctx.category_init_scope_key = None

    @staticmethod
    def _category_embedding_text(cat: CategoryConfig) -> str:
        name = cat.name.strip() or "Untitled"
        desc = cat.description.strip()
        return f"{name}: {desc}" if desc else name

    def _map_category_names_to_ids(self, names: list[str], ctx: Context) -> list[str]:
        if not names:
            return []
        mapped: list[str] = []
        seen: set[str] = set()
        for name in names:
            key = name.strip().lower()
            cid = ctx.category_name_to_id.get(key)
            if cid and cid not in seen:
                mapped.append(cid)
                seen.add(cid)
        return mapped

    async def _split_into_episodes(
        self, *, local_path: str, text: str | None, modality: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        configured_prompt = self.memorize_config.multimodal_preprocess_prompts.get(modality)
        if configured_prompt is None:
            template = PREPROCESS_PROMPTS.get(modality)
        elif isinstance(configured_prompt, str):
            template = configured_prompt
        else:
            template = self._resolve_custom_prompt(configured_prompt, {})

        if not template:
            return [{"text": text, "caption": None}]

        if modality == "audio":
            text = await self._prepare_audio_text(local_path, text, llm_client=llm_client)
            if text is None:
                return [{"text": None, "caption": None}]

        if self._modality_requires_text(modality) and not text:
            return [{"text": text, "caption": None}]

        return await self._dispatch_preprocessor(
            modality=modality,
            local_path=local_path,
            text=text,
            template=template,
            llm_client=llm_client,
        )

    async def _prepare_audio_text(self, local_path: str, text: str | None, llm_client: Any | None = None) -> str | None:
        if text:
            return text

        audio_extensions = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        text_extensions = {".txt", ".text", ".jsonl"}
        file_ext = pathlib.Path(local_path).suffix.lower()

        if file_ext in audio_extensions:
            try:
                client = llm_client or self._get_llm_client()
                transcribed = cast(str, await client.transcribe(local_path))
            except Exception:
                logger.exception("Audio transcription failed for %s", local_path)
                return None
            else:
                return transcribed

        if file_ext in text_extensions:
            path_obj = pathlib.Path(local_path)
            try:
                text_content = path_obj.read_text(encoding="utf-8")
            except OSError:
                logger.exception("Failed to read text file %s", local_path)
                return None
            else:
                return text_content

        logger.warning(f"Unknown audio file type: {file_ext}, skipping transcription")
        return None

    def _modality_requires_text(self, modality: str) -> bool:
        return modality in ("conversation", "document")

    async def _dispatch_preprocessor(
        self,
        *,
        modality: str,
        local_path: str,
        text: str | None,
        template: str,
        llm_client: Any | None = None,
    ) -> list[dict[str, str | None]]:
        if modality == "conversation" and text is not None:
            return await self._split_conversation_into_episodes(text, template, llm_client=llm_client)
        if modality == "video":
            return await self._preprocess_video(local_path, template, llm_client=llm_client)
        if modality == "image":
            return await self._preprocess_image(local_path, template, llm_client=llm_client)
        if modality == "document" and text is not None:
            return await self._preprocess_document(text, template, llm_client=llm_client)
        if modality == "audio" and text is not None:
            return await self._preprocess_audio(text, template, llm_client=llm_client)
        return [{"text": text, "caption": None}]

    async def _split_conversation_into_episodes(
        self, conversation_raw_text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        indexed_conversation_text = format_conversation_for_preprocess(conversation_raw_text)
        eps_per_seg = getattr(self.memorize_config, "episodes_per_segment", 3) or 3
        prompt = template.format(
            conversation=self._escape_prompt_value(indexed_conversation_text),
            episodes_per_segment=eps_per_seg,
        )
        client = llm_client or self._get_llm_client()
        preprocessor_response = await client.chat(prompt)
        _, episodes = self._parse_conversation_preprocess_with_episodes(
            preprocessor_response,
            indexed_conversation_text,
        )

        # Important: always use the original JSON-derived, indexed conversation text for downstream
        # episode detection and memory extraction. The LLM may rewrite the conversation and drop fields
        # like created_at, which would cause them to be lost.
        all_message_indices = self._extract_message_indices(indexed_conversation_text)
        if not episodes:
            return [{"text": indexed_conversation_text, "caption": None, "message_indices": all_message_indices}]

        indexed_lines = indexed_conversation_text.split("\n")
        max_idx = len(indexed_lines) - 1
        episode_resources: list[dict[str, Any]] = []
        pending_captions: list[tuple[int, str]] = []

        for episode in episodes:
            explicit_indices = episode.get("message_indices")
            if isinstance(explicit_indices, list) and explicit_indices:
                parsed_indices = self._dedupe_message_indices([
                    value for value in explicit_indices if isinstance(value, (int, float, str))
                ])
                indices = sorted(i for i in parsed_indices if 0 <= i <= max_idx)
            else:
                start = int(episode.get("start", 0))
                end = int(episode.get("end", max_idx))
                start = max(0, min(start, max_idx))
                end = max(0, min(end, max_idx))
                indices = list(range(start, end + 1))

            episode_text = "\n".join(indexed_lines[i] for i in indices if i <= max_idx)
            if episode_text.strip():
                caption_raw = episode.get("caption")
                caption = str(caption_raw).strip() if isinstance(caption_raw, str) else ""
                episode_resources.append({
                    "text": episode_text,
                    "caption": caption or None,
                    "message_indices": indices,
                })
                if not caption:
                    pending_captions.append((len(episode_resources) - 1, episode_text))

        if pending_captions:
            max_parallel = min(4, len(pending_captions))
            limiter = asyncio.Semaphore(max_parallel)

            async def summarize_one(resource_idx: int, episode_text: str) -> tuple[int, str | None]:
                async with limiter:
                    caption = await self._summarize_episode(episode_text, llm_client=client)
                    return resource_idx, caption

            caption_results = await asyncio.gather(
                *(summarize_one(resource_idx, episode_text) for resource_idx, episode_text in pending_captions)
            )
            for resource_idx, generated_caption in caption_results:
                episode_resources[resource_idx]["caption"] = generated_caption
        return (
            episode_resources
            if episode_resources
            else [{"text": indexed_conversation_text, "caption": None, "message_indices": all_message_indices}]
        )

    async def _summarize_episode(self, episode_text: str, llm_client: Any | None = None) -> str | None:
        system_prompt = (
            "Summarize the given conversation episode in 1-2 concise sentences. "
            "Focus on the main topic or theme discussed."
        )
        try:
            client = llm_client or self._get_llm_client(
                step_context={"operation": "memorize", "step_id": "episode_summary"},
            )
            response = await client.chat(episode_text, system_prompt=system_prompt)
            return response.strip() if response else None
        except Exception:
            logger.exception("Failed to summarize episode")
            return None

    async def _preprocess_video(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        try:
            if not VideoFrameExtractor.is_ffmpeg_available():
                logger.warning("ffmpeg not available, cannot process video. Returning None.")
                return [{"text": None, "caption": None}]

            frame_path = VideoFrameExtractor.extract_middle_frame(local_path)

            try:
                client = llm_client or self._get_llm_client()
                processed = await client.vision(prompt=template, image_path=frame_path, system_prompt=None)
                description, caption = self._parse_multimodal_response(processed, "detailed_description", "caption")
                return [{"text": description, "caption": caption}]
            finally:
                import pathlib

                try:
                    pathlib.Path(frame_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("Failed to clean up frame %s: %s", frame_path, e)

        except (OSError, RuntimeError) as e:
            logger.error("Video preprocessing failed: %s", e, exc_info=True)
            return [{"text": None, "caption": None}]

    async def _preprocess_image(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        client = llm_client or self._get_llm_client()
        processed = await client.vision(prompt=template, image_path=local_path, system_prompt=None)
        description, caption = self._parse_multimodal_response(processed, "detailed_description", "caption")
        return [{"text": description, "caption": caption}]

    async def _preprocess_document(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        prompt = template.format(document_text=self._escape_prompt_value(text))
        client = llm_client or self._get_llm_client()
        processed = await client.chat(prompt)
        processed_content, caption = self._parse_multimodal_response(processed, "processed_content", "caption")
        return [{"text": processed_content or text, "caption": caption}]

    async def _preprocess_audio(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        prompt = template.format(transcription=self._escape_prompt_value(text))
        client = llm_client or self._get_llm_client()
        processed = await client.chat(prompt)
        processed_content, caption = self._parse_multimodal_response(processed, "processed_content", "caption")
        return [{"text": processed_content or text, "caption": caption}]

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
        words = len((text or "").split())
        if words < 1:
            return 0
        return int(words / 0.75)

    @staticmethod
    def _compute_batch_max_items(total_message_count: int) -> int:
        if total_message_count >= 80:
            return 12
        if total_message_count >= 40:
            return 8
        return 6

    def _build_batch_extraction_text(self, episodes: Sequence[Mapping[str, Any]]) -> str:
        sections: list[str] = []
        summaries: list[str] = []
        for episode in episodes:
            episode_ref = self._parse_episode_ref(episode.get("episode_ref"))
            if episode_ref is None:
                continue
            episode_text = str(episode.get("text") or "").strip()
            if not episode_text:
                continue
            message_indices = self._dedupe_message_indices(episode.get("message_indices"))
            if message_indices:
                heading = f"## Episode {episode_ref} (messages {message_indices[0]}-{message_indices[-1]})"
            else:
                heading = f"## Episode {episode_ref}"
            sections.append(heading)
            sections.append(episode_text)
            episode_summary = (
                str(episode.get("episode_summary") or "").strip()
                or str(episode.get("caption") or "").strip()
            )
            if episode_summary:
                summaries.append(f"Episode {episode_ref}: {episode_summary}")
        if summaries:
            sections.append("## Episode Summaries")
            sections.extend(summaries)
        return "\n\n".join(s for s in sections if s).strip()

    @staticmethod
    def _message_is_primary_for_memorize(message: Mapping[str, Any]) -> bool:
        flag = message.get("memorize_chat")
        if isinstance(flag, bool):
            return flag
        return True

    @staticmethod
    def _message_index_for_sort(message: Mapping[str, Any]) -> int:
        raw = message.get("_message_index")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _format_episode_message_line(message: Mapping[str, Any]) -> str:
        idx = MemorizeMixin._message_index_for_sort(message)
        role = str(message.get("name") or message.get("role") or "user").strip() or "user"
        content = str(message.get("content") or "").strip()
        source = str(message.get("source_label") or "").strip()
        source_prefix = f"[{source}] " if source else ""
        return f"[{max(0, idx)}] {source_prefix}[{role}]: {content}"

    async def _summarize_background_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
    ) -> str | None:
        if not messages:
            return None
        rendered = "\n".join(
            self._format_episode_message_line(msg)
            for msg in sorted(messages, key=self._message_index_for_sort)
        ).strip()
        if not rendered:
            return None
        summary = await self._summarize_episode(rendered, llm_client=llm_client)
        return str(summary or "").strip() or None

    async def _render_episode_with_background_context(
        self,
        *,
        primary_messages: Sequence[Mapping[str, Any]],
        background_messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Render episode text with inline background summaries.

        Returns (rendered_text, background_summary_rows), where summary rows carry
        ``after_index`` and ``summary``.
        """
        prim = sorted(primary_messages, key=self._message_index_for_sort)
        bg = sorted(background_messages, key=self._message_index_for_sort)
        if not prim and not bg:
            return "", []
        if not bg:
            return "\n".join(self._format_episode_message_line(msg) for msg in prim).strip(), []

        grouped: dict[str, list[Mapping[str, Any]]] = {}
        group_order: list[str] = []
        for msg in bg:
            source_key = str(msg.get("source_conversation_id") or msg.get("source_label") or "background").strip() or "background"
            if source_key not in grouped:
                grouped[source_key] = []
                group_order.append(source_key)
            grouped[source_key].append(msg)

        summary_rows: list[dict[str, Any]] = []
        for source_key in group_order:
            group_msgs = grouped[source_key]
            summary = await self._summarize_background_messages(messages=group_msgs, llm_client=llm_client)
            if not summary:
                continue
            source_label = str(group_msgs[0].get("source_label") or source_key).strip() or source_key
            first_idx = self._message_index_for_sort(group_msgs[0])
            after_index = None
            for primary in prim:
                pidx = self._message_index_for_sort(primary)
                if pidx <= first_idx:
                    after_index = pidx
                else:
                    break
            summary_rows.append(
                {
                    "after_index": after_index,
                    "summary": summary,
                    "source_label": source_label,
                }
            )

        rendered_lines: list[str] = []
        for primary in prim:
            pidx = self._message_index_for_sort(primary)
            for row in summary_rows:
                if row.get("_emitted"):
                    continue
                if row.get("after_index") == pidx:
                    rendered_lines.append(
                        f"[Background:{row.get('source_label')}] {str(row.get('summary') or '').strip()}"
                    )
                    row["_emitted"] = True
            rendered_lines.append(self._format_episode_message_line(primary))

        prefix_lines: list[str] = []
        for row in summary_rows:
            if row.get("_emitted"):
                row.pop("_emitted", None)
                continue
            prefix_lines.append(
                f"[Background:{row.get('source_label')}] {str(row.get('summary') or '').strip()}"
            )
            row.pop("_emitted", None)
        if prefix_lines:
            rendered_lines = [*prefix_lines, *rendered_lines]

        return "\n".join(line for line in rendered_lines if line.strip()).strip(), summary_rows

    def _render_episode_with_summary_rows(
        self,
        *,
        primary_messages: Sequence[Mapping[str, Any]],
        summary_rows: Sequence[Mapping[str, Any]],
    ) -> str:
        prim = sorted(primary_messages, key=self._message_index_for_sort)
        rendered_lines: list[str] = []
        mutable_rows: list[dict[str, Any]] = [dict(row) for row in summary_rows if isinstance(row, Mapping)]
        for primary in prim:
            pidx = self._message_index_for_sort(primary)
            for row in mutable_rows:
                if row.get("_emitted"):
                    continue
                if row.get("after_index") == pidx:
                    label = str(row.get("source_label") or "background").strip() or "background"
                    rendered_lines.append(f"[Background:{label}] {str(row.get('summary') or '').strip()}")
                    row["_emitted"] = True
            rendered_lines.append(self._format_episode_message_line(primary))
        prefix_lines: list[str] = []
        for row in mutable_rows:
            if row.get("_emitted"):
                continue
            label = str(row.get("source_label") or "background").strip() or "background"
            prefix_lines.append(f"[Background:{label}] {str(row.get('summary') or '').strip()}")
        if prefix_lines:
            rendered_lines = [*prefix_lines, *rendered_lines]
        return "\n".join(line for line in rendered_lines if line.strip()).strip()

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
        return item_id.replace("-", "")[:6]

    def _extract_refs_from_summaries(self, summaries: dict[str, str]) -> set[str]:
        from memu.utils.references import extract_references

        refs: set[str] = set()
        for summary in summaries.values():
            refs.update(extract_references(summary))
        return refs

    async def _persist_item_references(
        self,
        *,
        updated_summaries: dict[str, str],
        category_updates: dict[str, list[tuple[str, str]]],
        store: Database,
    ) -> None:
        referenced_short_ids = self._extract_refs_from_summaries(updated_summaries)
        if not referenced_short_ids:
            return

        short_id_to_item_id: dict[str, str] = {}
        for item_tuples in category_updates.values():
            for item_id, _ in item_tuples:
                short_id = self._build_item_ref_id(item_id)
                short_id_to_item_id[short_id] = item_id

        for short_id in referenced_short_ids:
            matched_item_id = short_id_to_item_id.get(short_id)
            if matched_item_id:
                store.memory_item_repo.update_item(
                    item_id=matched_item_id,
                    extra={"ref_id": short_id},
                )

    @staticmethod
    def _looks_like_identifier_value(value: str) -> bool:
        text = value.strip()
        if not text:
            return True
        lowered = text.lower()
        if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", lowered):
            return True
        if re.fullmatch(r"[0-9a-f]{24,}", lowered):
            return True
        if " " in text:
            return False
        if len(text) < 12:
            return False
        if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
            return False
        has_digit = any(ch.isdigit() for ch in text)
        has_sep = ("-" in text) or ("_" in text)
        return has_digit and has_sep

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
        enable_refs = getattr(self.memorize_config, "enable_item_references", False)

        if enable_refs:
            from memu.prompts.category_summary import (
                CUSTOM_PROMPT_WITH_REFS as category_summary_custom_prompt,
            )
            from memu.prompts.category_summary import (
                PROMPT_WITH_REFS as category_summary_prompt,
            )

            tuple_memories = cast(list[tuple[str, str]], new_memories)
            new_items_text = "\n".join(
                f"- [{self._build_item_ref_id(item_id)}] {summary}"
                for item_id, summary in tuple_memories
                if summary.strip()
            )
        else:
            category_summary_prompt = CATEGORY_SUMMARY_PROMPT
            category_summary_custom_prompt = CATEGORY_SUMMARY_CUSTOM_PROMPT

            if new_memories and isinstance(new_memories[0], tuple):
                tuple_memories = cast(list[tuple[str, str]], new_memories)
                new_items_text = "\n".join(f"- {summary}" for item_id, summary in tuple_memories if summary.strip())
            else:
                str_memories = cast(list[str], new_memories)
                new_items_text = "\n".join(f"- {m}" for m in str_memories if m.strip())

        original = category.summary or ""
        category_config = self.category_config_map.get(category.name)
        configured_prompt = (
            category_config and category_config.summary_prompt
        ) or self.memorize_config.default_category_summary_prompt
        if configured_prompt is None:
            prompt = category_summary_prompt
        elif isinstance(configured_prompt, str):
            prompt = configured_prompt
        else:
            prompt = self._resolve_custom_prompt(configured_prompt, category_summary_custom_prompt)
        target_length = (
            category_config and category_config.target_length
        ) or self.memorize_config.default_category_summary_target_length
        user_scope = user or {}
        user_name = self._summary_user_name(user_scope, default="the user")
        raw_agent = (
            user_scope.get("soul_name")
            or user_scope.get("character_name")
            or user_scope.get("soul_id")
        )
        agent_name = str(raw_agent).strip() if raw_agent else "the assistant"
        if " - " in agent_name:
            agent_name = agent_name.split(" - ", 1)[0].strip() or agent_name

        return prompt.format(
            category=self._escape_prompt_value(category.name),
            original_content=self._escape_prompt_value(original or ""),
            new_memory_items_text=self._escape_prompt_value(new_items_text or "No new memory items."),
            target_length=target_length,
            user_name=self._escape_prompt_value(user_name),
            agent_name=self._escape_prompt_value(agent_name),
        )

    async def _update_category_summaries(
        self,
        updates: dict[str, list[tuple[str, str]]] | dict[str, list[str]],
        ctx: Context,
        store: Database,
        llm_client: Any | None = None,
        user: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        updated_summaries: dict[str, str] = {}
        if not updates:
            return updated_summaries
        tasks = []
        target_ids: list[str] = []
        client = llm_client or self._get_llm_client()
        for cid, memories in updates.items():
            cat = store.memory_category_repo.categories.get(cid)
            if not cat or not memories:
                continue
            prompt = self._build_category_summary_prompt(category=cat, new_memories=memories, user=user)
            tasks.append(client.chat(prompt))
            target_ids.append(cid)
        if not tasks:
            return updated_summaries
        summaries = await asyncio.gather(*tasks)
        for cid, summary in zip(target_ids, summaries, strict=True):
            cat = store.memory_category_repo.categories.get(cid)
            if not cat:
                continue
            cleaned_summary = summary.replace("```markdown", "").replace("```", "").strip()
            user_name = self._summary_user_name(user or {}, default="")
            if user_name and user_name.lower() not in ("user", "the user"):
                cleaned_summary = re.sub(
                    r"(?m)^(\s*[-*]\s*)(?:The user|the user|User|user)\b",
                    r"\1" + user_name,
                    cleaned_summary,
                )

            store.memory_category_repo.update_category(
                category_id=cid,
                summary=cleaned_summary,
            )
            updated_summaries[cid] = cleaned_summary
        return updated_summaries

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
    def _coerce_to_iterable(values: Any) -> Sequence[Any]:
        return parsing._coerce_to_iterable(values)

    @staticmethod
    def _extract_message_indices(text: str | None) -> list[int]:
        return parsing._extract_message_indices(text)

    @staticmethod
    def _extract_conversation_messages(raw_text: Any) -> list[tuple[int, dict[str, Any]]]:
        return parsing._extract_conversation_messages(raw_text)

    @staticmethod
    def _parse_message_happened_at(raw: Any) -> Any | None:
        return parsing._parse_message_happened_at(raw)

    def _extract_message_happened_at_map(self, raw_text: Any) -> dict[int, Any]:
        return parsing._extract_message_happened_at_map(raw_text)

    @staticmethod
    def _normalize_speaker_slug(prefix: str, raw: Any) -> str:
        return speakers._normalize_speaker_slug(prefix, raw)

    @staticmethod
    def _speaker_role_from_id(speaker_id: str | None) -> str | None:
        return speakers._speaker_role_from_id(speaker_id)

    @staticmethod
    def _normalize_coarse_role(role: str | None) -> str:
        return speakers._normalize_coarse_role(role)

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

    @staticmethod
    def _has_ambiguous_speaker_role(roster: Sequence[SpeakerRosterEntry]) -> bool:
        return speakers._has_ambiguous_speaker_role(roster)

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

    @staticmethod
    def _is_user_declared_relationship_entity(entity: Any) -> bool:
        return speakers._is_user_declared_relationship_entity(entity)

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

    @staticmethod
    def _episode_mentions_roster_entry(episode_text: Any, entry: SpeakerRosterEntry) -> bool:
        return speakers._episode_mentions_roster_entry(episode_text, entry)

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
        if not message_happened_at_map:
            return None
        for message_idx in source_message_ids or []:
            happened_at = message_happened_at_map.get(int(message_idx))
            if happened_at is not None:
                return happened_at
        for message_idx in sorted(message_happened_at_map):
            happened_at = message_happened_at_map.get(message_idx)
            if happened_at is not None:
                return happened_at
        return None

    def _prepare_episode(
        self,
        *,
        modality: str,
        text: str | None,
        message_indices: Any = None,
    ) -> tuple[str | None, list[int]]:
        if modality != "conversation" or not isinstance(text, str) or not text.strip():
            return None, []
        episode_text = format_conversation_for_preprocess(text)
        if not episode_text.strip():
            episode_text = text.strip()
        if isinstance(message_indices, list):
            indices = self._dedupe_message_indices([
                value for value in message_indices if isinstance(value, (int, float, str))
            ])
        else:
            indices = self._extract_message_indices(episode_text)
        return episode_text, indices

    def _parse_multimodal_response(self, raw: str, content_tag: str, caption_tag: str) -> tuple[str | None, str | None]:
        content = self._extract_tag_content(raw, content_tag)
        caption = self._extract_tag_content(raw, caption_tag)
        if not content:
            content = raw.strip()
        if not caption and content:
            first_sentence = content.split(".")[0]
            caption = first_sentence if len(first_sentence) <= 200 else first_sentence[:200]
        return content, caption

    def _parse_conversation_preprocess_with_episodes(
        self, raw: str, original_text: str
    ) -> tuple[str | None, list[dict[str, int | str]] | None]:
        conversation = self._extract_tag_content(raw, "conversation")
        episodes = self._extract_episodes_with_fallback(raw)
        return conversation, episodes

    def _extract_episodes_with_fallback(self, raw: str) -> list[dict[str, Any]] | None:
        episodes = self._episodes_from_json_payload(raw)
        if episodes is not None:
            return episodes
        try:
            blob = self._extract_json_blob(raw)
        except (ValueError, IndexError):
            logger.warning("Failed to extract episodes from conversation preprocess response: %.200s", raw)
            return None
        return self._episodes_from_json_payload(blob)

    def _episodes_from_json_payload(self, payload: str) -> list[dict[str, Any]] | None:
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        return self._episodes_from_parsed_data(parsed)

    @staticmethod
    def _episodes_from_parsed_data(parsed: Any) -> list[dict[str, Any]] | None:
        if not isinstance(parsed, dict):
            return None
        episodes_data = parsed.get("episodes")
        if not isinstance(episodes_data, list):
            return None
        episodes: list[dict[str, Any]] = []
        for ep in episodes_data:
            if not isinstance(ep, dict):
                continue
            if "message_indices" in ep and isinstance(ep["message_indices"], list):
                episode: dict[str, Any] = {"message_indices": ep["message_indices"]}
                if "caption" in ep and isinstance(ep["caption"], str):
                    episode["caption"] = ep["caption"]
                if "background_summaries" in ep and isinstance(ep["background_summaries"], list):
                    cleaned_background: list[dict[str, Any]] = []
                    for row in ep["background_summaries"]:
                        if not isinstance(row, dict):
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
                        cleaned_background.append(
                            {
                                "after_index": after_index,
                                "summary": summary,
                            }
                        )
                    episode["background_summaries"] = cleaned_background
                episodes.append(episode)
            elif "start" in ep and "end" in ep:
                try:
                    episode = {"start": int(ep["start"]), "end": int(ep["end"])}
                    if "caption" in ep and isinstance(ep["caption"], str):
                        episode["caption"] = ep["caption"]
                    episodes.append(episode)
                except (TypeError, ValueError):
                    continue
        return episodes or None

    @staticmethod
    def _extract_tag_content(raw: str, tag: str) -> str | None:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
        match = pattern.search(raw)
        if not match:
            return None
        content = match.group(1).strip()
        return content or None

    def _parse_memory_type_response(self, raw: str) -> list[dict[str, Any]]:
        return parsing._parse_memory_type_response(raw, self._extract_json_blob)

    def _find_xml_boundaries(self, raw: str) -> tuple[int, int, str] | None:
        return parsing._find_xml_boundaries(raw)

    def _parse_memory_element(self, memory_elem: Any) -> dict[str, Any] | None:
        return parsing._parse_memory_element(memory_elem)

    def _parse_memory_type_response_xml(self, raw: str) -> list[dict[str, Any]]:
        return parsing._parse_memory_type_response_xml(raw)
