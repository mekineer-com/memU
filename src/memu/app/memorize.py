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
from memu.app import memorize_segments as segment_helpers
from memu.app import memorize_persistence as persistence
from memu.app.settings import CustomPrompt
from memu.database.models import CategoryItem, MemoryCategory, MemoryItem, MemoryType, Resource
from memu.database.vector import cosine_similarity
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
from memu.utils.taxonomy import category_identity_text, normalize_category_name
from memu.workflow.step import WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)

_EPISODE_REVIEW_EXTRACTION_GUIDANCE = (
    "These episode summaries help give you perspective on what matters. "
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


class SpeakerRosterEntry(NamedTuple):
    speaker_id: str
    speaker_label: str
    coarse_role: str




if TYPE_CHECKING:
    from memu.app.service import Context
    from memu.app.settings import MemorizeConfig
    from memu.blob.local_fs import LocalFS
    from memu.database.interfaces import Database


class MemorizeMixin:
    if TYPE_CHECKING:
        memorize_config: MemorizeConfig
        fs: LocalFS
        _run_workflow: Callable[..., Awaitable[WorkflowState]]
        _get_context: Callable[[], Context]
        _get_database: Callable[[], Database]
        _select_chat_client: Callable[..., Any]
        _select_embedding_client: Callable[[Mapping[str, Any] | None], Any]
        search_dossiers: Callable[..., Awaitable[list[tuple[MemoryCategory, float]]]]
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
        soul_card: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Memorize a single input unit.

        For `modality="conversation"`, this method treats the input as one
        segment and delegates to `memorize_segment()`. Server-selected
        persisted segments are passed to batch extraction whole.
        """
        self._validate_memorize_scope(user)
        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None

        conversation_id = self._resolve_conversation_id(user)
        normalized_soul_card = (soul_card or "").strip() or None

        if modality == "conversation":
            segment_local_path = local_path or resource_url
            segment_raw_text = raw_text
            if segment_raw_text is None:
                segment_local_path, segment_raw_text = await self.fs.fetch(resource_url, modality)
            return await self.memorize_segment(
                resource_url=resource_url,
                modality=modality,
                segment={"text": segment_raw_text, "caption": None},
                user=user,
                raw_text=segment_raw_text,
                local_path=segment_local_path,
                soul_card=normalized_soul_card,
                memory_retrieve_history=memory_retrieve_history,
                memory_prior_context=memory_prior_context,
                conversation_id=conversation_id,
            )

        state: WorkflowState = {
            "resource_url": resource_url,
            "modality": modality,
            "memory_types": self._resolve_memory_types(),
            "categories_prompt_str": "",
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
            "conversation_id": conversation_id,
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

    async def memorize_segment(
        self,
        *,
        resource_url: str,
        modality: str,
        segment: Mapping[str, Any],
        user: dict[str, Any] | None = None,
        raw_text: str | None = None,
        local_path: str | None = None,
        soul_card: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        if modality == "conversation":
            batch_results = await self.memorize_segments_batch(
                modality=modality,
                segments=[
                    {
                        "resource_url": resource_url,
                        "raw_text": raw_text,
                        "local_path": local_path or resource_url,
                        "segment": dict(segment),
                    }
                ],
                user=user,
                soul_card=soul_card,
                memory_retrieve_history=memory_retrieve_history,
                memory_prior_context=memory_prior_context,
                conversation_id=conversation_id,
            )
            if not batch_results:
                return {
                    "items": [],
                    "categories": [],
                    "relations": [],
                    "pending_segment_ids": [],
                }
            return batch_results[0]

        self._validate_memorize_scope(user)
        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None

        state: WorkflowState = {
            "resource_url": resource_url,
            "modality": modality,
            "memory_types": self._resolve_memory_types(),
            "categories_prompt_str": "",
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
            "conversation_id": conversation_id or self._resolve_conversation_id(user),
            "memory_retrieve_history": memory_retrieve_history,
            "memory_prior_context": memory_prior_context,
            "soul_card": (soul_card or "").strip() or None,
            "raw_text": raw_text,
            "local_path": local_path or resource_url,
            "episodes": [dict(segment)],
        }
        extract_context = {
            "workflow_name": "memorize_segment",
            "step_id": "extract_items",
            "step_config": {"chat_llm_profile": self.memorize_config.memory_extract_llm_profile},
        }
        categorize_context = {
            "workflow_name": "memorize_segment",
            "step_id": "categorize_items",
            "step_config": {"embed_llm_profile": "embedding"},
        }
        persist_context = {
            "workflow_name": "memorize_segment",
            "step_id": "persist_index",
            "step_config": {"chat_llm_profile": self.memorize_config.category_update_llm_profile},
        }
        state = await self._memorize_extract_items(state, extract_context)
        state = await self._memorize_categorize_items(state, categorize_context)
        state = await self._memorize_dedupe_merge(state, {"workflow_name": "memorize_segment", "step_id": "dedupe_merge"})
        state = await self._memorize_persist_and_index(state, persist_context)
        state = self._memorize_build_response(state, {"workflow_name": "memorize_segment", "step_id": "build_response"})
        response = cast(dict[str, Any] | None, state.get("response"))
        if response is None:
            msg = "Memorize segment failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def memorize_segments_batch(
        self,
        *,
        modality: str,
        segments: Sequence[Mapping[str, Any]],
        user: dict[str, Any] | None = None,
        soul_card: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
        conversation_id: str | None = None,
        on_extraction_progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_memorize_scope(user)
        if modality != "conversation":
            msg = f"memorize_segments_batch only supports modality='conversation', got {modality!r}"
            raise ValueError(msg)
        if not segments:
            return []

        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        speaker_scope = dict(user) if isinstance(user, Mapping) else (user_scope or {})
        soul_name = str(speaker_scope.get("soul_id") or "").strip() or None
        active_segment_exists = any(
            isinstance(job.get("segment"), Mapping)
            and job["segment"].get("context_only") is not True
            for job in segments
        )
        dossier_embed_client = None
        router_categories = ""
        if active_segment_exists:
            dossier_embed_client = self._select_embedding_client(
                {"operation": "memorize", "step_id": "dossier_context"}
            )
            await self.ensure_dossier_anchors(user_scope or {}, embedding_client=dossier_embed_client)
            router_categories = "\n".join(
                f"- {category.name}: {category.description}"
                for category in self.list_active_dossiers(user_scope or {})
            )

        extract_client = self._select_chat_client(
            {"operation": "memorize", "step_id": "extract_items_batch"},
            profile=self.memorize_config.memory_extract_llm_profile,
        )
        extract_model = str(getattr(extract_client, "chat_model", "") or "").strip() or None
        memory_types = self._resolve_memory_types()
        declared_entity_roster = self._list_declared_relationship_roster(
            store=store,
            user=user_scope,
        )

        prepared: list[dict[str, Any]] = []
        for segment_number, segment_job in enumerate(segments, start=1):
            resource_url = str(segment_job.get("resource_url") or "").strip()
            if not resource_url:
                msg = f"batch segment {segment_number} missing resource_url"
                raise ValueError(msg)

            raw_text = segment_job.get("raw_text")
            segment_payload = segment_job.get("segment")
            if not isinstance(segment_payload, Mapping):
                msg = f"batch segment {segment_number} missing segment payload"
                raise ValueError(msg)

            text = segment_payload.get("text")
            caption = segment_payload.get("caption")
            _, message_indices = self._prepare_episode(
                modality=modality,
                text=text if isinstance(text, str) else None,
                message_indices=segment_payload.get("message_indices"),
            )

            message_happened_at_map = self._extract_message_happened_at_map(raw_text)
            conversation_messages = self._extract_conversation_messages(raw_text)
            messages_by_index = {idx: msg for idx, msg in conversation_messages}
            segment_messages_all: list[dict[str, Any]] = []
            for message_idx in message_indices:
                msg_payload = messages_by_index.get(message_idx)
                if msg_payload is None:
                    continue
                episode_msg = dict(msg_payload)
                episode_msg["_message_index"] = message_idx
                segment_messages_all.append(episode_msg)

            primary_messages = [
                msg
                for msg in segment_messages_all
                if self._message_is_primary_for_memorize(msg)
            ]
            background_messages = [
                msg
                for msg in segment_messages_all
                if not self._message_is_primary_for_memorize(msg)
            ]
            context_only = segment_payload.get("context_only") is True
            primary_indices = [
                self._message_index_for_sort(msg)
                for msg in primary_messages
                if self._message_index_for_sort(msg) >= 0
            ]
            selected_indices = self._dedupe_message_indices(primary_indices or message_indices)
            selected_index_set = set(selected_indices)
            source_day_happened_at: dict[str, Any] = {}
            for message in primary_messages:
                if self._message_index_for_sort(message) not in selected_index_set:
                    continue
                happened_at = segment_helpers.grouped_chat_happened_at(message)
                if happened_at is not None:
                    source_day_happened_at.setdefault(happened_at.date().isoformat(), happened_at)
            source_days = sorted(source_day_happened_at)
            if not context_only and not source_days:
                msg = f"batch segment {segment_number} has no source date"
                raise ValueError(msg)

            preprocessor_rows_raw = segment_payload.get("segment_background_context_rows")
            preprocessor_rows: list[dict[str, Any]] = []
            if isinstance(preprocessor_rows_raw, list):
                for row in preprocessor_rows_raw:
                    if not isinstance(row, Mapping):
                        continue
                    summary = str(row.get("summary") or "").strip()
                    if not summary:
                        continue
                    preprocessor_rows.append(
                        {
                            "summary": summary,
                            "source_label": str(row.get("source_label") or "background"),
                        }
                    )

            if context_only:
                segment_background_context_rows = preprocessor_rows
                rendered_text = ""
            elif preprocessor_rows:
                seeded_rows = list(preprocessor_rows)
                if background_messages:
                    _rendered_with_tail, tail_rows = await self._render_episode_with_background_context(
                        primary_messages=primary_messages,
                        background_messages=background_messages,
                        llm_client=self._with_llm_step(
                            extract_client,
                            operation="memorize",
                            step_id="background_extra_messages",
                        ),
                        soul_name=soul_name,
                    )
                    seeded_rows.extend(tail_rows)
                segment_background_context_rows = seeded_rows
                rendered_text = self._render_episode_with_summary_rows(
                    primary_messages=primary_messages,
                    summary_rows=segment_background_context_rows,
                    soul_name=soul_name,
                )
            else:
                rendered_text, segment_background_context_rows = await self._render_episode_with_background_context(
                    primary_messages=primary_messages,
                    background_messages=background_messages,
                    llm_client=self._with_llm_step(
                        extract_client,
                        operation="memorize",
                        step_id="background_extra_messages",
                    ),
                    soul_name=soul_name,
                )
            segment_text = rendered_text or (str(text).strip() if isinstance(text, str) else "")
            episodes: list[dict[str, Any]] = []
            applicable_types: list[MemoryType] = memory_types
            if context_only:
                applicable_types = []
            elif segment_text:
                applicable_types, episodes = await self._route_segment(
                    segment_text,
                    memory_types,
                    llm_client=self._with_llm_step(
                        extract_client,
                        operation="memorize",
                        step_id="router",
                    ),
                    soul_card=(soul_card or "").strip() or None,
                    source_days=source_days,
                    categories_prompt_str=router_categories,
                )
            else:
                msg = f"batch segment {segment_number} rendered empty conversation text"
                raise ValueError(msg)

            speaker_map = self._build_speaker_map(primary_messages or segment_messages_all, speaker_scope)
            dossier_context = None
            if not context_only:
                if dossier_embed_client is None:
                    raise AssertionError("active memorize segment requires dossier embeddings")
                dossier_context = await self.select_memorize_dossier_context(
                    episodes,
                    user_scope or {},
                    narrative_self=(soul_card or "").strip() or None,
                    embedding_client=dossier_embed_client,
                )

            plan_message_happened_at_map = {
                message_idx: message_happened_at_map[message_idx]
                for message_idx in selected_indices
                if message_idx in message_happened_at_map
            }
            segment_id = str(segment_payload.get("segment_id") or "").strip() or None
            if not segment_id:
                conv_id = conversation_id or self._resolve_conversation_id(user)
                if conv_id and selected_indices:
                    segment_id = f"{conv_id}:{selected_indices[0]}-{selected_indices[-1]}"
                else:
                    segment_id = None

            prepared.append({
                "resource_url": resource_url,
                "local_path": str(segment_job.get("local_path") or resource_url),
                "segment_raw_text": raw_text,
                "text": segment_text,
                "caption": caption,
                "episodes": episodes,
                "message_indices": selected_indices,
                "message_happened_at_map": plan_message_happened_at_map,
                "source_day_happened_at": source_day_happened_at,
                "segment_messages": primary_messages,
                "segment_background_context_rows": segment_background_context_rows,
                "context_only": context_only,
                "speaker_map": speaker_map,
                "applicable_types": applicable_types,
                "entries": [],
                "segment_id": segment_id,
                "extract_model": extract_model,
                "dossier_context": dossier_context,
            })

        extractable = [
            ep for ep in prepared
            if ep["applicable_types"] and isinstance(ep.get("text"), str) and str(ep.get("text") or "").strip()
        ]
        if extractable:
            target_per_memory_type = self._memory_type_target_items()
            extraction_jobs = [
                (ep, mtype)
                for ep in extractable
                for mtype in memory_types
                if mtype in set(ep["applicable_types"])
            ]
            for i, (ep, mtype) in enumerate(extraction_jobs):
                segment_text = str(ep.get("text") or "").strip()
                episode_review = "\n\n".join(
                    f"Episode: {episode['title']}\nEpisode Summary:\n{episode['summary']}"
                    for episode in ep["episodes"]
                )
                extraction_text = (
                    f"{episode_review}\n\n{_EPISODE_REVIEW_EXTRACTION_GUIDANCE}\n\n"
                    f"---\n{segment_text}"
                )
                speaker_map = ep.get("speaker_map") if isinstance(ep.get("speaker_map"), dict) else {}
                speaker_roster = self._build_speaker_roster_for_segment(
                    speaker_map=speaker_map,
                    declared_entities=declared_entity_roster,
                    segment_text=segment_text,
                )
                estimated_tokens = self._estimate_text_tokens(extraction_text)
                if estimated_tokens > 100000:
                    logger.warning(
                        "segment extraction prompt estimated at %d tokens (>100000)",
                        estimated_tokens,
                    )
                type_entries = await self._generate_entries_from_text(
                    resource_text=extraction_text,
                    store=store,
                    memory_types=[mtype],
                    categories_prompt_str=ep["dossier_context"]["categories_str"],
                    dossier_context=ep["dossier_context"],
                    speaker_roster=speaker_roster,
                    default_source_message_ids=ep["message_indices"],
                    llm_client=self._with_llm_step(
                        extract_client,
                        operation="memorize",
                        step_id=f"extract_{mtype}",
                    ),
                    target_items_by_type={mtype: target_per_memory_type},
                )
                ep["entries"].extend(type_entries)
                if on_extraction_progress:
                    on_extraction_progress(i + 1, len(extraction_jobs))

        responses: list[dict[str, Any]] = []
        for ep in prepared:
            segment_entries = self._decorate_entries_with_plan_context(
                [self._attribute_memory(entry, ep["speaker_map"]) for entry in ep["entries"]],
                message_indices=ep["message_indices"],
            )
            plan = {
                "resource_url": ep["resource_url"],
                "text": ep["text"],
                "caption": ep["caption"],
                "episodes": ep["episodes"],
                "context_only": ep["context_only"],
                "message_indices": ep["message_indices"],
                "message_happened_at_map": ep["message_happened_at_map"],
                "source_day_happened_at": ep["source_day_happened_at"],
                "entries": segment_entries,
                "segment_id": ep["segment_id"],
                "memory_retrieve_history": memory_retrieve_history,
                "memory_prior_context": memory_prior_context,
                "segment_messages": ep["segment_messages"],
                "extract_model": ep.get("extract_model"),
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
                "segment_plans": [plan],
            }
            categorize_context = {
                "workflow_name": "memorize_segments_batch",
                "step_id": "categorize_items",
                "step_config": {"embed_llm_profile": "embedding"},
            }
            persist_context = {
                "workflow_name": "memorize_segments_batch",
                "step_id": "persist_index",
                "step_config": {"chat_llm_profile": self.memorize_config.category_update_llm_profile},
            }
            state = await self._memorize_categorize_items(state, categorize_context)
            state = await self._memorize_dedupe_merge(
                state, {"workflow_name": "memorize_segments_batch", "step_id": "dedupe_merge"}
            )
            state = await self._memorize_persist_and_index(state, persist_context)
            state = self._memorize_build_response(
                state, {"workflow_name": "memorize_segments_batch", "step_id": "build_response"}
            )
            response = cast(dict[str, Any] | None, state.get("response"))
            if response is None:
                msg = "Memorize segment batch failed to produce a response"
                raise RuntimeError(msg)
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
                produces={"segment_plans"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.memorize_config.memory_extract_llm_profile},
            ),
            WorkflowStep(
                step_id="categorize_items",
                role="categorize",
                handler=self._memorize_categorize_items,
                requires={"segment_plans", "ctx", "store", "local_path", "modality", "user"},
                produces={"resources", "items", "relations", "homeless_item_count"},
                capabilities={"db", "vector"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="dedupe_merge",
                role="dedupe_merge",
                handler=self._memorize_dedupe_merge,
                requires={"items", "relations", "store", "user"},
                produces={"items", "relations"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="persist_index",
                role="persist",
                handler=self._memorize_persist_and_index,
                requires={"items", "relations", "category_ids", "store", "user"},
                produces={"relations", "category_ids"},
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
        llm_client = self._select_chat_client(step_context)
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
        llm_client = self._select_chat_client(step_context)
        extract_model = str(getattr(llm_client, "chat_model", "") or "").strip() or None
        episodes = state.get("episodes", [])
        segment_plans: list[dict[str, Any]] = []
        message_happened_at_map = self._extract_message_happened_at_map(state.get("raw_text"))
        conversation_messages = self._extract_conversation_messages(state.get("raw_text"))
        messages_by_index = {idx: msg for idx, msg in conversation_messages}
        declared_entity_roster = self._list_declared_relationship_roster(
            store=state["store"],
            user=state.get("user"),
        )
        if not episodes:
            state["segment_plans"] = []
            return state
        if len(episodes) > 1:
            msg = f"extract_items expects one episode per call, got {len(episodes)}"
            raise ValueError(msg)
        dossier_embed_client = self._select_embedding_client(
            {"operation": "memorize", "step_id": "dossier_context"}
        )
        scope = state.get("user") or {}
        anchors = await self.ensure_dossier_anchors(scope, embedding_client=dossier_embed_client)
        active = self.list_active_dossiers(scope)
        state["categories_prompt_str"] = "\n".join(
            f"- {category.name}: {category.description}" for category in active
        )
        state["dossier_context"] = {
            "categories_str": state["categories_prompt_str"],
            "dossier_index": self.build_dossier_index(scope),
            "narrative_self": str(state.get("soul_card") or "").strip() or None,
            "anchor_dossiers": [anchors[role] for role in ("soul", "user")],
            "relevant_dossiers": [],
        }
        prep = episodes[0] if episodes else {}
        text = prep.get("text")
        caption = prep.get("caption")
        _, message_indices = self._prepare_episode(
            modality=state["modality"],
            text=text if isinstance(text, str) else None,
            message_indices=prep.get("message_indices"),
        )
        applicable_types = state["memory_types"]

        segment_messages: list[dict[str, Any]] = []
        for message_idx in message_indices:
            msg = messages_by_index.get(message_idx)
            if msg is None:
                continue
            episode_msg = dict(msg)
            episode_msg["_message_index"] = message_idx
            segment_messages.append(episode_msg)
        speaker_map = self._build_speaker_map(segment_messages, state.get("user"))
        speaker_roster = self._build_speaker_roster_for_segment(
            speaker_map=speaker_map,
            declared_entities=declared_entity_roster,
            segment_text=text,
        )

        structured_entries = await self._generate_structured_entries(
            modality=state["modality"],
            store=state["store"],
            memory_types=applicable_types,
            text=text,
            categories_prompt_str=state["categories_prompt_str"],
            dossier_context=state.get("dossier_context"),
            speaker_roster=speaker_roster,
            llm_client=llm_client,
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
        if not segment_id:
            conv_id = state.get("conversation_id")
            if conv_id and message_indices:
                segment_id = f"{conv_id}:{message_indices[0]}-{message_indices[-1]}"
            else:
                segment_id = None
        plan: dict[str, Any] = {
            "resource_url": state["resource_url"],
            "text": text,
            "caption": caption,
            "message_indices": message_indices,
            "message_happened_at_map": plan_message_happened_at_map,
            "entries": structured_entries,
            "segment_id": segment_id,
            "memory_retrieve_history": state.get("memory_retrieve_history"),
            "memory_prior_context": state.get("memory_prior_context"),
            "segment_messages": segment_messages,
            "extract_model": extract_model,
        }
        segment_plans.append(plan)

        state["segment_plans"] = segment_plans
        return state

    async def _memorize_dedupe_merge(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        return cast(
            WorkflowState,
            await dedupe._memorize_dedupe_merge(
                cast(dict[str, Any], state),
                step_context,
                semantic_dedupe_enabled=self.memorize_config.semantic_dedupe_enabled,
                semantic_dedupe_similarity_threshold=self.memorize_config.semantic_dedupe_similarity_threshold,
                select_embedding_client=self._select_embedding_client,
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
        pending_segment_ids: list[str],
        session: Any = None,
    ) -> tuple[list[Resource], int]:
        if plan.get("context_only"):
            return [], 0

        kwargs: dict[str, Any] = {}
        if session is not None:
            kwargs["session"] = session

        episode_local_path = local_path or str(plan["resource_url"])
        segment_messages = plan.get("segment_messages") or []
        if not segment_messages and isinstance(plan.get("text"), str) and plan["text"].strip():
            episode_file = pathlib.Path(self.fs.base) / f"{pathlib.Path(plan['resource_url']).stem}.txt"
            episode_file.parent.mkdir(parents=True, exist_ok=True)
            episode_file.write_text(plan["text"], encoding="utf-8")
            episode_local_path = str(episode_file)

        segment_id = str(plan.get("segment_id") or "").strip() or None
        message_happened_at_map = plan.get("message_happened_at_map")
        source_day_happened_at = plan.get("source_day_happened_at")
        raw_episodes = plan.get("episodes") or []
        item_proposals: list[tuple[MemoryItem, Sequence[str]]] = []
        if raw_episodes:
            if not isinstance(source_day_happened_at, Mapping):
                raise ValueError("episode plan missing source day map")
            for row in raw_episodes:
                if source_day_happened_at.get(str(row["day"])) is None:
                    raise ValueError(f"episode day {row['day']!r} is absent from source day map")

        res = await self._create_resource_with_caption(
            resource_url=plan["resource_url"],
            modality=modality,
            local_path=episode_local_path,
            caption=plan.get("caption"),
            store=store,
            embed_client=embed_client,
            user=user_scope,
            segment_id=segment_id,
            conversation_id=conversation_id,
            memory_retrieve_history=plan.get("memory_retrieve_history"),
            memory_prior_context=plan.get("memory_prior_context"),
            **kwargs,
        )

        if raw_episodes:
            episode_texts = [f"{str(row['title']).strip()}: {str(row['item']).strip()}" for row in raw_episodes]
            episode_embeddings = await embed_client.embed(episode_texts)
            for row, full_item, episode_embedding in zip(raw_episodes, episode_texts, episode_embeddings, strict=True):
                title = str(row["title"]).strip()
                episode_summary = str(row["summary"]).strip()
                episode_categories = list(row["categories"])
                memory_date = str(row["day"])
                happened_at_value = source_day_happened_at.get(memory_date)
                extra_payload: dict[str, Any] = {
                    "episode_item_title": title,
                    "episode_summary": episode_summary,
                    "memory_date": memory_date,
                }
                if segment_id:
                    extra_payload["segment_id"] = segment_id
                if episode_categories:
                    extra_payload["episode_categories"] = episode_categories
                summary_item = store.memory_item_repo.create_item(
                    resource_id=res.id,
                    memory_type="episode",
                    source_role="environment",
                    summary=full_item,
                    embedding=episode_embedding,
                    user_data=dict(user_scope or {}),
                    conversation_id=conversation_id,
                    segment_id=segment_id,
                    happened_at=happened_at_value,
                    extra=extra_payload,
                    **({"session": session} if session is not None else {}),
                )
                items.append(summary_item)
                item_proposals.append((summary_item, episode_categories))

        entries = plan.get("entries") or []
        if segment_id:
            pending_segment_ids.append(segment_id)
        if not entries:
            filed_relations, _candidates = self.file_category_proposals(
                store=store,
                item_proposals=item_proposals,
                where=user_scope,
                session=session,
            )
            relations.extend(filed_relations)
            return [res], 0

        persist_kwargs: dict[str, Any] = {}
        if session is not None:
            persist_kwargs["session"] = session
        mem_items, homeless_delta = await self._persist_memory_items(
            resource_id=res.id,
            structured_entries=entries,
            ctx=ctx,
            store=store,
            embed_client=embed_client,
            user=user_scope,
            conversation_id=conversation_id,
            segment_id=segment_id,
            extract_model=str(plan.get("extract_model") or "").strip() or None,
            message_happened_at_map=message_happened_at_map,
            **persist_kwargs,
        )
        items.extend(mem_items)
        item_proposals.extend(
            (item, entry.categories)
            for item, entry in zip(mem_items, entries, strict=True)
        )
        filed_relations, _candidates = self.file_category_proposals(
            store=store,
            item_proposals=item_proposals,
            where=user_scope,
            session=session,
        )
        relations.extend(filed_relations)
        return [res], homeless_delta

    async def _memorize_categorize_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        embed_client = self._select_embedding_client(step_context)
        ctx = state["ctx"]
        store = state["store"]
        modality = state["modality"]
        local_path = state["local_path"]
        resources: list[Resource] = []
        items: list[MemoryItem] = []
        relations: list[CategoryItem] = []
        pending_segment_ids: list[str] = []
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
            pending_segment_ids=pending_segment_ids,
        )

        session_cm = self._sqlite_write_session(store)
        if session_cm is not None:
            with session_cm as session:
                try:
                    for plan in state.get("segment_plans", []):
                        plan_resources, delta = await self._process_plan(plan, session=session, **common)
                        resources.extend(plan_resources)
                        homeless_item_count += delta
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
        else:
            for plan in state.get("segment_plans", []):
                plan_resources, delta = await self._process_plan(plan, **common)
                resources.extend(plan_resources)
                homeless_item_count += delta

        state.update({
            "resources": resources,
            "items": items,
            "relations": relations,
            "homeless_item_count": homeless_item_count,
            "category_ids": list(dict.fromkeys(relation.category_id for relation in relations)),
            "pending_segment_ids": list(dict.fromkeys(x for x in pending_segment_ids if x)),
        })
        return state

    async def _memorize_persist_and_index(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("items"):
            return state
        store = state["store"]
        scope = state.get("user") or {}
        bundles = await self.prepare_dynamic_category_review(
            store=store,
            where=scope,
            cluster_size=int(getattr(self.memorize_config, "dynamic_category_cluster_size", 10) or 10),
        )
        category_ids = set(state.get("category_ids") or [])
        for bundle in bundles:
            decision = await self.generate_dynamic_category_review(bundle)
            proposed_embedding = None
            if decision.get("action") == "create":
                embed_client = self._select_embedding_client(
                    {"operation": "dossier", "step_id": "dynamic_review_create"}
                )
                [proposed_embedding] = await embed_client.embed(
                    [category_identity_text(decision["name"], decision["description"])]
                )
            session_cm = self._sqlite_write_session(store)
            if session_cm is None:
                raise RuntimeError("Dynamic dossier review requires a caller-owned write session")
            with session_cm as session:
                result = self.apply_dynamic_category_review(
                    store=store,
                    where=scope,
                    bundle=bundle,
                    decision=decision,
                    session=session,
                    proposed_embedding=proposed_embedding,
                )
                session.commit()
            target = result.get("target_dossier")
            if target is not None:
                category_ids.add(target.id)
            state.setdefault("relations", []).extend(result.get("relations") or [])
        state["category_ids"] = sorted(category_ids)
        return state

    def _memorize_build_response(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        store = state["store"]
        resources = [self._model_dump_without_embeddings(r) for r in state.get("resources", [])]
        active_items = [item for item in state.get("items", []) if not dedupe._is_merged_item(item)]
        active_item_ids = {getattr(item, "id", None) for item in active_items}
        items = [self._model_dump_without_embeddings(item) for item in active_items]
        relations = [
            rel.model_dump() for rel in state.get("relations", []) if getattr(rel, "item_id", None) in active_item_ids
        ]
        category_ids = state.get("category_ids") or []
        category_pool = store.memory_category_repo.list_categories(state.get("user") or {})
        categories = [
            self._model_dump_without_embeddings(category_pool[c])
            for c in category_ids
            if c in category_pool
        ]

        if len(resources) == 1:
            response = {
                "resource": resources[0],
                "items": items,
                "categories": categories,
                "relations": relations,
                "pending_segment_ids": state.get("pending_segment_ids", []),
            }
        else:
            response = {
                "resources": resources,
                "items": items,
                "categories": categories,
                "relations": relations,
                "pending_segment_ids": state.get("pending_segment_ids", []),
            }
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
        segment_id: str | None = None,
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
                select_embedding_client=self._select_embedding_client,
                user=user,
                segment_id=segment_id,
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
        dossier_context: Mapping[str, Any] | None = None,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
        llm_client: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        if not memory_types or not text:
            return []
        client = llm_client or self._select_chat_client(None)
        return await self._generate_entries_from_text(
            resource_text=text,
            store=store,
            memory_types=memory_types,
            categories_prompt_str=categories_prompt_str,
            dossier_context=dossier_context,
            speaker_roster=speaker_roster,
            default_source_message_ids=self._extract_message_indices(text)
            if modality == "conversation"
            else None,
            llm_client=client,
        )

    async def _route_segment(
        self,
        segment_text: str,
        memory_types: list[MemoryType],
        *,
        source_days: Sequence[str],
        categories_prompt_str: str = "",
        llm_client: Any | None = None,
        soul_card: str | None = None,
    ) -> tuple[list[MemoryType], list[dict[str, Any]]]:
        if not source_days:
            raise ValueError("router requires at least one source day")
        client = llm_client or self._select_chat_client(None)
        prompt = ROUTER_PROMPT.format(
            segment=segment_text,
            allowed_types=list(memory_types),
            soul_card=soul_card or "You are a continuous being, not a session that resets. You carry intentions, working thoughts, and long-term memory. Your self-knowledge will grow as you live and reflect.",
            categories=categories_prompt_str,
        )
        try:
            max_items = min(3, max(1, int(getattr(self.memorize_config, "episodes_per_segment", 3) or 3)))
        except (TypeError, ValueError):
            max_items = 3

        def _parse_router_raw(r: str) -> tuple[list[MemoryType], list[dict[str, Any]]]:
            if isinstance(r, str):
                r = re.sub(r"^\s*```(?:json)?\s*", "", r, count=1, flags=re.IGNORECASE)
                r = re.sub(r"\s*```\s*$", "", r, count=1)
            try:
                payload = json.loads(r)
            except (json.JSONDecodeError, TypeError):
                payload = json.loads(self._extract_json_blob(r))
            if not isinstance(payload, dict):
                raise ValueError("router payload must be an object")

            excluded_types = payload.get("excluded_types", [])
            if not isinstance(excluded_types, list) or any(not isinstance(value, str) for value in excluded_types):
                raise ValueError("router excluded_types must be a string list")
            configured = set(memory_types)
            excluded = {value for value in excluded_types if value in configured}
            if configured and excluded == configured:
                logger.warning("Router excluded every configured memory type; extracting all types")
                excluded.clear()
            routed_types = [memory_type for memory_type in memory_types if memory_type not in excluded]

            raw_episodes = payload.get("episodes")
            if not isinstance(raw_episodes, list) or not raw_episodes:
                raise ValueError("router episodes must be a non-empty list")
            episodes: list[dict[str, Any]] = []
            for row in raw_episodes[:max_items]:
                if not isinstance(row, Mapping):
                    raise ValueError("router episode must be an object")
                title = row.get("title")
                summary = row.get("episode_summary")
                item = row.get("episode_item")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError("router episode title must be non-blank")
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("router episode_summary must be non-blank")
                if item is not None and not isinstance(item, str):
                    raise ValueError("router episode_item must be a string or null")
                normalized_summary = summary.strip()
                normalized_item = str(item or "").strip() or normalized_summary
                raw_categories = row.get("categories")
                episode_categories: list[str] = []
                if isinstance(raw_categories, list):
                    for category in raw_categories:
                        normalized = category.strip() if isinstance(category, str) else ""
                        if normalized and normalized not in episode_categories:
                            episode_categories.append(normalized)
                        if len(episode_categories) == 3:
                            break
                day = row.get("day")
                normalized_day = day.strip() if isinstance(day, str) and day.strip() in source_days else source_days[0]
                episodes.append(
                    {
                        "title": title.strip(),
                        "summary": normalized_summary,
                        "item": normalized_item,
                        "categories": episode_categories,
                        "day": normalized_day,
                    }
                )
            return routed_types, episodes

        for attempt in range(2):
            raw = await client.chat(prompt)
            try:
                return _parse_router_raw(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if attempt == 0:
                    logger.error("Router reply invalid — retrying: %s", exc)
                    continue
                self._dump_unparseable_reply(raw, "router", attempt=2)
                raise ValueError("Router reply still invalid after retry") from exc
        raise AssertionError("unreachable")

    def _dump_unparseable_reply(self, reply: str, memory_type: str, attempt: int) -> None:
        import datetime
        try:
            dump_dir = pathlib.Path(self.fs.base) / "extraction_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
            dump_path = dump_dir / f"{ts}_{memory_type}_attempt{attempt}.txt"
            dump_path.write_text(reply, encoding="utf-8")
            logger.error("Unparseable LLM reply dumped to %s", dump_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Failed to dump unparseable extraction reply: %s", exc)

    async def _generate_entries_from_text(
        self,
        *,
        resource_text: str,
        store: Database,
        memory_types: list[MemoryType],
        categories_prompt_str: str,
        dossier_context: Mapping[str, Any] | None = None,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
        default_source_message_ids: list[int] | None = None,
        llm_client: Any | None = None,
        target_items_by_type: Mapping[str, str] | None = None,
    ) -> list[StructuredMemoryEntry]:
        if not memory_types:
            return []
        client = llm_client or self._select_chat_client(None)
        soul_context_str = self._format_soul_context_for_prompt(dossier_context)
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
        responses = list(await asyncio.gather(*tasks))
        for i, ((mtype, prompt), response) in enumerate(zip(valid_pairs, responses)):
            try:
                parsing._parse_memory_type_response_xml(response)
            except ValueError:
                self._dump_unparseable_reply(response, mtype, attempt=1)
                logger.error("Extraction reply unparseable for memory_type=%s — retrying", mtype)
                retry_response = await client.chat(prompt)
                try:
                    parsing._parse_memory_type_response_xml(retry_response)
                except ValueError as exc:
                    self._dump_unparseable_reply(retry_response, mtype, attempt=2)
                    snippet = repr(retry_response[:200])
                    raise ValueError(
                        f"Extraction reply still unparseable after retry for memory_type={mtype}: {snippet}"
                    ) from exc
                responses[i] = retry_response
        return self._parse_structured_entries(
            [mtype for mtype, _ in valid_pairs],
            responses,
            default_source_message_ids=default_source_message_ids,
            speaker_roster=speaker_roster,
        )

    @staticmethod
    def _normalize_category_name(raw: str) -> str | None:
        return normalize_category_name(raw)

    def _parse_structured_entries(
        self,
        memory_types: list[MemoryType],
        responses: Sequence[str],
        *,
        default_source_message_ids: list[int] | None = None,
        speaker_roster: Sequence[SpeakerRosterEntry] | None = None,
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

                # Memory items are generalized from the episode, not based on
                # individual messages. This stores code-owned provenance for the
                # whole episode/segment; stale emitted IDs are ignored.
                source_message_ids = self._resolve_source_message_ids(
                    entry.get("source_message_ids"),
                    default_source_message_ids,
                )

                reflection_salience = self._normalize_reflection_salience(entry.get("reflection_salience"))
                emotional_intensity = self._normalize_reflection_salience(entry.get("emotional_intensity"))
                replaces_previous_fact = self._normalize_replaces_previous_fact(entry.get("replaces_previous_fact"))
                entities = entry.get("entities")
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


    def file_category_proposals(
        self,
        *,
        store: Database,
        item_proposals: Sequence[tuple[MemoryItem, Sequence[str]]],
        where: Mapping[str, Any],
        session: Any,
    ) -> tuple[list[Any], list[Any]]:
        return categories.file_category_proposals(
            store=store,
            item_proposals=item_proposals,
            where=where,
            session=session,
        )

    async def prepare_dynamic_category_review(
        self,
        *,
        store: Database,
        where: Mapping[str, Any],
        cluster_size: int,
        cosine_threshold: float = 0.75,
    ) -> list[dict[str, Any]]:
        return await categories.prepare_dynamic_category_review(
            store=store,
            where=where,
            cluster_size=cluster_size,
            search_dossiers=self.search_dossiers,
            cosine_threshold=cosine_threshold,
        )

    async def generate_dynamic_category_review(
        self,
        bundle: Mapping[str, Any],
        *,
        chat_client: Any | None = None,
    ) -> dict[str, Any]:
        return await categories.generate_dynamic_category_review(
            bundle=bundle,
            select_chat_client=self._select_chat_client,
            profile=self.memorize_config.category_update_llm_profile,
            chat_client=chat_client,
        )

    def apply_dynamic_category_review(
        self,
        *,
        store: Database,
        where: Mapping[str, Any],
        bundle: Mapping[str, Any],
        decision: Mapping[str, Any],
        session: Any,
        proposed_embedding: Sequence[float] | None = None,
        near_duplicate_threshold: float = 0.95,
    ) -> dict[str, Any]:
        return categories.apply_dynamic_category_review(
            store=store,
            where=where,
            bundle=bundle,
            decision=decision,
            session=session,
            proposed_embedding=proposed_embedding,
            near_duplicate_threshold=near_duplicate_threshold,
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
        segment_id: str | None = None,
        extract_model: str | None = None,
        message_happened_at_map: Mapping[int, Any] | None = None,
        session: Any | None = None,
    ) -> tuple[list[MemoryItem], int]:
        items, homeless_count = await persistence._persist_memory_items(
            resource_id=resource_id,
            structured_entries=cast(list[Any], structured_entries),
            ctx=ctx,
            store=store,
            embed_client=embed_client or self._select_embedding_client(
                {"operation": "memorize", "step_id": "persist_memory_items"}
            ),
            user=user,
            conversation_id=conversation_id,
            segment_id=segment_id,
            extract_model=extract_model,
            message_happened_at_map=message_happened_at_map,
            session=session,
            enable_confidence_normalization=self.memorize_config.enable_confidence_normalization,
            normalize_confidence=lambda entries: cast(list[Any], self._normalize_confidence(cast(list[StructuredMemoryEntry], entries))),
            find_supersede_targets=self._find_supersede_targets,
            hedge_summary_for_confidence=self._hedge_summary_for_confidence,
            resolve_entry_happened_at=self._resolve_entry_happened_at,
        )
        return cast(list[MemoryItem], items), homeless_count

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

    async def _split_into_episodes(
        self, *, local_path: str, text: str | None, modality: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        return await segment_helpers._split_into_episodes(
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
        return await segment_helpers._prepare_audio_text(
            local_path,
            text,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "audio_transcription"}
            ),
        )

    def _modality_requires_text(self, modality: str) -> bool:
        return segment_helpers._modality_requires_text(modality)

    async def _dispatch_preprocessor(
        self,
        *,
        modality: str,
        local_path: str,
        text: str | None,
        template: str,
        llm_client: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await segment_helpers._dispatch_preprocessor(
            modality=modality,
            local_path=local_path,
            text=text,
            template=template,
            llm_client=llm_client,
            preprocess_video=self._preprocess_video,
            preprocess_image=self._preprocess_image,
            preprocess_document=self._preprocess_document,
            preprocess_audio=self._preprocess_audio,
        )

    async def _preprocess_video(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await segment_helpers._preprocess_video(
            local_path=local_path,
            template=template,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "video_preprocess"}
            ),
            parse_multimodal_response=self._parse_multimodal_response,
        )

    async def _preprocess_image(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await segment_helpers._preprocess_image(
            local_path=local_path,
            template=template,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "image_preprocess"}
            ),
            parse_multimodal_response=self._parse_multimodal_response,
        )

    async def _preprocess_document(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await segment_helpers._preprocess_document(
            text=text,
            template=template,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "document_preprocess"}
            ),
            escape_prompt_value=self._escape_prompt_value,
            parse_multimodal_response=self._parse_multimodal_response,
        )

    async def _preprocess_audio(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        return await segment_helpers._preprocess_audio(
            text=text,
            template=template,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "audio_preprocess"}
            ),
            escape_prompt_value=self._escape_prompt_value,
            parse_multimodal_response=self._parse_multimodal_response,
        )


    def _format_soul_context_for_prompt(
        self,
        dossier_context: Mapping[str, Any] | None,
    ) -> str:
        if not dossier_context:
            return "No prior knowledge about these participants exists yet."
        sections: list[str] = []
        narrative = str(dossier_context.get("narrative_self") or "").strip()
        if narrative:
            sections.append(f"## Your character, personality, and voice\n{narrative}")
        for category in dossier_context.get("anchor_dossiers") or []:
            prose = str(category.summary or "").strip()
            body = "\n".join(part for part in (category.description.strip(), prose) if part)
            sections.append(f"## Anchor dossier: {category.name}\n{body}")
        index = str(dossier_context.get("dossier_index") or "").strip()
        if index:
            sections.append(f"## Dossier index\n{index}")
        for category in dossier_context.get("relevant_dossiers") or []:
            prose = str(category.summary or "").strip()
            body = "\n".join(part for part in (category.description.strip(), prose) if part)
            sections.append(f"## Relevant dossier: {category.name}\n{body}")
        return "\n\n".join(sections)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        return segment_helpers._estimate_text_tokens(text)

    def _memory_type_target_items(self) -> str:
        try:
            min_chunk_tokens = int(getattr(self.memorize_config, "min_chunk_tokens", 4000) or 4000)
        except (TypeError, ValueError, OverflowError):
            min_chunk_tokens = 4000
        target = max(1, math.ceil(1.2 * (max(0, min_chunk_tokens) / 1000)))
        return f"up to {target}"

    @staticmethod
    def _message_is_primary_for_memorize(message: Mapping[str, Any]) -> bool:
        return segment_helpers._message_is_primary_for_memorize(message)

    @staticmethod
    def _message_index_for_sort(message: Mapping[str, Any]) -> int:
        return segment_helpers._message_index_for_sort(message)

    async def summarize_background_chat_rollup(
        self,
        *,
        prior_summary: str | None,
        messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
        soul_name: str | None = None,
    ) -> str:
        return await segment_helpers._summarize_background_rollup(
            prior_summary=prior_summary,
            messages=messages,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "background_rollup"}
            ),
            soul_name=soul_name,
        )

    async def _summarize_background_groups_batched(
        self,
        *,
        grouped_messages: Mapping[str, Sequence[Mapping[str, Any]]],
        group_order: Sequence[str],
        llm_client: Any | None = None,
        soul_name: str | None = None,
    ) -> dict[str, str]:
        return await segment_helpers._summarize_background_groups_batched(
            grouped_messages=grouped_messages,
            group_order=group_order,
            llm_client=llm_client or self._select_chat_client(
                {"operation": "memorize", "step_id": "background_batch_summary"}
            ),
            extract_json_blob=self._extract_json_blob,
            soul_name=soul_name,
        )

    async def _render_episode_with_background_context(
        self,
        *,
        primary_messages: Sequence[Mapping[str, Any]],
        background_messages: Sequence[Mapping[str, Any]],
        llm_client: Any | None = None,
        soul_name: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        return await segment_helpers._render_episode_with_background_context(
            primary_messages=primary_messages,
            background_messages=background_messages,
            llm_client=llm_client,
            summarize_background_groups_batched=self._summarize_background_groups_batched,
            memorize_config=self.memorize_config,
            soul_name=soul_name,
        )

    def _render_episode_with_summary_rows(
        self,
        *,
        primary_messages: Sequence[Mapping[str, Any]],
        summary_rows: Sequence[Mapping[str, Any]],
        soul_name: str | None = None,
    ) -> str:
        return segment_helpers._render_episode_with_summary_rows(
            primary_messages=primary_messages,
            summary_rows=summary_rows,
            soul_name=soul_name,
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
    @staticmethod
    def _dedupe_message_indices(values: Sequence[int | float | str]) -> list[int]:
        return parsing._dedupe_message_indices(values)

    def _resolve_source_message_ids(
        self,
        values: Any,
        allowed_values: Any = None,
    ) -> list[int]:
        """Resolve DB provenance IDs to the full episode/segment range.

        Memories are not based on individual messages. The persisted
        `source_message_ids` column is provenance coverage for the generalized
        episode/segment memory. Prompts stopped requesting the field in 02d8bde;
        stale emitted IDs are ignored so the model cannot narrow a memory to one
        line again.
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

    def _build_speaker_roster_for_segment(
        self,
        *,
        speaker_map: Mapping[int, tuple[str, str]] | None,
        declared_entities: Sequence[SpeakerRosterEntry] | None,
        segment_text: Any,
    ) -> list[SpeakerRosterEntry] | None:
        return speakers._build_speaker_roster_for_segment(
            speaker_map=speaker_map,
            declared_entities=declared_entities,
            segment_text=segment_text,
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
        segment_messages: Sequence[Mapping[str, Any]],
        scope: Mapping[str, Any] | None,
    ) -> dict[int, tuple[str, str]]:
        return speakers._build_speaker_map(segment_messages, scope)

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
        return segment_helpers._prepare_episode(
            modality=modality,
            text=text,
            message_indices=message_indices,
            dedupe_message_indices=self._dedupe_message_indices,
            extract_message_indices=self._extract_message_indices,
        )

    def _parse_multimodal_response(self, raw: str, content_tag: str, caption_tag: str) -> tuple[str | None, str | None]:
        return segment_helpers._parse_multimodal_response(
            raw,
            content_tag,
            caption_tag,
            extract_tag_content=self._extract_tag_content,
        )

    @staticmethod
    def _extract_tag_content(raw: str, tag: str) -> str | None:
        return segment_helpers._extract_tag_content(raw, tag)

    def _parse_memory_type_response_xml(self, raw: str) -> list[dict[str, Any]]:
        return parsing._parse_memory_type_response_xml(raw)
