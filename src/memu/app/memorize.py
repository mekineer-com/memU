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
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from pydantic import BaseModel

from memu.app.settings import CategoryConfig, CustomPrompt
from memu.database.models import CategoryItem, MemoryCategory, MemoryItem, MemoryType, Resource
from memu.prompts.category_summary import (
    CUSTOM_PROMPT as CATEGORY_SUMMARY_CUSTOM_PROMPT,
)
from memu.prompts.category_summary import (
    PROMPT as CATEGORY_SUMMARY_PROMPT,
)
from memu.prompts.diary.diary_worthy import PROMPT as DIARY_WORTHY_PROMPT
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


class StructuredMemoryEntry(NamedTuple):
    memory_type: MemoryType
    content: str
    categories: list[str]
    source_role: str | None
    confidence: float | None
    source_message_ids: list[int]
    reflection_salience: float | None


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

    def _segment_entry_sort_key(self, entry: StructuredMemoryEntry) -> tuple[float, int]:
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
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if 0.0 <= parsed <= 1.0:
            return parsed
        return None

    @staticmethod
    def _hedge_summary_for_confidence(summary: str, confidence: float | None) -> str:
        text = str(summary or "").strip()
        if not text or confidence is None or confidence >= 0.6:
            return text
        lowered = text[:1].lower() + text[1:] if text[:1].isupper() else text
        if confidence < 0.35:
            return f"I have a faint suspicion that {lowered}"
        return f"I have an inkling that {lowered}"

    def _default_reflection_salience(
        self,
        *,
        memory_type: MemoryType,
        confidence: float | None,
        diary_worthy: bool,
    ) -> float:
        """Heuristic fallback when extraction prompt doesn't emit reflection_salience.

        Calibration rationale:
        - 0.85 diary-worthy base: these memories cleared the diary gate, so they
          are likely defining.  0.35 non-diary base: useful but not formative.
        - Events get +0.05 because they anchor temporal narrative more than
          profiles do (profiles are identity, events are story).
        - Confidence adjustments: extraction certainty correlates weakly with
          personal importance, so small nudges (0.05-0.10) rather than large
          swings.  Low confidence already means tentative wording, so lower salience
          keeps them out of the backbone.

        The Plex exporter uses these values for inclusion tiers:
          backbone >= 0.85, plex-worthy >= 0.7, excluded < 0.7 (auto-scaled).
        """
        base = 0.85 if diary_worthy else 0.35
        if memory_type == "event":
            base += 0.05
        if confidence is None:
            return max(0.0, min(1.0, base))
        if confidence < 0.35:
            base -= 0.10
        elif confidence < 0.6:
            base -= 0.05
        elif confidence >= 0.9:
            base += 0.05
        return max(0.0, min(1.0, base))

    async def memorize(
        self,
        *,
        resource_url: str,
        modality: str,
        user: dict[str, Any] | None = None,
        raw_text: str | None = None,
        local_path: str | None = None,
    ) -> dict[str, Any]:
        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        conversation_id: str | None = None
        if isinstance(user, dict):
            for key in (
                "conversation_id",
                "conversationId",
                "conversationID",
                "conversationid",
                "session_id",
                "sessionId",
                "sessionID",
                "sessionid",
                "session_date",
                "sessionDate",
                "sessiondate",
            ):
                raw = user.get(key)
                if raw is None:
                    continue
                candidate = str(raw).strip()
                if candidate:
                    conversation_id = candidate
                    break
        if not conversation_id and isinstance(user_scope, dict):
            raw_sid = user_scope.get("session_id")
            if isinstance(raw_sid, str) and raw_sid.strip():
                conversation_id = raw_sid.strip()

        memory_types = self._resolve_memory_types()

        state: WorkflowState = {
            "resource_url": resource_url,
            "modality": modality,
            "memory_types": memory_types,
            "categories_prompt_str": self._category_prompt_str,
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
            "conversation_id": conversation_id,
        }

        # Optional fast path for callers that already have the resource text.
        # This avoids forcing the caller to write a temporary file and avoids
        # duplicating local files into blob_config.resources_dir.
        if raw_text is not None:
            state["raw_text"] = raw_text
            state["local_path"] = local_path or resource_url

        result = await self._run_workflow("memorize", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Memorize workflow failed to produce a response"
            raise RuntimeError(msg)
        homeless_count = int(result.get("homeless_item_count") or 0)
        homeless_trigger = int(getattr(self.memorize_config, "homeless_trigger_count", 20) or 20)
        logger.info(
            "category-centroid: homeless items this run=%s trigger=%s",
            homeless_count,
            homeless_trigger,
        )
        return response

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
                step_id="preprocess_multimodal",
                role="preprocess",
                handler=self._memorize_preprocess_multimodal,
                requires={"local_path", "modality", "raw_text"},
                produces={"preprocessed_resources"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.memorize_config.preprocess_llm_profile},
            ),
            WorkflowStep(
                step_id="extract_items",
                role="extract",
                handler=self._memorize_extract_items,
                requires={
                    "preprocessed_resources",
                    "memory_types",
                    "categories_prompt_str",
                    "modality",
                    "resource_url",
                },
                produces={"resource_plans"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.memorize_config.memory_extract_llm_profile},
            ),
            WorkflowStep(
                step_id="categorize_items",
                role="categorize",
                handler=self._memorize_categorize_items,
                requires={"resource_plans", "ctx", "store", "local_path", "modality", "user"},
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
        # If a caller pre-provided the resource contents, don't fetch/copy anything.
        if state.get("raw_text") is not None and state.get("local_path") is not None:
            return state

        local_path, raw_text = await self.fs.fetch(state["resource_url"], state["modality"])
        state.update({"local_path": local_path, "raw_text": raw_text})
        return state

    async def _memorize_preprocess_multimodal(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        preprocessed = await self._preprocess_resource_url(
            local_path=state["local_path"],
            text=state.get("raw_text"),
            modality=state["modality"],
            llm_client=llm_client,
        )
        if not preprocessed:
            preprocessed = [{"text": state.get("raw_text"), "caption": None}]
        state["preprocessed_resources"] = preprocessed
        return state

    async def _memorize_extract_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        preprocessed_resources = state.get("preprocessed_resources", [])
        resource_plans: list[dict[str, Any]] = []
        total_segments = len(preprocessed_resources) or 1
        diary_worthy_ids: list[int] = []

        for idx, prep in enumerate(preprocessed_resources):
            res_url = self._segment_resource_url(state["resource_url"], idx, total_segments)
            text = prep.get("text")
            caption = prep.get("caption")
            diary_segment_text, message_indices = self._prepare_diary_segment(
                modality=state["modality"],
                text=text if isinstance(text, str) else None,
                message_indices=prep.get("message_indices"),
            )
            diary_worthy = False
            if diary_segment_text:
                diary_worthy = await self._classify_diary_worthy_segment(
                    diary_segment_text,
                    llm_client=llm_client,
                )
                if diary_worthy:
                    diary_worthy_ids.extend(message_indices)

            structured_entries = await self._generate_structured_entries(
                resource_url=res_url,
                modality=state["modality"],
                memory_types=state["memory_types"],
                text=text,
                categories_prompt_str=state["categories_prompt_str"],
                llm_client=llm_client,
            )
            structured_entries = self._decorate_entries_with_plan_context(
                structured_entries,
                message_indices=message_indices,
                diary_worthy=diary_worthy,
            )

            resource_plans.append({
                "resource_url": res_url,
                "text": text,
                "caption": caption,
                "message_indices": message_indices,
                "diary_worthy": diary_worthy,
                "entries": structured_entries,
            })

        state["resource_plans"] = resource_plans
        state["diary_worthy_ids"] = self._dedupe_message_indices(diary_worthy_ids)
        return state

    async def _memorize_dedupe_merge(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        _ = step_context
        items = list(state.get("items") or [])
        state["items"] = items

        if not self.memorize_config.semantic_dedupe_enabled:
            return state
        if len(items) < 1:
            return state

        dedupe_scope = self._build_semantic_dedupe_scope(state.get("user"))
        if dedupe_scope is None:
            logger.info("dedupe: skipped (missing soul_id/user_id scope)")
            return state

        store = state["store"]
        # Pull from repository and only active items.
        active_pool = dict(store.memory_item_repo.list_items(dedupe_scope))
        if len(active_pool) < 2:
            return state

        new_item_ids: list[str] = []
        seen_new: set[str] = set()
        for item in items:
            item_id = getattr(item, "id", None)
            if not isinstance(item_id, str) or not item_id or item_id in seen_new:
                continue
            seen_new.add(item_id)
            pool_item = active_pool.get(item_id)
            if pool_item is None:
                continue
            if self._is_merged_item(pool_item):
                continue
            if self._item_embedding(pool_item) is None:
                continue
            new_item_ids.append(item_id)
        if not new_item_ids:
            return state

        try:
            threshold = float(self.memorize_config.semantic_dedupe_similarity_threshold)
        except Exception:
            threshold = 0.89
        threshold = max(0.0, min(1.0, threshold))
        merged_map: dict[str, str] = {}
        dedupe_embed_client: Any | None = None
        dedupe_embed_cache: dict[str, list[float] | None] = {}
        summary_tokens: dict[str, set[str]] = {}
        token_index: dict[str, set[str]] = {}
        token_freq: dict[str, int] = {}

        for pool_item_id, pool_item in active_pool.items():
            tokens = self._dedupe_summary_tokens(getattr(pool_item, "summary", ""))
            summary_tokens[pool_item_id] = tokens
            for token in tokens:
                token_index.setdefault(token, set()).add(pool_item_id)
                token_freq[token] = token_freq.get(token, 0) + 1

        for new_item_id in new_item_ids:
            anchor = active_pool.get(new_item_id)
            if anchor is None or self._is_merged_item(anchor):
                continue
            anchor_embedding = self._item_embedding(anchor)
            if anchor_embedding is None:
                continue

            candidates: list[tuple[float, str]] = []
            candidate_ids = self._prefilter_dedupe_candidate_ids(
                anchor_id=new_item_id,
                anchor=anchor,
                active_pool=active_pool,
                merged_map=merged_map,
                summary_tokens=summary_tokens,
                token_index=token_index,
                token_freq=token_freq,
            )
            for candidate_id in candidate_ids:
                candidate = active_pool.get(candidate_id)
                if candidate is None:
                    continue
                candidate_embedding = self._item_embedding(candidate)
                compare_anchor: list[float] | None = anchor_embedding
                compare_candidate: list[float] | None = candidate_embedding

                # Old rows may have vectors from a different embedding model/dimension.
                # Re-embed summaries on-demand so dedupe still works after model changes.
                if candidate_embedding is None or len(anchor_embedding) != len(candidate_embedding):
                    if dedupe_embed_client is None:
                        dedupe_embed_client = self._get_llm_client("embedding")
                    compare_anchor = await self._dedupe_reembed_for_similarity(
                        item=anchor,
                        embed_client=dedupe_embed_client,
                        cache=dedupe_embed_cache,
                    )
                    compare_candidate = await self._dedupe_reembed_for_similarity(
                        item=candidate,
                        embed_client=dedupe_embed_client,
                        cache=dedupe_embed_cache,
                    )
                if compare_anchor is None or compare_candidate is None or len(compare_anchor) != len(compare_candidate):
                    continue
                similarity = self._cosine_similarity(compare_anchor, compare_candidate)
                if similarity >= threshold:
                    candidates.append((similarity, candidate_id))
            if not candidates:
                continue

            # Deterministic order: highest similarity first, then stable by id.
            candidates.sort(key=lambda row: (-row[0], row[1]))
            for similarity, candidate_id in candidates:
                current_anchor = active_pool.get(new_item_id)
                candidate = active_pool.get(candidate_id)
                if current_anchor is None or candidate is None:
                    continue
                if self._is_merged_item(current_anchor) or self._is_merged_item(candidate):
                    continue

                survivor, redundant = self._choose_survivor_and_redundant(current_anchor, candidate)
                if survivor.id == redundant.id:
                    continue

                store.memory_item_repo.update_item(item_id=redundant.id, merged_into=survivor.id)
                merged_map[redundant.id] = survivor.id
                active_pool.pop(redundant.id, None)
                logger.info("dedupe: merged %s into %s (sim=%.3f)", redundant.id, survivor.id, similarity)

                # Once a new item is merged into an existing survivor, stop comparing it.
                if redundant.id == new_item_id:
                    break

        if not merged_map:
            return state

        merged_ids = set(merged_map.keys())
        state["items"] = [item for item in items if getattr(item, "id", None) not in merged_ids]
        state["relations"] = [
            rel for rel in (state.get("relations") or []) if getattr(rel, "item_id", None) not in merged_ids
        ]
        state["category_updates"] = self._filter_merged_from_category_updates(state.get("category_updates"), merged_ids)
        return state

    @staticmethod
    def _extract_scope_field(
        scope: Mapping[str, Any],
        *,
        keys: Sequence[str],
    ) -> tuple[str, str] | None:
        for key in keys:
            raw = scope.get(key)
            if raw is None:
                continue
            value = str(raw).strip()
            if value:
                return key, value
        return None

    def _build_semantic_dedupe_scope(self, scope: Mapping[str, Any] | None) -> dict[str, str] | None:
        if not isinstance(scope, Mapping):
            return None
        user_field = self._extract_scope_field(scope, keys=("user_id", "userId"))
        soul_field = self._extract_scope_field(scope, keys=("soul_id", "soulId"))
        if user_field is None or soul_field is None:
            return None
        _user_key, user_value = user_field
        _soul_key, soul_value = soul_field
        return {
            "user_id": user_value,
            "soul_id": soul_value,
        }

    @staticmethod
    def _normalize_embedding_vector(embedding: Any) -> list[float] | None:
        if not isinstance(embedding, Sequence):
            return None
        if isinstance(embedding, (str, bytes, bytearray)):
            return None
        normalized: list[float] = []
        for value in embedding:
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                return None
        return normalized if normalized else None

    @staticmethod
    def _is_merged_item(item: Any) -> bool:
        merged_into = getattr(item, "merged_into", None)
        return isinstance(merged_into, str) and merged_into.strip() != ""

    @staticmethod
    def _item_embedding(item: Any) -> list[float] | None:
        return MemorizeMixin._normalize_embedding_vector(getattr(item, "embedding", None))

    async def _dedupe_reembed_for_similarity(
        self,
        *,
        item: Any,
        embed_client: Any,
        cache: dict[str, list[float] | None],
    ) -> list[float] | None:
        item_id = str(getattr(item, "id", "")).strip()
        summary = str(getattr(item, "summary", "")).strip()
        cache_key = item_id or summary
        if not cache_key:
            return None
        if cache_key in cache:
            return cache[cache_key]
        if not summary:
            cache[cache_key] = None
            return None

        try:
            vectors = await embed_client.embed([summary])
        except Exception:
            logger.warning("dedupe: fallback re-embed failed for %s", item_id or "<no-id>", exc_info=True)
            cache[cache_key] = None
            return None

        vector: list[float] | None = None
        if isinstance(vectors, list) and vectors:
            raw = vectors[0]
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                normalized: list[float] = []
                for value in raw:
                    try:
                        normalized.append(float(value))
                    except (TypeError, ValueError):
                        normalized = []
                        break
                if normalized:
                    vector = normalized

        cache[cache_key] = vector
        return vector

    @staticmethod
    def _summary_len(item: Any) -> int:
        summary = getattr(item, "summary", "")
        return len(str(summary).strip())

    def _choose_survivor_and_redundant(self, left: MemoryItem, right: MemoryItem) -> tuple[MemoryItem, MemoryItem]:
        left_len = self._summary_len(left)
        right_len = self._summary_len(right)
        if left_len > right_len:
            return left, right
        if right_len > left_len:
            return right, left
        left_id = str(getattr(left, "id", ""))
        right_id = str(getattr(right, "id", ""))
        if left_id <= right_id:
            return left, right
        return right, left

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _filter_merged_from_category_updates(
        updates: Any,
        merged_ids: set[str],
    ) -> dict[str, list[tuple[str, str]]]:
        if not isinstance(updates, dict):
            return {}
        filtered: dict[str, list[tuple[str, str]]] = {}
        for category_id, item_tuples in updates.items():
            if not isinstance(item_tuples, list):
                continue
            kept: list[tuple[str, str]] = []
            for entry in item_tuples:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    continue
                item_id = str(entry[0]).strip()
                summary = str(entry[1])
                if not item_id or item_id in merged_ids:
                    continue
                kept.append((item_id, summary))
            if kept:
                filtered[str(category_id)] = kept
        return filtered

    @staticmethod
    def _dedupe_summary_tokens(summary: Any) -> set[str]:
        text = str(summary or "").lower()
        if not text:
            return set()
        stopwords = {
            "about",
            "after",
            "before",
            "being",
            "during",
            "from",
            "have",
            "just",
            "said",
            "some",
            "still",
            "that",
            "their",
            "them",
            "then",
            "there",
            "they",
            "this",
            "through",
            "very",
            "when",
            "where",
            "while",
            "with",
            "would",
        }
        out: set[str] = set()
        for token in re.findall(r"[a-z0-9]{4,}", text):
            if token in stopwords:
                continue
            out.add(token)
        return out

    @staticmethod
    def _looks_like_speech_act_event(summary: Any) -> bool:
        text = str(summary or "").strip()
        if not text:
            return False
        return bool(
            re.match(
                r"^(?:i|we|you|they|he|she|[A-Z][A-Za-z0-9_'-]*(?:\s+[A-Z][A-Za-z0-9_'-]*)*)\s+"
                r"(?:shared|mentioned|stated|said|noted|clarified|explained|described|summarized|emphasized|expressed|voiced|wrote|told|admitted|revealed)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _dedupe_source_role(item: Any) -> str | None:
        raw = getattr(item, "source_role", None)
        if not isinstance(raw, str):
            return None
        value = raw.strip()
        return value or None

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
        anchor_role = self._dedupe_source_role(anchor)
        anchor_tokens = summary_tokens.get(anchor_id) or set()
        candidate_scores: dict[str, int] = {}

        selected_tokens = sorted(
            anchor_tokens,
            key=lambda token: (token_freq.get(token, 0), -len(token), token),
        )[:4]

        for token in selected_tokens:
            for candidate_id in token_index.get(token, set()):
                if candidate_id == anchor_id or candidate_id in merged_map:
                    continue
                candidate = active_pool.get(candidate_id)
                if candidate is None or self._is_merged_item(candidate):
                    continue
                candidate_role = self._dedupe_source_role(candidate)
                if anchor_role and candidate_role and anchor_role != candidate_role:
                    continue
                overlap = len(anchor_tokens & (summary_tokens.get(candidate_id) or set()))
                if overlap <= 0:
                    continue
                prev = candidate_scores.get(candidate_id, 0)
                if overlap > prev:
                    candidate_scores[candidate_id] = overlap

        if not candidate_scores:
            for candidate_id, candidate in active_pool.items():
                if candidate_id == anchor_id or candidate_id in merged_map:
                    continue
                if self._is_merged_item(candidate):
                    continue
                candidate_role = self._dedupe_source_role(candidate)
                if anchor_role and candidate_role and anchor_role != candidate_role:
                    continue
                candidate_scores[candidate_id] = 0

        ordered = sorted(candidate_scores.items(), key=lambda row: (-row[1], row[0]))
        return [candidate_id for candidate_id, _score in ordered[:64]]

    def _build_category_centroids(
        self,
        *,
        store: Database,
        user: Mapping[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        where = dict(user or {}) if isinstance(user, Mapping) else {}
        relations = store.category_item_repo.list_relations(where)
        if not relations:
            return {}

        items_by_id = store.memory_item_repo.list_items(where)
        if not items_by_id:
            return {}

        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for rel in relations:
            item = items_by_id.get(rel.item_id)
            if item is None or self._is_merged_item(item):
                continue
            embedding = self._item_embedding(item)
            if embedding is None:
                continue
            total = sums.get(rel.category_id)
            if total is None:
                sums[rel.category_id] = list(embedding)
                counts[rel.category_id] = 1
                continue
            if len(total) != len(embedding):
                continue
            for idx, value in enumerate(embedding):
                total[idx] += value
            counts[rel.category_id] = counts.get(rel.category_id, 0) + 1

        centroids: dict[str, list[float]] = {}
        for category_id, total in sums.items():
            count = counts.get(category_id, 0)
            if count <= 0:
                continue
            centroids[category_id] = [value / count for value in total]
        return centroids

    def _apply_category_centroid_gate(
        self,
        *,
        structured_entries: list[StructuredMemoryEntry],
        item_embeddings: Sequence[Any],
        ctx: Context,
        category_centroids: Mapping[str, Sequence[float]],
    ) -> tuple[list[StructuredMemoryEntry], set[int]]:
        if not structured_entries or not category_centroids:
            return structured_entries, set()

        try:
            threshold = float(getattr(self.memorize_config, "category_centroid_threshold", 0.65) or 0.65)
        except Exception:
            threshold = 0.65
        threshold = max(0.0, min(1.0, threshold))

        updated: list[StructuredMemoryEntry] = []
        gated_indexes: set[int] = set()

        for idx, (
            (memory_type, summary_text, cat_names, source_role, confidence, source_message_ids, reflection_salience),
            raw_embedding,
        ) in enumerate(zip(structured_entries, item_embeddings, strict=True)):
            embedding = self._normalize_embedding_vector(raw_embedding)
            if embedding is None:
                updated.append(
                    StructuredMemoryEntry(
                        memory_type,
                        summary_text,
                        cat_names,
                        source_role,
                        confidence,
                        source_message_ids,
                        reflection_salience,
                    )
                )
                continue

            existing_names: list[str] = []
            unknown_names: list[str] = []
            for name in cat_names or []:
                key = name.strip().lower()
                if key and key in ctx.category_name_to_id:
                    existing_names.append(name)
                else:
                    unknown_names.append(name)
            if not existing_names:
                updated.append(
                    StructuredMemoryEntry(
                        memory_type,
                        summary_text,
                        cat_names,
                        source_role,
                        confidence,
                        source_message_ids,
                        reflection_salience,
                    )
                )
                continue

            max_similarity: float | None = None
            for centroid in category_centroids.values():
                if len(centroid) != len(embedding):
                    continue
                similarity = self._cosine_similarity(embedding, centroid)
                if max_similarity is None or similarity > max_similarity:
                    max_similarity = similarity

            if max_similarity is None or max_similarity >= threshold:
                updated.append(
                    StructuredMemoryEntry(
                        memory_type,
                        summary_text,
                        cat_names,
                        source_role,
                        confidence,
                        source_message_ids,
                        reflection_salience,
                    )
                )
                continue

            gated_indexes.add(idx)
            updated.append(
                StructuredMemoryEntry(
                    memory_type,
                    summary_text,
                    unknown_names,
                    source_role,
                    confidence,
                    source_message_ids,
                    reflection_salience,
                )
            )

        return updated, gated_indexes

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
        pending_diary_memory_ids: list[str] = []
        user_scope = state.get("user", {})
        category_centroids = self._build_category_centroids(store=store, user=user_scope)
        homeless_item_count = 0
        diary_threshold = 0.75

        session_cm = self._sqlite_write_session(store)
        if session_cm is not None:
            with session_cm as session:
                try:
                    for plan in state.get("resource_plans", []):
                        res = await self._create_resource_with_caption(
                            resource_url=plan["resource_url"],
                            modality=modality,
                            local_path=local_path,
                            caption=plan.get("caption"),
                            store=store,
                            embed_client=embed_client,
                            user=user_scope,
                            session=session,
                        )
                        resources.append(res)

                        entries = plan.get("entries") or []
                        if not entries:
                            continue

                        mem_items, rels, cat_updates, homeless_delta = await self._persist_memory_items(
                            resource_id=res.id,
                            structured_entries=entries,
                            ctx=ctx,
                            store=store,
                            category_centroids=category_centroids,
                            embed_client=embed_client,
                            user=user_scope,
                            conversation_id=state.get("conversation_id"),
                            session=session,
                        )
                        items.extend(mem_items)
                        for item in mem_items:
                            salience = getattr(item, "reflection_salience", None)
                            if isinstance(salience, (int, float)) and float(salience) >= diary_threshold:
                                pending_diary_memory_ids.append(item.id)
                        relations.extend(rels)
                        homeless_item_count += homeless_delta
                        for cat_id, mems in cat_updates.items():
                            category_updates.setdefault(cat_id, []).extend(mems)
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
        else:
            for plan in state.get("resource_plans", []):
                res = await self._create_resource_with_caption(
                    resource_url=plan["resource_url"],
                    modality=modality,
                    local_path=local_path,
                    caption=plan.get("caption"),
                    store=store,
                    embed_client=embed_client,
                    user=user_scope,
                )
                resources.append(res)

                entries = plan.get("entries") or []
                if not entries:
                    continue

                mem_items, rels, cat_updates, homeless_delta = await self._persist_memory_items(
                    resource_id=res.id,
                    structured_entries=entries,
                    ctx=ctx,
                    store=store,
                    category_centroids=category_centroids,
                    embed_client=embed_client,
                    user=user_scope,
                    conversation_id=state.get("conversation_id"),
                )
                items.extend(mem_items)
                for item in mem_items:
                    salience = getattr(item, "reflection_salience", None)
                    if isinstance(salience, (int, float)) and float(salience) >= diary_threshold:
                        pending_diary_memory_ids.append(item.id)
                relations.extend(rels)
                homeless_item_count += homeless_delta
                for cat_id, mems in cat_updates.items():
                    category_updates.setdefault(cat_id, []).extend(mems)

        state.update({
            "resources": resources,
            "items": items,
            "relations": relations,
            "category_updates": category_updates,
            "homeless_item_count": homeless_item_count,
            "pending_diary_memory_ids": list(dict.fromkeys(pending_diary_memory_ids)),
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
                "pending_diary_memory_ids": state.get("pending_diary_memory_ids", []),
            }
        else:
            response = {
                "resources": resources,
                "items": items,
                "categories": categories,
                "relations": relations,
                "pending_diary_memory_ids": state.get("pending_diary_memory_ids", []),
            }
        state["response"] = response
        return state

    def _segment_resource_url(self, base_url: str, idx: int, total_segments: int) -> str:
        if total_segments <= 1:
            return base_url
        path = pathlib.Path(base_url)
        return f"{path.stem}_#segment_{idx}{path.suffix}"

    async def _fetch_and_preprocess_resource(
        self, resource_url: str, modality: str, llm_client: Any | None = None
    ) -> tuple[str, list[dict[str, str | None]]]:
        """
        Fetch and preprocess a resource.

        Returns:
            Tuple of (local_path, preprocessed_resources)
            where preprocessed_resources is a list of dicts with 'text' and 'caption'
        """
        local_path, text = await self.fs.fetch(resource_url, modality)
        preprocessed_resources = await self._preprocess_resource_url(
            local_path=local_path,
            text=text,
            modality=modality,
            llm_client=llm_client,
        )
        return local_path, preprocessed_resources

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
        session: Any | None = None,
    ) -> Resource:
        caption_text = caption.strip() if caption else None
        if caption_text:
            client = embed_client or self._get_llm_client("embedding")
            caption_embedding = (await client.embed([caption_text]))[0]
        else:
            caption_embedding = None

        resource_kwargs = {
            "url": resource_url,
            "modality": modality,
            "local_path": local_path,
            "caption": caption_text,
            "embedding": caption_embedding,
            "user_data": dict(user or {}),
        }
        if session is not None:
            res = cast(Any, store.resource_repo).create_resource(**resource_kwargs, session=session)
        else:
            res = store.resource_repo.create_resource(**resource_kwargs)
        # if caption:
        #     caption_text = caption.strip()
        #     if caption_text:
        #         res.caption = caption_text
        #         client = embed_client or self._get_llm_client()
        #         res.embedding = (await client.embed([caption_text]))[0]
        #         res.updated_at = pendulum.now()
        return cast(Resource, res)

    @staticmethod
    def _sqlite_write_session(store: Database) -> Any | None:
        try:
            from memu.database.sqlite.sqlite import SQLiteStore
        except Exception:
            return None
        if isinstance(store, SQLiteStore):
            return store._sessions.session()
        return None

    def _resolve_memory_types(self) -> list[MemoryType]:
        configured_types = self.memorize_config.memory_types or DEFAULT_MEMORY_TYPES
        return [cast(MemoryType, mtype) for mtype in configured_types]

    def _resolve_summary_prompt(self, modality: str, override: str | None) -> str | None:
        memo_settings = self.memorize_config
        result = memo_settings.multimodal_preprocess_prompts.get(modality)
        if override:
            return override
        if result is None:
            return (
                memo_settings.default_category_summary_prompt
                if isinstance(memo_settings.default_category_summary_prompt, str)
                else None
            )
        return result if isinstance(result, str) else None

    def _resolve_multimodal_preprocess_prompt(self, modality: str) -> str | None:
        memo_settings = self.memorize_config
        result = memo_settings.multimodal_preprocess_prompts.get(modality)
        return result if isinstance(result, str) else None

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
            # raise ValueError(f"No valid blocks contained in custom prompt: {prompt}")
            return ""
        sorted_blocks = sorted(valid_blocks)
        return "\n\n".join(block for (_, _, block) in sorted_blocks if block is not None)

    async def _generate_structured_entries(
        self,
        *,
        resource_url: str,
        modality: str,
        memory_types: list[MemoryType],
        text: str | None,
        categories_prompt_str: str,
        segments: list[dict[str, int | str]] | None = None,
        llm_client: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        if not memory_types:
            return []

        client = llm_client or self._get_llm_client()
        if text:
            entries = await self._generate_text_entries(
                resource_text=text,
                modality=modality,
                memory_types=memory_types,
                categories_prompt_str=categories_prompt_str,
                segments=segments,
                llm_client=client,
            )
            return entries
            # if entries:
            #     return entries
            # no_result_entry = self._build_no_result_fallback(memory_types[0], resource_url, modality)
            # return [no_result_entry]

        return []
        # return self._build_no_text_fallback(memory_types, resource_url, modality)

    async def _generate_text_entries(
        self,
        *,
        resource_text: str,
        modality: str,
        memory_types: list[MemoryType],
        categories_prompt_str: str,
        segments: list[dict[str, int | str]] | None,
        llm_client: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        if modality == "conversation" and segments:
            segment_entries = await self._generate_entries_for_segments(
                resource_text=resource_text,
                segments=segments,
                memory_types=memory_types,
                categories_prompt_str=categories_prompt_str,
                llm_client=llm_client,
            )
            if segment_entries:
                return segment_entries
        return await self._generate_entries_from_text(
            resource_text=resource_text,
            memory_types=memory_types,
            categories_prompt_str=categories_prompt_str,
            default_source_message_ids=self._extract_message_indices(resource_text)
            if modality == "conversation"
            else None,
            llm_client=llm_client,
        )

    async def _generate_entries_for_segments(
        self,
        *,
        resource_text: str,
        segments: list[dict[str, int | str]],
        memory_types: list[MemoryType],
        categories_prompt_str: str,
        llm_client: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        entries: list[StructuredMemoryEntry] = []
        lines = resource_text.split("\n")
        max_idx = len(lines) - 1
        for segment in segments:
            start_idx = int(segment.get("start", 0))
            end_idx = int(segment.get("end", max_idx))
            segment_text = self._extract_segment_text(lines, start_idx, end_idx)
            if not segment_text:
                continue
            applicable_types = await self._route_segment(segment_text, memory_types, llm_client)
            if not applicable_types:
                continue
            segment_entries = await self._generate_entries_from_text(
                resource_text=segment_text,
                memory_types=applicable_types,
                categories_prompt_str=categories_prompt_str,
                default_source_message_ids=self._extract_message_indices(segment_text),
                llm_client=llm_client,
            )
            segment_entries = sorted(segment_entries, key=self._segment_entry_sort_key, reverse=True)[:3]
            entries.extend(segment_entries)
        return entries

    async def _route_segment(
        self,
        segment_text: str,
        memory_types: list[MemoryType],
        llm_client: Any | None = None,
    ) -> list[MemoryType]:
        if not memory_types:
            return []
        client = llm_client or self._get_llm_client()
        prompt = ROUTER_PROMPT.format(
            segment=segment_text,
            allowed_types=list(memory_types),
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
                return memory_types
        if not isinstance(payload, dict):
            return memory_types
        routed_types = payload.get("types")
        if not isinstance(routed_types, list):
            return memory_types
        allowed_types = {
            routed_type
            for routed_type in routed_types
            if isinstance(routed_type, str) and routed_type in set(memory_types)
        }
        return [mtype for mtype in memory_types if mtype in allowed_types]

    async def _generate_entries_from_text(
        self,
        *,
        resource_text: str,
        memory_types: list[MemoryType],
        categories_prompt_str: str,
        default_source_message_ids: list[int] | None = None,
        llm_client: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        if not memory_types:
            return []
        client = llm_client or self._get_llm_client()
        prompts = [
            self._build_memory_type_prompt(
                memory_type=mtype,
                resource_text=resource_text,
                categories_str=categories_prompt_str,
            )
            for mtype in memory_types
        ]
        valid_prompts = [prompt for prompt in prompts if prompt.strip()]
        # These prompts are instructions that request structured output, not text summaries.
        tasks = [client.chat(prompt_text) for prompt_text in valid_prompts]
        responses = await asyncio.gather(*tasks)
        return self._parse_structured_entries(
            memory_types,
            responses,
            default_source_message_ids=default_source_message_ids,
        )

    @staticmethod
    def _normalize_category_name(raw: str) -> str | None:
        if not isinstance(raw, str):
            return None
        s = raw.strip().lower()
        if not s:
            return None
        # Normalize common human labels into stable snake_case names.
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s or None

    def _parse_structured_entries(
        self,
        memory_types: list[MemoryType],
        responses: Sequence[str],
        *,
        default_source_message_ids: list[int] | None = None,
    ) -> list[StructuredMemoryEntry]:
        entries: list[StructuredMemoryEntry] = []
        for mtype, response in zip(memory_types, responses, strict=True):
            parsed = self._parse_memory_type_response_xml(response)
            # if not parsed:
            #     fallback_entry = response.strip()
            #     if fallback_entry:
            #         entries.append((mtype, fallback_entry, []))
            #     continue
            for entry in parsed:
                content = (entry.get("content") or "").strip()
                if not content:
                    continue
                source_role_raw = entry.get("source_role")
                source_role = None
                if isinstance(source_role_raw, str):
                    normalized_role = source_role_raw.strip().lower()
                    if normalized_role in {"soul", "user", "environment"}:
                        source_role = normalized_role

                confidence = None
                confidence_raw = entry.get("confidence")
                if confidence_raw is not None:
                    try:
                        parsed_confidence = float(confidence_raw)
                    except (TypeError, ValueError):
                        parsed_confidence = None
                    if parsed_confidence is not None and 0.0 <= parsed_confidence <= 1.0:
                        confidence = parsed_confidence

                source_message_ids = self._resolve_source_message_ids(
                    entry.get("source_message_ids") if isinstance(entry.get("source_message_ids"), list) else None,
                    default_source_message_ids,
                )

                reflection_salience = self._normalize_reflection_salience(entry.get("reflection_salience"))

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
                    )
                )
        return self._prune_extracted_entry_duplicates(entries)

    def _prune_extracted_entry_duplicates(
        self,
        entries: list[StructuredMemoryEntry],
    ) -> list[StructuredMemoryEntry]:
        if len(entries) < 2:
            return entries

        profile_tokens: list[tuple[str | None, set[str]]] = []
        for (
            memory_type,
            summary,
            _cat_names,
            source_role,
            _confidence,
            _source_message_ids,
            _reflection_salience,
        ) in entries:
            if memory_type != "profile":
                continue
            tokens = self._dedupe_summary_tokens(summary)
            if tokens:
                profile_tokens.append((source_role, tokens))

        seen_exact: set[tuple[str, str | None, str]] = set()
        kept: list[StructuredMemoryEntry] = []
        for (
            memory_type,
            summary,
            cat_names,
            source_role,
            confidence,
            source_message_ids,
            reflection_salience,
        ) in entries:
            normalized_summary = re.sub(r"\s+", " ", str(summary or "").strip())
            exact_key = (memory_type, source_role, normalized_summary.casefold())
            if exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)

            if memory_type == "event" and profile_tokens and self._looks_like_speech_act_event(normalized_summary):
                event_tokens = self._dedupe_summary_tokens(normalized_summary)
                if event_tokens:
                    drop_event = False
                    for profile_role, profile_summary_tokens in profile_tokens:
                        if source_role and profile_role and source_role != profile_role:
                            continue
                        overlap = len(event_tokens & profile_summary_tokens)
                        if overlap <= 0:
                            continue
                        union = len(event_tokens | profile_summary_tokens)
                        if union and (overlap / union) >= 0.45:
                            drop_event = True
                            break
                    if drop_event:
                        continue

                kept.append(
                    StructuredMemoryEntry(
                        memory_type,
                        normalized_summary,
                        cat_names,
                        source_role,
                        confidence,
                        source_message_ids,
                        reflection_salience,
                    )
                )

        return kept

    def _extract_segment_text(self, lines: list[str], start_idx: int, end_idx: int) -> str | None:
        segment_lines = []
        for line in lines:
            match = re.match(r"\[(\d+)\]", line)
            if not match:
                continue
            idx = int(match.group(1))
            if start_idx <= idx <= end_idx:
                segment_lines.append(line)
        return "\n".join(segment_lines) if segment_lines else None

    def _decorate_entries_with_plan_context(
        self,
        entries: list[StructuredMemoryEntry],
        *,
        message_indices: list[int],
        diary_worthy: bool,
    ) -> list[StructuredMemoryEntry]:
        if not entries:
            return entries
        decorated: list[StructuredMemoryEntry] = []
        default_ids = self._dedupe_message_indices(message_indices)
        for memory_type, content, cats, source_role, confidence, source_message_ids, reflection_salience in entries:
            resolved_ids = self._resolve_source_message_ids(source_message_ids, default_ids)
            resolved_salience = (
                reflection_salience
                if reflection_salience is not None
                else self._default_reflection_salience(
                    memory_type=memory_type,
                    confidence=confidence,
                    diary_worthy=diary_worthy,
                )
            )
            decorated.append(
                StructuredMemoryEntry(
                    memory_type,
                    content,
                    cats,
                    source_role,
                    confidence,
                    resolved_ids,
                    resolved_salience,
                )
            )
        return decorated

    def _build_no_text_fallback(
        self, memory_types: list[MemoryType], resource_url: str, modality: str
    ) -> list[StructuredMemoryEntry]:
        fallback = f"Resource {resource_url} ({modality}) stored. No text summary in v0."
        return [
            StructuredMemoryEntry(mtype, f"{fallback} (memory type: {mtype}).", [], None, None, [], None)
            for mtype in memory_types
        ]

    def _build_no_result_fallback(
        self, memory_type: MemoryType, resource_url: str, modality: str
    ) -> StructuredMemoryEntry:
        fallback = f"Resource {resource_url} ({modality}) stored. No structured memories generated."
        return StructuredMemoryEntry(memory_type, fallback, [], None, None, [], None)

    async def _maybe_create_dynamic_categories(
        self,
        *,
        structured_entries: list[StructuredMemoryEntry],
        ctx: Context,
        store: Database,
        embed_client: Any,
        user: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> list[StructuredMemoryEntry]:
        """Resolve category names, optionally creating new *main* categories.

        Inspired by memU v0.1.8 cluster logic:
        - First, only accept existing categories.
        - Then, in a separate step, propose NEW categories under strict rules.

        Returns a new structured_entries list with category names mapped to existing/new categories.
        """

        if not getattr(self.memorize_config, "allow_dynamic_categories", False):
            return structured_entries

        await self._ensure_categories_ready(ctx, store, user)

        max_total = int(getattr(self.memorize_config, "max_categories_total", 0) or 0)
        min_mentions = int(getattr(self.memorize_config, "dynamic_category_min_mentions", 10) or 10)
        policy = str(getattr(self.memorize_config, "dynamic_category_policy", "") or "").strip()
        default_desc = str(getattr(self.memorize_config, "dynamic_category_description", "") or "").strip()

        cur_total = len(getattr(ctx, "category_ids", []) or [])
        remaining = (max_total - cur_total) if max_total else None
        if remaining is not None and remaining <= 0:
            # No capacity for new categories; drop unknowns.
            filtered: list[StructuredMemoryEntry] = []
            for (
                mtype,
                content,
                cats,
                source_role,
                confidence,
                source_message_ids,
                reflection_salience,
            ) in structured_entries:
                kept = [c for c in (cats or []) if c in ctx.category_name_to_id]
                filtered.append(
                    StructuredMemoryEntry(
                        mtype,
                        content,
                        kept,
                        source_role,
                        confidence,
                        source_message_ids,
                        reflection_salience,
                    )
                )
            return filtered

        # Split known vs unknown categories, while counting unknown mentions.
        unknown_counts: dict[str, int] = {}
        unknown_examples: dict[str, list[str]] = {}
        per_entry_unknowns: list[list[str]] = []
        filtered_entries: list[StructuredMemoryEntry] = []

        for (
            mtype,
            content,
            cats,
            source_role,
            confidence,
            source_message_ids,
            reflection_salience,
        ) in structured_entries:
            known: list[str] = []
            unknown: list[str] = []
            for c in cats or []:
                n = self._normalize_category_name(c)
                if not n:
                    continue
                if n in ctx.category_name_to_id:
                    known.append(n)
                else:
                    unknown.append(n)
                    unknown_counts[n] = unknown_counts.get(n, 0) + 1
                    ex_list = unknown_examples.setdefault(n, [])
                    if len(ex_list) < 3:
                        ex_list.append(content)

            # Deduplicate known while preserving order.
            seen: set[str] = set()
            known_dedup: list[str] = []
            for k in known:
                if k not in seen:
                    known_dedup.append(k)
                    seen.add(k)
            filtered_entries.append(
                StructuredMemoryEntry(
                    mtype,
                    content,
                    known_dedup,
                    source_role,
                    confidence,
                    source_message_ids,
                    reflection_salience,
                )
            )
            per_entry_unknowns.append(unknown)

        if not unknown_counts:
            return filtered_entries

        # Limit prompt size: keep top candidates by count.
        candidates_sorted = sorted(unknown_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_candidates = [name for name, _ in candidates_sorted[:30]]

        # Build planning prompt context.
        existing_lines: list[str] = []
        for cfg in self.memorize_config.memory_categories or []:
            nm = (cfg.name or "").strip()
            if not nm:
                continue
            desc = (cfg.description or "").strip()
            existing_lines.append(f"- {nm}: {desc}" if desc else f"- {nm}")
        existing_block = "\n".join(existing_lines) if existing_lines else "(none)"

        cand_lines: list[str] = []
        for cand in top_candidates:
            cnt = unknown_counts.get(cand, 0)
            examples = unknown_examples.get(cand, [])
            ex_block = "\n".join(f"    - {e}" for e in examples) if examples else ""
            cand_lines.append(f"- {cand} (count={cnt})\n{ex_block}" if ex_block else f"- {cand} (count={cnt})")
        candidates_block = "\n".join(cand_lines)
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
- Only propose NEW categories if the topic is a broad life domain AND it is mentioned at least {min_mentions} times across the extracted memories OR it is clearly important.
- If multiple close candidates (e.g., health/medical) should be merged, merge them and accumulate the count.
- New category names must be 1-3 words, letters/spaces only (no underscores); we will normalize to snake_case later.
- Do not recreate an existing category under a new name.

Output ONLY valid JSON with this shape:
{{"create": [{{"name": "...", "description": "...", "from": ["candidate1", "candidate2"], "important": true|false}}], "map": [{{"from": "candidate", "to": "existing_category"}}]}}
""".strip()

        user_prompt = f"""EXISTING CATEGORIES:
{existing_block}

CANDIDATE CATEGORY LABELS (from extraction, not in existing list):
{candidates_block}

Decide which candidates should map into existing categories, and which (if any) justify creating a NEW life-domain category.""".strip()

        mapping: dict[str, str] = {}
        new_defs: dict[str, str] = {}  # normalized_name -> description
        planner_fallback_reason: str | None = None

        def _valid_new_name(raw: str) -> bool:
            raw = (raw or "").strip()
            if not raw:
                return False
            if len(raw.split()) > 3:
                return False
            return bool(re.fullmatch(r"[A-Za-z ]+", raw))

        # Ask the LLM to produce a plan; fall back to count-only if it fails.
        try:
            planner_profile = getattr(self.memorize_config, "category_update_llm_profile", "default")
            planner = self._get_llm_client(planner_profile)
            resp = await planner.chat(user_prompt, system_prompt=system_prompt, temperature=0.2)

            m = re.search(r"\{[\s\S]*\}", resp or "")
            plan = None
            if m:
                import json

                plan = json.loads(m.group(0))
            else:
                planner_fallback_reason = "planner_no_json"

            if isinstance(plan, dict):
                # Create directives
                for entry in plan.get("create", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    raw_name = str(entry.get("name", "") or "").strip()
                    if not _valid_new_name(raw_name):
                        continue
                    desc = str(entry.get("description", "") or "").strip() or default_desc
                    important = bool(entry.get("important", False))
                    src = entry.get("from", []) or []
                    if not isinstance(src, list):
                        src = [src]
                    src_norm: list[str] = []
                    for s in src:
                        if isinstance(s, str):
                            sn = self._normalize_category_name(s)
                            if sn and sn in unknown_counts:
                                src_norm.append(sn)
                    if not src_norm:
                        continue

                    total = sum(unknown_counts.get(s, 0) for s in src_norm)
                    if total < min_mentions and not important:
                        continue

                    norm_name = self._normalize_category_name(raw_name)
                    if not norm_name:
                        continue
                    if norm_name in ctx.category_name_to_id:
                        for s in src_norm:
                            mapping[s] = norm_name
                        continue

                    new_defs.setdefault(norm_name, desc)
                    for s in src_norm:
                        mapping[s] = norm_name

                # Map directives
                for entry in plan.get("map", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    src = self._normalize_category_name(str(entry.get("from", "") or ""))
                    tgt = self._normalize_category_name(str(entry.get("to", "") or ""))
                    if not src or src not in unknown_counts:
                        continue
                    if tgt and (tgt in ctx.category_name_to_id or tgt in new_defs):
                        mapping[src] = tgt
            elif planner_fallback_reason is None:
                planner_fallback_reason = "planner_invalid_plan"
        except Exception:
            planner_fallback_reason = "planner_exception"
            logger.warning("dynamic-category planner failed; using heuristic fallback", exc_info=True)

        # Fallback: create frequent candidates directly (no LLM)
        if not new_defs and not mapping:
            if planner_fallback_reason is None:
                planner_fallback_reason = "planner_no_usable_directives"
            logger.warning(
                "dynamic-category planner fallback active (%s); applying heuristic category creation for %d candidates",
                planner_fallback_reason,
                len(candidates_sorted),
            )
            for cand, cnt in candidates_sorted:
                if cnt < min_mentions:
                    continue
                mapping[cand] = cand
                new_defs.setdefault(cand, default_desc)

        # Apply capacity limits.
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

        # Rebuild entries with mapped categories.
        updated: list[StructuredMemoryEntry] = []
        for (mtype, content, known, source_role, confidence, source_message_ids, reflection_salience), unk in zip(
            filtered_entries, per_entry_unknowns, strict=True
        ):
            cats = list(known)
            for u in unk or []:
                tgt = mapping.get(u)
                if not tgt:
                    continue
                tn = self._normalize_category_name(tgt)
                if tn and tn in ctx.category_name_to_id and tn not in cats:
                    cats.append(tn)
            updated.append(
                StructuredMemoryEntry(
                    mtype,
                    content,
                    cats,
                    source_role,
                    confidence,
                    source_message_ids,
                    reflection_salience,
                )
            )

        return updated

    async def _persist_memory_items(
        self,
        *,
        resource_id: str,
        structured_entries: list[StructuredMemoryEntry],
        ctx: Context,
        store: Database,
        category_centroids: Mapping[str, Sequence[float]] | None = None,
        embed_client: Any | None = None,
        user: Mapping[str, Any] | None = None,
        conversation_id: str | None = None,
        session: Any | None = None,
    ) -> tuple[list[MemoryItem], list[CategoryItem], dict[str, list[tuple[str, str]]], int]:
        """
        Persist memory items and track category updates.

        Returns:
            Tuple of (items, relations, category_updates, homeless_count)
            where category_updates maps category_id -> list of (item_id, summary) tuples
        """
        summary_payloads = [content for _, content, _, _, _, _, _ in structured_entries]
        client = embed_client or self._get_llm_client()
        item_embeddings = await client.embed(summary_payloads) if summary_payloads else []
        items: list[MemoryItem] = []
        rels: list[CategoryItem] = []
        # Changed: now stores (item_id, summary) tuples for reference support
        category_memory_updates: dict[str, list[tuple[str, str]]] = {}
        centroid_gated_indexes: set[int] = set()

        if category_centroids:
            structured_entries, centroid_gated_indexes = self._apply_category_centroid_gate(
                structured_entries=structured_entries,
                item_embeddings=item_embeddings,
                ctx=ctx,
                category_centroids=category_centroids,
            )

        reinforce = self.memorize_config.enable_item_reinforcement
        structured_entries = await self._maybe_create_dynamic_categories(
            structured_entries=structured_entries,
            ctx=ctx,
            store=store,
            embed_client=client,
            user=user,
            session=session,
        )
        homeless_count = sum(
            1 for idx, entry in enumerate(structured_entries) if idx in centroid_gated_indexes and not entry[2]
        )
        for (
            memory_type,
            summary_text,
            cat_names,
            source_role,
            confidence,
            source_message_ids,
            reflection_salience,
        ), emb in zip(structured_entries, item_embeddings, strict=True):
            resolved_summary = self._hedge_summary_for_confidence(summary_text, confidence)
            item_kwargs = {
                "resource_id": resource_id,
                "memory_type": memory_type,
                "summary": resolved_summary,
                "embedding": emb,
                "user_data": dict(user or {}),
                "reinforce": reinforce,
                "source_role": source_role,
                "confidence": confidence,
                "source_message_ids": source_message_ids,
                "reflection_salience": reflection_salience,
                "conversation_id": conversation_id,
            }
            if session is not None:
                item = cast(Any, store.memory_item_repo).create_item(**item_kwargs, session=session)
            else:
                item = store.memory_item_repo.create_item(**item_kwargs)
            items.append(item)
            if reinforce and item.extra.get("reinforcement_count", 1) > 1:
                # existing item
                continue
            mapped_cat_ids = self._map_category_names_to_ids(cat_names, ctx)
            for cid in mapped_cat_ids:
                rel_kwargs = {"item_id": item.id, "category_id": cid, "user_data": dict(user or {})}
                if session is not None:
                    rel = cast(Any, store.category_item_repo).link_item_category(**rel_kwargs, session=session)
                else:
                    rel = store.category_item_repo.link_item_category(**rel_kwargs)
                rels.append(rel)
                # Store (item_id, summary) tuple for reference support
                category_memory_updates.setdefault(cid, []).append((item.id, resolved_summary))

        return items, rels, category_memory_updates, homeless_count

    @staticmethod
    def _category_scope_key(user_scope: Mapping[str, Any] | None) -> str:
        if not isinstance(user_scope, Mapping):
            return "__global__"
        user_id = str(user_scope.get("user_id") or user_scope.get("userId") or "").strip()
        soul_id = str(user_scope.get("soul_id") or user_scope.get("soulId") or "").strip()
        if not user_id and not soul_id:
            return "__global__"
        return f"user={user_id}|soul={soul_id}"

    def _start_category_initialization(
        self,
        ctx: Context,
        store: Database,
        user_scope: Mapping[str, Any] | None = None,
    ) -> None:
        scope_key = self._category_scope_key(user_scope)
        if ctx.categories_ready and ctx.category_scope_key == scope_key:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop:
            ctx.category_init_scope_key = scope_key
            ctx.category_init_task = loop.create_task(
                self._initialize_categories(ctx, store, user_scope, scope_key=scope_key)
            )
        else:
            asyncio.run(self._initialize_categories(ctx, store, user_scope, scope_key=scope_key))

    async def _ensure_categories_ready(
        self, ctx: Context, store: Database, user_scope: Mapping[str, Any] | None = None
    ) -> None:
        scope_key = self._category_scope_key(user_scope)
        if ctx.categories_ready and ctx.category_scope_key == scope_key:
            return
        if ctx.category_init_task:
            await ctx.category_init_task
            ctx.category_init_task = None
            ctx.category_init_scope_key = None
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

    async def _preprocess_resource_url(
        self, *, local_path: str, text: str | None, modality: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        """
        Preprocess resource based on modality.

        General preprocessing dispatcher for all modalities:
        - Text-based modalities (conversation, document): require text content
        - Audio modality: transcribe audio file first, then process as text
        - Media modalities (video, image): process media files directly

        Args:
            local_path: Local file path to the resource
            text: Text content if available (for text-based modalities)
            modality: Resource modality type

        Returns:
            List of preprocessed resources, each with 'text' and 'caption'
        """
        configured_prompt = self.memorize_config.multimodal_preprocess_prompts.get(modality)
        if configured_prompt is None:
            template = PREPROCESS_PROMPTS.get(modality)
        elif isinstance(configured_prompt, str):
            template = configured_prompt
        else:
            # No custom prompts configured for preprocssing for now,
            # If the user decide to use their custom prompt, they must provide ALL prompt blocks.
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
        """Ensure audio resources provide text either via transcription or file read."""
        if text:
            return text

        audio_extensions = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        text_extensions = {".txt", ".text"}
        file_ext = pathlib.Path(local_path).suffix.lower()

        if file_ext in audio_extensions:
            try:
                logger.info(f"Transcribing audio file: {local_path}")
                client = llm_client or self._get_llm_client()
                transcribed = cast(str, await client.transcribe(local_path))
                logger.info(f"Audio transcription completed: {len(transcribed)} characters")
            except Exception:
                logger.exception("Audio transcription failed for %s", local_path)
                return None
            else:
                return transcribed

        if file_ext in text_extensions:
            path_obj = pathlib.Path(local_path)
            try:
                text_content = path_obj.read_text(encoding="utf-8")
                logger.info(f"Read pre-transcribed text file: {len(text_content)} characters")
            except Exception:
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
            return await self._preprocess_conversation(text, template, llm_client=llm_client)
        if modality == "video":
            return await self._preprocess_video(local_path, template, llm_client=llm_client)
        if modality == "image":
            return await self._preprocess_image(local_path, template, llm_client=llm_client)
        if modality == "document" and text is not None:
            return await self._preprocess_document(text, template, llm_client=llm_client)
        if modality == "audio" and text is not None:
            return await self._preprocess_audio(text, template, llm_client=llm_client)
        return [{"text": text, "caption": None}]

    async def _preprocess_conversation(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, Any]]:
        """Preprocess conversation data with segmentation, returns list of resources (one per segment)."""
        preprocessed_text = format_conversation_for_preprocess(text)
        prompt = template.format(conversation=self._escape_prompt_value(preprocessed_text))
        client = llm_client or self._get_llm_client()
        processed = await client.chat(prompt)
        _conv, segments = self._parse_conversation_preprocess_with_segments(processed, preprocessed_text)

        # Important: always use the original JSON-derived, indexed conversation text for downstream
        # segmentation and memory extraction. The LLM may rewrite the conversation and drop fields
        # like created_at, which would cause them to be lost.
        conversation_text = preprocessed_text
        all_indices = self._extract_message_indices(conversation_text)
        # If no segments, return single resource
        if not segments:
            return [{"text": conversation_text, "caption": None, "message_indices": all_indices}]

        # Generate caption for each segment and return as separate resources
        lines = conversation_text.split("\n")
        max_idx = len(lines) - 1
        resources: list[dict[str, Any]] = []
        pending_captions: list[tuple[int, str]] = []

        for segment in segments:
            start = int(segment.get("start", 0))
            end = int(segment.get("end", max_idx))
            start = max(0, min(start, max_idx))
            end = max(0, min(end, max_idx))
            segment_text = "\n".join(lines[start : end + 1])

            if segment_text.strip():
                caption_raw = segment.get("caption")
                caption = str(caption_raw).strip() if isinstance(caption_raw, str) else ""
                resources.append({
                    "text": segment_text,
                    "caption": caption or None,
                    "message_indices": list(range(start, end + 1)),
                })
                if not caption:
                    pending_captions.append((len(resources) - 1, segment_text))

        if pending_captions:
            max_parallel = min(4, len(pending_captions))
            limiter = asyncio.Semaphore(max_parallel)

            async def summarize_one(resource_idx: int, segment_text: str) -> tuple[int, str | None]:
                async with limiter:
                    caption = await self._summarize_segment(segment_text, llm_client=client)
                    return resource_idx, caption

            caption_results = await asyncio.gather(
                *(summarize_one(resource_idx, segment_text) for resource_idx, segment_text in pending_captions)
            )
            for resource_idx, generated_caption in caption_results:
                resources[resource_idx]["caption"] = generated_caption
        return (
            resources if resources else [{"text": conversation_text, "caption": None, "message_indices": all_indices}]
        )

    async def _summarize_segment(self, segment_text: str, llm_client: Any | None = None) -> str | None:
        """Summarize a single conversation segment."""
        system_prompt = (
            "Summarize the given conversation segment in 1-2 concise sentences. "
            "Focus on the main topic or theme discussed."
        )
        try:
            client = llm_client or self._get_llm_client()
            response = await client.chat(segment_text, system_prompt=system_prompt)
            return response.strip() if response else None
        except Exception:
            logger.exception("Failed to summarize segment")
            return None

    async def _preprocess_video(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        """
        Preprocess video data - extract description and caption using Vision API.

        Extracts the middle frame from the video and analyzes it using Vision API.

        Args:
            local_path: Path to the video file
            template: Prompt template for video analysis

        Returns:
            List with single resource containing text (description) and caption
        """
        try:
            # Check if ffmpeg is available
            if not VideoFrameExtractor.is_ffmpeg_available():
                logger.warning("ffmpeg not available, cannot process video. Returning None.")
                return [{"text": None, "caption": None}]

            # Extract middle frame from video
            logger.info(f"Extracting frame from video: {local_path}")
            frame_path = VideoFrameExtractor.extract_middle_frame(local_path)

            try:
                # Call Vision API with extracted frame
                logger.info(f"Analyzing video frame with Vision API: {frame_path}")
                client = llm_client or self._get_llm_client()
                processed = await client.vision(prompt=template, image_path=frame_path, system_prompt=None)
                description, caption = self._parse_multimodal_response(processed, "detailed_description", "caption")
                return [{"text": description, "caption": caption}]
            finally:
                # Clean up temporary frame file
                import pathlib

                try:
                    pathlib.Path(frame_path).unlink(missing_ok=True)
                    logger.debug(f"Cleaned up temporary frame: {frame_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up frame {frame_path}: {e}")

        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}", exc_info=True)
            return [{"text": None, "caption": None}]

    async def _preprocess_image(
        self, local_path: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        """
        Preprocess image data - extract description and caption using Vision API.

        Args:
            local_path: Path to the image file
            template: Prompt template for image analysis

        Returns:
            List with single resource containing text (description) and caption
        """
        # Call Vision API with image
        client = llm_client or self._get_llm_client()
        processed = await client.vision(prompt=template, image_path=local_path, system_prompt=None)
        description, caption = self._parse_multimodal_response(processed, "detailed_description", "caption")
        return [{"text": description, "caption": caption}]

    async def _preprocess_document(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        """Preprocess document data - condense and extract caption"""
        prompt = template.format(document_text=self._escape_prompt_value(text))
        client = llm_client or self._get_llm_client()
        processed = await client.chat(prompt)
        processed_content, caption = self._parse_multimodal_response(processed, "processed_content", "caption")
        return [{"text": processed_content or text, "caption": caption}]

    async def _preprocess_audio(
        self, text: str, template: str, llm_client: Any | None = None
    ) -> list[dict[str, str | None]]:
        """Preprocess audio data - format transcription and extract caption"""
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

        # If enabled, tell the model it's allowed to propose new categories.
        try:
            if getattr(self.memorize_config, "allow_dynamic_categories", False):
                max_total = int(getattr(self.memorize_config, "max_categories_total", 0) or 0)
                policy = str(getattr(self.memorize_config, "dynamic_category_policy", "") or "").strip()
                note = "\n\n" + (policy + "\n\n" if policy else "")
                note += (
                    "If none of the existing categories fit, you may propose a NEW category name. "
                    "Keep it broad (a life domain), not a specific event. "
                    f"Max total categories: {max_total or 'unlimited'}."
                )
                return base + note
        except Exception:
            return base
        return base

    def _add_conversation_indices(self, conversation: str) -> str:
        """
        Add [INDEX] markers to each line of the conversation.

        Args:
            conversation: Raw conversation text with lines

        Returns:
            Conversation with [INDEX] markers prepended to each non-empty line
        """
        lines = conversation.split("\n")
        indexed_lines = []
        index = 0

        for line in lines:
            stripped = line.strip()
            if stripped:  # Only index non-empty lines
                indexed_lines.append(f"[{index}] {line}")
                index += 1
            else:
                # Preserve empty lines without indexing
                indexed_lines.append(line)

        return "\n".join(indexed_lines)

    def _build_memory_type_prompt(self, *, memory_type: MemoryType, resource_text: str, categories_str: str) -> str:
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

        # When dynamic categories are enabled, remove the legacy hard-stop instruction.
        if getattr(self.memorize_config, "allow_dynamic_categories", False):
            template = re.sub(r"(?im)^.*do not create new memory categories.*\n?", "", template)
        safe_resource = self._escape_prompt_value(resource_text)
        safe_categories = self._escape_prompt_value(categories_str)
        return template.format(resource=safe_resource, categories_str=safe_categories)

    def _build_item_ref_id(self, item_id: str) -> str:
        return item_id.replace("-", "")[:6]

    def _extract_refs_from_summaries(self, summaries: dict[str, str]) -> set[str]:
        """
        Extract all [ref:xxx] references from summary texts.

        Args:
            summaries: dict mapping category_id -> summary text

        Returns:
            Set of all referenced short IDs (the xxx part from [ref:xxx])
        """
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
        """
        Persist ref_id to items that are referenced in category summaries.

        This function:
        1. Extracts all [ref:xxx] patterns from updated summaries
        2. Builds a mapping of short_id -> full item_id for all items in category_updates
        3. For items whose short_id appears in the references, updates their extra column
           with {"ref_id": short_id}
        """
        # Extract all referenced short IDs from summaries
        referenced_short_ids = self._extract_refs_from_summaries(updated_summaries)
        if not referenced_short_ids:
            return

        # Build mapping of short_id -> full item_id for all items in category_updates
        short_id_to_item_id: dict[str, str] = {}
        for item_tuples in category_updates.values():
            for item_id, _ in item_tuples:
                short_id = self._build_item_ref_id(item_id)
                short_id_to_item_id[short_id] = item_id

        # Update extra column for referenced items
        for short_id in referenced_short_ids:
            matched_item_id = short_id_to_item_id.get(short_id)
            if matched_item_id:
                store.memory_item_repo.update_item(
                    item_id=matched_item_id,
                    extra={"ref_id": short_id},
                )

    def _build_category_summary_prompt(
        self,
        *,
        category: MemoryCategory,
        new_memories: list[str] | list[tuple[str, str]],
        user: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the prompt for updating a category summary.

        Args:
            category: The category to update
            new_memories: Either list of summary strings (legacy) or list of (item_id, summary) tuples (with refs)
        """
        # Check if references are enabled and we have (id, summary) tuples
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
        # Prefer human-readable names for summaries.
        # user scope varies by integration; accept common keys.
        user_scope = user or {}
        raw_user = (
            user_scope.get("user_name")
            or user_scope.get("userName")
            or user_scope.get("user_id")
            or user_scope.get("userId")
        )
        raw_agent = (
            user_scope.get("soul_name")
            or user_scope.get("soulName")
            or user_scope.get("character_name")
            or user_scope.get("characterName")
            or user_scope.get("soul_id")
            or user_scope.get("soulId")
        )
        user_name = str(raw_user).strip() if raw_user else "the user"
        agent_name = str(raw_agent).strip() if raw_agent else "the assistant"
        # Strip ST timestamp suffixes like "Siri - 2026-...Z".
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
        """
        Update category summaries based on new memory items.

        Returns:
            dict mapping category_id -> updated summary text
        """
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
            # If prompts still output "The user ...", rewrite to the real user name.
            user_scope = user or {}
            raw_user = (
                user_scope.get("user_name")
                or user_scope.get("userName")
                or user_scope.get("user_id")
                or user_scope.get("userId")
            )
            user_name = str(raw_user).strip() if raw_user else ""
            if user_name and user_name.lower() not in ("user", "the user"):
                import re

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

    def _parse_conversation_preprocess(self, raw: str) -> tuple[str | None, str | None]:
        conversation = self._extract_tag_content(raw, "conversation")
        summary = self._extract_tag_content(raw, "summary")
        return conversation, summary

    @staticmethod
    def _dedupe_message_indices(values: Sequence[int | float | str]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for value in values:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                continue
            if candidate in seen or candidate < 0:
                continue
            seen.add(candidate)
            out.append(candidate)
        return out

    def _resolve_source_message_ids(
        self,
        values: Sequence[int | float | str] | None,
        allowed_values: Sequence[int | float | str] | None = None,
    ) -> list[int]:
        """Clamp extracted source IDs to the valid segment range.

        If the model emits IDs outside the segment, drop them silently rather
        than falling back to the entire segment — an empty anchor is honest,
        a full-segment anchor is noise.
        """
        parsed = self._dedupe_message_indices(values or [])
        allowed = self._dedupe_message_indices(allowed_values or [])
        if not allowed:
            return parsed
        allowed_set = set(allowed)
        return [candidate for candidate in parsed if candidate in allowed_set]

    @staticmethod
    def _extract_message_indices(text: str | None) -> list[int]:
        if not isinstance(text, str) or not text.strip():
            return []
        out: list[int] = []
        for line in text.splitlines():
            match = re.match(r"\[(\d+)\]\s", line)
            if match is None:
                continue
            try:
                out.append(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return out

    def _prepare_diary_segment(
        self,
        *,
        modality: str,
        text: str | None,
        message_indices: Any = None,
    ) -> tuple[str | None, list[int]]:
        if modality != "conversation" or not isinstance(text, str) or not text.strip():
            return None, []
        segment_text = format_conversation_for_preprocess(text)
        if not segment_text.strip():
            segment_text = text.strip()
        if isinstance(message_indices, list):
            indices = self._dedupe_message_indices([
                value for value in message_indices if isinstance(value, (int, float, str))
            ])
        else:
            indices = self._extract_message_indices(segment_text)
        return segment_text, indices

    def _parse_diary_worthy_response(self, raw: str) -> bool:
        if not isinstance(raw, str) or not raw.strip():
            return False
        payload = None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            try:
                payload = json.loads(self._extract_json_blob(raw))
            except Exception:
                return False
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("worthy") is True)

    async def _classify_diary_worthy_segment(self, segment_text: str, llm_client: Any | None = None) -> bool:
        prompt = DIARY_WORTHY_PROMPT.format(exchange=self._escape_prompt_value(segment_text))
        client = llm_client or self._get_llm_client()
        try:
            raw = await client.chat(prompt, temperature=0.0)
        except Exception:
            return False
        return self._parse_diary_worthy_response(raw)

    def _parse_multimodal_response(self, raw: str, content_tag: str, caption_tag: str) -> tuple[str | None, str | None]:
        """
        Parse multimodal preprocessing response (video, image, document, audio).
        Extracts content and caption from XML-like tags.

        Args:
            raw: Raw LLM response
            content_tag: Tag name for main content (e.g., "detailed_description", "processed_content")
            caption_tag: Tag name for caption (typically "caption")

        Returns:
            Tuple of (content, caption)
        """
        content = self._extract_tag_content(raw, content_tag)
        caption = self._extract_tag_content(raw, caption_tag)

        # Fallback: if no tags found, try to use raw response as content
        if not content:
            content = raw.strip()

        # Fallback for caption: use first sentence of content if no caption found
        if not caption and content:
            first_sentence = content.split(".")[0]
            caption = first_sentence if len(first_sentence) <= 200 else first_sentence[:200]

        return content, caption

    def _parse_conversation_preprocess_with_segments(
        self, raw: str, original_text: str
    ) -> tuple[str | None, list[dict[str, int | str]] | None]:
        """
        Parse conversation preprocess response and extract segments.
        Returns: (conversation_text, segments)
        """
        conversation = self._extract_tag_content(raw, "conversation")
        segments = self._extract_segments_with_fallback(raw)
        return conversation, segments

    def _extract_segments_with_fallback(self, raw: str) -> list[dict[str, int | str]] | None:
        segments = self._segments_from_json_payload(raw)
        if segments is not None:
            return segments
        try:
            blob = self._extract_json_blob(raw)
        except Exception:
            logging.exception("Failed to extract segments from conversation preprocess response")
            return None
        return self._segments_from_json_payload(blob)

    def _segments_from_json_payload(self, payload: str) -> list[dict[str, int | str]] | None:
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        return self._segments_from_parsed_data(parsed)

    @staticmethod
    def _segments_from_parsed_data(parsed: Any) -> list[dict[str, int | str]] | None:
        if not isinstance(parsed, dict):
            return None
        segments_data = parsed.get("segments")
        if not isinstance(segments_data, list):
            return None
        segments: list[dict[str, int | str]] = []
        for seg in segments_data:
            if isinstance(seg, dict) and "start" in seg and "end" in seg:
                try:
                    segment: dict[str, int | str] = {
                        "start": int(seg["start"]),
                        "end": int(seg["end"]),
                    }
                    if "caption" in seg and isinstance(seg["caption"], str):
                        segment["caption"] = seg["caption"]
                    segments.append(segment)
                except (TypeError, ValueError):
                    continue
        return segments or None

    @staticmethod
    def _extract_tag_content(raw: str, tag: str) -> str | None:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
        match = pattern.search(raw)
        if not match:
            return None
        content = match.group(1).strip()
        return content or None

    def _parse_memory_type_response(self, raw: str) -> list[dict[str, Any]]:
        if not raw:
            return []
        raw = raw.strip()
        if not raw:
            return []
        payload = None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            try:
                blob = self._extract_json_blob(raw)
                payload = json.loads(blob)
            except Exception:
                return []
        if not isinstance(payload, dict):
            return []
        items = payload.get("memories_items")
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            normalized.append(entry)
        return normalized

    def _find_xml_boundaries(self, raw: str) -> tuple[int, int, str] | None:
        """Find the start index, end index, and closing tag for XML root element."""
        root_tags = ["item", "profile", "behaviors", "events", "knowledge", "skills"]
        for tag in root_tags:
            opening = f"<{tag}>"
            closing = f"</{tag}>"
            start_idx = raw.find(opening)
            if start_idx != -1:
                end_idx = raw.rfind(closing)
                if end_idx != -1:
                    return (start_idx, end_idx, closing)
        return None

    def _parse_memory_element(self, memory_elem: Element) -> dict[str, Any] | None:
        """Parse a single memory XML element into a dict."""
        memory_dict: dict[str, Any] = {}

        content_elem = memory_elem.find("content")
        if content_elem is not None and content_elem.text:
            memory_dict["content"] = content_elem.text.strip()

        categories_elem = memory_elem.find("categories")
        if categories_elem is not None:
            categories = [cat_elem.text.strip() for cat_elem in categories_elem.findall("category") if cat_elem.text]
            memory_dict["categories"] = categories

        source_role_elem = memory_elem.find("source_role")
        if source_role_elem is not None and source_role_elem.text:
            raw_role = source_role_elem.text.strip().lower()
            if raw_role in {"soul", "user", "environment"}:
                memory_dict["source_role"] = raw_role

        confidence_elem = memory_elem.find("confidence")
        if confidence_elem is not None and confidence_elem.text:
            try:
                confidence = float(confidence_elem.text.strip())
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None and 0.0 <= confidence <= 1.0:
                memory_dict["confidence"] = confidence

        salience_elem = memory_elem.find("reflection_salience")
        if salience_elem is not None and salience_elem.text:
            try:
                reflection_salience = float(salience_elem.text.strip())
            except (TypeError, ValueError):
                reflection_salience = None
            if reflection_salience is not None and 0.0 <= reflection_salience <= 1.0:
                memory_dict["reflection_salience"] = reflection_salience

        source_message_ids_elem = memory_elem.find("source_message_ids")
        if source_message_ids_elem is not None:
            message_ids: list[int] = []
            for id_elem in source_message_ids_elem.findall("id"):
                if id_elem.text is None:
                    continue
                try:
                    message_ids.append(int(id_elem.text.strip()))
                except (TypeError, ValueError):
                    continue
            if message_ids:
                memory_dict["source_message_ids"] = message_ids

        if memory_dict.get("content") and memory_dict.get("categories"):
            return memory_dict
        return None

    def _parse_memory_type_response_xml(self, raw: str) -> list[dict[str, Any]]:
        """
        Parse XML memory extraction output into a list of memory items.

        Expected XML format (root tag varies by memory type):
        <profile|behaviors|events|knowledge|skills>
            <memory>
                <content>...</content>
                <categories>
                    <category>...</category>
                </categories>
                <source_role>soul|user|environment</source_role>  <!-- optional -->
                <confidence>0.0-1.0</confidence>                 <!-- optional -->
            </memory>
        </...>
        """
        if not raw or not raw.strip():
            return []
        raw = raw.strip()

        try:
            boundaries = self._find_xml_boundaries(raw)
            if boundaries is None:
                logger.warning("Could not find valid root tag in XML response")
                return []

            start_idx, end_idx, end_tag = boundaries
            xml_content = raw[start_idx : end_idx + len(end_tag)]
            xml_content = xml_content.replace("&", "&amp;")

            root = ET.fromstring(xml_content)
            result: list[dict[str, Any]] = []

            for memory_elem in root.findall("memory"):
                parsed = self._parse_memory_element(memory_elem)
                if parsed:
                    result.append(parsed)

        except ET.ParseError:
            logger.exception("Failed to parse XML")
            return []
        else:
            return result
