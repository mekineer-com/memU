from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from memu.database.vector import cosine_similarity

logger = logging.getLogger(__name__)


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


def _build_semantic_dedupe_scope(scope: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(scope, Mapping):
        return None
    user_field = _extract_scope_field(scope, keys=("user_id",))
    soul_field = _extract_scope_field(scope, keys=("soul_id",))
    if user_field is None or soul_field is None:
        return None
    _user_key, user_value = user_field
    _soul_key, soul_value = soul_field
    return {
        "user_id": user_value,
        "soul_id": soul_value,
    }


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


def _is_merged_item(item: Any) -> bool:
    merged_into = getattr(item, "merged_into", None)
    return isinstance(merged_into, str) and merged_into.strip() != ""


def _item_embedding(item: Any) -> list[float] | None:
    return _normalize_embedding_vector(getattr(item, "embedding", None))


async def _dedupe_reembed_for_similarity(
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


def _summary_len(item: Any) -> int:
    summary = getattr(item, "summary", "")
    return len(str(summary).strip())


def _choose_survivor_and_redundant(left: Any, right: Any) -> tuple[Any, Any]:
    left_time = getattr(left, "happened_at", None) or getattr(left, "created_at", None)
    right_time = getattr(right, "happened_at", None) or getattr(right, "created_at", None)
    if left_time is not None and right_time is not None:
        try:
            if left_time > right_time:
                return left, right
            if right_time > left_time:
                return right, left
        except TypeError:
            left_iso = str(getattr(left_time, "isoformat", lambda: left_time)())
            right_iso = str(getattr(right_time, "isoformat", lambda: right_time)())
            if left_iso > right_iso:
                return left, right
            if right_iso > left_iso:
                return right, left
    elif left_time is not None:
        return left, right
    elif right_time is not None:
        return right, left

    left_len = _summary_len(left)
    right_len = _summary_len(right)
    if left_len > right_len:
        return left, right
    if right_len > left_len:
        return right, left
    left_id = str(getattr(left, "id", ""))
    right_id = str(getattr(right, "id", ""))
    if left_id <= right_id:
        return left, right
    return right, left


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


def _dedupe_source_role(item: Any) -> str | None:
    raw = getattr(item, "source_role", None)
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _dedupe_speaker_id(item: Any) -> str | None:
    raw = getattr(item, "speaker_id", None)
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _prefilter_dedupe_candidate_ids(
    *,
    anchor_id: str,
    anchor: Any,
    active_pool: Mapping[str, Any],
    merged_map: Mapping[str, str],
    summary_tokens: Mapping[str, set[str]],
    token_index: Mapping[str, set[str]],
    token_freq: Mapping[str, int],
) -> list[str]:
    anchor_role = _dedupe_source_role(anchor)
    anchor_speaker_id = _dedupe_speaker_id(anchor)
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
            if candidate is None or _is_merged_item(candidate):
                continue
            candidate_role = _dedupe_source_role(candidate)
            candidate_speaker_id = _dedupe_speaker_id(candidate)
            if anchor_role != candidate_role:
                continue
            if anchor_speaker_id != candidate_speaker_id:
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
            if _is_merged_item(candidate):
                continue
            candidate_role = _dedupe_source_role(candidate)
            candidate_speaker_id = _dedupe_speaker_id(candidate)
            if anchor_role != candidate_role:
                continue
            if anchor_speaker_id != candidate_speaker_id:
                continue
            candidate_scores[candidate_id] = 0

    ordered = sorted(candidate_scores.items(), key=lambda row: (-row[1], row[0]))
    return [candidate_id for candidate_id, _score in ordered[:64]]


async def _memorize_dedupe_merge(
    state: dict[str, Any],
    _step_context: Any,
    *,
    semantic_dedupe_enabled: bool,
    semantic_dedupe_similarity_threshold: float,
    select_embedding_client: Callable[..., Any],
) -> dict[str, Any]:
    items = list(state.get("items") or [])
    state["items"] = items

    if not semantic_dedupe_enabled:
        return state
    if len(items) < 1:
        return state

    dedupe_scope = _build_semantic_dedupe_scope(state.get("user"))
    if dedupe_scope is None:
        return state

    store = state["store"]
    active_pool = dict(store.memory_item_repo.list_items(dedupe_scope))
    if len(active_pool) < 2:
        return state

    new_item_ids: list[str] = []
    seen_new: set[str] = set()
    for item in items:
        item_id = getattr(item, "id", None)
        if not item_id or item_id in seen_new:
            continue
        seen_new.add(item_id)
        pool_item = active_pool.get(item_id)
        if pool_item is None:
            continue
        if _is_merged_item(pool_item):
            continue
        if _item_embedding(pool_item) is None:
            continue
        new_item_ids.append(item_id)
    if not new_item_ids:
        return state

    threshold = max(0.0, min(1.0, float(semantic_dedupe_similarity_threshold)))
    merged_map: dict[str, str] = {}
    dedupe_embed_client: Any | None = None
    dedupe_embed_cache: dict[str, list[float] | None] = {}
    summary_tokens: dict[str, set[str]] = {}
    token_index: dict[str, set[str]] = {}
    token_freq: dict[str, int] = {}

    for pool_item_id, pool_item in active_pool.items():
        tokens = _dedupe_summary_tokens(getattr(pool_item, "summary", ""))
        summary_tokens[pool_item_id] = tokens
        for token in tokens:
            token_index.setdefault(token, set()).add(pool_item_id)
            token_freq[token] = token_freq.get(token, 0) + 1

    for new_item_id in new_item_ids:
        anchor = active_pool.get(new_item_id)
        if anchor is None or _is_merged_item(anchor):
            continue
        anchor_embedding = _item_embedding(anchor)
        if anchor_embedding is None:
            continue

        candidates: list[tuple[float, str]] = []
        candidate_ids = _prefilter_dedupe_candidate_ids(
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
            candidate_embedding = _item_embedding(candidate)
            compare_anchor: list[float] | None = anchor_embedding
            compare_candidate: list[float] | None = candidate_embedding

            if candidate_embedding is None or len(anchor_embedding) != len(candidate_embedding):
                if dedupe_embed_client is None:
                    dedupe_embed_client = select_embedding_client(
                        {"operation": "memorize", "step_id": "semantic_dedupe_reembed"}
                    )
                compare_anchor = await _dedupe_reembed_for_similarity(
                    item=anchor,
                    embed_client=dedupe_embed_client,
                    cache=dedupe_embed_cache,
                )
                compare_candidate = await _dedupe_reembed_for_similarity(
                    item=candidate,
                    embed_client=dedupe_embed_client,
                    cache=dedupe_embed_cache,
                )
            if compare_anchor is None or compare_candidate is None or len(compare_anchor) != len(compare_candidate):
                continue
            similarity = cosine_similarity(compare_anchor, compare_candidate)
            if similarity >= threshold:
                candidates.append((similarity, candidate_id))
        if not candidates:
            continue

        candidates.sort(key=lambda row: (-row[0], row[1]))
        for _similarity, candidate_id in candidates:
            current_anchor = active_pool.get(new_item_id)
            candidate = active_pool.get(candidate_id)
            if current_anchor is None or candidate is None:
                continue
            if _is_merged_item(current_anchor) or _is_merged_item(candidate):
                continue

            survivor, redundant = _choose_survivor_and_redundant(current_anchor, candidate)
            if survivor.id == redundant.id:
                continue

            store.memory_item_repo.update_item(item_id=redundant.id, merged_into=survivor.id)
            merged_map[redundant.id] = survivor.id
            active_pool.pop(redundant.id, None)
            if redundant.id == new_item_id:
                break

    if not merged_map:
        return state

    merged_ids = set(merged_map.keys())
    remaining_items: list[Any] = []
    for item in items:
        item_id = getattr(item, "id", None)
        if item_id in merged_ids:
            continue
        refreshed = active_pool.get(item_id)
        remaining_items.append(refreshed if refreshed is not None else item)
    state["items"] = remaining_items
    state["relations"] = [
        rel for rel in (state.get("relations") or []) if getattr(rel, "item_id", None) not in merged_ids
    ]
    state["category_updates"] = _filter_merged_from_category_updates(state.get("category_updates"), merged_ids)
    return state


def _supersede_similarity_threshold(raw_threshold: Any) -> float:
    threshold = float(raw_threshold or 0.75)
    return max(0.0, min(1.0, threshold))


async def _find_supersede_targets(
    *,
    structured_entries: list[Any],
    store: Any,
    embed_client: Any,
    user: Mapping[str, Any] | None = None,
    threshold: float,
) -> dict[int, str]:
    replace_requests = [
        (idx, entry)
        for idx, entry in enumerate(structured_entries)
        if isinstance(entry.replaces_previous_fact, str) and entry.replaces_previous_fact.strip()
    ]
    if not replace_requests:
        return {}

    replace_texts = [str(entry.replaces_previous_fact).strip() for _, entry in replace_requests]
    replace_vectors = await embed_client.embed(replace_texts)
    targets: dict[int, str] = {}

    for (idx, entry), raw_vector in zip(replace_requests, replace_vectors, strict=True):
        vector = _normalize_embedding_vector(raw_vector)
        if vector is None:
            logger.warning(
                "supersede: embedding normalization failed for replacement text, skipping: %.80s",
                entry.replaces_previous_fact,
            )
            continue
        where = dict(user or {})
        where["memory_type"] = entry.memory_type
        if entry.source_role:
            where["source_role"] = entry.source_role
        hits = store.memory_item_repo.vector_search_items(query_vec=vector, top_k=3, where=where)
        for candidate_id, similarity in hits:
            if similarity >= threshold:
                targets[idx] = candidate_id
                break

    return targets
