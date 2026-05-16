from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from memu.database.models import Triple


def _build_item_ref_id(item_id: str) -> str:
    return item_id.replace("-", "")[:6]


def _extract_refs_from_summaries(summaries: dict[str, str]) -> set[str]:
    from memu.utils.references import extract_references

    refs: set[str] = set()
    for summary in summaries.values():
        refs.update(extract_references(summary))
    return refs


async def _persist_item_references(
    *,
    updated_summaries: dict[str, str],
    category_updates: dict[str, list[tuple[str, str]]],
    store: Any,
    build_item_ref_id: Callable[[str], str],
) -> None:
    referenced_short_ids = _extract_refs_from_summaries(updated_summaries)
    if not referenced_short_ids:
        return

    short_id_to_item_id: dict[str, str] = {}
    for item_tuples in category_updates.values():
        for item_id, _summary in item_tuples:
            short_id = build_item_ref_id(item_id)
            short_id_to_item_id[short_id] = item_id

    for short_id in referenced_short_ids:
        matched_item_id = short_id_to_item_id.get(short_id)
        if matched_item_id:
            store.memory_item_repo.update_item(
                item_id=matched_item_id,
                extra={"ref_id": short_id},
            )


def _looks_like_identifier_value(value: str) -> bool:
    import re

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


def _resolve_entry_happened_at(
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


async def _create_resource_with_caption(
    *,
    resource_url: str,
    modality: str,
    local_path: str,
    caption: str | None,
    store: Any,
    embed_client: Any | None,
    get_embedding_client: Callable[..., Any],
    user: Mapping[str, Any] | None,
    episode_id: str | None,
    conversation_id: str | None,
    memory_retrieve_history: list[str] | None,
    memory_prior_context: list[str] | None,
    session: Any | None,
) -> Any:
    caption_text = caption.strip() if caption else None
    if caption_text:
        client = embed_client or get_embedding_client("embedding")
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
        return store.resource_repo.create_resource(**resource_kwargs, session=session)
    return store.resource_repo.create_resource(**resource_kwargs)


def _sqlite_write_session(store: Any) -> Any | None:
    try:
        from memu.database.sqlite.sqlite import SQLiteStore
    except ImportError:
        return None
    if isinstance(store, SQLiteStore):
        return store._sessions.session()
    return None


async def _persist_memory_items(
    *,
    resource_id: str,
    structured_entries: list[Any],
    ctx: Any,
    store: Any,
    embed_client: Any | None,
    get_llm_client: Callable[..., Any],
    user: Mapping[str, Any] | None,
    conversation_id: str | None,
    episode_id: str | None,
    message_happened_at_map: Mapping[int, Any] | None,
    session: Any | None,
    maybe_create_dynamic_categories: Callable[..., Awaitable[list[Any]]],
    enable_confidence_normalization: bool,
    normalize_confidence: Callable[[list[Any]], list[Any]],
    find_supersede_targets: Callable[..., Awaitable[dict[int, str]]],
    hedge_summary_for_confidence: Callable[[str, float | None], str],
    resolve_entry_happened_at: Callable[[Sequence[int] | None, Mapping[int, Any] | None], Any | None],
    map_category_names_to_ids: Callable[[list[str], Any], list[str]],
) -> tuple[list[Any], list[Any], dict[str, list[tuple[str, str]]], int]:
    summary_payloads = [entry.content for entry in structured_entries]
    client = embed_client or get_llm_client()
    item_embeddings = await client.embed(summary_payloads) if summary_payloads else []
    items: list[Any] = []
    rels: list[Any] = []
    category_memory_updates: dict[str, list[tuple[str, str]]] = {}
    superseded_targets: set[str] = set()

    structured_entries = await maybe_create_dynamic_categories(
        structured_entries=structured_entries,
        item_embeddings=item_embeddings,
        ctx=ctx,
        store=store,
        embed_client=client,
        user=user,
        session=session,
    )
    if enable_confidence_normalization:
        structured_entries = normalize_confidence(structured_entries)
    homeless_count = sum(1 for entry in structured_entries if not entry.categories)
    supersede_targets = await find_supersede_targets(
        structured_entries=structured_entries,
        store=store,
        embed_client=client,
        user=user,
    )
    for idx, (entry, emb) in enumerate(zip(structured_entries, item_embeddings, strict=True)):
        resolved_summary = hedge_summary_for_confidence(entry.content, entry.confidence)
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
            "happened_at": resolve_entry_happened_at(entry.source_message_ids, message_happened_at_map),
            "reflection_salience": entry.reflection_salience,
            "emotional_intensity": entry.emotional_intensity,
            "conversation_id": conversation_id,
            "episode_id": episode_id,
        }
        if session is not None:
            item = store.memory_item_repo.create_item(**item_kwargs, session=session)
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
                store.triple_repo.add(
                    Triple(
                        subject_id=item.id,
                        subject_kind="memory",
                        predicate="mentions",
                        object_id=entity_record.id,
                        object_kind="entity",
                        source_memory_id=item.id,
                    ),
                    user_data=dict(user or {}),
                    session=session,
                )
        target_item_id = supersede_targets.get(idx)
        if target_item_id and target_item_id != item.id and target_item_id not in superseded_targets:
            superseded_targets.add(target_item_id)
            store.triple_repo.add(
                Triple(
                    subject_id=target_item_id,
                    subject_kind="memory",
                    predicate="evolved_into",
                    object_id=item.id,
                    object_kind="memory",
                    source_memory_id=item.id,
                ),
                user_data=dict(user or {}),
                session=session,
            )
        mapped_cat_ids = map_category_names_to_ids(entry.categories, ctx)
        if resolved_summary.strip():
            for cid in mapped_cat_ids:
                category_memory_updates.setdefault(cid, []).append((item.id, resolved_summary))
                rel_kwargs = {"item_id": item.id, "category_id": cid, "user_data": dict(user or {})}
                if session is not None:
                    rel = store.category_item_repo.link_item_category(**rel_kwargs, session=session)
                else:
                    rel = store.category_item_repo.link_item_category(**rel_kwargs)
                rels.append(rel)

    return items, rels, category_memory_updates, homeless_count
