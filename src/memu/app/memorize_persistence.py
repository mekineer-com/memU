from __future__ import annotations

import mimetypes
from pathlib import Path
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from memu.app.memorize_dedupe import _dedupe_source_role, _dedupe_speaker_id
from memu.database.models import Triple, entity_is_ignored



def _resolve_entry_happened_at(
    memory_date: str | None,
    source_day_happened_at: Mapping[str, Any] | None,
) -> Any | None:
    if not memory_date or not source_day_happened_at:
        return None
    return source_day_happened_at.get(memory_date)


async def _prepare_resource_with_caption(
    *,
    resource_url: str,
    modality: str,
    local_path: str,
    caption: str | None,
    embed_client: Any | None,
    select_embedding_client: Callable[..., Any],
    user: Mapping[str, Any] | None,
    segment_id: str | None,
    conversation_id: str | None,
    memory_retrieve_history: list[str] | None,
    memory_prior_context: list[str] | None,
) -> dict[str, Any]:
    caption_text = caption.strip() if caption else None
    if modality in {"image", "audio", "video"}:
        mime_type = mimetypes.guess_type(local_path)[0]
        if not mime_type:
            raise ValueError(f"Cannot infer media type for {local_path}")
        client = embed_client or select_embedding_client(
            {"operation": "memorize", "step_id": "resource_media_embedding"}
        )
        caption_embedding = await client.embed_media(Path(local_path).read_bytes(), mime_type)
    elif caption_text:
        client = embed_client or select_embedding_client(
            {"operation": "memorize", "step_id": "resource_caption_embedding"}
        )
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
    if segment_id:
        resource_kwargs["segment_id"] = segment_id
    if conversation_id:
        resource_kwargs["conversation_id"] = conversation_id
    if memory_retrieve_history:
        resource_kwargs["memory_retrieve_history"] = memory_retrieve_history
    if memory_prior_context:
        resource_kwargs["memory_prior_context"] = memory_prior_context
    return resource_kwargs


async def _create_resource_with_caption(*, store: Any, **kwargs: Any) -> Any:
    resource_kwargs = await _prepare_resource_with_caption(**kwargs)
    return store.resource_repo.create_resource(**resource_kwargs)


def _sqlite_write_session(store: Any) -> Any | None:
    try:
        from memu.database.sqlite.sqlite import SQLiteStore
    except ImportError:
        return None
    if isinstance(store, SQLiteStore):
        return store._sessions.session()
    return None


async def _prepare_memory_items(
    *,
    structured_entries: list[Any],
    store: Any,
    embed_client: Any,
    user: Mapping[str, Any] | None,
    enable_confidence_normalization: bool,
    normalize_confidence: Callable[[list[Any]], list[Any]],
    find_supersede_targets: Callable[..., Awaitable[dict[int, str]]],
) -> tuple[list[Any], list[list[float]], dict[int, str]]:
    summary_payloads = [entry.content for entry in structured_entries]
    item_embeddings = await embed_client.embed(summary_payloads) if summary_payloads else []

    if enable_confidence_normalization:
        structured_entries = normalize_confidence(structured_entries)
    supersede_targets = await find_supersede_targets(
        structured_entries=structured_entries,
        store=store,
        embed_client=embed_client,
        user=user,
    )
    return structured_entries, item_embeddings, supersede_targets


def _persist_memory_items(
    *,
    resource_id: str,
    structured_entries: list[Any],
    item_embeddings: list[list[float]],
    supersede_targets: dict[int, str],
    store: Any,
    user: Mapping[str, Any] | None,
    conversation_id: str | None,
    segment_id: str | None,
    extract_model: str | None,
    source_day_happened_at: Mapping[str, Any] | None,
    session: Any,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    superseded_targets: set[str] = set()
    # Capture only existing copies before this batch creates any generalized items.
    supersede_copies: dict[int, list[str]] = {}
    for idx, target_item_id in supersede_targets.items():
        entry = structured_entries[idx]
        speaker_id = _dedupe_speaker_id(entry)
        if speaker_id is None:
            continue
        role = _dedupe_source_role(entry)
        where = {**dict(user or {}), "memory_type": entry.memory_type}
        target = store.memory_item_repo.list_items_by_ids(
            {target_item_id}, where, session=session,
        ).get(target_item_id)
        if target is None or _dedupe_speaker_id(target) != speaker_id or _dedupe_source_role(target) != role:
            continue
        copies = store.memory_item_repo.list_items(
            {**where, "summary": target.summary,
             "speaker_id": target.speaker_id, "source_role": target.source_role},
            include_embeddings=False, session=session,
        )
        supersede_copies[idx] = [
            copy.id for copy in copies.values()
            if _dedupe_source_role(copy) == role
        ]
    homeless_count = sum(1 for entry in structured_entries if not entry.categories)
    normalized_extract_model = str(extract_model or "").strip() or None
    for idx, (entry, emb) in enumerate(zip(structured_entries, item_embeddings, strict=True)):
        item_kwargs = {
            "resource_id": resource_id,
            "memory_type": entry.memory_type,
            "summary": entry.content,
            "embedding": emb,
            "user_data": dict(user or {}),
            "source_role": entry.source_role,
            "speaker_id": entry.speaker_id,
            "speaker_label": entry.speaker_label,
            "confidence": entry.confidence,
            "source_message_ids": entry.source_message_ids,
            "happened_at": _resolve_entry_happened_at(entry.memory_date, source_day_happened_at),
            "reflection_salience": entry.reflection_salience,
            "emotional_intensity": entry.emotional_intensity,
            "conversation_id": conversation_id,
            "segment_id": segment_id,
        }
        if normalized_extract_model is not None:
            item_kwargs["extra"] = {"model": normalized_extract_model}
        item = store.memory_item_repo.create_item(**item_kwargs, session=session)
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
                if entity_is_ignored(entity_record):
                    continue
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
        for target_item_id in supersede_copies.get(idx, []):
            if target_item_id in superseded_targets:
                continue
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
    if normalized_extract_model is not None and items:
        store.memory_item_repo.refresh_model_score_calibration(model=normalized_extract_model, session=session)

    return items, homeless_count
