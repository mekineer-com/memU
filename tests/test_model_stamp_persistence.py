from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memu.app import memorize_persistence as persistence


class _StubMemoryRepo:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_item(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(id="item-1", summary=kwargs.get("summary", ""))


class _StubCategoryItemRepo:
    def link_item_category(self, **kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)


class _StubTripleRepo:
    def add(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _StubEntityRepo:
    def get_or_create(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(id="entity-1")


class _StubStore:
    def __init__(self) -> None:
        self.memory_item_repo = _StubMemoryRepo()
        self.category_item_repo = _StubCategoryItemRepo()
        self.triple_repo = _StubTripleRepo()
        self.entity_repo = _StubEntityRepo()


class _StubEmbedClient:
    async def embed(self, payloads: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in payloads]


@pytest.mark.asyncio
async def test_persist_memory_items_stamps_extract_model_in_extra() -> None:
    store = _StubStore()
    entry = SimpleNamespace(
        memory_type="profile",
        content="Marcos likes tea",
        source_role="user",
        speaker_id="user:marcos",
        speaker_label="Marcos",
        confidence=0.8,
        source_message_ids=[1],
        reflection_salience=0.7,
        emotional_intensity=0.4,
        categories=[],
        entities=[],
    )
    async def _maybe_create_dynamic_categories(**kwargs: Any) -> list[Any]:
        return list(kwargs["structured_entries"])

    async def _find_supersede_targets(**_kwargs: Any) -> dict[int, str]:
        return {}

    items, rels, cat_updates, homeless = await persistence._persist_memory_items(
        resource_id="res-1",
        structured_entries=[entry],
        ctx=SimpleNamespace(),
        store=store,
        embed_client=_StubEmbedClient(),
        get_llm_client=lambda: _StubEmbedClient(),
        user={"user_id": "u1", "soul_id": "s1"},
        conversation_id="conv-1",
        episode_id="ep-1",
        extract_model="claude-opus-4-6",
        message_happened_at_map={1: "2026-05-22T12:00:00Z"},
        session=None,
        maybe_create_dynamic_categories=_maybe_create_dynamic_categories,
        enable_confidence_normalization=False,
        normalize_confidence=lambda entries: entries,
        find_supersede_targets=_find_supersede_targets,
        hedge_summary_for_confidence=lambda summary, _confidence: summary,
        resolve_entry_happened_at=lambda source_ids, happened_map: happened_map.get(source_ids[0]) if source_ids else None,
        map_category_names_to_ids=lambda _names, _ctx: [],
    )

    assert len(items) == 1
    assert rels == []
    assert cat_updates == {}
    assert homeless == 1
    assert len(store.memory_item_repo.calls) == 1
    assert store.memory_item_repo.calls[0]["extra"] == {"model": "claude-opus-4-6"}


@pytest.mark.asyncio
async def test_persist_memory_items_without_extract_model_does_not_set_extra() -> None:
    store = _StubStore()
    entry = SimpleNamespace(
        memory_type="profile",
        content="Marcos likes tea",
        source_role="user",
        speaker_id="user:marcos",
        speaker_label="Marcos",
        confidence=0.8,
        source_message_ids=[1],
        reflection_salience=0.7,
        emotional_intensity=0.4,
        categories=[],
        entities=[],
    )
    async def _maybe_create_dynamic_categories(**kwargs: Any) -> list[Any]:
        return list(kwargs["structured_entries"])

    async def _find_supersede_targets(**_kwargs: Any) -> dict[int, str]:
        return {}

    await persistence._persist_memory_items(
        resource_id="res-1",
        structured_entries=[entry],
        ctx=SimpleNamespace(),
        store=store,
        embed_client=_StubEmbedClient(),
        get_llm_client=lambda: _StubEmbedClient(),
        user={"user_id": "u1", "soul_id": "s1"},
        conversation_id="conv-1",
        episode_id="ep-1",
        extract_model=None,
        message_happened_at_map={1: "2026-05-22T12:00:00Z"},
        session=None,
        maybe_create_dynamic_categories=_maybe_create_dynamic_categories,
        enable_confidence_normalization=False,
        normalize_confidence=lambda entries: entries,
        find_supersede_targets=_find_supersede_targets,
        hedge_summary_for_confidence=lambda summary, _confidence: summary,
        resolve_entry_happened_at=lambda source_ids, happened_map: happened_map.get(source_ids[0]) if source_ids else None,
        map_category_names_to_ids=lambda _names, _ctx: [],
    )

    assert len(store.memory_item_repo.calls) == 1
    assert "extra" not in store.memory_item_repo.calls[0]
