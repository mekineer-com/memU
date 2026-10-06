from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memu.app import memorize_persistence as persistence


class _StubMemoryRepo:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.calibration_calls: list[str] = []

    def create_item(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(id="item-1", summary=kwargs.get("summary", ""))

    def refresh_model_score_calibration(self, *, model: str, session: Any | None = None) -> dict[str, int]:
        self.calibration_calls.append(model)
        return {}


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
        content="TestOwner likes tea",
        source_role="user",
        speaker_id="user:testowner",
        speaker_label="TestOwner",
        confidence=0.8,
        source_message_ids=[1],
        reflection_salience=0.7,
        emotional_intensity=0.4,
        memory_date="2026-05-22",
        categories=[],
        entities=[],
    )
    async def _find_supersede_targets(**_kwargs: Any) -> dict[int, str]:
        return {}

    entries, embeddings, targets = await persistence._prepare_memory_items(
        structured_entries=[entry], store=store, embed_client=_StubEmbedClient(),
        user={"user_id": "u1", "soul_id": "s1"},
        enable_confidence_normalization=False,
        normalize_confidence=lambda entries: entries,
        find_supersede_targets=_find_supersede_targets,
    )
    items, homeless = persistence._persist_memory_items(
        resource_id="res-1",
        structured_entries=entries,
        item_embeddings=embeddings,
        supersede_targets=targets,
        store=store,
        user={"user_id": "u1", "soul_id": "s1"},
        conversation_id="conv-1",
        segment_id="ep-1",
        extract_model="claude-opus-4-6",
        source_day_happened_at={"2026-05-22": "2026-05-22T12:00:00Z"},
        session=None,
        hedge_summary_for_confidence=lambda summary, _confidence: summary,
    )

    assert len(items) == 1
    assert homeless == 1
    assert len(store.memory_item_repo.calls) == 1
    assert store.memory_item_repo.calls[0]["happened_at"] == "2026-05-22T12:00:00Z"
    assert store.memory_item_repo.calls[0]["extra"] == {"model": "claude-opus-4-6"}
    assert store.memory_item_repo.calibration_calls == ["claude-opus-4-6"]


@pytest.mark.asyncio
async def test_persist_memory_items_without_extract_model_does_not_set_extra() -> None:
    store = _StubStore()
    entry = SimpleNamespace(
        memory_type="profile",
        content="TestOwner likes tea",
        source_role="user",
        speaker_id="user:testowner",
        speaker_label="TestOwner",
        confidence=0.8,
        source_message_ids=[1],
        reflection_salience=0.7,
        emotional_intensity=0.4,
        memory_date="2026-05-22",
        categories=[],
        entities=[],
    )
    async def _find_supersede_targets(**_kwargs: Any) -> dict[int, str]:
        return {}

    entries, embeddings, targets = await persistence._prepare_memory_items(
        structured_entries=[entry], store=store, embed_client=_StubEmbedClient(),
        user={"user_id": "u1", "soul_id": "s1"},
        enable_confidence_normalization=False,
        normalize_confidence=lambda entries: entries,
        find_supersede_targets=_find_supersede_targets,
    )
    persistence._persist_memory_items(
        resource_id="res-1",
        structured_entries=entries,
        item_embeddings=embeddings,
        supersede_targets=targets,
        store=store,
        user={"user_id": "u1", "soul_id": "s1"},
        conversation_id="conv-1",
        segment_id="ep-1",
        extract_model=None,
        source_day_happened_at={"2026-05-22": "2026-05-22T12:00:00Z"},
        session=None,
        hedge_summary_for_confidence=lambda summary, _confidence: summary,
    )

    assert len(store.memory_item_repo.calls) == 1
    assert "extra" not in store.memory_item_repo.calls[0]
    assert store.memory_item_repo.calibration_calls == []
