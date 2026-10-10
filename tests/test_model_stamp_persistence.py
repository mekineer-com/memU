from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memu.app import memorize_persistence as persistence
from memu.app.memorize import MemorizeMixin, StructuredMemoryEntry


def _confidence_entries(scores: list[float | None]) -> list[StructuredMemoryEntry]:
    return [StructuredMemoryEntry(
        memory_type="profile", content="TestOwner likes tea", categories=[], source_role="user",
        confidence=score, source_message_ids=[1], reflection_salience=0.7,
    ) for score in scores]


@pytest.mark.parametrize(("scores", "expected"), [
    ([0.9] * 5 + [0.85], [0.77] * 5 + [0.36]),
    ([0.9] * 30 + [0.85, None], [0.73] * 30 + [None, None]),
    ([0.85] * 30 + [0.9, None], [0.67] * 30 + [None, None]),
])
def test_confidence_normalization_preserves_existing_bounds_policy(scores, expected) -> None:
    entries = _confidence_entries(scores)
    normalized = MemorizeMixin._normalize_confidence(entries)
    assert [entry.confidence for entry in normalized] == expected
    assert [entry.content for entry in normalized] == [entry.content for entry in entries]


def test_confidence_normalization_matches_shifted_model_scores() -> None:
    model_a = MemorizeMixin._normalize_confidence(_confidence_entries([0.9] * 5 + [0.85]))
    model_b = MemorizeMixin._normalize_confidence(_confidence_entries([0.8] * 5 + [0.75]))
    assert [entry.confidence for entry in model_a] == [entry.confidence for entry in model_b]


@pytest.mark.parametrize("scores", [
    [], [None] * 6, [0.9] * 4 + [0.85, None], [0.8] * 6,
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
])
def test_confidence_normalization_preserves_existing_gate(scores) -> None:
    entries = _confidence_entries(scores)
    assert MemorizeMixin._normalize_confidence(entries) is entries


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
    def __init__(self) -> None:
        self.payloads: list[list[str]] = []

    async def embed(self, payloads: list[str]) -> list[list[float]]:
        self.payloads.append(list(payloads))
        return [[0.1, 0.2, 0.3] for _ in payloads]


@pytest.mark.asyncio
async def test_persist_memory_items_stamps_extract_model_in_extra() -> None:
    store = _StubStore()
    entry = StructuredMemoryEntry(
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

    raw_entries = [entry._replace(confidence=score) for score in [0.9] * 5 + [0.85]]
    embed_client = _StubEmbedClient()
    entries, embeddings, targets = await persistence._prepare_memory_items(
        structured_entries=raw_entries, store=store, embed_client=embed_client,
        user={"user_id": "u1", "soul_id": "s1"},
        enable_confidence_normalization=True,
        normalize_confidence=MemorizeMixin._normalize_confidence,
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
    )

    assert len(items) == 6
    assert homeless == 6
    assert len(store.memory_item_repo.calls) == 6
    assert embed_client.payloads == [[entry.content] * 6]
    assert [call["summary"] for call in store.memory_item_repo.calls] == [entry.content] * 6
    assert [item.summary for item in items] == [entry.content] * 6
    assert [call["embedding"] for call in store.memory_item_repo.calls] == embeddings == [[0.1, 0.2, 0.3]] * 6
    assert [call["confidence"] for call in store.memory_item_repo.calls] == [0.77] * 5 + [0.36]
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
    )

    assert len(store.memory_item_repo.calls) == 1
    assert "extra" not in store.memory_item_repo.calls[0]
    assert store.memory_item_repo.calibration_calls == []
