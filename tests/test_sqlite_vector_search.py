from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel
from sqlalchemy import event

from memu.database.models import Triple
from memu.database.sqlite.sqlite import SQLiteStore
from memu.database.vector import (
    autocut_first_cluster,
    cosine_topk,
    reciprocal_rank_fusion,
    relative_score_fusion,
)


class SearchScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def test_relative_score_fusion_supports_first_cluster_autocut() -> None:
    fused = relative_score_fusion(
        [("alpha", 0.90), ("beta", 0.89), ("gamma", 0.20), ("delta", 0.10)],
        [("beta", 0.90), ("alpha", 0.88), ("gamma", 0.20), ("delta", 0.10)],
    )

    assert [item_id for item_id, _score in autocut_first_cluster(fused)] == ["beta", "alpha"]


@pytest.fixture
def store() -> SQLiteStore:
    value = SQLiteStore(dsn="sqlite:///:memory:", scope_model=SearchScope)
    yield value
    value.close()


def test_sql_vector_search_matches_numpy_and_scope(store: SQLiteStore, caplog: pytest.LogCaptureFixture) -> None:
    scope = {"user_id": "u", "soul_id": "s"}
    rows = [
        ("a", [1.0, 0.0]),
        ("b", [0.8, 0.2]),
        ("c", [0.2, 0.8]),
    ]
    created = [
        store.memory_item_repo.create_item(
            memory_type="knowledge", summary=item_id, embedding=embedding, user_data=scope
        )
        for item_id, embedding in rows
    ]
    store.memory_item_repo.create_item(
        memory_type="knowledge",
        summary="closer but wrong scope",
        embedding=[1.0, 0.0],
        user_data={"user_id": "other", "soul_id": "s"},
    )
    store.memory_item_repo.create_item(
        memory_type="knowledge", summary="wrong dimension", embedding=[1.0, 0.0, 0.0], user_data=scope
    )

    with caplog.at_level(logging.ERROR):
        actual = store.memory_item_repo.vector_search_items([1.0, 0.0], 3, scope)
    expected = cosine_topk([1.0, 0.0], [(item.id, vector) for item, (_, vector) in zip(created, rows, strict=True)], 3)

    assert [item_id for item_id, _ in actual] == [item_id for item_id, _ in expected]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected], abs=1e-6)
    assert "found dims: [3]" in caplog.text


def test_hybrid_search_and_embedding_free_materialization(store: SQLiteStore, monkeypatch) -> None:
    scope = {"user_id": "u", "soul_id": "s"}
    lexical = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="platypus habitat", embedding=[0.0, 1.0], user_data=scope
    )
    store.memory_item_repo.create_item(
        memory_type="knowledge", summary="different topic", embedding=[1.0, 0.0], user_data=scope
    )

    pool = store.memory_item_repo.list_items(scope)
    vector_hits = cosine_topk([1.0, 0.0], [(item.id, item.embedding) for item in pool.values()], 2)
    fts_hits = store.memory_item_repo.fts_search_items("platypus", 2, pool_ids=set(pool))
    expected = reciprocal_rank_fusion(vector_hits, fts_hits)
    actual = store.memory_item_repo.vector_search_items(
        [1.0, 0.0], 2, scope, fts_enabled=True, fts_query="platypus", fts_top_k=2
    )
    assert lexical.id in {item_id for item_id, _ in actual}
    assert [item_id for item_id, _ in actual] == [item_id for item_id, _ in expected]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected])

    statements = []
    event.listen(
        store._sessions.engine, "before_cursor_execute", lambda _c, _cu, sql, _p, _ctx, _many: statements.append(sql)
    )
    monkeypatch.setattr(store.memory_item_repo, "_normalize_embedding", lambda _value: pytest.fail("decoded"))
    pool = store.memory_item_repo.list_items(scope, include_embeddings=False)
    assert pool and all(item.embedding == [] for item in pool.values())
    assert "memory_items.embedding" not in statements[-1]


def test_superseded_salience_uses_actual_metadata(store: SQLiteStore, monkeypatch) -> None:
    scope = {"user_id": "u", "soul_id": "s"}
    old = store.memory_item_repo.create_item(
        memory_type="knowledge",
        summary="old",
        embedding=[1.0, 0.0],
        reflection_salience=0.9,
        emotional_intensity=0.8,
        user_data=scope,
    )
    new = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="new", embedding=[0.9, 0.1], user_data=scope
    )
    store.triple_repo.add(
        Triple(
            subject_id=old.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=new.id,
            object_kind="memory",
        ),
        user_data=scope,
    )
    captured = []

    def capture(candidates, **_kwargs):
        captured.extend(candidates)
        return [(item_id, score) for item_id, score, *_rest in candidates]

    monkeypatch.setattr("memu.database.sqlite.repositories.memory_item_repo.rerank_by_salience", capture)
    active_hits = store.memory_item_repo.vector_search_items([1.0, 0.0], 2, scope)
    assert old.id not in {item_id for item_id, _ in active_hits}
    store.memory_item_repo.vector_search_items([1.0, 0.0], 2, scope, ranking="salience", include_superseded=True)

    old_candidate = next(candidate for candidate in captured if candidate[0] == old.id)
    assert old_candidate[3:5] == (0.9, 0.8)
