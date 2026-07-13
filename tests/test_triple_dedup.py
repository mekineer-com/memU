"""Triple dedup + symmetric-predicate canonicalization tests.

Mempalace dedups currently-valid identical triples on insert and treats
symmetric predicates (conflicts_with, parallels) as direction-independent.
Memu's triple_repo now does both.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from memu.app.service import MemoryService
from memu.database.models import Triple


class TripleDedupScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


@pytest.fixture(scope="module")
def store():
    # Constructed once per module so all tests share the same SQLAlchemy
    # metadata binding (constructing a MemoryService in multiple tests
    # collides in the declarative registry). Each test uses a unique scope
    # + memory ids so writes can't cross-contaminate.
    svc = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": TripleDedupScope},
    )
    return svc._get_database()


def _triple(subject_id: str, predicate: str, object_id: str) -> Triple:
    return Triple(
        subject_id=subject_id,
        subject_kind="memory",
        predicate=predicate,
        object_id=object_id,
        object_kind="memory",
        confidence=0.8,
        source_memory_id=subject_id,
    )


def test_identical_triple_inserted_twice_dedups_to_one_row(store) -> None:
    scope = {"user_id": "tdu_identical", "soul_id": "s"}
    first = store.triple_repo.add(_triple("id_a", "shaped_by", "id_b"), user_data=scope)
    second = store.triple_repo.add(_triple("id_a", "shaped_by", "id_b"), user_data=scope)

    assert first.id == second.id
    edges = store.triple_repo.get_edges_from("id_a", predicate="shaped_by", where=scope)
    assert len(edges) == 1


def test_symmetric_predicate_dedups_across_endpoint_order(store) -> None:
    scope = {"user_id": "tdu_symmetric", "soul_id": "s"}
    store.triple_repo.add(_triple("sym_b", "parallels", "sym_a"), user_data=scope)
    store.triple_repo.add(_triple("sym_a", "parallels", "sym_b"), user_data=scope)

    # Canonical order is sorted; "sym_a" < "sym_b" → subject=sym_a, object=sym_b.
    from_a = store.triple_repo.get_edges_from("sym_a", predicate="parallels", where=scope)
    from_b = store.triple_repo.get_edges_from("sym_b", predicate="parallels", where=scope)
    assert len(from_a) == 1
    assert from_a[0].subject_id == "sym_a"
    assert from_a[0].object_id == "sym_b"
    assert len(from_b) == 0


def test_asymmetric_predicate_keeps_both_directions(store) -> None:
    scope = {"user_id": "tdu_asym", "soul_id": "s"}
    store.triple_repo.add(_triple("asym_a", "caused_by", "asym_b"), user_data=scope)
    store.triple_repo.add(_triple("asym_b", "caused_by", "asym_a"), user_data=scope)

    assert len(store.triple_repo.get_edges_from("asym_a", predicate="caused_by", where=scope)) == 1
    assert len(store.triple_repo.get_edges_from("asym_b", predicate="caused_by", where=scope)) == 1


def test_dedup_is_scoped_per_user(store) -> None:
    scope_a = {"user_id": "tdu_scope_alpha", "soul_id": "s"}
    scope_b = {"user_id": "tdu_scope_beta", "soul_id": "s"}

    store.triple_repo.add(_triple("sc_a", "shaped_by", "sc_b"), user_data=scope_a)
    store.triple_repo.add(_triple("sc_a", "shaped_by", "sc_b"), user_data=scope_b)

    assert len(store.triple_repo.get_edges_from("sc_a", predicate="shaped_by", where=scope_a)) == 1
    assert len(store.triple_repo.get_edges_from("sc_a", predicate="shaped_by", where=scope_b)) == 1


def test_invalidate_finds_canonicalized_row_regardless_of_argument_order(store) -> None:
    scope = {"user_id": "tdu_invalidate", "soul_id": "s"}
    store.triple_repo.add(_triple("inv_a", "conflicts_with", "inv_b"), user_data=scope)
    # Caller passes the reversed endpoint order — canonicalization should still find it.
    store.triple_repo.invalidate("inv_b", "conflicts_with", "inv_a", scope=scope)

    current = store.triple_repo.get_edges_from(
        "inv_a", predicate="conflicts_with", current_only=True, where=scope
    )
    assert current == []


def test_bulk_memory_edges_keep_scope_direction_predicate_and_current_filters(store) -> None:
    scope = {"user_id": "tdu_bulk", "soul_id": "s"}
    other_scope = {"user_id": "tdu_bulk_other", "soul_id": "s"}
    outgoing = store.triple_repo.add(_triple("bulk_a", "caused_by", "bulk_b"), user_data=scope)
    incoming = store.triple_repo.add(_triple("bulk_c", "shaped_by", "bulk_a"), user_data=scope)
    ignored_predicate = store.triple_repo.add(_triple("bulk_a", "unrelated", "bulk_d"), user_data=scope)
    other = store.triple_repo.add(_triple("bulk_a", "caused_by", "bulk_e"), user_data=other_scope)
    expired = store.triple_repo.add(_triple("bulk_a", "parallels", "bulk_f"), user_data=scope)
    store.triple_repo.invalidate("bulk_a", "parallels", "bulk_f", scope=scope)

    edges = store.triple_repo.list_edges_for_memories(
        {"bulk_a"},
        {"caused_by", "shaped_by", "parallels"},
        where=scope,
    )

    assert {edge.id for edge in edges} == {outgoing.id, incoming.id}
    assert ignored_predicate.id not in {edge.id for edge in edges}
    assert other.id not in {edge.id for edge in edges}
    assert expired.id not in {edge.id for edge in edges}
