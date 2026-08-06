from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from memu.database.sqlite.sqlite import SQLiteStore


class TaxonomyScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


SCOPE = {"user_id": "test-user", "soul_id": "test-soul"}
OTHER_SCOPE = {"user_id": "other-user", "soul_id": "other-soul"}


def _store(tmp_path, name: str = "taxonomy.db") -> SQLiteStore:
    return SQLiteStore(dsn=f"sqlite:///{tmp_path / name}", scope_model=TaxonomyScope)


def _item(store: SQLiteStore, scope: dict[str, str], summary: str = "memory"):
    return store.memory_item_repo.create_item(
        memory_type="episode",
        summary=summary,
        embedding=[1.0, 0.0],
        user_data=scope,
    )


def _category(
    store: SQLiteStore,
    scope: dict[str, str],
    name: str,
    **taxonomy,
):
    return store.memory_category_repo.get_or_create_category(
        name=name,
        description=f"{name} description",
        embedding=[0.0, 1.0],
        user_data=scope,
        **taxonomy,
    )


def test_fresh_schema_is_additive_and_runtime_creation_stays_inert(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    category = _category(store, SCOPE, "ordinary")

    with store._sessions.engine.connect() as conn:
        tables = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
        item_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(memory_items)")}
        category_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(categories)")}

    assert {"dossier_candidates", "memory_ref_counters"} <= tables
    assert "memory_ref" in item_columns
    assert {"kind", "lore_subtype", "entity_id", "anchor_role", "last_evidence_at", "last_revised_at"} <= category_columns
    assert item.memory_ref is None
    assert category.kind is None
    assert store.dossier_candidate_repo.list_candidates(SCOPE) == []


def test_legacy_reopen_adds_schema_without_assigning_taxonomy(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    store = _store(tmp_path, path.name)
    item = _item(store, SCOPE, "unchanged")
    category = _category(store, SCOPE, "legacy")
    store.close()

    with sqlite3.connect(path) as conn:
        for name in (
            "ix_categories__activity_scoped",
            "ix_categories__anchor_scoped",
            "ix_categories__entity_scoped",
            "ix_memory_items__ref_scoped",
        ):
            conn.execute(f"DROP INDEX {name}")
        conn.execute("DROP TABLE dossier_candidates")
        conn.execute("DROP TABLE memory_ref_counters")
        conn.execute("ALTER TABLE memory_items DROP COLUMN memory_ref")
        for column in ("kind", "lore_subtype", "entity_id", "anchor_role", "last_evidence_at", "last_revised_at"):
            conn.execute(f"ALTER TABLE categories DROP COLUMN {column}")

    reopened = _store(tmp_path, path.name)
    restored_item = reopened.memory_item_repo.get_item(item.id, include_superseded=True)
    restored_category = reopened.memory_category_repo.list_categories(SCOPE)[category.id]
    assert restored_item is not None and restored_item.summary == "unchanged" and restored_item.memory_ref is None
    assert restored_category.name == "legacy" and restored_category.kind is None
    assert reopened.dossier_candidate_repo.list_candidates(SCOPE) == []
    reopened.close()

    reopened_again = _store(tmp_path, path.name)
    reopened_item = reopened_again.memory_item_repo.get_item(item.id, include_superseded=True)
    assert reopened_item is not None and reopened_item.memory_ref is None


def test_memory_ref_allocator_is_scoped_atomic_and_never_rewinds(tmp_path) -> None:
    store = _store(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(lambda _: store.memory_item_repo.allocate_memory_ref(SCOPE), range(20)))
    assert sorted(refs) == list(range(1, 21))
    assert store.memory_item_repo.allocate_memory_ref(OTHER_SCOPE) == 1

    item = _item(store, SCOPE)
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = 20 WHERE id = ?", (item.id,))
    store.memory_item_repo.hard_delete_item(item.id, SCOPE)
    assert store.memory_item_repo.allocate_memory_ref(SCOPE) == 21


def test_memory_ref_caller_session_rollback_does_not_commit_counter(tmp_path) -> None:
    store = _store(tmp_path)
    with store._sessions.session() as session:
        assert store.memory_item_repo.allocate_memory_ref(SCOPE, session=session) == 1
        session.rollback()
    assert store.memory_item_repo.allocate_memory_ref(SCOPE) == 1


def test_memory_ref_backfill_is_deterministic_and_includes_merged_items(tmp_path) -> None:
    store = _store(tmp_path)
    first = _item(store, SCOPE, "first")
    second = _item(store, SCOPE, "second")
    same_time = "2026-01-01 00:00:00"
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET created_at = ? WHERE id IN (?, ?)", (same_time, first.id, second.id))
        conn.exec_driver_sql("UPDATE memory_items SET merged_into = ? WHERE id = ?", (first.id, second.id))
        timestamps_before = dict(
            conn.exec_driver_sql(
                "SELECT id, updated_at FROM memory_items WHERE id IN (?, ?)", (first.id, second.id)
            ).fetchall()
        )

    expected_ids = sorted([first.id, second.id])
    assigned = store.memory_item_repo.backfill_memory_refs(SCOPE)
    assert assigned == {item_id: index for index, item_id in enumerate(expected_ids, start=1)}
    assert store.memory_item_repo.backfill_memory_refs(SCOPE) == assigned
    with store._sessions.engine.connect() as conn:
        timestamps_after = dict(
            conn.exec_driver_sql(
                "SELECT id, updated_at FROM memory_items WHERE id IN (?, ?)", (first.id, second.id)
            ).fetchall()
        )
    assert timestamps_after == timestamps_before
    merged = store.memory_item_repo.get_item_by_memory_ref(assigned[second.id], SCOPE)
    assert merged is not None and merged.id == second.id
    assert store.memory_item_repo.get_item_by_memory_ref(assigned[first.id], OTHER_SCOPE) is None


def test_memory_ref_backfill_repairs_counter_and_rejects_mixed_state(tmp_path) -> None:
    repaired = _store(tmp_path, "repair.db")
    first = _item(repaired, SCOPE, "first")
    second = _item(repaired, SCOPE, "second")
    with repaired._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = 4 WHERE id = ?", (first.id,))
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = 9 WHERE id = ?", (second.id,))
    assert set(repaired.memory_item_repo.backfill_memory_refs(SCOPE).values()) == {4, 9}
    assert repaired.memory_item_repo.allocate_memory_ref(SCOPE) == 10

    mixed = _store(tmp_path, "mixed.db")
    assigned = _item(mixed, SCOPE, "assigned")
    unassigned = _item(mixed, SCOPE, "unassigned")
    with mixed._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = 1 WHERE id = ?", (assigned.id,))
    with pytest.raises(RuntimeError, match="mixed assigned/unassigned"):
        mixed.memory_item_repo.backfill_memory_refs(SCOPE)
    with mixed._sessions.engine.connect() as conn:
        rows = dict(conn.exec_driver_sql("SELECT id, memory_ref FROM memory_items").fetchall())
        counter_count = conn.exec_driver_sql("SELECT COUNT(*) FROM memory_ref_counters").scalar()
    assert rows == {assigned.id: 1, unassigned.id: None}
    assert counter_count == 0


def test_dossier_fields_anchors_and_activity_order_round_trip(tmp_path) -> None:
    store = _store(tmp_path)
    old = datetime(2025, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 1, 1, tzinfo=timezone.utc)
    soul = _category(store, SCOPE, "self", kind="lore", anchor_role="soul", lore_subtype="identity")
    user = _category(store, SCOPE, "user", kind="lore", anchor_role="user")
    alpha = _category(store, SCOPE, "Alpha", kind="topic", entity_id="entity-a", last_evidence_at=old)
    beta = _category(store, SCOPE, "beta", kind="topic", last_evidence_at=new)
    null_b = _category(store, SCOPE, "bravo", kind="topic")
    null_a = _category(store, SCOPE, "Able", kind="topic")

    anchors = store.memory_category_repo.list_anchor_categories(SCOPE)
    assert anchors == {"soul": soul, "user": user}
    assert [row.id for row in store.memory_category_repo.list_categories_by_activity(SCOPE, kind="topic")] == [
        beta.id,
        alpha.id,
        null_a.id,
        null_b.id,
    ]
    assert store.memory_category_repo.list_anchor_categories(OTHER_SCOPE) == {}
    assert store.memory_category_repo.list_categories(SCOPE)[alpha.id].entity_id == "entity-a"
    revised = store.memory_category_repo.update_category(
        category_id=alpha.id,
        lore_subtype="person",
        last_revised_at=new,
    )
    assert revised.kind == "topic"
    assert revised.entity_id == "entity-a"
    assert revised.lore_subtype == "person"
    assert revised.last_revised_at == new.replace(tzinfo=None)
    with pytest.raises(IntegrityError):
        _category(store, SCOPE, "second self", kind="lore", anchor_role="soul")
    _category(store, OTHER_SCOPE, "other self", kind="lore", anchor_role="soul")


def test_candidate_lifecycle_is_durable_idempotent_and_scope_safe(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    first = store.dossier_candidate_repo.add_candidate(
        proposed_name="Health Span",
        item_id=item.id,
        where=SCOPE,
        segment_id="first-segment",
        memory_day="20260806",
    )
    duplicate = store.dossier_candidate_repo.add_candidate(
        proposed_name="health-span!",
        item_id=item.id,
        where=SCOPE,
        segment_id="later-segment",
        memory_day="2026-08-07",
    )
    other = store.dossier_candidate_repo.add_candidate(
        proposed_name="Health Span",
        item_id=item.id,
        where=OTHER_SCOPE,
    )
    assert duplicate.id == first.id
    assert duplicate.proposed_name == "Health Span"
    assert duplicate.segment_id == "first-segment"
    assert duplicate.memory_day == "2026-08-06"
    assert other.id != first.id
    store.close()

    reopened = _store(tmp_path)
    assert [row.id for row in reopened.dossier_candidate_repo.list_candidates(SCOPE)] == [first.id]
    with pytest.raises(ValueError):
        reopened.dossier_candidate_repo.add_candidate(
            proposed_name="bad day",
            item_id=item.id,
            where=SCOPE,
            memory_day="not-a-day",
        )


def test_candidate_batch_resolution_is_atomic_and_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    one = store.dossier_candidate_repo.add_candidate(proposed_name="one", item_id=item.id, where=SCOPE)
    two = store.dossier_candidate_repo.add_candidate(proposed_name="two", item_id=item.id, where=SCOPE)
    target = _category(store, SCOPE, "target", kind="topic")
    other_target = _category(store, SCOPE, "other target", kind="topic")

    resolved = store.dossier_candidate_repo.resolve_candidates([one.id, two.id], target.id, SCOPE)
    assert {row.resolved_category_id for row in resolved} == {target.id}
    timestamps = {row.resolved_at for row in resolved}
    assert len(timestamps) == 1
    assert store.dossier_candidate_repo.list_candidates(SCOPE) == []
    assert {row.id for row in store.dossier_candidate_repo.resolve_candidates([one.id, two.id], target.id, SCOPE)} == {
        one.id,
        two.id,
    }

    three = store.dossier_candidate_repo.add_candidate(proposed_name="three", item_id=item.id, where=SCOPE)
    with pytest.raises(ValueError, match="another category"):
        store.dossier_candidate_repo.resolve_candidates([one.id, three.id], other_target.id, SCOPE)
    unresolved = store.dossier_candidate_repo.list_candidates(SCOPE)
    assert [row.id for row in unresolved] == [three.id]
    with pytest.raises(KeyError, match="not found in scope"):
        store.dossier_candidate_repo.resolve_candidates([three.id, "missing"], target.id, SCOPE)
    assert [row.id for row in store.dossier_candidate_repo.list_candidates(SCOPE)] == [three.id]


def test_new_taxonomy_methods_require_complete_scope(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="soul_id"):
        store.memory_item_repo.allocate_memory_ref({"user_id": "test-user"})
    with pytest.raises(ValueError, match="soul_id"):
        store.dossier_candidate_repo.list_candidates({"user_id": "test-user"})
