from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from memu.app import memorize_categories
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


def test_fresh_schema_is_additive_and_runtime_creation_allocates_reference(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    category = _category(store, SCOPE, "ordinary")

    with store._sessions.engine.connect() as conn:
        tables = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
        item_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(memory_items)")}
        category_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(categories)")}
        candidate_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(dossier_candidates)")}

    assert {"dossier_candidates", "memory_ref_counters"} <= tables
    assert "memory_ref" in item_columns
    assert "last_considered_at" in candidate_columns
    assert {
        "kind",
        "lore_subtype",
        "entity_id",
        "anchor_role",
        "last_evidence_at",
        "last_revised_at",
        "previous_description",
        "approved_description",
    } <= category_columns
    assert item.memory_ref == 1
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
    assert item.memory_ref == 21
    store.memory_item_repo.hard_delete_item(item.id, SCOPE)
    assert store.memory_item_repo.allocate_memory_ref(SCOPE) == 22


def test_memory_ref_caller_session_rollback_does_not_commit_counter(tmp_path) -> None:
    store = _store(tmp_path)
    with store._sessions.session() as session:
        assert store.memory_item_repo.allocate_memory_ref(SCOPE, session=session) == 1
        session.rollback()
    assert store.memory_item_repo.allocate_memory_ref(SCOPE) == 1


def test_create_item_allocates_reference_in_caller_transaction(tmp_path) -> None:
    store = _store(tmp_path)
    with store._sessions.session() as session:
        item = store.memory_item_repo.create_item(
            memory_type="episode",
            summary="transactional memory",
            embedding=[1.0, 0.0],
            user_data=SCOPE,
            session=session,
        )
        assert item.memory_ref == 1
        session.rollback()

    assert store.memory_item_repo.list_items(SCOPE) == {}
    assert store.memory_item_repo.allocate_memory_ref(SCOPE) == 1


def test_memory_ref_allocator_rejects_non_positive_counter(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.memory_item_repo.allocate_memory_ref(SCOPE) == 1
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_ref_counters SET next_value = 0")

    with pytest.raises(RuntimeError, match="counter must be positive"):
        store.memory_item_repo.allocate_memory_ref(SCOPE)
    with store._sessions.engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT next_value FROM memory_ref_counters").scalar_one() == 0


def test_memory_ref_backfill_is_deterministic_and_includes_merged_items(tmp_path) -> None:
    store = _store(tmp_path)
    first = _item(store, SCOPE, "first")
    second = _item(store, SCOPE, "second")
    same_time = "2026-01-01 00:00:00"
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET created_at = ? WHERE id IN (?, ?)", (same_time, first.id, second.id))
        conn.exec_driver_sql("UPDATE memory_items SET merged_into = ? WHERE id = ?", (first.id, second.id))
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = NULL")
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
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = NULL WHERE id = ?", (unassigned.id,))
        conn.exec_driver_sql("DELETE FROM memory_ref_counters")
    with pytest.raises(RuntimeError, match="mixed assigned/unassigned"):
        mixed.memory_item_repo.backfill_memory_refs(SCOPE)
    with mixed._sessions.engine.connect() as conn:
        rows = dict(conn.exec_driver_sql("SELECT id, memory_ref FROM memory_items").fetchall())
        counter_count = conn.exec_driver_sql("SELECT COUNT(*) FROM memory_ref_counters").scalar()
    assert rows == {assigned.id: 1, unassigned.id: None}
    assert counter_count == 0

    invalid = _store(tmp_path, "invalid.db")
    invalid_item = _item(invalid, SCOPE)
    with invalid._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = 0 WHERE id = ?", (invalid_item.id,))
    with pytest.raises(RuntimeError, match="non-positive"):
        invalid.memory_item_repo.backfill_memory_refs(SCOPE)


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
    cleared = store.memory_category_repo.update_category(
        category_id=alpha.id,
        lore_subtype=None,
        entity_id=None,
        last_revised_at=None,
    )
    assert cleared.lore_subtype is None
    assert cleared.entity_id is None
    assert cleared.last_revised_at is None
    with pytest.raises(IntegrityError):
        _category(store, SCOPE, "second self", kind="lore", anchor_role="soul")
    _category(store, OTHER_SCOPE, "other self", kind="lore", anchor_role="soul")


def test_description_approval_backfill_runs_once(tmp_path) -> None:
    path = tmp_path / "description-approval.db"
    store = _store(tmp_path, path.name)
    category = _category(store, SCOPE, "Health", kind="topic")
    store.close()

    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE categories DROP COLUMN previous_description")
        conn.execute("ALTER TABLE categories DROP COLUMN approved_description")

    reopened = _store(tmp_path, path.name)
    restored = reopened.memory_category_repo.list_categories(SCOPE)[category.id]
    assert restored.approved_description == "Health description"
    reopened.close()

    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE categories SET approved_description = NULL WHERE id = ?", (category.id,))

    reopened_again = _store(tmp_path, path.name)
    assert reopened_again.memory_category_repo.list_categories(SCOPE)[category.id].approved_description is None


def test_candidate_lifecycle_is_durable_idempotent_and_scope_safe(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    other_item = _item(store, OTHER_SCOPE)
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
        item_id=other_item.id,
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


def test_clear_categories_removes_only_scoped_taxonomy_state(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    other_item = _item(store, OTHER_SCOPE)
    category = _category(store, SCOPE, "Health", kind="topic")
    other_category = _category(store, OTHER_SCOPE, "Work", kind="topic")
    relation = store.category_item_repo.link_item_category(item.id, category.id, SCOPE)
    other_relation = store.category_item_repo.link_item_category(
        other_item.id, other_category.id, OTHER_SCOPE
    )
    candidate = store.dossier_candidate_repo.add_candidate(
        proposed_name="Wellbeing", item_id=item.id, where=SCOPE
    )
    store.dossier_candidate_repo.resolve_candidates([candidate.id], category.id, SCOPE)
    store.dossier_candidate_repo.add_candidate(
        proposed_name="Daily Care", item_id=item.id, where=SCOPE
    )
    store.dossier_candidate_repo.add_candidate(
        proposed_name="Career", item_id=other_item.id, where=OTHER_SCOPE
    )

    assert store.memory_category_repo.clear_categories(SCOPE) == {category.id: category}
    assert store.memory_category_repo.list_categories(SCOPE) == {}
    assert store.category_item_repo.list_relations(SCOPE) == []
    assert store.dossier_candidate_repo.list_candidates(SCOPE, unresolved_only=False) == []
    assert store.memory_item_repo.get_item(item.id) is not None
    assert list(store.memory_category_repo.list_categories(OTHER_SCOPE)) == [other_category.id]
    assert [row.id for row in store.category_item_repo.list_relations(OTHER_SCOPE)] == [
        other_relation.id
    ]
    assert len(store.dossier_candidate_repo.list_candidates(OTHER_SCOPE)) == 1
    assert relation not in store.category_item_repo.relations

    store.dossier_candidate_repo.add_candidate(
        proposed_name="Candidate Only", item_id=item.id, where=SCOPE
    )
    assert store.memory_category_repo.clear_categories(SCOPE) == {}
    assert store.dossier_candidate_repo.list_candidates(SCOPE) == []


def test_legacy_candidate_table_gains_consideration_column(tmp_path) -> None:
    path = tmp_path / "legacy-candidate.db"
    store = _store(tmp_path, path.name)
    item = _item(store, SCOPE)
    candidate = store.dossier_candidate_repo.add_candidate(
        proposed_name="candidate", item_id=item.id, where=SCOPE
    )
    store.close()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE dossier_candidates DROP COLUMN last_considered_at")

    reopened = _store(tmp_path, path.name)
    restored = reopened.dossier_candidate_repo.list_candidates(SCOPE)
    assert [row.id for row in restored] == [candidate.id]
    assert restored[0].last_considered_at is None


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


def test_category_proposal_filing_keeps_known_and_unknown_and_is_atomic(tmp_path) -> None:
    store = _store(tmp_path)
    item = _item(store, SCOPE)
    known = _category(store, SCOPE, "Known", kind="topic")
    _category(store, SCOPE, "Identity", kind="lore", anchor_role="soul")

    with store._sessions.session() as session:
        relations, candidates = memorize_categories.file_category_proposals(
            store=store,
            item_proposals=[(item, ["Known", "New Domain", "Identity"])],
            where=SCOPE,
            session=session,
        )
        session.commit()
    assert [relation.category_id for relation in relations] == [known.id]
    assert [candidate.normalized_name for candidate in candidates] == ["new_domain", "identity"]

    with store._sessions.session() as session:
        repeated = memorize_categories.file_category_proposals(
            store=store,
            item_proposals=[(item, ["known"]), (item, ["new-domain!", "identity"])],
            where=SCOPE,
            session=session,
        )
        session.commit()
    assert repeated[0][0].id == relations[0].id
    assert [row.id for row in repeated[1]] == [row.id for row in candidates]

    with store._sessions.session() as session:
        with pytest.raises(ValueError, match="more than three"):
            memorize_categories.file_category_proposals(
                store=store,
                item_proposals=[(item, ["one", "two"]), (item, ["three", "four"])],
                where=SCOPE,
                session=session,
            )
        session.rollback()
    assert [row.normalized_name for row in store.dossier_candidate_repo.list_candidates(SCOPE)] == [
        "new_domain",
        "identity",
    ]


@pytest.mark.asyncio
async def test_dynamic_category_review_accumulates_all_candidates_and_obeys_threshold(tmp_path) -> None:
    store = _store(tmp_path)
    anchor = _category(store, SCOPE, "Self", kind="lore", anchor_role="soul")
    exact = _category(store, SCOPE, "Shared Topic", kind="topic")
    store.memory_category_repo.update_category(category_id=exact.id, summary="private prose")

    for index in range(9):
        item = _item(store, SCOPE, f"memory {index}")
        store.dossier_candidate_repo.add_candidate(
            proposed_name="Shared Topic" if index == 0 else "new topic",
            item_id=item.id,
            where=SCOPE,
        )

    async def search(query, **kwargs):
        assert query
        assert anchor.id not in {category.id for category in kwargs["categories"]}
        return []

    assert await memorize_categories.prepare_dynamic_category_review(
        store=store,
        where=SCOPE,
        cluster_size=10,
        search_dossiers=search,
    ) == []
    tenth = _item(store, SCOPE, "memory 9")
    store.dossier_candidate_repo.add_candidate(
        proposed_name="new topic", item_id=tenth.id, where=SCOPE
    )
    store.close()

    reopened = _store(tmp_path)
    bundles = await memorize_categories.prepare_dynamic_category_review(
        store=reopened,
        where=SCOPE,
        cluster_size=10,
        search_dossiers=search,
    )
    assert len(bundles) == 1
    assert bundles[0]["memory_count"] == 10
    assert len(bundles[0]["candidate_ids"]) == 10
    assert [category.id for category in bundles[0]["existing_dossiers"]] == [exact.id]
    assert bundles[0]["existing_dossiers"][0].summary is None

    with reopened._sessions.session() as session:
        reopened.dossier_candidate_repo.mark_candidates_considered(
            bundles[0]["candidate_ids"], datetime.now(timezone.utc), SCOPE, session
        )
        session.commit()
    assert await memorize_categories.prepare_dynamic_category_review(
        store=reopened,
        where=SCOPE,
        cluster_size=10,
        search_dossiers=search,
    ) == []
    new_item = _item(reopened, SCOPE, "new evidence")
    reopened.dossier_candidate_repo.add_candidate(
        proposed_name="new topic", item_id=new_item.id, where=SCOPE
    )
    assert len(
        await memorize_categories.prepare_dynamic_category_review(
            store=reopened,
            where=SCOPE,
            cluster_size=10,
            search_dossiers=search,
        )
    ) == 1


@pytest.mark.asyncio
async def test_dynamic_category_review_counts_canonical_memories_and_fails_on_bad_lineage(tmp_path) -> None:
    store = _store(tmp_path)
    source = _item(store, SCOPE, "source")
    middle = _item(store, SCOPE, "middle")
    survivor = _item(store, SCOPE, "survivor")
    other = _item(store, SCOPE, "other")
    store.memory_item_repo.update_item(item_id=source.id, merged_into=middle.id)
    store.memory_item_repo.update_item(item_id=middle.id, merged_into=survivor.id)
    store.dossier_candidate_repo.add_candidate(proposed_name="topic", item_id=source.id, where=SCOPE)
    store.dossier_candidate_repo.add_candidate(proposed_name="topic", item_id=middle.id, where=SCOPE)
    store.dossier_candidate_repo.add_candidate(proposed_name="topic", item_id=other.id, where=SCOPE)

    async def no_hits(_query, **_kwargs):
        return []

    bundles = await memorize_categories.prepare_dynamic_category_review(
        store=store,
        where=SCOPE,
        cluster_size=2,
        search_dossiers=no_hits,
    )
    assert bundles[0]["memory_count"] == 2
    assert len(bundles[0]["candidate_ids"]) == 3

    store.memory_item_repo.update_item(item_id=survivor.id, merged_into=source.id)
    with pytest.raises(ValueError, match="merge cycle"):
        await memorize_categories.prepare_dynamic_category_review(
            store=store,
            where=SCOPE,
            cluster_size=2,
            search_dossiers=no_hits,
        )

    broken = _store(tmp_path, "broken.db")
    deleted = _item(broken, SCOPE)
    live = _item(broken, SCOPE)
    broken.dossier_candidate_repo.add_candidate(proposed_name="topic", item_id=deleted.id, where=SCOPE)
    broken.dossier_candidate_repo.add_candidate(proposed_name="topic", item_id=live.id, where=SCOPE)
    broken.memory_item_repo.hard_delete_item(deleted.id, SCOPE)
    with pytest.raises(KeyError, match="not found in scope"):
        await memorize_categories.prepare_dynamic_category_review(
            store=broken,
            where=SCOPE,
            cluster_size=2,
            search_dossiers=no_hits,
        )


@pytest.mark.asyncio
async def test_dynamic_category_review_generation_is_strict_and_bundle_bound(tmp_path) -> None:
    store = _store(tmp_path)
    items = [_item(store, SCOPE, f"memory {index}") for index in range(2)]
    candidates = [
        store.dossier_candidate_repo.add_candidate(
            proposed_name="garden rituals",
            item_id=item.id,
            where=SCOPE,
            memory_day=f"2026-08-0{index + 1}",
        )
        for index, item in enumerate(items)
    ]
    existing = _category(store, SCOPE, "Garden Life", kind="lore")
    soul_anchor = _category(store, SCOPE, "Test Soul", kind="lore", anchor_role="soul")
    store.memory_category_repo.update_category(
        category_id=soul_anchor.id,
        summary="## Becoming\nI find wonder in little rituals [M999].",
    )
    async def nearby(_query, **_kwargs):
        return [(existing, 0.9)]

    bundle = (
        await memorize_categories.prepare_dynamic_category_review(
            store=store,
            where=SCOPE,
            cluster_size=2,
            search_dossiers=nearby,
        )
    )[0]
    accepted, rejected = [candidate.id for candidate in candidates]
    create_xml = f"""<dynamic_dossier_review cluster_id="{bundle['cluster_id']}">
  <action>create</action>
  <accepted_candidate_ids><candidate_id>{accepted}</candidate_id></accepted_candidate_ids>
  <rejected_candidate_ids><candidate_id>{rejected}</candidate_id></rejected_candidate_ids>
  <title>Garden Magic</title><description>A bright little world of growing things.</description><kind>lore</kind>
</dynamic_dossier_review>"""

    class ChatClient:
        def __init__(self, response: str) -> None:
            self.response = response
            self.calls: list[tuple[str, str | None]] = []

        async def chat(self, prompt: str, system_prompt: str | None = None) -> str:
            self.calls.append((prompt, system_prompt))
            return self.response

    client = ChatClient(create_xml)
    decision = await memorize_categories.generate_dynamic_category_review(
        bundle=bundle,
        select_chat_client=lambda *_args, **_kwargs: pytest.fail("injected client ignored"),
        profile="category",
        chat_client=client,
    )
    assert decision == {
        "cluster_id": bundle["cluster_id"],
        "action": "create",
        "accepted_candidate_ids": [accepted],
        "rejected_candidate_ids": [rejected],
        "name": "Garden Magic",
        "description": "A bright little world of growing things.",
        "kind": "lore",
    }
    assert all(candidate.id in client.calls[0][0] for candidate in candidates)
    assert "[M1]" in client.calls[0][0] and "[M2]" in client.calls[0][0]
    assert "I find wonder in little rituals." in client.calls[0][0]
    assert "[M999]" not in client.calls[0][0]

    existing_xml = f"""<dynamic_dossier_review cluster_id="{bundle['cluster_id']}">
  <action>existing</action>
  <accepted_candidate_ids><candidate_id>{accepted}</candidate_id></accepted_candidate_ids>
  <rejected_candidate_ids><candidate_id>{rejected}</candidate_id></rejected_candidate_ids>
  <existing_dossier_id>{existing.id}</existing_dossier_id>
</dynamic_dossier_review>"""
    assert memorize_categories.parse_dynamic_category_review(existing_xml, bundle) == {
        "cluster_id": bundle["cluster_id"],
        "action": "existing",
        "accepted_candidate_ids": [accepted],
        "rejected_candidate_ids": [rejected],
        "existing_dossier_id": existing.id,
    }

    defer_xml = f"""<dynamic_dossier_review cluster_id="{bundle['cluster_id']}">
  <action>defer</action><accepted_candidate_ids></accepted_candidate_ids>
  <rejected_candidate_ids><candidate_id>{accepted}</candidate_id><candidate_id>{rejected}</candidate_id></rejected_candidate_ids>
</dynamic_dossier_review>"""
    assert memorize_categories.parse_dynamic_category_review(defer_xml, bundle) == {
        "cluster_id": bundle["cluster_id"],
        "action": "defer",
        "accepted_candidate_ids": [],
        "rejected_candidate_ids": [accepted, rejected],
    }

    invalid = [
        f"```xml\n{create_xml}\n```",
        create_xml.replace(bundle["cluster_id"], "foreign-cluster", 1),
        create_xml.replace(rejected, "foreign-candidate"),
        create_xml.replace(f"<candidate_id>{rejected}</candidate_id>", ""),
        existing_xml.replace(existing.id, "foreign-dossier"),
        existing_xml.replace(f"<candidate_id>{accepted}</candidate_id>", ""),
        existing_xml.replace(
            "</dynamic_dossier_review>", "<kind>lore</kind></dynamic_dossier_review>"
        ),
        create_xml.replace(
            f"<candidate_id>{accepted}</candidate_id>",
            f"<candidate_id>{accepted}</candidate_id><candidate_id>{accepted}</candidate_id>",
        ),
        defer_xml.replace(
            "<accepted_candidate_ids></accepted_candidate_ids>",
            f"<accepted_candidate_ids><candidate_id>{accepted}</candidate_id></accepted_candidate_ids>",
        ),
        create_xml.replace("<kind>lore</kind>", "<kind>other</kind>"),
        create_xml.replace("<title>Garden Magic</title>", "<title>Garden [M999]</title>"),
    ]
    for response in invalid:
        with pytest.raises(ValueError):
            memorize_categories.parse_dynamic_category_review(response, bundle)


@pytest.mark.asyncio
async def test_dynamic_category_review_apply_is_atomic_and_collision_safe(tmp_path) -> None:
    store = _store(tmp_path)
    items = [_item(store, SCOPE, f"memory {index}") for index in range(2)]
    for item in items:
        store.dossier_candidate_repo.add_candidate(proposed_name="new topic", item_id=item.id, where=SCOPE)

    async def no_hits(_query, **_kwargs):
        return []

    bundle = (
        await memorize_categories.prepare_dynamic_category_review(
            store=store,
            where=SCOPE,
            cluster_size=2,
            search_dossiers=no_hits,
        )
    )[0]
    accepted, rejected = bundle["candidate_ids"]
    decision = {
        "cluster_id": bundle["cluster_id"],
        "action": "create",
        "accepted_candidate_ids": [accepted],
        "rejected_candidate_ids": [rejected],
        "existing_dossier_id": None,
        "name": "Created Topic",
        "description": "A created dossier",
        "kind": "topic",
    }
    with store._sessions.session() as session:
        result = memorize_categories.apply_dynamic_category_review(
            store=store,
            where=SCOPE,
            bundle=bundle,
            decision=decision,
            proposed_embedding=[1.0, 0.0],
            session=session,
        )
        session.rollback()
    assert result["status"] == "created"
    assert result["target_dossier"].approved_description is None
    assert all(category.name != "Created Topic" for category in store.memory_category_repo.list_categories(SCOPE).values())
    assert all(row.last_considered_at is None for row in store.dossier_candidate_repo.list_candidates(SCOPE))

    existing = _category(store, SCOPE, "Near Duplicate", kind="topic")
    store.memory_category_repo.update_category(category_id=existing.id, embedding=[1.0, 0.0])
    with store._sessions.session() as session:
        collision = memorize_categories.apply_dynamic_category_review(
            store=store,
            where=SCOPE,
            bundle=bundle,
            decision=decision,
            proposed_embedding=[1.0, 0.0],
            near_duplicate_threshold=0.9,
            session=session,
        )
        session.commit()
    assert collision["status"] == "collision"
    assert collision["target_dossier"].id == existing.id
    remaining = store.dossier_candidate_repo.list_candidates(SCOPE)
    assert len(remaining) == 2
    assert all(row.last_considered_at is not None for row in remaining)

    legacy_exact = _category(store, SCOPE, "Created Topic")
    with store._sessions.session() as session:
        exact_collision = memorize_categories.apply_dynamic_category_review(
            store=store,
            where=SCOPE,
            bundle=bundle,
            decision=decision,
            proposed_embedding=[0.0, 1.0],
            session=session,
        )
        session.commit()
    assert exact_collision["status"] == "collision"
    assert exact_collision["target_dossier"].id == legacy_exact.id

    bundle["existing_dossiers"] = [store.memory_category_repo.list_categories(SCOPE)[existing.id]]
    existing_decision = {
        "cluster_id": bundle["cluster_id"],
        "action": "existing",
        "accepted_candidate_ids": [accepted],
        "rejected_candidate_ids": [rejected],
        "existing_dossier_id": existing.id,
        "name": None,
        "description": None,
        "kind": None,
    }
    with store._sessions.session() as session:
        applied = memorize_categories.apply_dynamic_category_review(
            store=store,
            where=SCOPE,
            bundle=bundle,
            decision=existing_decision,
            session=session,
        )
        session.commit()
    assert applied["status"] == "existing"
    assert [row.id for row in store.dossier_candidate_repo.list_candidates(SCOPE)] == [rejected]
    accepted_item_id = next(
        candidate.item_id
        for memory in bundle["memories"]
        for candidate in memory["candidates"]
        if candidate.id == accepted
    )
    assert store.category_item_repo.get_item_categories(accepted_item_id)[0].category_id == existing.id
    with store._sessions.session() as session:
        replay = memorize_categories.apply_dynamic_category_review(
            store=store,
            where=SCOPE,
            bundle=bundle,
            decision=existing_decision,
            session=session,
        )
    assert replay["status"] == "existing"


@pytest.mark.asyncio
async def test_dynamic_category_review_defer_stamps_only_valid_decisions(tmp_path) -> None:
    store = _store(tmp_path)
    for index in range(2):
        item = _item(store, SCOPE, f"memory {index}")
        store.dossier_candidate_repo.add_candidate(
            proposed_name="deferred topic", item_id=item.id, where=SCOPE
        )

    async def no_hits(_query, **_kwargs):
        return []

    bundle = (
        await memorize_categories.prepare_dynamic_category_review(
            store=store,
            where=SCOPE,
            cluster_size=2,
            search_dossiers=no_hits,
        )
    )[0]
    invalid = {
        "cluster_id": bundle["cluster_id"],
        "action": "defer",
        "accepted_candidate_ids": [bundle["candidate_ids"][0]],
        "rejected_candidate_ids": [bundle["candidate_ids"][1]],
        "existing_dossier_id": None,
        "name": None,
        "description": None,
        "kind": None,
    }
    with store._sessions.session() as session:
        with pytest.raises(ValueError, match="cannot contain"):
            memorize_categories.apply_dynamic_category_review(
                store=store,
                where=SCOPE,
                bundle=bundle,
                decision=invalid,
                session=session,
            )
        session.rollback()
    assert all(
        row.last_considered_at is None
        for row in store.dossier_candidate_repo.list_candidates(SCOPE)
    )

    deferred = {
        **invalid,
        "accepted_candidate_ids": [],
        "rejected_candidate_ids": bundle["candidate_ids"],
    }
    with store._sessions.session() as session:
        result = memorize_categories.apply_dynamic_category_review(
            store=store,
            where=SCOPE,
            bundle=bundle,
            decision=deferred,
            session=session,
        )
        session.commit()
    assert result["status"] == "deferred"
    assert all(
        row.last_considered_at is not None
        for row in store.dossier_candidate_repo.list_candidates(SCOPE)
    )
    considered_at = [
        row.last_considered_at for row in store.dossier_candidate_repo.list_candidates(SCOPE)
    ]
    with store._sessions.session() as session:
        memorize_categories.apply_dynamic_category_review(
            store=store, where=SCOPE, bundle=bundle, decision=deferred, session=session
        )
        session.commit()
    assert [
        row.last_considered_at for row in store.dossier_candidate_repo.list_candidates(SCOPE)
    ] == considered_at
    assert await memorize_categories.prepare_dynamic_category_review(
        store=store,
        where=SCOPE,
        cluster_size=2,
        search_dossiers=no_hits,
    ) == []

    race = _store(tmp_path, "review-race.db")
    race_items = [_item(race, SCOPE, f"race {index}") for index in range(2)]
    for item in race_items:
        race.dossier_candidate_repo.add_candidate(
            proposed_name="race", item_id=item.id, where=SCOPE
        )
    race_bundle = (
        await memorize_categories.prepare_dynamic_category_review(
            store=race, where=SCOPE, cluster_size=2, search_dossiers=no_hits
        )
    )[0]
    race.memory_item_repo.update_item(item_id=race_items[0].id, summary="changed")
    race_decision = {
        "cluster_id": race_bundle["cluster_id"],
        "action": "defer",
        "accepted_candidate_ids": [],
        "rejected_candidate_ids": race_bundle["candidate_ids"],
    }
    with race._sessions.session() as session, pytest.raises(ValueError, match="context changed"):
        memorize_categories.apply_dynamic_category_review(
            store=race,
            where=SCOPE,
            bundle=race_bundle,
            decision=race_decision,
            session=session,
        )
