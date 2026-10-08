# Historical startup-upgrade test excerpts from tests/test_taxonomy_storage.py; reference only.
# Helpers/imports remain in the original file history; these are not active tests.

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
            "ix_category_items__category_scoped",
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
    with reopened._sessions.engine.connect() as conn:
        index_names = {row[1] for row in conn.exec_driver_sql("PRAGMA index_list(category_items)")}
        query_plan = conn.exec_driver_sql(
            "EXPLAIN QUERY PLAN SELECT * FROM category_items "
            "WHERE user_id = ? AND soul_id = ? AND category_id = ?",
            (SCOPE["user_id"], SCOPE["soul_id"], category.id),
        ).fetchall()
    assert "ix_category_items__category_scoped" in index_names
    assert "ix_category_items__category_scoped" in str(query_plan)
    reopened.close()

    reopened_again = _store(tmp_path, path.name)
    reopened_item = reopened_again.memory_item_repo.get_item(item.id, include_superseded=True)
    assert reopened_item is not None and reopened_item.memory_ref is None


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
