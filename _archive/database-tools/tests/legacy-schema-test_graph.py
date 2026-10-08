# Historical startup-upgrade test excerpts from tests/test_graph.py; reference only.
# Helpers/imports remain in the original file history; these are not active tests.

def test_sqlite_approval_backfill_runs_only_when_column_is_added(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
CREATE TABLE memory_items (
  id TEXT PRIMARY KEY,
  created_at DATETIME,
  updated_at DATETIME,
  resource_id TEXT,
  memory_type TEXT NOT NULL,
  summary TEXT NOT NULL,
  embedding BLOB,
  happened_at DATETIME,
  source_role TEXT,
  speaker_id TEXT,
  speaker_label TEXT,
  confidence FLOAT,
  source_message_ids JSON,
  reflection_salience FLOAT,
  emotional_intensity FLOAT,
  conversation_id TEXT,
  segment_id TEXT,
  unresolved TEXT,
  merged_into TEXT,
  extra JSON,
  user_id TEXT,
  soul_id TEXT
);
INSERT INTO memory_items (id, created_at, updated_at, memory_type, summary, embedding, extra, user_id, soul_id)
VALUES ('old', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'episode', 'old approved', X'CDCCCC3D', '{}', 'backfill', 's');
"""
    )
    conn.close()
    dsn = f"sqlite:///{db_path}"
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": dsn}},
        user_config={"model": GraphScope},
    )
    scope = {"user_id": "backfill", "soul_id": "s"}
    old = service._get_database().memory_item_repo.get_item("old")
    assert old.approved_at is not None

    new = service._get_database().memory_item_repo.create_item(
        memory_type="episode",
        summary="new pending",
        embedding=[0.2],
        user_data=scope,
    )
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": dsn}},
        user_config={"model": GraphScope},
    )
    pending_ids = {node["memory_id"] for node in service.graph_list_pending(where=scope)["items"]}
    assert pending_ids == {new.id}
