from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from memu.app.service import MemoryService
from memu.database.sqlite.sqlite import SQLiteStore


class SpeakerPhase1Scope(BaseModel):
    user_id: str | None = None


def test_create_item_round_trips_speaker_fields() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": SpeakerPhase1Scope},
    )
    store = service._get_database()
    item = store.memory_item_repo.create_item(
        memory_type="event",
        summary="phase1 speaker fields roundtrip",
        embedding=[0.2, 0.4, 0.6],
        user_data={"user_id": "speaker_phase1"},
        source_role="user",
        speaker_id="user:speaker_phase1",
        speaker_label="Speaker Phase1",
    )

    loaded = store.memory_item_repo.get_item(item.id)
    assert loaded is not None
    assert loaded.speaker_id == "user:speaker_phase1"
    assert loaded.speaker_label == "Speaker Phase1"


def test_sqlite_store_backfills_speaker_columns_for_legacy_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-speaker-phase1.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
CREATE TABLE memu_memory_items (
    id TEXT PRIMARY KEY,
    created_at DATETIME,
    updated_at DATETIME,
    user_id TEXT,
    memory_type TEXT,
    summary TEXT
)
"""
    )
    con.commit()
    con.close()

    store = SQLiteStore(dsn=f"sqlite:///{db_path}", scope_model=SpeakerPhase1Scope)
    store.close()

    con = sqlite3.connect(db_path)
    cols = [row[1] for row in con.execute("PRAGMA table_info(memu_memory_items)").fetchall()]
    con.close()
    assert "speaker_id" in cols
    assert "speaker_label" in cols
