from __future__ import annotations

import importlib.util
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/migrate-entity-speaker-ids.py"
    spec = importlib.util.spec_from_file_location("migrate_entity_speaker_ids", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_script()


def _create_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
CREATE TABLE entities (
  id TEXT PRIMARY KEY, name TEXT, normalized TEXT, properties JSON,
  user_id TEXT, soul_id TEXT
);
CREATE TABLE memory_items (
  id TEXT PRIMARY KEY, speaker_id TEXT, user_id TEXT, soul_id TEXT
);
"""
        )
        conn.executemany(
            "INSERT INTO entities VALUES (?, ?, ?, ?, 'test-user', 'test-soul')",
            [
                ("a1b2c3d4", "Rowan O'Neil", "rowan_o'neil", json.dumps({"aliases": ["Ro"]})),
                ("b2c3d4e5", "Taylor", "taylor", "{}"),
                ("c3d4e5f6", "Taylor", "taylor", "{}"),
            ],
        )
        conn.executemany(
            "INSERT INTO memory_items VALUES (?, ?, 'test-user', 'test-soul')",
            [
                ("m1", "entity:rowan_o_neil"),
                ("m2", "entity:ro"),
                ("m3", "entity:a1b2c3d4"),
            ],
        )
        conn.commit()


def test_dry_run_maps_legacy_names_and_aliases_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "test-soul.db"
    _create_db(path)

    report = migration.scan_database(path)

    assert report["errors"] == []
    assert report["already_current"] == 1
    assert {row["memory_id"] for row in report["rewrites"]} == {"m1", "m2"}
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT speaker_id FROM memory_items WHERE id='m1'").fetchone()[0] == "entity:rowan_o_neil"


def test_ambiguous_name_blocks_apply_before_backup(tmp_path: Path) -> None:
    path = tmp_path / "test-soul.db"
    _create_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO memory_items VALUES ('m4', 'entity:taylor', 'test-user', 'test-soul')")
        conn.commit()

    with pytest.raises(migration.MigrationError, match="unresolved speaker references"):
        migration.apply_migration(path)

    assert not (tmp_path / "_backups").exists()


def test_apply_backs_up_and_rewrites_atomically(tmp_path: Path) -> None:
    path = tmp_path / "test-soul.db"
    _create_db(path)

    result = migration.apply_migration(path)

    assert Path(result["backup"]).is_file()
    assert result["after"]["rewrites"] == []
    assert result["after"]["already_current"] == 3
    with closing(sqlite3.connect(path)) as conn:
        speaker_ids = dict(conn.execute("SELECT id, speaker_id FROM memory_items"))
    assert set(speaker_ids.values()) == {"entity:a1b2c3d4"}
