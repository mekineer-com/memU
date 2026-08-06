from __future__ import annotations

import importlib.util
import json
import sqlite3
import struct
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/migrate-embeddings-to-blob.py"
    spec = importlib.util.spec_from_file_location("migrate_embeddings_to_blob", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_script()


def _create_db(path: Path, *, journal_mode: str = "DELETE") -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        for table in migration.TABLES:
            conn.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, embedding TEXT)")
            conn.execute(f"INSERT INTO {table} VALUES (?, ?)", (f"{table}-text", json.dumps([0.25, 0.75])))
        conn.commit()


def _storage(path: Path) -> dict[str, list[tuple[str, str, bytes | str]]]:
    with closing(sqlite3.connect(path)) as conn:
        return {
            table: conn.execute(f"SELECT id, typeof(embedding), embedding FROM {table} ORDER BY id").fetchall()
            for table in migration.TABLES
        }


def test_apply_converts_all_tables_and_preserves_existing_blob(tmp_path) -> None:
    path = tmp_path / "soul.db"
    _create_db(path)
    existing_blob = struct.pack("2f", 1.25, 1.75)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO memory_items VALUES ('already-blob', ?)", (existing_blob,))
        conn.commit()

    before = migration.scan_database(path)
    result = migration.apply_migration(path)
    after = migration.scan_database(path)
    storage = _storage(path)

    assert before["errors"] == []
    assert after["errors"] == []
    assert all(row[1] == "blob" for rows in storage.values() for row in rows)
    assert next(row[2] for row in storage["memory_items"] if row[0] == "already-blob") == existing_blob
    assert all(
        after["tables"][table]["fingerprint"] == before["tables"][table]["fingerprint"] for table in migration.TABLES
    )

    backup = Path(result["backup"])
    assert backup.is_file()
    backup_storage = _storage(backup)
    assert all(
        next(row[1] for row in backup_storage[table] if row[0] == f"{table}-text") == "text"
        for table in migration.TABLES
    )


def test_dry_run_does_not_write(tmp_path) -> None:
    path = tmp_path / "dry.db"
    _create_db(path)
    before = path.read_bytes()

    report = migration.scan_database(path)

    assert report["errors"] == []
    assert path.read_bytes() == before
    assert all(report["tables"][table]["storage"]["text"] == 1 for table in migration.TABLES)


@pytest.mark.parametrize(
    "invalid",
    [
        "not-json",
        json.dumps(["not-numeric"]),
        json.dumps([float("nan")]),
        json.dumps([1e39]),
        json.dumps([]),
        b"",
        b"bad",
    ],
)
def test_invalid_value_blocks_apply_before_backup(tmp_path, invalid) -> None:
    path = tmp_path / "invalid.db"
    _create_db(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("UPDATE resources SET embedding = ?", (invalid,))
        conn.commit()

    with pytest.raises(migration.MigrationError, match="validation failed"):
        migration.apply_migration(path)

    assert not (tmp_path / "_backups").exists()
    assert _storage(path)["memory_items"][0][1] == "text"


def test_conversion_failure_rolls_back_every_table(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rollback.db"
    _create_db(path)
    original = migration._convert_text_rows

    def fail_after_first_table(conn, table):
        original(conn, table)
        if table == "resources":
            raise RuntimeError("simulated conversion failure")

    monkeypatch.setattr(migration, "_convert_text_rows", fail_after_first_table)
    with pytest.raises(RuntimeError, match="simulated conversion failure"):
        migration.apply_migration(path)

    assert all(row[1] == "text" for rows in _storage(path).values() for row in rows)


def test_busy_wal_checkpoint_fails_before_backup(tmp_path) -> None:
    path = tmp_path / "busy.db"
    _create_db(path, journal_mode="WAL")
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM resources").fetchall()
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("INSERT INTO resources VALUES ('after-reader', '[1.0]')")
        writer.commit()

    try:
        with pytest.raises(migration.MigrationError, match="WAL checkpoint is busy"):
            migration.apply_migration(path)
    finally:
        reader.close()

    assert not (tmp_path / "_backups").exists()


def test_writer_cannot_commit_during_backup(tmp_path, monkeypatch) -> None:
    path = tmp_path / "locked.db"
    _create_db(path, journal_mode="WAL")
    original = migration._create_backup
    writer_blocked = False

    def backup_while_writer_tries(source_path, destination):
        nonlocal writer_blocked
        with closing(sqlite3.connect(source_path, timeout=0)) as writer:
            try:
                writer.execute("INSERT INTO resources VALUES ('racer', '[1.0]')")
            except sqlite3.OperationalError as exc:
                writer_blocked = "locked" in str(exc)
            else:
                writer.commit()
        return original(source_path, destination)

    monkeypatch.setattr(migration, "_create_backup", backup_while_writer_tries)
    migration.apply_migration(path)

    assert writer_blocked
    assert all(row[0] != "racer" for row in _storage(path)["resources"])


def test_backup_failure_leaves_no_final_file_or_source_mutation(tmp_path, monkeypatch) -> None:
    path = tmp_path / "backup-failure.db"
    _create_db(path)

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic rename failure")

    monkeypatch.setattr(migration.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic rename failure"):
        migration.apply_migration(path)

    backup_dir = tmp_path / "_backups"
    assert list(backup_dir.iterdir()) == []
    assert all(row[1] == "text" for rows in _storage(path).values() for row in rows)


def test_post_commit_verification_failure_reports_backup_and_stop(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "verify-failure.db"
    _create_db(path)

    def fail_verification(_path, _before):
        raise migration.MigrationError("simulated verification failure")

    monkeypatch.setattr(migration, "_verify_after", fail_verification)
    result = migration.main([str(path), "--apply"])
    error = capsys.readouterr().err

    assert result == 1
    assert "conversion committed; verification failed" in error
    assert "backup:" in error
    assert "do not restart services" in error
    assert all(row[1] == "blob" for rows in _storage(path).values() for row in rows)
