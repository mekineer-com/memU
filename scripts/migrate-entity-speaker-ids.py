#!/usr/bin/env python3
"""Migrate legacy entity speaker slugs to stable entity IDs in a stopped memU DB.

If post-commit verification fails, restore the printed backup before restarting services.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memu.database.models import normalize_entity_name


class MigrationError(RuntimeError):
    pass


def _legacy_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _aliases(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        properties = raw if isinstance(raw, dict) else json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid entity properties JSON: {exc}") from exc
    aliases = properties.get("aliases", []) if isinstance(properties, dict) else []
    return [str(value) for value in aliases] if isinstance(aliases, list) else []


def _scan_connection(conn: sqlite3.Connection, database: Path) -> dict[str, Any]:
    for table in ("entities", "memory_items"):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
            raise MigrationError(f"missing table: {table}")

    entities = conn.execute(
        "SELECT id, name, normalized, properties, user_id, soul_id FROM entities"
    ).fetchall()
    memories = conn.execute(
        """
SELECT id, speaker_id, user_id, soul_id
FROM memory_items
WHERE speaker_id LIKE 'entity:%'
ORDER BY id
"""
    ).fetchall()

    entity_ids: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    candidates: dict[tuple[Any, Any, str], set[str]] = defaultdict(set)
    for entity_id, name, normalized, properties, user_id, soul_id in entities:
        scope = (user_id, soul_id)
        entity_id = str(entity_id)
        entity_ids[scope].add(entity_id)
        values = [str(name or ""), str(normalized or ""), *_aliases(properties)]
        for value in values:
            for key in {value.strip().lower(), normalize_entity_name(value), _legacy_slug(value)} - {""}:
                candidates[(*scope, key)].add(entity_id)

    rewrites: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    already_current = 0
    for memory_id, speaker_id, user_id, soul_id in memories:
        old_speaker_id = str(speaker_id)
        tail = old_speaker_id.removeprefix("entity:")
        scope = (user_id, soul_id)
        if tail in entity_ids[scope]:
            already_current += 1
            continue
        matches = sorted(candidates.get((*scope, tail.lower()), set()))
        if len(matches) != 1:
            errors.append(
                {
                    "memory_id": str(memory_id),
                    "speaker_id": old_speaker_id,
                    "matches": matches,
                }
            )
            continue
        rewrites.append(
            {
                "memory_id": str(memory_id),
                "old_speaker_id": old_speaker_id,
                "new_speaker_id": f"entity:{matches[0]}",
            }
        )

    return {
        "database": str(database),
        "entity_speaker_rows": len(memories),
        "already_current": already_current,
        "rewrites": rewrites,
        "errors": errors,
    }


def scan_database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MigrationError(f"database does not exist: {path}")
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return _scan_connection(conn, path)


def _create_backup(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = destination / f"{path.stem}-pre-entity-speaker-ids-{stamp}{path.suffix or '.db'}"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{backup_path.name}.", suffix=".tmp", dir=destination)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(temporary_path)) as target:
            source.backup(target)
        with closing(sqlite3.connect(temporary_path)) as backup:
            if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise MigrationError("backup integrity check failed")
        os.replace(temporary_path, backup_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return backup_path


def apply_migration(path: Path, backup_dir: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise MigrationError(f"database does not exist: {path}")

    conn = sqlite3.connect(path, timeout=0)
    backup_path: Path | None = None
    try:
        conn.execute("PRAGMA busy_timeout=0")
        busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise MigrationError("WAL checkpoint is busy; stop every process using this database")
        conn.execute("BEGIN IMMEDIATE")
        try:
            before = _scan_connection(conn, path)
            if before["errors"]:
                raise MigrationError("unresolved speaker references; no changes made")
            backup_path = _create_backup(path, backup_dir or path.parent / "_backups")
            conn.executemany(
                "UPDATE memory_items SET speaker_id = ? WHERE id = ? AND speaker_id = ?",
                [
                    (row["new_speaker_id"], row["memory_id"], row["old_speaker_id"])
                    for row in before["rewrites"]
                ],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    after = scan_database(path)
    if (
        after["errors"]
        or after["rewrites"]
        or after["entity_speaker_rows"] != before["entity_speaker_rows"]
    ):
        raise MigrationError(
            f"migration committed but verification failed; backup: {backup_path}; do not restart services"
        )
    return {"backup": str(backup_path), "before": before, "after": after}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--apply", action="store_true", help="back up and migrate the stopped database")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = apply_migration(args.database, args.backup_dir) if args.apply else scan_database(args.database)
    except (MigrationError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(result.get("errors")))


if __name__ == "__main__":
    raise SystemExit(main())
