#!/usr/bin/env python3
# ruff: noqa: S608, TRY003
"""Inspect or migrate memU SQLite embeddings to canonical float32 BLOBs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TABLES = ("resources", "memory_items", "categories")
SQLITE_VEC_VERSION = "v0.1.9"
SQLITE_VEC_PATH = Path(__file__).resolve().parents[1] / "src/memu/database/sqlite/vec0.so"


class MigrationError(RuntimeError):
    pass


def _canonical_blob(value: Any, storage_type: str) -> bytes:
    if storage_type == "text":
        try:
            values = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("malformed JSON") from exc
        if not isinstance(values, list):
            raise TypeError("JSON embedding is not a list")
        try:
            values = [float(item) for item in values]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding contains a non-numeric value") from exc
        if not values:
            raise ValueError("zero-dimensional embedding")
        if not all(math.isfinite(item) for item in values):
            raise ValueError("embedding contains a non-finite value")
        try:
            blob = struct.pack(f"{len(values)}f", *values)
        except (OverflowError, struct.error) as exc:
            raise ValueError("embedding cannot be represented as float32") from exc
        if not all(math.isfinite(item) for item in struct.unpack(f"{len(values)}f", blob)):
            raise ValueError("embedding cannot be represented as finite float32")
        return blob

    if storage_type == "blob":
        blob = bytes(value)
        if not blob:
            raise ValueError("zero-dimensional embedding")
        if len(blob) % 4:
            raise ValueError(f"malformed BLOB length: {len(blob)} bytes")
        values = struct.unpack(f"{len(blob) // 4}f", blob)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("embedding contains a non-finite value")
        return blob

    raise TypeError(f"unsupported SQLite storage type: {storage_type}")


def _fingerprint(rows: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for row_id, blob in sorted(rows):
        encoded_id = row_id.encode("utf-8")
        digest.update(struct.pack("!I", len(encoded_id)))
        digest.update(encoded_id)
        digest.update(blob)
    return digest.hexdigest()


def _open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def scan_database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MigrationError(f"database does not exist: {path}")

    report: dict[str, Any] = {"database": str(path), "tables": {}, "errors": []}
    with closing(_open_read_only(path)) as conn:
        for table in TABLES:
            try:
                total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                rows = conn.execute(
                    f"SELECT id, embedding, typeof(embedding) FROM {table} WHERE embedding IS NOT NULL ORDER BY id"
                ).fetchall()
            except sqlite3.Error as exc:
                raise MigrationError(f"cannot inspect {table}: {exc}") from exc

            storage = {"text": 0, "blob": 0, "other": 0}
            dimensions: dict[int, int] = {}
            canonical_rows: list[tuple[str, bytes]] = []
            for raw_id, value, storage_type in rows:
                bucket = storage_type if storage_type in {"text", "blob"} else "other"
                storage[bucket] += 1
                try:
                    blob = _canonical_blob(value, storage_type)
                except (TypeError, ValueError, struct.error) as exc:
                    report["errors"].append(f"{table}:{raw_id}: {exc}")
                    continue
                dimension = len(blob) // 4
                dimensions[dimension] = dimensions.get(dimension, 0) + 1
                canonical_rows.append((str(raw_id), blob))

            report["tables"][table] = {
                "total": total,
                "non_null": len(rows),
                "storage": storage,
                "dimensions": dimensions,
                "fingerprint": _fingerprint(canonical_rows),
            }
    return report


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    if not SQLITE_VEC_PATH.is_file():
        raise MigrationError(f"sqlite-vec extension not found at {SQLITE_VEC_PATH}")
    conn.enable_load_extension(True)
    try:
        conn.load_extension(str(SQLITE_VEC_PATH))
    finally:
        conn.enable_load_extension(False)
    version = conn.execute("SELECT vec_version()").fetchone()[0]
    if version != SQLITE_VEC_VERSION:
        raise MigrationError(f"sqlite-vec version mismatch: expected {SQLITE_VEC_VERSION}, got {version}")


def _convert_text_rows(conn: sqlite3.Connection, table: str) -> None:
    rows = conn.execute(
        f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL AND typeof(embedding) = 'text'"
    ).fetchall()
    for row_id, value in rows:
        conn.execute(f"UPDATE {table} SET embedding = ? WHERE id = ?", (_canonical_blob(value, "text"), row_id))


def _create_backup(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = destination / f"{path.stem}-pre-sqlite-vec-{stamp}{path.suffix or '.db'}"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{backup_path.name}.", suffix=".tmp", dir=destination)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(temporary_path)) as target:
            source.backup(target)
        os.replace(temporary_path, backup_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return backup_path


def _verify_after(path: Path, before: dict[str, Any]) -> dict[str, Any]:
    after = scan_database(path)
    if after["errors"]:
        raise MigrationError("post-migration validation failed: " + "; ".join(after["errors"]))

    with closing(_open_read_only(path)) as conn:
        _load_sqlite_vec(conn)
        for table in TABLES:
            previous = before["tables"][table]
            current = after["tables"][table]
            if current["total"] != previous["total"] or current["non_null"] != previous["non_null"]:
                raise MigrationError(f"row counts changed in {table}")
            if current["storage"] != {"text": 0, "blob": current["non_null"], "other": 0}:
                raise MigrationError(f"non-BLOB embeddings remain in {table}: {current['storage']}")
            if current["dimensions"] != previous["dimensions"]:
                raise MigrationError(f"embedding dimensions changed in {table}")
            if current["fingerprint"] != previous["fingerprint"]:
                raise MigrationError(f"embedding fingerprint changed in {table}")
            invalid_type_count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL AND vec_type(embedding) IS NOT 'float32'"
            ).fetchone()[0]
            if invalid_type_count:
                raise MigrationError(f"sqlite-vec rejected {invalid_type_count} embeddings in {table}")
    return after


def apply_migration(path: Path, backup_dir: Path | None = None) -> dict[str, Any]:
    with closing(sqlite3.connect(":memory:")) as extension_check:
        _load_sqlite_vec(extension_check)

    before = scan_database(path)
    if before["errors"]:
        raise MigrationError("validation failed: " + "; ".join(before["errors"]))

    conn = sqlite3.connect(path, timeout=0)
    try:
        conn.execute("PRAGMA busy_timeout=0")
        busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise MigrationError("WAL checkpoint is busy; stop every process using this database")
        conn.execute("BEGIN IMMEDIATE")
        try:
            backup_path = _create_backup(path, backup_dir or path.parent / "_backups")
            for table in TABLES:
                _convert_text_rows(conn, table)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    try:
        after = _verify_after(path, before)
    except Exception as exc:
        raise MigrationError(
            f"conversion committed; verification failed: {exc}; backup: {backup_path}; do not restart services"
        ) from exc
    return {"backup": str(backup_path), "before": before, "after": after}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--apply", action="store_true", help="back up and convert the database")
    parser.add_argument("--backup-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = apply_migration(args.database, args.backup_dir) if args.apply else scan_database(args.database)
    except (MigrationError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.apply and result["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
