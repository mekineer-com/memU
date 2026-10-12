from __future__ import annotations

import sqlite3

import pytest
from pydantic import BaseModel
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError

from memu.database.sqlite import session as session_module
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.sqlite.sqlite import SQLiteStore


@pytest.mark.parametrize("operation", ["mkdir", "touch"])
def test_sqlite_file_setup_preserves_filesystem_error(tmp_path, monkeypatch, operation) -> None:
    failure = PermissionError("fictional storage is unavailable")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(type(tmp_path), operation, fail)
    with pytest.raises(PermissionError) as caught:
        SQLiteSessionManager(dsn=f"sqlite:///{tmp_path / 'unavailable.db'}")
    assert caught.value is failure


def test_missing_sqlite_vec_fails_loudly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / session_module.SQLITE_VEC_EXTENSION_PATH.name
    monkeypatch.setattr(session_module, "SQLITE_VEC_EXTENSION_PATH", missing)
    manager = SQLiteSessionManager(dsn="sqlite:///:memory:")
    try:
        with pytest.raises(RuntimeError, match=r"python scripts/install-sqlite-vec\.py"):
            manager.engine.connect()
    finally:
        manager.close()


def test_pragma_initialization_failure_refuses_connection() -> None:
    def connect():
        conn = sqlite3.connect(":memory:")
        conn.set_authorizer(
            lambda action, name, _value, _db, _source: sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA and name == "foreign_keys" else sqlite3.SQLITE_OK
        )
        return conn

    manager = SQLiteSessionManager(dsn="sqlite:///:memory:", engine_kwargs={"creator": connect})
    try:
        with pytest.raises(DatabaseError, match="not authorized"):
            with manager.engine.connect():
                pytest.fail("connection accepted failed integrity setup")
    finally:
        manager.close()


def test_fts_initialization_failure_refuses_store() -> None:
    class FtsScope(BaseModel):
        user_id: str | None = None

    def deny_fts(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().startswith("CREATE VIRTUAL TABLE"):
            raise sqlite3.OperationalError("fictional FTS creation failure")

    event.listen(Engine, "before_cursor_execute", deny_fts)
    try:
        with pytest.raises(sqlite3.OperationalError, match="fictional FTS creation failure"):
            SQLiteStore(dsn="sqlite:///:memory:", scope_model=FtsScope)
    finally:
        event.remove(Engine, "before_cursor_execute", deny_fts)
