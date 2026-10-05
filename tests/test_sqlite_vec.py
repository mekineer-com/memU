from __future__ import annotations

import pytest

from memu.database.sqlite import session as session_module
from memu.database.sqlite.session import SQLiteSessionManager


def test_missing_sqlite_vec_fails_loudly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / session_module.SQLITE_VEC_EXTENSION_PATH.name
    monkeypatch.setattr(session_module, "SQLITE_VEC_EXTENSION_PATH", missing)
    manager = SQLiteSessionManager(dsn="sqlite:///:memory:")
    try:
        with pytest.raises(RuntimeError, match=r"python scripts/install-sqlite-vec\.py"):
            manager.engine.connect()
    finally:
        manager.close()
