from __future__ import annotations

import pytest

from memu.database.sqlite import session as session_module
from memu.database.sqlite.session import SQLiteSessionManager


def test_sqlite_vec_loaded_on_normal_connection() -> None:
    manager = SQLiteSessionManager(dsn="sqlite:///:memory:")
    try:
        with manager.engine.connect() as conn:
            assert conn.exec_driver_sql("SELECT vec_version()").scalar() == "v0.1.9"
            vector_type, length = conn.exec_driver_sql(
                "SELECT vec_type(vec_f32('[1,2]')), vec_length(vec_f32('[1,2]'))"
            ).one()
            assert (vector_type, length) == ("float32", 2)
            distance = conn.exec_driver_sql("SELECT vec_distance_cosine('[1, 0]', '[0.9, 0.1]')").scalar()
            assert float(distance) < 0.01
    finally:
        manager.close()


def test_missing_sqlite_vec_fails_loudly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing-vec0.so"
    monkeypatch.setattr(session_module, "SQLITE_VEC_EXTENSION_PATH", missing)
    manager = SQLiteSessionManager(dsn="sqlite:///:memory:")
    try:
        with pytest.raises(RuntimeError, match=r"scripts/build-sqlite-vec\.sh"):
            manager.engine.connect()
    finally:
        manager.close()
