"""SQLite session manager for database connections.

This module is intentionally opinionated for "it just works" on-disk SQLite:
  - Creates parent directories for file-backed DBs.
  - Applies pragmatic PRAGMAs for better multi-request reliability.

MemU may be used in a web server (FastAPI) where multiple requests can write
concurrently. SQLite can handle this, but defaults are easy to trip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, create_engine

logger = logging.getLogger(__name__)

SQLITE_VEC_VERSION = "v0.1.9"
SQLITE_VEC_EXTENSION_PATH = Path(__file__).with_name("vec0.so")


class SQLiteSessionManager:
    """Handle engine lifecycle and session creation for SQLite store."""

    def __init__(self, *, dsn: str, engine_kwargs: dict[str, Any] | None = None) -> None:
        """Initialize SQLite session manager.

        Args:
            dsn: SQLite connection string (e.g., "sqlite:///path/to/db.sqlite").
            engine_kwargs: Optional keyword arguments for create_engine.
        """
        self._ensure_sqlite_file_parent(dsn)

        kw: dict[str, Any] = {
            # Allow use across threads (FastAPI, background workers, etc.)
            "connect_args": {
                "check_same_thread": False,
                # Wait instead of immediately throwing "database is locked".
                # (seconds; sqlite3 default is 5.0)
                "timeout": 30,
            },
            # Helps recover from stale pooled connections.
            "pool_pre_ping": True,
        }
        if engine_kwargs:
            kw.update(engine_kwargs)

        self._engine = create_engine(dsn, **kw)
        self._apply_pragmas()
        self.embedding_profile: str | None = None

    def guard_embedding_write(self, session: Session, dimensions: int) -> None:
        profile = self.embedding_profile
        if profile is None:
            return
        expected_dimensions = int(profile.rsplit(":", 1)[1])
        if dimensions != expected_dimensions:
            raise ValueError(
                f"embedding dimension mismatch for {profile}: expected {expected_dimensions}, got {dimensions}"
            )
        row = session.connection().exec_driver_sql(
            "SELECT profile FROM embedding_profile WHERE id = 1"
        ).first()
        if row is None:
            session.connection().exec_driver_sql(
                "INSERT INTO embedding_profile (id, profile) VALUES (1, ?)", (profile,)
            )
        elif row[0] != profile:
            raise RuntimeError(
                f"embedding profile mismatch: database={row[0]} configured={profile}"
            )

    @staticmethod
    def _sqlite_file_from_dsn(dsn: str) -> Path | None:
        """Return a filesystem path for file-backed sqlite DSNs.

        Returns None for :memory: and unknown formats.
        """
        u = None
        try:
            u = make_url(dsn)
        except Exception:
            u = None
        if u and str(u.drivername or "").startswith("sqlite"):
            db = u.database
            if not db or db == ":memory:":
                return None
            return Path(db)

        # Fallback string parse
        base = (dsn or "").split("?", 1)[0]
        if ":memory:" in base:
            return None
        if base.startswith("sqlite:////"):
            return Path("/" + base[len("sqlite:////") :])
        if base.startswith("sqlite:///"):
            return Path(base[len("sqlite:///") :])
        return None

    @classmethod
    def _ensure_sqlite_file_parent(cls, dsn: str) -> None:
        """Create parent dirs for file-backed sqlite DBs.

        Without this, sqlite often fails with "unable to open database file".
        """
        try:
            p = cls._sqlite_file_from_dsn(dsn)
            if p is None:
                return
            p = p.expanduser()
            if not p.is_absolute():
                # Resolve relative paths against CWD (sqlite behavior).
                p = (Path.cwd() / p).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            # Touch ensures the file is visible immediately.
            p.touch(exist_ok=True)
        except Exception:
            # Best-effort; memU will raise a clearer error later if open fails.
            return

    def _apply_pragmas(self) -> None:
        """Apply PRAGMAs for better reliability under concurrent access."""

        @event.listens_for(self._engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn: Any, _conn_record: Any) -> None:
            build_command = "scripts/build-sqlite-vec.sh"
            if not SQLITE_VEC_EXTENSION_PATH.is_file():
                msg = f"sqlite-vec extension not found at {SQLITE_VEC_EXTENSION_PATH}; run {build_command}"
                raise RuntimeError(msg)
            dbapi_conn.enable_load_extension(True)
            try:
                try:
                    dbapi_conn.load_extension(str(SQLITE_VEC_EXTENSION_PATH))
                except Exception as exc:
                    msg = f"failed to load sqlite-vec extension at {SQLITE_VEC_EXTENSION_PATH}; run {build_command}"
                    raise RuntimeError(msg) from exc
            finally:
                dbapi_conn.enable_load_extension(False)
            version = dbapi_conn.execute("SELECT vec_version()").fetchone()[0]
            if version != SQLITE_VEC_VERSION:
                msg = (
                    f"sqlite-vec version mismatch at {SQLITE_VEC_EXTENSION_PATH}: "
                    f"expected {SQLITE_VEC_VERSION}, got {version}; run {build_command}"
                )
                raise RuntimeError(msg)

            try:
                cur = dbapi_conn.cursor()
                # WAL => readers don't block writers (and vice-versa) as much.
                cur.execute("PRAGMA journal_mode=WAL")
                # Reasonable tradeoff for speed vs durability.
                cur.execute("PRAGMA synchronous=NORMAL")
                # Wait for locks instead of failing fast.
                cur.execute("PRAGMA busy_timeout=30000")
                # Basic integrity.
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
            except Exception:
                # Don't crash just because pragmas aren't supported.
                return

    def session(self) -> Session:
        """Create a new database session."""
        return Session(self._engine, expire_on_commit=False)

    def close(self) -> None:
        """Close the database engine and release resources."""
        try:
            self._engine.dispose()
        except SQLAlchemyError:
            logger.exception("Failed to close SQLite engine")

    @property
    def engine(self) -> Any:
        """Return the underlying SQLAlchemy engine."""
        return self._engine


__all__ = ["SQLiteSessionManager"]
