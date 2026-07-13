"""Base repository class for SQLite backend."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from typing import Any

import pendulum

from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState


class SQLiteRepoBase:
    """Base class for SQLite repository implementations."""

    def __init__(
        self,
        *,
        state: DatabaseState,
        sqla_models: Any,
        sessions: SQLiteSessionManager,
        scope_fields: list[str],
    ) -> None:
        self._state = state
        self._sqla_models = sqla_models
        self._sessions = sessions
        self._scope_fields = scope_fields

    def _normalize_embedding(self, embedding: Any) -> list[float] | None:
        """Normalize canonical BLOBs and transitional legacy values."""
        if embedding is None:
            return None
        if isinstance(embedding, (bytes, bytearray, memoryview)):
            blob = bytes(embedding)
            if len(blob) % 4:
                msg = f"Malformed embedding BLOB length: {len(blob)} bytes"
                raise ValueError(msg)
            values = list(struct.unpack(f"{len(blob) // 4}f", blob))
        else:
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError as exc:
                    msg = "Malformed legacy JSON embedding"
                    raise ValueError(msg) from exc
                if not isinstance(embedding, list):
                    msg = "Legacy JSON embedding must be a list"
                    raise TypeError(msg)
            try:
                values = [float(x) for x in embedding]
            except (ValueError, TypeError, OverflowError) as exc:
                msg = "Embedding must contain only numeric values"
                raise ValueError(msg) from exc
        if not all(math.isfinite(value) for value in values):
            msg = "Embedding must contain only finite values"
            raise ValueError(msg)
        return values

    def _get_row_embedding(self, row: Any) -> Any:
        """Read embedding from the canonical embedding column."""
        return getattr(row, "embedding", None)

    def _set_row_embedding(self, row: Any, embedding: list[float] | None) -> None:
        """Write embedding to the canonical embedding column."""
        prepared = self._prepare_embedding(embedding)
        row.embedding = prepared

    def _prepare_embedding(self, embedding: list[float] | None) -> bytes | None:
        """Serialize an embedding to the canonical native float32 BLOB."""
        if embedding is None:
            return None
        values = self._normalize_embedding(embedding) or []
        if not values:
            msg = "Embedding vector cannot be empty"
            raise ValueError(msg)
        blob = struct.pack(f"{len(values)}f", *values)
        if not all(math.isfinite(value) for value in struct.unpack(f"{len(values)}f", blob)):
            msg = "Embedding cannot be represented as finite float32"
            raise ValueError(msg)
        return blob

    def _now(self) -> pendulum.DateTime:
        """Get current UTC time."""
        return pendulum.now("UTC")

    def _build_filters(self, model: Any, where: Mapping[str, Any] | None) -> list[Any]:
        """Build SQLAlchemy filter expressions from where clause."""
        if not where:
            return []
        filters: list[Any] = []
        for raw_key, expected in where.items():
            if expected is None:
                continue
            field, op = [*raw_key.split("__", 1), None][:2]
            column = getattr(model, str(field), None)
            if column is None:
                msg = f"Unknown filter field '{field}' for model '{model.__name__}'"
                raise ValueError(msg)
            if op == "in":
                if isinstance(expected, str):
                    filters.append(column == expected)
                else:
                    filters.append(column.in_(expected))
            else:
                filters.append(column == expected)
        return filters


__all__ = ["SQLiteRepoBase"]
