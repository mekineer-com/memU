"""SQLite memory item repository implementation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pendulum
from sqlmodel import delete, select

from memu.database.models import MemoryItem, MemoryType
from memu.database.repositories.memory_item import MemoryItemRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState
from memu.database.vector import cosine_topk, reciprocal_rank_fusion, rerank_by_salience

logger = logging.getLogger(__name__)


class SQLiteMemoryItemRepo(SQLiteRepoBase, MemoryItemRepo):
    """SQLite implementation of memory item repository."""

    def __init__(
        self,
        *,
        state: DatabaseState,
        memory_item_model: type[Any],
        sqla_models: SQLiteSQLAModels,
        sessions: SQLiteSessionManager,
        scope_fields: list[str],
    ) -> None:
        """Initialize memory item repository.

        Args:
            state: Shared database state for caching.
            memory_item_model: SQLModel class for memory items.
            sqla_models: SQLAlchemy model container.
            sessions: Session manager for database connections.
            scope_fields: List of user scope field names.
        """
        super().__init__(
            state=state,
            sqla_models=sqla_models,
            sessions=sessions,
            scope_fields=scope_fields,
        )
        self._memory_item_model = memory_item_model

    @staticmethod
    def _resolve_conversation_id(conversation_id: str | None, user_data: Mapping[str, Any]) -> str | None:
        if isinstance(conversation_id, str) and conversation_id.strip():
            return conversation_id.strip()
        raw = user_data.get("conversation_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    def _active_item_filter(self, model: Any, *, include_superseded: bool = False) -> Any | None:
        merged_into_col = getattr(model, "merged_into", None)
        from sqlalchemy import and_, func, or_, select

        active_conditions: list[Any] = []
        if merged_into_col is not None:
            active_conditions.append(or_(merged_into_col.is_(None), func.trim(merged_into_col) == ""))
        triple_model = getattr(self._sqla_models, "Triple", None)
        if triple_model is not None and not include_superseded:
            evolved_filters: list[Any] = [
                triple_model.subject_id == model.id,
                triple_model.predicate == "evolved_into",
                triple_model.valid_to.is_(None),
            ]
            triple_subject_kind = getattr(triple_model, "subject_kind", None)
            if triple_subject_kind is not None:
                evolved_filters.append(triple_subject_kind == "memory")
            for field in self._scope_fields:
                triple_scope = getattr(triple_model, field, None)
                item_scope = getattr(model, field, None)
                if triple_scope is not None and item_scope is not None:
                    evolved_filters.append(triple_scope == item_scope)
            active_conditions.append(~select(1).where(*evolved_filters).exists())

        if not active_conditions:
            return None
        if len(active_conditions) == 1:
            return active_conditions[0]
        return and_(*active_conditions)

    def _to_memory_item(
        self,
        row: Any,
        *,
        embedding: list[float] | None = None,
    ) -> MemoryItem:
        return MemoryItem(
            id=row.id,
            resource_id=row.resource_id,
            memory_type=row.memory_type,
            summary=row.summary,
            embedding=embedding if embedding is not None else self._normalize_embedding(self._get_row_embedding(row)),
            happened_at=getattr(row, "happened_at", None),
            source_role=getattr(row, "source_role", None),
            speaker_id=getattr(row, "speaker_id", None),
            speaker_label=getattr(row, "speaker_label", None),
            confidence=getattr(row, "confidence", None),
            source_message_ids=getattr(row, "source_message_ids", None),
            reflection_salience=getattr(row, "reflection_salience", None),
            emotional_intensity=getattr(row, "emotional_intensity", None),
            conversation_id=getattr(row, "conversation_id", None),
            episode_id=getattr(row, "episode_id", None),
            unresolved=getattr(row, "unresolved", None),
            merged_into=getattr(row, "merged_into", None),
            extra=getattr(row, "extra", {}) or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def get_item(self, item_id: str, *, include_superseded: bool = False) -> MemoryItem | None:
        """Get a memory item by ID.

        Args:
            item_id: The item ID to look up.
            include_superseded: When True, return the item even if it has a live
                outbound ``evolved_into`` edge (i.e. it's been replaced).

        Returns:
            MemoryItem if found, None otherwise.
        """
        with self._sessions.session() as session:
            filters = [self._memory_item_model.id == item_id]
            active_filter = self._active_item_filter(
                self._memory_item_model, include_superseded=include_superseded
            )
            if active_filter is not None:
                filters.append(active_filter)
            stmt = select(self._memory_item_model).where(*filters)
            row = session.exec(stmt).first()

        if row is None:
            return None

        return self._to_memory_item(row)

    def list_items(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        include_superseded: bool = False,
    ) -> dict[str, MemoryItem]:
        """List memory items matching the where clause.

        Args:
            where: Optional filter conditions.
            include_superseded: When True, also include items that have a live
                outbound ``evolved_into`` edge (older versions).

        Returns:
            Dictionary of item ID to MemoryItem mapping.
        """
        with self._sessions.session() as session:
            stmt = select(self._memory_item_model)
            filters = self._build_filters(self._memory_item_model, where)
            active_filter = self._active_item_filter(
                self._memory_item_model, include_superseded=include_superseded
            )
            if active_filter is not None:
                filters.append(active_filter)
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()

        result: dict[str, MemoryItem] = {}
        for row in rows:
            item = self._to_memory_item(row)
            result[row.id] = item

        return result

    def list_items_by_ref_ids(
        self, ref_ids: list[str], where: Mapping[str, Any] | None = None
    ) -> dict[str, MemoryItem]:
        """List items by their ref_id in the extra column.

        Args:
            ref_ids: List of ref_ids to query.
            where: Additional filter conditions.

        Returns:
            Dict mapping item_id -> MemoryItem for items whose extra.ref_id is in ref_ids.
        """
        if not ref_ids:
            return {}

        from sqlalchemy import func

        with self._sessions.session() as session:
            stmt = select(self._memory_item_model)
            filters = self._build_filters(self._memory_item_model, where)
            active_filter = self._active_item_filter(self._memory_item_model)
            if active_filter is not None:
                filters.append(active_filter)
            # Add filter for json_extract(extra, '$.ref_id') IN ref_ids (only rows with ref_id key)
            ref_id_col = func.json_extract(self._memory_item_model.extra, "$.ref_id")
            filters.append(ref_id_col.isnot(None))
            filters.append(ref_id_col.in_(ref_ids))
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()

        result: dict[str, MemoryItem] = {}
        for row in rows:
            item = self._to_memory_item(row)
            result[row.id] = item

        return result

    def clear_items(self, where: Mapping[str, Any] | None = None) -> dict[str, MemoryItem]:
        """Clear items matching the where clause.

        Args:
            where: Optional filter conditions.

        Returns:
            Dictionary of deleted item ID to MemoryItem mapping.
        """
        filters = self._build_filters(self._memory_item_model, where)
        with self._sessions.session() as session:
            # First get the objects to delete
            stmt = select(self._memory_item_model)
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()

            deleted: dict[str, MemoryItem] = {}
            for row in rows:
                item = self._to_memory_item(row)
                deleted[row.id] = item

            if not deleted:
                return {}

            # Delete from FTS index
            conn = session.connection()
            for item_id in deleted:
                conn.exec_driver_sql("DELETE FROM memory_items_fts WHERE item_id = ?", (item_id,))

            # Delete from database
            del_stmt = delete(self._memory_item_model)
            if filters:
                del_stmt = del_stmt.where(*filters)
            session.exec(del_stmt)
            session.commit()

        return deleted

    def create_item(
        self,
        *,
        resource_id: str | None = None,
        memory_type: MemoryType,
        summary: str,
        embedding: list[float],
        user_data: dict[str, Any],
        tool_record: dict[str, Any] | None = None,
        source_role: str | None = None,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        confidence: float | None = None,
        source_message_ids: list[int] | None = None,
        happened_at: datetime | None = None,
        reflection_salience: float | None = None,
        emotional_intensity: float | None = None,
        conversation_id: str | None = None,
        episode_id: str | None = None,
        unresolved: str | None = None,
        session: Any | None = None,
    ) -> MemoryItem:
        if session is None:
            with self._sessions.session() as session:
                item = self.create_item(
                    resource_id=resource_id,
                    memory_type=memory_type,
                    summary=summary,
                    embedding=embedding,
                    user_data=user_data,
                    tool_record=tool_record,
                    source_role=source_role,
                    speaker_id=speaker_id,
                    speaker_label=speaker_label,
                    confidence=confidence,
                    source_message_ids=source_message_ids,
                    happened_at=happened_at,
                    reflection_salience=reflection_salience,
                    emotional_intensity=emotional_intensity,
                    conversation_id=conversation_id,
                    episode_id=episode_id,
                    unresolved=unresolved,
                    session=session,
                )
                session.commit()
                return item

        # Build extra dict with tool_record fields at top level
        extra: dict[str, Any] = {}
        if tool_record:
            if tool_record.get("when_to_use") is not None:
                extra["when_to_use"] = tool_record["when_to_use"]
            if tool_record.get("metadata") is not None:
                extra["metadata"] = tool_record["metadata"]
            if tool_record.get("tool_calls") is not None:
                extra["tool_calls"] = tool_record["tool_calls"]

        create_user_data = dict(user_data or {})
        create_user_data.pop("conversation_id", None)
        conv_id = self._resolve_conversation_id(conversation_id, user_data)
        now = self._now()
        row = self._memory_item_model(
            resource_id=resource_id,
            memory_type=memory_type,
            summary=summary,
            embedding=None,
            source_role=source_role,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            confidence=confidence,
            source_message_ids=source_message_ids,
            happened_at=happened_at,
            reflection_salience=reflection_salience,
            emotional_intensity=emotional_intensity,
            conversation_id=conv_id,
            episode_id=episode_id,
            unresolved=unresolved,
            extra=extra if extra else {},
            created_at=now,
            updated_at=now,
            **create_user_data,
        )
        self._set_row_embedding(row, embedding)
        session.add(row)
        session.flush()
        session.refresh(row)
        self._fts_upsert(session, row.id, summary, memory_type)

        return self._to_memory_item(row, embedding=embedding)

    def update_item(
        self,
        *,
        item_id: str,
        memory_type: MemoryType | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        extra: dict[str, Any] | None = None,
        tool_record: dict[str, Any] | None = None,
        merged_into: str | None = None,
        unresolved: str | None = None,
        session: Any | None = None,
    ) -> MemoryItem:
        """Update an existing memory item.

        Args:
            item_id: ID of item to update.
            memory_type: New memory type (optional).
            summary: New summary text (optional).
            embedding: New embedding vector (optional).
            extra: Extra data to merge into existing extra dict (optional).
            tool_record: Tool-related fields (when_to_use, metadata, tool_calls) to merge into extra.

        Returns:
            Updated MemoryItem object.

        Raises:
            KeyError: If item not found.
        """
        if session is None:
            with self._sessions.session() as managed_session:
                item = self.update_item(
                    item_id=item_id,
                    memory_type=memory_type,
                    summary=summary,
                    embedding=embedding,
                    extra=extra,
                    tool_record=tool_record,
                    merged_into=merged_into,
                    unresolved=unresolved,
                    session=managed_session,
                )
                managed_session.commit()
                return item

        stmt = select(self._memory_item_model).where(self._memory_item_model.id == item_id)
        row = session.exec(stmt).first()

        if row is None:
            msg = f"Item with id {item_id} not found"
            raise KeyError(msg)

        if memory_type is not None:
            row.memory_type = memory_type
        if summary is not None:
            row.summary = summary
        if embedding is not None:
            self._set_row_embedding(row, embedding)
        if merged_into is not None:
            row.merged_into = merged_into
        if unresolved is not None:
            row.unresolved = unresolved

        # Merge extra and tool_record into existing extra dict
        current_extra = row.extra or {}
        if extra is not None:
            current_extra = {**current_extra, **extra}
        if tool_record is not None:
            # Merge tool_record fields at top level
            for key in ("when_to_use", "metadata", "tool_calls"):
                if tool_record.get(key) is not None:
                    current_extra[key] = tool_record[key]
        if extra is not None or tool_record is not None:
            row.extra = current_extra

        row.updated_at = self._now()
        session.add(row)
        session.flush()
        session.refresh(row)

        # Sync FTS: remove if item became inactive, otherwise upsert
        if merged_into:
            self._fts_delete(session, item_id)
        elif summary is not None:
            self._fts_upsert(session, item_id, row.summary, row.memory_type)

        return self._to_memory_item(row)

    def delete_item(self, item_id: str) -> None:
        """Delete a memory item.

        Args:
            item_id: ID of item to delete.
        """
        with self._sessions.session() as session:
            stmt = select(self._memory_item_model).where(self._memory_item_model.id == item_id)
            row = session.exec(stmt).first()
            if row:
                self._fts_delete(session, item_id)
                session.delete(row)
                session.commit()

    # ── FTS5 helpers ──────────────────────────────────────────────────

    def _fts_upsert(self, session: Any, item_id: str, summary: str, memory_type: str) -> None:
        """Insert or replace an item in the FTS5 index."""
        conn = session.connection()
        conn.exec_driver_sql("DELETE FROM memory_items_fts WHERE item_id = ?", (item_id,))
        conn.exec_driver_sql(
            "INSERT INTO memory_items_fts(summary, memory_type, item_id) VALUES (?, ?, ?)",
            (summary, memory_type, item_id),
        )

    def _fts_delete(self, session: Any, item_id: str) -> None:
        """Remove an item from the FTS5 index."""
        conn = session.connection()
        conn.exec_driver_sql("DELETE FROM memory_items_fts WHERE item_id = ?", (item_id,))

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape a user query for FTS5 MATCH safety.

        Wraps each word in double quotes so FTS5 special characters
        (*, -, OR, AND, NEAR, etc.) are treated as literals.
        """
        words = query.split()
        if not words:
            return ""
        escaped_words = (w.replace('"', '""') for w in words)
        return " ".join(f'"{w}"' for w in escaped_words)

    def fts_search_items(
        self,
        query: str,
        top_k: int,
        pool_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text BM25 search on memory item summaries.

        Returns (item_id, score) tuples with scores negated so higher = better
        (SQLite FTS5 rank is negative, lower = better match).
        Results are filtered to pool_ids if provided (scope filtering).
        """
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        with self._sessions.session() as session:
            conn = session.connection()
            rows = conn.exec_driver_sql(
                "SELECT item_id, rank FROM memory_items_fts "
                "WHERE memory_items_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (safe_query, top_k * 3 if pool_ids else top_k),
            ).fetchall()

        results: list[tuple[str, float]] = []
        for item_id, rank in rows:
            if pool_ids is not None and item_id not in pool_ids:
                continue
            results.append((item_id, -rank))  # negate: higher = better
            if len(results) >= top_k:
                break
        return results

    # ── Vector + hybrid search ─────────────────────────────────────

    def vector_search_items(
        self,
        query_vec: list[float],
        top_k: int,
        where: Mapping[str, Any] | None = None,
        *,
        ranking: str = "similarity",
        recency_decay_days: float = 30.0,
        fts_query: str | None = None,
        fts_enabled: bool = False,
        fts_top_k: int = 20,
        rrf_k: int = 60,
        include_superseded: bool = False,
    ) -> list[tuple[str, float]]:
        """Vector similarity search with optional FTS5 hybrid fusion.

        When fts_enabled and fts_query are provided, runs BM25 keyword search
        alongside cosine similarity and merges results via Reciprocal Rank Fusion
        before applying salience reranking.
        """
        pool = self.list_items(where, include_superseded=include_superseded)

        # Expand candidate pool when doing hybrid search
        vector_k = max(top_k, fts_top_k) if fts_enabled else top_k
        vector_hits = cosine_topk(query_vec, [(i.id, i.embedding) for i in pool.values()], k=vector_k)

        # Hybrid: fuse vector + FTS via RRF
        if fts_enabled and fts_query:
            fts_hits = self.fts_search_items(fts_query, fts_top_k, pool_ids=set(pool.keys()))
            # FTS returned nothing (stop-words only, etc.) — fall back to vector
            hits = reciprocal_rank_fusion(vector_hits, fts_hits, k=rrf_k) if fts_hits else vector_hits
        else:
            hits = vector_hits

        if ranking == "salience":
            candidate_ids = [item_id for item_id, _ in hits[:vector_k]]
            if not candidate_ids:
                return []

            with self._sessions.session() as session:
                stmt = select(
                    self._memory_item_model.id,
                    self._memory_item_model.reflection_salience,
                    self._memory_item_model.emotional_intensity,
                    self._memory_item_model.created_at,
                ).where(self._memory_item_model.id.in_(candidate_ids))
                active_filter = self._active_item_filter(self._memory_item_model)
                if active_filter is not None:
                    stmt = stmt.where(active_filter)
                rows = session.exec(stmt).all()

            item_meta: dict[str, tuple[float | None, float | None, datetime]] = {
                item_id: (
                    float(sal) if sal is not None else None,
                    float(emo) if emo is not None else None,
                    cat,
                )
                for item_id, sal, emo, cat in rows
            }

            candidates: list[tuple[str, float, datetime, float | None, float | None]] = []
            for item_id, score in hits[:vector_k]:
                sal, emo, cat = item_meta.get(item_id, (None, None, datetime.min))
                candidates.append((item_id, score, cat, sal, emo))

            return rerank_by_salience(candidates, recency_decay_days=recency_decay_days)[:top_k]

        return hits[:top_k]

    @staticmethod
    def _parse_datetime(dt_str: str | None) -> pendulum.DateTime | None:
        """Parse ISO datetime string from extra dict."""
        if dt_str is None:
            return None
        try:
            parsed = pendulum.parse(dt_str)
        except (ValueError, TypeError):
            return None
        else:
            if isinstance(parsed, pendulum.DateTime):
                return parsed
            return None

    def load_existing(self) -> None:
        """No-op: SQLite repo does not keep an in-memory item cache."""
        return None


__all__ = ["SQLiteMemoryItemRepo"]
