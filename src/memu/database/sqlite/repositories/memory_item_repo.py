"""SQLite memory item repository implementation."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import case, func, or_, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import defer
from sqlmodel import delete, select

from memu.database.models import MemoryItem, MemoryType
from memu.database.repositories.memory_item import MemoryItemRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState
from memu.database.vector import reciprocal_rank_fusion, rerank_by_salience

logger = logging.getLogger(__name__)

_SCORE_FIELDS: tuple[str, ...] = ("confidence", "reflection_salience", "emotional_intensity")
_MISSING_CALIBRATION_WARNED: set[tuple[str, str]] = set()


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

    @staticmethod
    def _next_recalibration_threshold(sample_size: int) -> int:
        if sample_size < 50:
            return 50
        if sample_size < 100:
            return 100
        if sample_size < 250:
            return 250
        if sample_size < 500:
            return 500
        return ((sample_size // 500) + 1) * 500

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        idx = max(0, min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1)))))
        return sorted_values[idx]

    def _build_percentile_params(self, values: list[float]) -> dict[str, float]:
        sorted_values = sorted(values)
        return {
            "p10": self._percentile(sorted_values, 0.10),
            "p25": self._percentile(sorted_values, 0.25),
            "p50": self._percentile(sorted_values, 0.50),
            "p75": self._percentile(sorted_values, 0.75),
            "p90": self._percentile(sorted_values, 0.90),
        }

    def refresh_model_score_calibration(self, *, model: str, session: Any | None = None) -> dict[str, int]:
        normalized_model = model.strip()
        if not normalized_model:
            msg = "refresh_model_score_calibration requires non-empty model"
            raise ValueError(msg)

        if session is None:
            with self._sessions.session() as managed_session:
                refreshed = self.refresh_model_score_calibration(model=normalized_model, session=managed_session)
                managed_session.commit()
                return refreshed

        from sqlalchemy import func

        refreshed_samples: dict[str, int] = {}
        conn = session.connection()
        for field in _SCORE_FIELDS:
            score_col = getattr(self._memory_item_model, field)
            model_col = func.json_extract(self._memory_item_model.extra, "$.model")
            values_stmt = (
                select(score_col)
                .where(model_col == normalized_model)
                .where(score_col.isnot(None))
            )
            values = [float(v) for v in session.exec(values_stmt).all() if v is not None]
            if not values:
                continue
            sample_size = len(values)
            existing = conn.exec_driver_sql(
                "SELECT sample_size FROM model_score_calibration WHERE model = ? AND field = ? AND version = 1",
                (normalized_model, field),
            ).fetchone()
            if existing is not None:
                previous_sample_size = int(existing[0] or 0)
                threshold = self._next_recalibration_threshold(previous_sample_size)
                if sample_size < threshold:
                    refreshed_samples[field] = previous_sample_size
                    continue
            params = self._build_percentile_params(values)
            conn.exec_driver_sql(
                """
INSERT INTO model_score_calibration(model, field, version, params_json, sample_size, updated_at)
VALUES (?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
ON CONFLICT(model, field, version) DO UPDATE SET
  params_json = excluded.params_json,
  sample_size = excluded.sample_size,
  updated_at = excluded.updated_at
""",
                (normalized_model, field, json.dumps(params), sample_size),
            )
            refreshed_samples[field] = sample_size
        return refreshed_samples

    def _load_score_calibration_map(self, *, models: set[str]) -> dict[tuple[str, str], dict[str, float]]:
        if not models:
            return {}
        with self._sessions.session() as session:
            conn = session.connection()
            placeholders = ",".join("?" for _ in models)
            rows = conn.exec_driver_sql(
                f"""
SELECT model, field, params_json
FROM model_score_calibration
WHERE version = 1 AND model IN ({placeholders})
""",
                tuple(sorted(models)),
            ).fetchall()
        calibration_map: dict[tuple[str, str], dict[str, float]] = {}
        for model, field, params_json in rows:
            try:
                parsed = json.loads(str(params_json))
            except json.JSONDecodeError:
                logger.warning("model_score_calibration has invalid json for model=%s field=%s", model, field)
                continue
            if not isinstance(parsed, dict):
                logger.warning("model_score_calibration has non-dict params for model=%s field=%s", model, field)
                continue
            try:
                calibration_map[(str(model), str(field))] = {
                    "p10": float(parsed["p10"]),
                    "p25": float(parsed["p25"]),
                    "p50": float(parsed["p50"]),
                    "p75": float(parsed["p75"]),
                    "p90": float(parsed["p90"]),
                }
            except (KeyError, TypeError, ValueError):
                logger.warning("model_score_calibration missing percentile keys for model=%s field=%s", model, field)
        return calibration_map

    @staticmethod
    def _warn_missing_calibration_once(model: str, field: str) -> None:
        key = (model, field)
        if key in _MISSING_CALIBRATION_WARNED:
            return
        _MISSING_CALIBRATION_WARNED.add(key)
        logger.warning("No score calibration found for model=%s field=%s; using identity mapping", model, field)

    def _active_item_filter(
        self,
        model: Any,
        *,
        include_superseded: bool = False,
        include_merged: bool = False,
    ) -> Any | None:
        merged_into_col = getattr(model, "merged_into", None)
        from sqlalchemy import and_, func, or_, select

        active_conditions: list[Any] = []
        if merged_into_col is not None and not include_merged:
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
            memory_ref=getattr(row, "memory_ref", None),
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
            segment_id=getattr(row, "segment_id", None),
            unresolved=getattr(row, "unresolved", None),
            merged_into=getattr(row, "merged_into", None),
            approved_at=getattr(row, "approved_at", None),
            extra=getattr(row, "extra", {}) or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def allocate_memory_ref(self, where: Mapping[str, Any], session: Any | None = None) -> int:
        scope = self._require_scope(where)
        if session is None:
            with self._sessions.session() as managed_session:
                memory_ref = self.allocate_memory_ref(scope, session=managed_session)
                managed_session.commit()
                return memory_ref

        model = self._sqla_models.MemoryRefCounter
        now = self._now()
        statement = sqlite_insert(model).values(
            counter_key="memory",
            next_value=2,
            created_at=now,
            updated_at=now,
            **scope,
        )
        upsert = statement.on_conflict_do_update(
            index_elements=[*[getattr(model, field) for field in self._scope_fields], model.counter_key],
            set_={"next_value": model.next_value + 1, "updated_at": now},
        ).returning(model.next_value)
        return int(session.execute(upsert).scalar_one()) - 1

    def get_item_by_memory_ref(self, memory_ref: int, where: Mapping[str, Any]) -> MemoryItem | None:
        scope = self._require_scope(where)
        with self._sessions.session() as session:
            row = session.exec(
                select(self._memory_item_model).where(
                    self._memory_item_model.memory_ref == memory_ref,
                    *self._build_filters(self._memory_item_model, scope),
                )
            ).first()
        return None if row is None else self._to_memory_item(row)

    def _set_memory_ref_counter(self, session: Any, scope: Mapping[str, Any], next_value: int) -> None:
        model = self._sqla_models.MemoryRefCounter
        row = session.exec(
            select(model).where(
                model.counter_key == "memory",
                *self._build_filters(model, scope),
            )
        ).first()
        if row is None:
            now = self._now()
            session.add(
                model(
                    counter_key="memory",
                    next_value=next_value,
                    created_at=now,
                    updated_at=now,
                    **scope,
                )
            )
        elif row.next_value < next_value:
            row.next_value = next_value
            row.updated_at = self._now()
            session.add(row)

    def backfill_memory_refs(self, where: Mapping[str, Any]) -> dict[str, int]:
        scope = self._require_scope(where)
        with self._sessions.session() as session:
            try:
                session.execute(text("BEGIN IMMEDIATE"))
                rows = session.exec(
                    select(self._memory_item_model)
                    .where(*self._build_filters(self._memory_item_model, scope))
                    .order_by(self._memory_item_model.created_at, self._memory_item_model.id)
                ).all()
                populated = [row for row in rows if row.memory_ref is not None]
                if populated and len(populated) != len(rows):
                    raise RuntimeError("Memory reference backfill refused mixed assigned/unassigned state")

                if not populated:
                    for memory_ref, row in enumerate(rows, start=1):
                        row.memory_ref = memory_ref
                        session.add(row)
                else:
                    refs = [int(row.memory_ref) for row in rows]
                    if len(refs) != len(set(refs)):
                        raise RuntimeError("Memory reference backfill found duplicate references")

                result = {row.id: int(row.memory_ref) for row in rows if row.memory_ref is not None}
                if result:
                    self._set_memory_ref_counter(session, scope, max(result.values()) + 1)
                session.commit()
                return result
            except Exception:
                session.rollback()
                raise

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
        include_embeddings: bool = True,
    ) -> dict[str, MemoryItem]:
        """List memory items matching the where clause.

        Args:
            where: Optional filter conditions.
            include_superseded: When True, also include items that have a live
                outbound ``evolved_into`` edge (older versions).
            include_embeddings: When False, omit embedding BLOBs from the query.

        Returns:
            Dictionary of item ID to MemoryItem mapping.
        """
        with self._sessions.session() as session:
            stmt = select(self._memory_item_model)
            if not include_embeddings:
                stmt = stmt.options(defer(self._memory_item_model.embedding))
            filters = self._build_filters(self._memory_item_model, where)
            active_filter = self._active_item_filter(
                self._memory_item_model, include_superseded=include_superseded
            )
            if active_filter is not None:
                filters.append(active_filter)
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()

        return {
            row.id: self._to_memory_item(row, embedding=None if include_embeddings else [])
            for row in rows
        }

    def _list_graph_items(
        self,
        stmt: Any,
        where: Mapping[str, Any] | None,
        include_superseded: bool,
        *,
        include_merged: bool = False,
        include_embeddings: bool = False,
        session: Any | None = None,
    ) -> dict[str, MemoryItem]:
        filters = self._build_filters(self._memory_item_model, where)
        active_filter = self._active_item_filter(
            self._memory_item_model,
            include_superseded=include_superseded,
            include_merged=include_merged,
        )
        if active_filter is not None:
            filters.append(active_filter)
        if filters:
            stmt = stmt.where(*filters)
        if session is None:
            with self._sessions.session() as managed_session:
                rows = managed_session.exec(stmt).all()
        else:
            rows = session.exec(stmt).all()
        return {row.id: self._to_memory_item(row, embedding=None if include_embeddings else []) for row in rows}

    def list_recent_items(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        limit: int,
        include_superseded: bool = False,
    ) -> dict[str, MemoryItem]:
        when = func.coalesce(self._memory_item_model.happened_at, self._memory_item_model.created_at)
        stmt = (
            select(self._memory_item_model)
            .order_by(when.desc(), self._memory_item_model.id.desc())
            .limit(max(1, int(limit)))
        )
        return self._list_graph_items(stmt, where, include_superseded)

    def list_canvas_items(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        limit: int,
    ) -> tuple[dict[str, MemoryItem], int]:
        filters = self._build_filters(self._memory_item_model, where)
        active_filter = self._active_item_filter(self._memory_item_model, include_superseded=False)
        if active_filter is not None:
            filters.append(active_filter)
        filters.append(self._memory_item_model.embedding.is_not(None))
        stmt = (
            select(self._memory_item_model)
            .where(*filters)
            .order_by(self._memory_item_model.updated_at.desc().nulls_last(), self._memory_item_model.id.desc())
            .limit(max(1, int(limit)))
        )
        count_stmt = select(func.count(self._memory_item_model.id)).where(*filters)
        with self._sessions.session() as session:
            rows = session.exec(stmt).all()
            total = int(session.exec(count_stmt).one())
        return {row.id: self._to_memory_item(row) for row in rows}, total

    def list_items_by_ids(
        self,
        item_ids: set[str],
        where: Mapping[str, Any] | None = None,
        *,
        include_superseded: bool = False,
        include_merged: bool = False,
        include_embeddings: bool = False,
        session: Any | None = None,
    ) -> dict[str, MemoryItem]:
        if not item_ids:
            return {}
        stmt = select(self._memory_item_model).where(self._memory_item_model.id.in_(item_ids))
        return self._list_graph_items(
            stmt,
            where,
            include_superseded,
            include_merged=include_merged,
            include_embeddings=include_embeddings,
            session=session,
        )

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

    def approve_item(self, item_id: str, where: Mapping[str, Any] | None = None) -> MemoryItem:
        with self._sessions.session() as session:
            filters = [self._memory_item_model.id == item_id, *self._build_filters(self._memory_item_model, where)]
            active_filter = self._active_item_filter(self._memory_item_model, include_superseded=False)
            if active_filter is not None:
                filters.append(active_filter)
            row = session.exec(select(self._memory_item_model).where(*filters)).first()
            if row is None:
                msg = f"Item with id {item_id} not found"
                raise KeyError(msg)
            if row.approved_at is None:
                row.approved_at = self._now()
                session.add(row)
                session.commit()
                session.refresh(row)
            return self._to_memory_item(row)

    def hard_delete_item(self, item_id: str, where: Mapping[str, Any] | None = None) -> MemoryItem:
        category_item_model = self._sqla_models.CategoryItem
        triple_model = self._sqla_models.Triple
        with self._sessions.session() as session:
            filters = [self._memory_item_model.id == item_id, *self._build_filters(self._memory_item_model, where)]
            active_filter = self._active_item_filter(self._memory_item_model, include_superseded=False)
            if active_filter is not None:
                filters.append(active_filter)
            row = session.exec(select(self._memory_item_model).where(*filters)).first()
            if row is None:
                msg = f"Item with id {item_id} not found"
                raise KeyError(msg)
            deleted = self._to_memory_item(row)
            scope_filters = self._build_filters(category_item_model, where)
            session.exec(delete(category_item_model).where(category_item_model.item_id == item_id, *scope_filters))
            triple_scope_filters = self._build_filters(triple_model, where)
            session.exec(
                delete(triple_model).where(
                    or_(
                        (triple_model.subject_kind == "memory") & (triple_model.subject_id == item_id),
                        (triple_model.object_kind == "memory") & (triple_model.object_id == item_id),
                        triple_model.source_memory_id == item_id,
                    ),
                    *triple_scope_filters,
                )
            )
            conn = session.connection()
            conn.exec_driver_sql("DELETE FROM memory_item_edit_history WHERE memory_item_id = ?", (item_id,))
            self._fts_delete(session, item_id)
            session.delete(row)
            session.commit()

        self._state.relations[:] = [rel for rel in self._state.relations if rel.item_id != item_id]
        return deleted

    def create_item(
        self,
        *,
        resource_id: str | None = None,
        memory_type: MemoryType,
        summary: str,
        embedding: list[float],
        user_data: dict[str, Any],
        extra: dict[str, Any] | None = None,
        source_role: str | None = None,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        confidence: float | None = None,
        source_message_ids: list[int] | None = None,
        happened_at: datetime | None = None,
        reflection_salience: float | None = None,
        emotional_intensity: float | None = None,
        conversation_id: str | None = None,
        segment_id: str | None = None,
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
                    extra=extra,
                    source_role=source_role,
                    speaker_id=speaker_id,
                    speaker_label=speaker_label,
                    confidence=confidence,
                    source_message_ids=source_message_ids,
                    happened_at=happened_at,
                    reflection_salience=reflection_salience,
                    emotional_intensity=emotional_intensity,
                    conversation_id=conversation_id,
                    segment_id=segment_id,
                    unresolved=unresolved,
                    session=session,
                )
                session.commit()
                return item

        merged_extra: dict[str, Any] = dict(extra or {})
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
            segment_id=segment_id,
            unresolved=unresolved,
            extra=merged_extra if merged_extra else {},
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

        current_extra = row.extra or {}
        if extra is not None:
            current_extra = {**current_extra, **extra}
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

    def update_summary_with_history(
        self,
        *,
        item_id: str,
        summary: str,
        embedding: list[float],
        where: Mapping[str, Any] | None = None,
        edited_by: str | None = None,
        approved: bool = False,
    ) -> MemoryItem:
        """Insert edit history and update the item in one transaction."""
        with self._sessions.session() as session:
            filters = [self._memory_item_model.id == item_id, *self._build_filters(self._memory_item_model, where)]
            active_filter = self._active_item_filter(self._memory_item_model, include_superseded=False)
            if active_filter is not None:
                filters.append(active_filter)
            row = session.exec(select(self._memory_item_model).where(*filters)).first()
            if row is None:
                msg = f"Item with id {item_id} not found"
                raise KeyError(msg)

            conn = session.connection()
            conn.exec_driver_sql(
                """
INSERT INTO memory_item_edit_history
    (id, memory_item_id, summary_before, summary_after, embedding_before, edited_by, scope_json)
VALUES (?, ?, ?, ?, ?, ?, ?)
""",
                (
                    str(uuid.uuid4()),
                    item_id,
                    row.summary,
                    summary,
                    self._get_row_embedding(row),
                    edited_by,
                    json.dumps(dict(where or {}), sort_keys=True),
                ),
            )
            row.summary = summary
            self._set_row_embedding(row, embedding)
            # Human edit (approved=True): "Save + approve" stamps a pending item once;
            # an already-approved item just saves. Soul edits require re-approval.
            if approved:
                if row.approved_at is None:
                    row.approved_at = self._now()
            else:
                row.approved_at = None
            row.updated_at = self._now()
            session.add(row)
            session.flush()
            session.refresh(row)
            self._fts_upsert(session, item_id, row.summary, row.memory_type)
            item = self._to_memory_item(row)
            session.commit()
            return item

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
        vector_k = max(top_k, fts_top_k) if fts_enabled else top_k
        query_blob = self._prepare_embedding(query_vec)
        query_dimension = len(query_vec)
        embedding = self._memory_item_model.embedding
        filters = self._build_filters(self._memory_item_model, where)
        active_filter = self._active_item_filter(
            self._memory_item_model, include_superseded=include_superseded
        )
        if active_filter is not None:
            filters.append(active_filter)

        with self._sessions.session() as session:
            dimensions = dict(
                session.exec(
                    select(func.vec_length(embedding), func.count())
                    .where(embedding.is_not(None), *filters)
                    .group_by(func.vec_length(embedding))
                ).all()
            )
            mismatched = sorted(dim for dim in dimensions if dim != query_dimension)
            if mismatched:
                logger.error(
                    "cosine_topk: skipped %d vector(s) with mismatched dimension "
                    "(expected %d, found dims: %s)",
                    sum(dimensions[dim] for dim in mismatched),
                    query_dimension,
                    mismatched,
                )

            distance = case(
                (func.vec_length(embedding) == query_dimension, func.vec_distance_cosine(embedding, query_blob)),
                else_=None,
            ).label("distance")
            rows = session.exec(
                select(self._memory_item_model.id, distance)
                .where(embedding.is_not(None), *filters, distance.is_not(None))
                .order_by(distance.asc(), self._memory_item_model.id.asc())
                .limit(vector_k)
            ).all()
            vector_hits = [(item_id, 1.0 - float(item_distance)) for item_id, item_distance in rows]

        # Hybrid: fuse vector + FTS via RRF
        if fts_enabled and fts_query:
            with self._sessions.session() as session:
                pool_ids = set(
                    session.exec(select(self._memory_item_model.id).where(*filters)).all()
                )
            fts_hits = self.fts_search_items(fts_query, fts_top_k, pool_ids=pool_ids)
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
                    self._memory_item_model.extra,
                ).where(self._memory_item_model.id.in_(candidate_ids))
                active_filter = self._active_item_filter(
                    self._memory_item_model, include_superseded=include_superseded
                )
                if active_filter is not None:
                    stmt = stmt.where(active_filter)
                rows = session.exec(stmt).all()

            item_meta: dict[str, tuple[float | None, float | None, datetime, str | None]] = {
                item_id: (
                    float(sal) if sal is not None else None,
                    float(emo) if emo is not None else None,
                    cat,
                    str((extra or {}).get("model", "")).strip() or None,
                )
                for item_id, sal, emo, cat, extra in rows
            }
            models = {m for _, _, _, m in item_meta.values() if m}
            calibration_map = self._load_score_calibration_map(models=models)
            for model in sorted(models):
                for field in ("reflection_salience", "emotional_intensity"):
                    if (model, field) not in calibration_map:
                        self._warn_missing_calibration_once(model, field)

            candidates: list[tuple[str, float, datetime, float | None, float | None, str | None]] = []
            for item_id, score in hits[:vector_k]:
                sal, emo, cat, model = item_meta.get(item_id, (None, None, datetime.min, None))
                candidates.append((item_id, score, cat, sal, emo, model))

            return rerank_by_salience(
                candidates,
                recency_decay_days=recency_decay_days,
                calibrations=calibration_map,
            )[:top_k]

        return hits[:top_k]


__all__ = ["SQLiteMemoryItemRepo"]
