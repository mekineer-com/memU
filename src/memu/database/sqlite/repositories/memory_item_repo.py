"""SQLite memory item repository implementation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pendulum
from sqlmodel import delete, select

from memu.database.inmemory.vector import cosine_topk, cosine_topk_salience
from memu.database.models import MemoryItem, MemoryType, compute_content_hash
from memu.database.repositories.memory_item import MemoryItemRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState

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
        self.items = self._state.items

    @staticmethod
    def _resolve_conversation_id(conversation_id: str | None, user_data: Mapping[str, Any]) -> str | None:
        if isinstance(conversation_id, str) and conversation_id.strip():
            return conversation_id.strip()
        raw = user_data.get("conversation_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        legacy = user_data.get("session_id")
        if isinstance(legacy, str) and legacy.strip():
            return legacy.strip()
        return None

    @staticmethod
    def _active_item_filter(model: Any) -> Any | None:
        merged_into_col = getattr(model, "merged_into", None)
        if merged_into_col is None:
            return None
        from sqlalchemy import func, or_

        return or_(merged_into_col.is_(None), func.trim(merged_into_col) == "")

    def _to_memory_item(
        self,
        row: Any,
        *,
        embedding: list[float] | None = None,
        scope: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        item_scope = dict(scope) if scope is not None else self._scope_kwargs_from(row)
        item_scope.pop("conversation_id", None)
        return MemoryItem(
            id=row.id,
            resource_id=row.resource_id,
            memory_type=row.memory_type,
            summary=row.summary,
            embedding=embedding if embedding is not None else self._normalize_embedding(self._get_row_embedding(row)),
            happened_at=getattr(row, "happened_at", None),
            source_role=getattr(row, "source_role", None),
            confidence=getattr(row, "confidence", None),
            source_message_ids=getattr(row, "source_message_ids", None),
            reflection_salience=getattr(row, "reflection_salience", None),
            conversation_id=getattr(row, "conversation_id", None),
            affective_tags=getattr(row, "affective_tags", None),
            unresolved=getattr(row, "unresolved", None),
            merged_into=getattr(row, "merged_into", None),
            superseded_by=getattr(row, "superseded_by", None),
            extra=getattr(row, "extra", {}) or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
            **item_scope,
        )

    def get_item(self, item_id: str) -> MemoryItem | None:
        """Get a memory item by ID.

        Args:
            item_id: The item ID to look up.

        Returns:
            MemoryItem if found, None otherwise.
        """
        # Check cache first
        if item_id in self.items:
            return self.items[item_id]

        with self._sessions.session() as session:
            filters = [self._memory_item_model.id == item_id]
            active_filter = self._active_item_filter(self._memory_item_model)
            if active_filter is not None:
                filters.append(active_filter)
            stmt = select(self._memory_item_model).where(*filters)
            row = session.exec(stmt).first()

        if row is None:
            return None

        item = self._to_memory_item(row)
        self.items[row.id] = item
        return item

    def list_items(self, where: Mapping[str, Any] | None = None) -> dict[str, MemoryItem]:
        """List memory items matching the where clause.

        Args:
            where: Optional filter conditions.

        Returns:
            Dictionary of item ID to MemoryItem mapping.
        """
        with self._sessions.session() as session:
            stmt = select(self._memory_item_model)
            filters = self._build_filters(self._memory_item_model, where)
            active_filter = self._active_item_filter(self._memory_item_model)
            if active_filter is not None:
                filters.append(active_filter)
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()

        result: dict[str, MemoryItem] = {}
        for row in rows:
            item = self._to_memory_item(row)
            result[row.id] = item
            self.items[row.id] = item

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
            self.items[row.id] = item

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

            # Delete from database
            del_stmt = delete(self._memory_item_model)
            if filters:
                del_stmt = del_stmt.where(*filters)
            session.exec(del_stmt)
            session.commit()

            # Clean up cache
            for item_id in deleted:
                self.items.pop(item_id, None)

        return deleted

    def create_item(
        self,
        *,
        resource_id: str | None = None,
        memory_type: MemoryType,
        summary: str,
        embedding: list[float],
        user_data: dict[str, Any],
        reinforce: bool = False,
        tool_record: dict[str, Any] | None = None,
        source_role: str | None = None,
        confidence: float | None = None,
        source_message_ids: list[int] | None = None,
        reflection_salience: float | None = None,
        conversation_id: str | None = None,
        affective_tags: dict[str, Any] | None = None,
        unresolved: str | None = None,
        session: Any | None = None,
    ) -> MemoryItem:
        """Create a new memory item.

        Args:
            resource_id: Associated resource ID.
            memory_type: Type of memory.
            summary: Memory summary text.
            embedding: Embedding vector.
            user_data: User scope data.
            reinforce: If True, reinforce existing item instead of creating duplicate.
            tool_record: Tool-related fields (when_to_use, metadata, tool_calls) to store in extra.

        Returns:
            Created MemoryItem object.
        """
        if reinforce and memory_type != "tool":
            return self.create_item_reinforce(
                resource_id=resource_id,
                memory_type=memory_type,
                summary=summary,
                embedding=embedding,
                user_data=user_data,
                source_role=source_role,
                confidence=confidence,
                conversation_id=conversation_id,
                affective_tags=affective_tags,
                unresolved=unresolved,
                session=session,
            )

        if session is None:
            with self._sessions.session() as session:
                item = self.create_item(
                    resource_id=resource_id,
                    memory_type=memory_type,
                    summary=summary,
                    embedding=embedding,
                    user_data=user_data,
                    reinforce=reinforce,
                    tool_record=tool_record,
                    source_role=source_role,
                    confidence=confidence,
                    source_message_ids=source_message_ids,
                    reflection_salience=reflection_salience,
                    conversation_id=conversation_id,
                    affective_tags=affective_tags,
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
            confidence=confidence,
            source_message_ids=source_message_ids,
            reflection_salience=reflection_salience,
            conversation_id=conv_id,
            affective_tags=affective_tags,
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

        item = self._to_memory_item(row, embedding=embedding, scope=user_data)
        self.items[row.id] = item
        return item

    def create_item_reinforce(
        self,
        *,
        resource_id: str | None = None,
        memory_type: MemoryType,
        summary: str,
        embedding: list[float],
        user_data: dict[str, Any],
        source_role: str | None = None,
        confidence: float | None = None,
        source_message_ids: list[int] | None = None,
        reflection_salience: float | None = None,
        conversation_id: str | None = None,
        affective_tags: dict[str, Any] | None = None,
        unresolved: str | None = None,
        session: Any | None = None,
    ) -> MemoryItem:
        """Create or reinforce a memory item with deduplication.

        If an item with the same content hash exists in the same scope,
        reinforce it instead of creating a duplicate.

        Args:
            resource_id: Associated resource ID.
            memory_type: Type of memory.
            summary: Memory summary text.
            embedding: Embedding vector.
            user_data: User scope data.

        Returns:
            Created or reinforced MemoryItem object.
        """
        from sqlalchemy import func

        content_hash = compute_content_hash(summary, memory_type)
        conv_id = self._resolve_conversation_id(conversation_id, user_data)

        if session is None:
            with self._sessions.session() as session:
                item = self.create_item_reinforce(
                    resource_id=resource_id,
                    memory_type=memory_type,
                    summary=summary,
                    embedding=embedding,
                    user_data=user_data,
                    source_role=source_role,
                    confidence=confidence,
                    source_message_ids=source_message_ids,
                    reflection_salience=reflection_salience,
                    conversation_id=conversation_id,
                    affective_tags=affective_tags,
                    unresolved=unresolved,
                    session=session,
                )
                session.commit()
                return item

        # Check for existing item with same hash in same scope (deduplication)
        # Use json_extract(extra, '$.content_hash') for query
        content_hash_col = func.json_extract(self._memory_item_model.extra, "$.content_hash")
        filters = [content_hash_col == content_hash]
        filters.extend(self._build_filters(self._memory_item_model, user_data))
        active_filter = self._active_item_filter(self._memory_item_model)
        if active_filter is not None:
            filters.append(active_filter)

        existing = session.exec(select(self._memory_item_model).where(*filters)).first()

        if existing:
            # Reinforce existing memory instead of creating duplicate
            current_extra = existing.extra or {}
            current_count = current_extra.get("reinforcement_count", 1)
            existing.extra = {
                **current_extra,
                "reinforcement_count": current_count + 1,
                "last_reinforced_at": self._now().isoformat(),
            }
            if source_role is not None:
                existing.source_role = source_role
            if confidence is not None:
                existing.confidence = confidence
            if source_message_ids is not None:
                existing.source_message_ids = source_message_ids
            if reflection_salience is not None:
                existing.reflection_salience = reflection_salience
            if conv_id is not None:
                existing.conversation_id = conv_id
            if affective_tags is not None:
                existing.affective_tags = affective_tags
            if unresolved is not None:
                existing.unresolved = unresolved
            existing.updated_at = self._now()
            session.add(existing)
            session.flush()
            session.refresh(existing)
            item = self._to_memory_item(existing)
            self.items[existing.id] = item
            return item

        # Create new item with salience tracking in extra
        now = self._now()
        create_user_data = dict(user_data or {})
        create_user_data.pop("conversation_id", None)
        item_extra = create_user_data.pop("extra", {}) if "extra" in create_user_data else {}
        item_extra.update({
            "content_hash": content_hash,
            "reinforcement_count": 1,
            "last_reinforced_at": now.isoformat(),
        })

        row = self._memory_item_model(
            resource_id=resource_id,
            memory_type=memory_type,
            summary=summary,
            embedding=None,
            source_role=source_role,
            confidence=confidence,
            source_message_ids=source_message_ids,
            reflection_salience=reflection_salience,
            conversation_id=conv_id,
            affective_tags=affective_tags,
            unresolved=unresolved,
            extra=item_extra,
            created_at=now,
            updated_at=now,
            **create_user_data,
        )
        self._set_row_embedding(row, embedding)

        session.add(row)
        session.flush()
        session.refresh(row)

        item = self._to_memory_item(row, embedding=embedding)
        self.items[row.id] = item
        return item

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
        superseded_by: str | None = None,
        affective_tags: dict[str, Any] | None = None,
        unresolved: str | None = None,
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
        with self._sessions.session() as session:
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
            if superseded_by is not None:
                row.superseded_by = superseded_by
            if affective_tags is not None:
                row.affective_tags = affective_tags
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
            session.commit()
            session.refresh(row)

        item = self._to_memory_item(row)
        self.items[row.id] = item
        return item

    def delete_item(self, item_id: str) -> None:
        """Delete a memory item.

        Args:
            item_id: ID of item to delete.
        """
        with self._sessions.session() as session:
            stmt = select(self._memory_item_model).where(self._memory_item_model.id == item_id)
            row = session.exec(stmt).first()
            if row:
                session.delete(row)
                session.commit()

        if item_id in self.items:
            del self.items[item_id]

    def vector_search_items(
        self,
        query_vec: list[float],
        top_k: int,
        where: Mapping[str, Any] | None = None,
        *,
        ranking: str = "similarity",
        recency_decay_days: float = 30.0,
    ) -> list[tuple[str, float]]:
        """Perform vector similarity search on memory items.

        Uses brute-force cosine similarity since SQLite doesn't have native vector support.

        Args:
            query_vec: Query embedding vector.
            top_k: Maximum number of results to return.
            where: Optional filter conditions.
            ranking: Ranking strategy - "similarity" (default) or "salience".
            recency_decay_days: Half-life for recency decay in salience ranking.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        # Load items from database with filters
        pool = self.list_items(where)

        if ranking == "salience":
            # Salience-aware ranking: similarity x reinforcement x recency
            # Read values from extra dict
            corpus = [
                (
                    i.id,
                    i.embedding,
                    (i.extra or {}).get("reinforcement_count", 1),
                    self._parse_datetime((i.extra or {}).get("last_reinforced_at")),
                )
                for i in pool.values()
            ]
            return cosine_topk_salience(query_vec, corpus, k=top_k, recency_decay_days=recency_decay_days)

        # Default: pure cosine similarity (backward compatible)
        hits = cosine_topk(query_vec, [(i.id, i.embedding) for i in pool.values()], k=top_k)
        return hits

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
        """Load all existing items from database into cache."""
        self.list_items()


__all__ = ["SQLiteMemoryItemRepo"]
