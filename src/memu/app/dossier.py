from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, cast

from memu.database.models import DossierKind, MemoryCategory, MemoryItem
from memu.database.vector import cosine_topk

if TYPE_CHECKING:
    from memu.app.settings import MemorizeConfig
    from memu.database.interfaces import Database

DOSSIER_INDEX_LIMIT = 20
DOSSIER_KINDS: tuple[DossierKind, ...] = ("lore", "topic", "goal")
MEMORY_REF_PATTERN = re.compile(r"^\[M([1-9][0-9]*)\]$")
AnchorRole = Literal["soul", "user"]


def _identity_text(name: Any, description: Any) -> str:
    name = str(name or "").strip() or "Untitled"
    description = str(description or "").strip()
    return f"{name}: {description}" if description else name


def dossier_identity_text(category: Any) -> str:
    return _identity_text(getattr(category, "name", ""), getattr(category, "description", ""))


def _scope(where: Mapping[str, Any] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in ("user_id", "soul_id"):
        value = str((where or {}).get(field) or "").strip()
        if not value:
            raise ValueError(f"Complete dossier scope required; missing: {field}")
        values[field] = value
    return values


def _activity_key(category: MemoryCategory) -> tuple[int, float, str, str]:
    happened = category.last_evidence_at
    if happened is None:
        return (1, 0.0, category.name.casefold(), category.id)
    if happened.tzinfo is None:
        happened = happened.replace(tzinfo=UTC)
    return (0, -happened.timestamp(), category.name.casefold(), category.id)


def _embedding_vector(values: Sequence[float], *, label: str) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} embedding is invalid") from exc
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} embedding must contain finite values")
    return vector


def _content_text(category: MemoryCategory) -> str:
    identity = dossier_identity_text(category)
    summary = str(category.summary or "").strip()
    return f"{identity}\n{summary}" if summary else identity


def _render_index_field(value: str) -> str:
    return " ".join(value.split())


class DossierMixin:
    if TYPE_CHECKING:
        memorize_config: MemorizeConfig
        _dossier_content_embedding_cache: dict[str, tuple[str, list[float]]]
        _get_database: Callable[[], Database]
        _select_embedding_client: Callable[[Mapping[str, Any] | None], Any]

    async def ensure_dossier_anchors(
        self,
        where: Mapping[str, Any],
        *,
        embedding_client: Any | None = None,
    ) -> dict[str, MemoryCategory]:
        scope = _scope(where)
        if scope["user_id"].casefold() == scope["soul_id"].casefold():
            raise ValueError("Dossier soul and user identities must differ")

        store = self._get_database()
        categories = store.memory_category_repo.list_categories(scope)
        anchors = store.memory_category_repo.list_anchor_categories(scope)
        expected: dict[AnchorRole, str] = {"soul": scope["soul_id"], "user": scope["user_id"]}
        for role, category in anchors.items():
            if role not in ("soul", "user"):
                raise ValueError(f"Invalid {role} dossier anchor for scope")
            if category.name != expected[cast(AnchorRole, role)] or category.kind != "lore":
                raise ValueError(f"Invalid {role} dossier anchor for scope")

        missing = [(role, name) for role, name in expected.items() if role not in anchors]
        for role, name in missing:
            collision = next((cat for cat in categories.values() if cat.name.casefold() == name.casefold()), None)
            if collision is not None:
                raise ValueError(f"Cannot seed {role} dossier anchor; title already exists in scope: {name}")

        if not missing:
            return anchors

        descriptions = {
            role: f"Identity, history, relationships, and lived experience of {name}."
            for role, name in missing
        }
        texts = [_identity_text(name, descriptions[role]) for role, name in missing]
        client = embedding_client or self._select_embedding_client(
            {"operation": "dossier", "step_id": "seed_anchors"}
        )
        embeddings = await client.embed(texts)
        if len(embeddings) != len(missing):
            raise ValueError("Anchor embedding response count does not match input")

        for (role, name), raw_embedding in zip(missing, embeddings, strict=True):
            embedding = _embedding_vector(raw_embedding, label=f"{role} anchor")
            anchor = store.memory_category_repo.get_or_create_category(
                name=name,
                description=descriptions[role],
                embedding=embedding,
                user_data=scope,
                kind="lore",
                lore_subtype="person",
                anchor_role=role,
            )
            if anchor.anchor_role != role or anchor.kind != "lore":
                raise ValueError(f"Cannot seed {role} dossier anchor; title was claimed concurrently")
            anchors[role] = anchor
        return anchors

    async def update_dossier(
        self,
        category_id: str,
        where: Mapping[str, Any],
        *,
        name: str | None = None,
        description: str | None = None,
        kind: DossierKind | None | EllipsisType = ...,
        lore_subtype: str | None | EllipsisType = ...,
        entity_id: str | None | EllipsisType = ...,
        last_evidence_at: datetime | None | EllipsisType = ...,
        last_revised_at: datetime | None | EllipsisType = ...,
        embedding_client: Any | None = None,
    ) -> MemoryCategory:
        scope = _scope(where)
        store = self._get_database()
        current = store.memory_category_repo.list_categories(scope).get(category_id)
        if current is None:
            raise KeyError(f"Dossier with id {category_id} not found in scope")

        final_name = current.name if name is None else name.strip()
        if not final_name:
            raise ValueError("Dossier title is required")
        final_description = current.description if description is None else description.strip()
        final_kind = current.kind if kind is ... else kind
        if final_kind is not None and final_kind not in DOSSIER_KINDS:
            raise ValueError(f"Invalid dossier kind: {final_kind}")
        if current.anchor_role is not None:
            if final_name != current.name:
                raise ValueError("Dossier anchors cannot be renamed")
            if final_kind != "lore":
                raise ValueError("Dossier anchors must remain Lore dossiers")

        final_identity = _identity_text(final_name, final_description)
        embedding = None
        if dossier_identity_text(current) != final_identity:
            client = embedding_client or self._select_embedding_client(
                {"operation": "dossier", "step_id": "update_identity"}
            )
            raw_embeddings = await client.embed([final_identity])
            if len(raw_embeddings) != 1:
                raise ValueError("Dossier embedding response count does not match input")
            embedding = _embedding_vector(raw_embeddings[0], label="dossier identity")

        return store.memory_category_repo.update_category(
            category_id=category_id,
            name=final_name if name is not None else None,
            description=final_description if description is not None else None,
            embedding=embedding,
            kind=kind,
            lore_subtype=lore_subtype,
            entity_id=entity_id,
            last_evidence_at=last_evidence_at,
            last_revised_at=last_revised_at,
        )

    def list_active_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        anchors = store.memory_category_repo.list_anchor_categories(scope)
        expected: dict[AnchorRole, str] = {"soul": scope["soul_id"], "user": scope["user_id"]}
        active: list[MemoryCategory] = []
        for role, category in anchors.items():
            if role not in ("soul", "user"):
                raise ValueError(f"Invalid {role} dossier anchor for scope")
            if category.name != expected[cast(AnchorRole, role)] or category.kind != "lore":
                raise ValueError(f"Invalid {role} dossier anchor for scope")
            active.append(category)

        limit = max(0, int(self.memorize_config.active_dossiers_per_kind))
        for kind in DOSSIER_KINDS:
            evidenced = [
                category
                for category in store.memory_category_repo.list_categories_by_activity(scope, kind=kind)
                if category.last_evidence_at is not None
            ]
            active.extend(evidenced[:limit])
        return sorted(active, key=_activity_key)

    def list_inactive_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        active_ids = {category.id for category in self.list_active_dossiers(scope)}
        inactive = [
            category
            for category in store.memory_category_repo.list_categories(scope).values()
            if category.kind in DOSSIER_KINDS and category.anchor_role is None and category.id not in active_ids
        ]
        return sorted(inactive, key=_activity_key)

    def build_dossier_index(self, where: Mapping[str, Any]) -> str:
        lines: list[str] = []
        for category in self.list_active_dossiers(where)[:DOSSIER_INDEX_LIMIT]:
            name = _render_index_field(category.name)
            description = _render_index_field(category.description)
            lines.append(f"- {name}: {description}" if description else f"- {name}")
        return "\n".join(lines)

    async def search_dossiers(
        self,
        query_embedding: Sequence[float],
        *,
        where: Mapping[str, Any],
        view: Literal["identity", "content"],
        activity: Literal["active", "inactive", "all"],
        limit: int,
        min_score: float,
        embedding_client: Any | None = None,
    ) -> list[tuple[MemoryCategory, float]]:
        if view not in {"identity", "content"}:
            raise ValueError(f"Unknown dossier search view: {view}")
        if activity not in {"active", "inactive", "all"}:
            raise ValueError(f"Unknown dossier activity set: {activity}")
        if view == "content" and activity != "active":
            raise ValueError("Dossier content search is limited to active dossiers")
        if limit < 1:
            raise ValueError("Dossier search limit must be positive")
        if not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
            raise ValueError("Dossier search min_score must be between -1 and 1")

        query = _embedding_vector(query_embedding, label="query")
        scope = _scope(where)
        store = self._get_database()
        if activity == "active":
            categories = self.list_active_dossiers(scope)
        elif activity == "inactive":
            categories = self.list_inactive_dossiers(scope)
        else:
            categories = [
                category
                for category in store.memory_category_repo.list_categories(scope).values()
                if category.kind in DOSSIER_KINDS
            ]
        if not categories:
            return []

        corpus: list[tuple[str, list[float]]] = []
        if view == "identity":
            for category in categories:
                if category.embedding is None:
                    raise ValueError(f"Dossier {category.id} is missing its identity embedding")
                corpus.append((category.id, _embedding_vector(category.embedding, label=f"dossier {category.id}")))
        else:
            cache = self._dossier_content_embedding_cache
            missing: list[tuple[str, str]] = []
            for category in categories:
                text = _content_text(category)
                cached = cache.get(category.id)
                if cached is None or cached[0] != text:
                    missing.append((category.id, text))
            if missing:
                client = embedding_client or self._select_embedding_client(
                    {"operation": "dossier", "step_id": "content_embedding"}
                )
                raw_embeddings = await client.embed([text for _, text in missing])
                if len(raw_embeddings) != len(missing):
                    raise ValueError("Dossier content embedding response count does not match input")
                for (category_id, text), raw_embedding in zip(missing, raw_embeddings, strict=True):
                    cache[category_id] = (
                        text,
                        _embedding_vector(raw_embedding, label=f"dossier {category_id} content"),
                    )
            corpus = [(category.id, cache[category.id][1]) for category in categories]

        wrong_dimension = [category_id for category_id, vector in corpus if len(vector) != len(query)]
        if wrong_dimension:
            raise ValueError(
                f"Dossier embedding dimension mismatch for: {', '.join(sorted(wrong_dimension))}"
            )

        category_by_id = {
            category.id: (
                category.model_copy(update={"summary": None, "previous_summary": None, "approved_summary": None})
                if view == "identity"
                else category
            )
            for category in categories
        }
        scored = cosine_topk(query, corpus, k=len(corpus))
        results = [
            (category_by_id[category_id], score)
            for category_id, score in scored
            if score >= min_score
        ]
        results.sort(key=lambda hit: (-hit[1], hit[0].name.casefold(), hit[0].id))
        return results[:limit]

    @staticmethod
    def format_memory_ref(value: int | MemoryItem) -> str:
        memory_ref = value.memory_ref if isinstance(value, MemoryItem) else value
        if isinstance(memory_ref, bool) or not isinstance(memory_ref, int) or memory_ref < 1:
            raise ValueError("Memory reference must be a positive integer")
        return f"[M{memory_ref}]"

    @staticmethod
    def parse_memory_ref(value: str) -> int:
        match = MEMORY_REF_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"Invalid memory reference: {value}")
        return int(match.group(1))

    def resolve_memory_ref(self, value: int | str, where: Mapping[str, Any]) -> MemoryItem:
        if isinstance(value, str):
            memory_ref = self.parse_memory_ref(value)
        else:
            self.format_memory_ref(value)
            memory_ref = value
        scope = _scope(where)
        item = self._get_database().memory_item_repo.get_item_by_memory_ref(memory_ref, scope)
        if item is None:
            raise KeyError(f"Memory reference [M{memory_ref}] not found in scope")
        return item


__all__ = ["DossierMixin", "dossier_identity_text"]
