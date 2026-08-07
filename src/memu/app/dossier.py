from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, cast

from memu.database.models import DossierKind, MemoryCategory, MemoryItem
from memu.database.vector import cosine_topk, reciprocal_rank_fusion
from memu.utils.taxonomy import (
    DOSSIER_KINDS,
    category_identity_text,
    dossier_scope as _scope,
    embedding_vector as _embedding_vector,
)

if TYPE_CHECKING:
    from memu.app.settings import MemorizeConfig, RetrieveConfig
    from memu.database.interfaces import Database

DOSSIER_INDEX_LIMIT = 20
MEMORY_REF_PATTERN = re.compile(r"^\[M([1-9][0-9]*)\]$")
MEMORY_REF_SCAN_PATTERN = re.compile(r"\[M(?:[0-9][^\]\r\n]*)?\]")
AnchorRole = Literal["soul", "user"]


def _activity_key(category: MemoryCategory) -> tuple[int, float, str, str]:
    happened = category.last_evidence_at
    if happened is None:
        return (1, 0.0, category.name.casefold(), category.id)
    if happened.tzinfo is None:
        happened = happened.replace(tzinfo=UTC)
    return (0, -happened.timestamp(), category.name.casefold(), category.id)


def _timestamp(value: datetime) -> float:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).timestamp()


def _item_time(item: MemoryItem) -> datetime:
    return item.happened_at or item.created_at


def _item_sort_key(item: MemoryItem) -> tuple[float, int, str]:
    return (_timestamp(_item_time(item)), int(item.memory_ref or 0), item.id)


def _load_linked_items(
    store: Any,
    relations: Sequence[Any],
    scope: Mapping[str, str],
) -> tuple[dict[str, MemoryItem], set[str]]:
    item_ids = {relation.item_id for relation in relations}
    if not item_ids:
        return {}, set()
    items = store.memory_item_repo.list_items_by_ids(
        item_ids,
        scope,
        include_superseded=True,
        include_merged=True,
    )
    missing = item_ids - items.keys()
    if missing:
        raise KeyError(f"Dossier memberships reference missing memories: {sorted(missing)}")
    active_ids = store.memory_item_repo.list_items_by_ids(item_ids, scope).keys()
    return items, item_ids - active_ids


def _content_text(category: MemoryCategory) -> str:
    identity = category_identity_text(category.name, category.description)
    summary = str(category.summary or "").strip()
    return f"{identity}\n{summary}" if summary else identity


def _render_index_field(value: str) -> str:
    return " ".join(value.split())


def _render_dossier_index_line(category: MemoryCategory) -> str:
    name = _render_index_field(category.name)
    description = _render_index_field(category.description)
    return f"- {name}: {description}" if description else f"- {name}"


def _validate_anchors(
    anchors: Mapping[str, MemoryCategory], scope: Mapping[str, str]
) -> None:
    expected: dict[AnchorRole, str] = {"soul": scope["soul_id"], "user": scope["user_id"]}
    for role, category in anchors.items():
        if (
            role not in expected
            or category.name != expected[cast(AnchorRole, role)]
            or category.kind != "lore"
        ):
            raise ValueError(f"Invalid {role} dossier anchor for scope")


class DossierMixin:
    if TYPE_CHECKING:
        memorize_config: MemorizeConfig
        retrieve_config: RetrieveConfig
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
        anchors = {
            cast(str, category.anchor_role): category
            for category in categories.values()
            if category.anchor_role is not None
        }
        _validate_anchors(anchors, scope)
        expected: dict[AnchorRole, str] = {"soul": scope["soul_id"], "user": scope["user_id"]}

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
        texts = [category_identity_text(name, descriptions[role]) for role, name in missing]
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

        final_identity = category_identity_text(final_name, final_description)
        embedding = None
        if category_identity_text(current.name, current.description) != final_identity:
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

    def list_due_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        categories = store.memory_category_repo.list_categories(scope)
        relations = store.category_item_repo.list_relations(scope)
        _linked_items, inactive_ids = _load_linked_items(store, relations, scope)

        actionable: list[tuple[float, MemoryCategory]] = []
        for category in categories.values():
            if category.kind not in DOSSIER_KINDS:
                continue
            category_relations = [
                relation for relation in relations if relation.category_id == category.id
            ]
            if not category_relations:
                continue
            if category.last_revised_at is None:
                due_relations = category_relations
            else:
                revised_at = _timestamp(category.last_revised_at)
                due_relations = [
                    relation
                    for relation in category_relations
                    if _timestamp(relation.created_at) > revised_at
                    or relation.item_id in inactive_ids
                ]
            if due_relations:
                actionable.append(
                    (min(_timestamp(relation.created_at) for relation in due_relations), category)
                )

        actionable.sort(
            key=lambda row: (
                row[0],
                str(row[1].kind),
                row[1].name.casefold(),
                row[1].id,
            )
        )
        return [category for _timestamp_value, category in actionable]

    def list_dossiers_revised_since(
        self,
        where: Mapping[str, Any],
        *,
        revised_after: datetime | None,
    ) -> list[MemoryCategory]:
        scope = _scope(where)
        cutoff = None if revised_after is None else _timestamp(revised_after)
        categories = [
            category
            for category in self._get_database().memory_category_repo.list_categories(scope).values()
            if category.kind in DOSSIER_KINDS
            and category.last_revised_at is not None
            and (cutoff is None or _timestamp(category.last_revised_at) > cutoff)
        ]
        return sorted(
            categories,
            key=lambda category: (
                category.last_evidence_at is None,
                _timestamp(category.last_evidence_at) if category.last_evidence_at else 0.0,
                str(category.kind),
                category.name.casefold(),
                category.id,
            ),
        )

    @staticmethod
    def extract_memory_refs(text: str) -> list[int]:
        return list(
            dict.fromkeys(
                DossierMixin.parse_memory_ref(token)
                for token in MEMORY_REF_SCAN_PATTERN.findall(text or "")
            )
        )

    def prepare_dossier_revision(
        self,
        category_id: str,
        where: Mapping[str, Any],
        *,
        active_life_goals: Sequence[str] = (),
        removed_life_goals: Sequence[str] = (),
    ) -> dict[str, Any]:
        scope = _scope(where)
        store = self._get_database()
        category = store.memory_category_repo.list_categories(scope).get(category_id)
        if category is None or category.kind not in DOSSIER_KINDS:
            raise KeyError(f"Dossier with id {category_id} not found in scope")

        relations = [
            relation
            for relation in store.category_item_repo.list_relations(scope)
            if relation.category_id == category.id
        ]
        linked_items, linked_inactive_ids = _load_linked_items(store, relations, scope)
        revised_at = None if category.last_revised_at is None else _timestamp(category.last_revised_at)
        pending_relations = [
            relation
            for relation in relations
            if revised_at is None or _timestamp(relation.created_at) > revised_at
        ]
        pending_ids = {relation.item_id for relation in pending_relations}
        if not pending_ids and not linked_inactive_ids:
            raise ValueError(f"Dossier {category_id} is not due for revision")

        cited_items: dict[str, MemoryItem] = {}
        for memory_ref in self.extract_memory_refs(category.summary or ""):
            item = store.memory_item_repo.get_item_by_memory_ref(memory_ref, scope)
            if item is None:
                raise KeyError(f"Memory reference [M{memory_ref}] not found in scope")
            cited_items[item.id] = item

        top_k = max(0, int(self.retrieve_config.item.top_k))
        candidate_items: dict[str, MemoryItem] = {}
        if top_k:
            query = _embedding_vector(category.embedding, label=f"dossier {category.id} identity")
            excluded_ids = set(linked_items) | set(cited_items)
            hits = store.memory_item_repo.vector_search_items(
                query,
                top_k + len(excluded_ids),
                scope,
                ranking="similarity",
            )
            candidate_ids = [item_id for item_id, _score in hits if item_id not in excluded_ids][
                :top_k
            ]
            candidate_items = store.memory_item_repo.list_items_by_ids(set(candidate_ids), scope)
            missing = set(candidate_ids) - candidate_items.keys()
            if missing:
                raise KeyError(f"Dossier candidate memories disappeared: {sorted(missing)}")

        prompt_items = {
            item_id: item
            for item_id, item in linked_items.items()
            if item_id in pending_ids
        }
        prompt_items.update(cited_items)
        prompt_items.update(candidate_items)
        all_shown_items = {
            **prompt_items,
            **{item_id: linked_items[item_id] for item_id in linked_inactive_ids},
        }
        missing_refs = sorted(
            item_id
            for item_id, item in all_shown_items.items()
            if item.memory_ref is None
        )
        if missing_refs:
            raise ValueError(f"Dossier revision memories lack stable references: {missing_refs}")

        active_shown_ids = store.memory_item_repo.list_items_by_ids(set(all_shown_items), scope).keys()
        shown_inactive_ids = set(all_shown_items) - active_shown_ids
        cleanup = [
            {
                "item_id": item_id,
                "memory_ref": linked_items[item_id].memory_ref,
                "lineage_state": "merged" if linked_items[item_id].merged_into else "superseded",
                "merged_into": linked_items[item_id].merged_into,
            }
            for item_id in sorted(linked_inactive_ids)
        ]
        return {
            "dossier": category,
            "category_updated_at": category.updated_at,
            "relation_tokens": sorted(
                (relation.id, relation.created_at, relation.updated_at) for relation in relations
            ),
            "linked_item_ids": sorted(linked_items),
            "linked_inactive_item_ids": sorted(linked_inactive_ids),
            "cited_unlinked_item_ids": sorted(set(cited_items) - set(linked_items)),
            "shown_inactive_item_ids": sorted(shown_inactive_ids),
            "shown_item_tokens": sorted(
                (
                    item.id,
                    item.updated_at,
                    item.memory_ref,
                    item.merged_into,
                )
                for item in all_shown_items.values()
            ),
            "cited_items": sorted(cited_items.values(), key=_item_sort_key),
            "pending_items": sorted(
                (linked_items[item_id] for item_id in pending_ids), key=_item_sort_key
            ),
            "cleanup_memberships": cleanup,
            "candidate_items": sorted(candidate_items.values(), key=_item_sort_key),
            "untouched_item_ids": sorted(
                set(linked_items) - pending_ids - set(cited_items) - linked_inactive_ids
            ),
            "active_life_goals": [
                goal.strip() for goal in active_life_goals if category.kind == "goal" and goal.strip()
            ],
            "removed_life_goals": [
                goal.strip() for goal in removed_life_goals if category.kind == "goal" and goal.strip()
            ],
        }

    def list_active_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        anchors = store.memory_category_repo.list_anchor_categories(scope)
        _validate_anchors(anchors, scope)
        active = list(anchors.values())

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
        return "\n".join(
            _render_dossier_index_line(category)
            for category in self.list_active_dossiers(where)[:DOSSIER_INDEX_LIMIT]
        )

    async def select_memorize_dossier_context(
        self,
        episodes: Sequence[Mapping[str, Any]],
        where: Mapping[str, Any],
        *,
        narrative_self: str | None,
        embedding_client: Any | None = None,
    ) -> dict[str, Any]:
        scope = _scope(where)
        if not 1 <= len(episodes) <= 3:
            raise ValueError("Memorize dossier context requires one to three episodes")

        episode_texts: list[str] = []
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise ValueError("Each episode must be a mapping")
            title = episode.get("title")
            summary = episode.get("summary")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Each episode requires a nonblank title")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Each episode requires a nonblank summary")
            episode_texts.append(category_identity_text(title, summary))

        anchors = await self.ensure_dossier_anchors(scope, embedding_client=embedding_client)
        client = embedding_client or self._select_embedding_client(
            {"operation": "dossier", "step_id": "memorize_context"}
        )
        raw_embeddings = await client.embed(episode_texts)
        if len(raw_embeddings) != len(episode_texts):
            raise ValueError("Episode embedding response count does not match input")
        query_embeddings = [
            _embedding_vector(raw, label=f"episode {index + 1}")
            for index, raw in enumerate(raw_embeddings)
        ]

        store = self._get_database()
        active = self.list_active_dossiers(scope)
        anchor_ids = {category.id for category in anchors.values()}
        active_candidates = [category for category in active if category.id not in anchor_ids]
        active_by_id = {category.id: category for category in active_candidates}
        active_ids = {category.id for category in active}
        inactive = sorted(
            [
                category
                for category in store.memory_category_repo.list_categories(scope).values()
                if category.kind in DOSSIER_KINDS
                and category.anchor_role is None
                and category.id not in active_ids
            ],
            key=_activity_key,
        )

        relevant: list[MemoryCategory] = []
        selected_ids: set[str] = set()
        for query in query_embeddings:
            identity = await self.search_dossiers(
                query,
                where=scope,
                view="identity",
                activity="active",
                limit=3,
                min_score=-1.0,
                categories=active_candidates,
            )
            content = await self.search_dossiers(
                query,
                where=scope,
                view="content",
                activity="active",
                limit=3,
                min_score=-1.0,
                embedding_client=client,
                categories=active_candidates,
            )
            ranked = reciprocal_rank_fusion(
                [(category.id, score) for category, score in identity],
                [(category.id, score) for category, score in content],
            )
            for category_id, _score in ranked[:3]:
                if category_id not in selected_ids:
                    relevant.append(active_by_id[category_id])
                    selected_ids.add(category_id)

        inactive_by_id = {category.id: category for category in inactive}
        inactive_rankings: list[list[tuple[str, float]]] = []
        for query in query_embeddings:
            hits = await self.search_dossiers(
                query,
                where=scope,
                view="identity",
                activity="inactive",
                limit=10,
                min_score=-1.0,
                categories=inactive,
            )
            inactive_rankings.append([(category.id, score) for category, score in hits])
        inactive_ids = [
            category_id
            for category_id, _score in reciprocal_rank_fusion(*inactive_rankings)[:10]
        ]

        category_lines = [f"- {_render_index_field(category.name)}" for category in active]
        category_lines.extend(
            _render_dossier_index_line(inactive_by_id[category_id]) for category_id in inactive_ids
        )
        narrative = (
            narrative_self.strip()
            if isinstance(narrative_self, str) and narrative_self.strip()
            else None
        )
        return {
            "categories_str": "\n".join(category_lines),
            "dossier_index": "\n".join(
                _render_dossier_index_line(category)
                for category in active_candidates[:DOSSIER_INDEX_LIMIT]
            ),
            "narrative_self": narrative,
            "anchor_dossiers": [anchors[role] for role in ("soul", "user")],
            "relevant_dossiers": relevant[:9],
        }

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
        categories: Sequence[MemoryCategory] | None = None,
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
        search_categories = list(categories) if categories is not None else None
        if search_categories is None:
            store = self._get_database()
            if activity == "active":
                search_categories = self.list_active_dossiers(scope)
            elif activity == "inactive":
                search_categories = self.list_inactive_dossiers(scope)
            else:
                search_categories = [
                    category
                    for category in store.memory_category_repo.list_categories(scope).values()
                    if category.kind in DOSSIER_KINDS
                ]
        if not search_categories:
            return []

        corpus: list[tuple[str, list[float]]] = []
        if view == "identity":
            for category in search_categories:
                if category.embedding is None:
                    raise ValueError(f"Dossier {category.id} is missing its identity embedding")
                corpus.append((category.id, _embedding_vector(category.embedding, label=f"dossier {category.id}")))
        else:
            cache = self._dossier_content_embedding_cache
            missing: list[tuple[str, str]] = []
            for category in search_categories:
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
            corpus = [(category.id, cache[category.id][1]) for category in search_categories]

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
            for category in search_categories
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


__all__ = ["DossierMixin"]
