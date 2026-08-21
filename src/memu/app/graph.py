from __future__ import annotations

import base64
import logging
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import numpy as np
from sqlalchemy import text

from memu.app.category_summary_journal import append_category_summary_journal
from memu.app.dossier import DossierRevisionStaleError
from memu.database.models import Triple, entity_is_ignored, normalize_entity_name
from memu.database.vector import cosine_similarity, cosine_topk
from memu.utils.taxonomy import DOSSIER_KINDS

SEMANTIC_PREDICATES = ["caused_by", "evokes", "conflicts_with", "parallels", "shaped_by"]
ENTITY_PROPERTY_KEYS = {"origin", "active", "relationship", "aliases", "source_refs", "ignored"}
logger = logging.getLogger(__name__)


def _normalize_search_memory_ref(value: str) -> int | None:
    match = re.fullmatch(r"(?:[Mm]([1-9]\d*)|\[[Mm]([1-9]\d*)\])", value)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


class EntityMergeConflictError(ValueError):
    def __init__(self, canonical: dict[str, Any], duplicate: dict[str, Any], conflicts: list[str]) -> None:
        super().__init__("; ".join(conflicts))
        self.canonical = canonical
        self.duplicate = duplicate
        self.conflicts = conflicts


class EntityActionConflictError(ValueError):
    def __init__(self, conflicts: list[str]) -> None:
        super().__init__("; ".join(conflicts))
        self.conflicts = conflicts


class DossierMembershipConflictError(ValueError):
    pass


def _stable_entity_values(values: Iterable[Any], *, exclude: str = "") -> list[str]:
    seen = {normalize_entity_name(exclude)} if exclude else set()
    result: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        normalized = normalize_entity_name(clean)
        if clean and normalized not in seen:
            seen.add(normalized)
            result.append(clean)
    return result


def _stable_text_values(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))


def _entity_conflict_payload(entity: Any) -> dict[str, Any]:
    return {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "properties": dict(entity.properties or {}),
    }


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _label(text: str, limit: int = 80) -> str:
    clean = " ".join(str(text or "").split())
    return clean[: limit - 1] + "..." if len(clean) > limit else clean


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


DUPE_CLUSTER_THRESHOLD = 0.76


def _cluster_by_embedding(items: list[Any]) -> dict[str, dict[str, Any]]:
    """Cluster items by pairwise cosine similarity (transitive closure).

    Returns {item_id: {"similar_to": [...], "similarity": float}} for clustered
    items only; non-clustered items are absent (absence = not a dupe).
    """
    # ponytail: O(n^2) pairwise scan — pending lists are small; revisit if that changes.
    n = len(items)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        vec_i = items[i].embedding
        if not vec_i:
            continue
        for j in range(i + 1, n):
            vec_j = items[j].embedding
            if not vec_j:
                continue
            score = cosine_similarity(vec_i, vec_j)
            if score >= DUPE_CLUSTER_THRESHOLD:
                pairs.append((i, j, score))

    if not pairs:
        return {}

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j, _score in pairs:
        union(i, j)

    best_score: dict[int, float] = {}
    for i, j, score in pairs:
        best_score[i] = max(best_score.get(i, 0.0), score)
        best_score[j] = max(best_score.get(j, 0.0), score)

    groups: dict[int, list[int]] = {}
    for idx in best_score:
        groups.setdefault(find(idx), []).append(idx)

    result: dict[str, dict[str, Any]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        group_key = min(items[m].id for m in members)
        for idx in members:
            others = [f"memory:{items[m].id}" for m in members if m != idx]
            result[items[idx].id] = {
                "similar_to": others,
                "similarity": round(best_score[idx], 3),
                "_group_key": group_key,
            }
    return result


def _atomic_similarity_edges(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_size: dict[int, list[tuple[str, list[float]]]] = {}
    for atom in atoms:
        embedding = atom.get("embedding") or []
        if embedding:
            by_size.setdefault(len(embedding), []).append((str(atom["id"]), embedding))

    scored: list[tuple[str, str, float]] = []
    for group in by_size.values():
        ids = [atom_id for atom_id, _embedding in group]
        matrix = np.asarray([embedding for _atom_id, embedding in group], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 0
        if valid.sum() < 2:
            continue
        ids = [atom_id for atom_id, keep in zip(ids, valid, strict=True) if keep]
        matrix = matrix[valid]
        norms = norms[valid]
        scores = matrix @ matrix.T / (norms[:, None] * norms[None, :])
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                score = float(scores[left, right])
                if score >= 0.5:
                    scored.append((ids[left], ids[right], score))

    scored.sort(key=lambda edge: edge[2], reverse=True)
    per_atom: dict[str, int] = {}
    edges: list[dict[str, Any]] = []
    for source, target, weight in scored:
        source, target = sorted((source, target))
        if per_atom.get(source, 0) >= 3 or per_atom.get(target, 0) >= 3:
            continue
        per_atom[source] = per_atom.get(source, 0) + 1
        per_atom[target] = per_atom.get(target, 0) + 1
        edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "kind": "similarity",
            "predicate": "similarity",
        })
    return edges


def _pack_embedding(values: list[float]) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


class GraphMixin:
    def _graph_active_category_ids(self, where: Mapping[str, Any] | None) -> set[str]:
        return {category.id for category in self.list_active_dossiers(where or {})}

    def _scoped_category_node(
        self,
        category: Any,
        *,
        where: Mapping[str, Any] | None,
        active_category_ids: set[str],
        memory_refs: list[int] | None = None,
    ) -> dict[str, Any]:
        citations = []
        # ponytail: N+1 citation lookups; batch WHERE IN if scale matters.
        for memory_ref in memory_refs if memory_refs is not None else self.extract_memory_refs(category.summary or ""):
            try:
                item = self.resolve_memory_ref(memory_ref, where or {})
            except KeyError:
                continue
            citations.append({
                "ref": self.format_memory_ref(memory_ref),
                "memory_id": item.id,
                "summary": item.summary,
            })
        return self._category_node(
            category,
            active=category.id in active_category_ids,
            citations=citations,
        )

    def _category_detail_node(
        self,
        category: Any,
        *,
        where: Mapping[str, Any] | None,
        active_category_ids: set[str],
    ) -> dict[str, Any]:
        store = self._get_database()
        memory_refs = self.extract_memory_refs(category.summary or "")
        node = self._scoped_category_node(
            category,
            where=where,
            active_category_ids=active_category_ids,
            memory_refs=memory_refs,
        )
        relations = store.category_item_repo.list_relations(
            {**dict(where or {}), "category_id": category.id}
        )
        member_ids = {relation.item_id for relation in relations}
        members = store.memory_item_repo.list_items_by_ids(
            member_ids,
            where,
            include_superseded=True,
            include_merged=True,
        )
        active_ids = set(store.memory_item_repo.list_items_by_ids(member_ids, where))
        cited_refs = set(memory_refs)
        node["members"] = [
            {
                "id": f"memory:{item.id}",
                "memory_id": item.id,
                "memory_ref": self.format_memory_ref(item.memory_ref) if item.memory_ref is not None else None,
                "summary": item.summary,
                "memory_type": item.memory_type,
                "happened_at": _iso(item.happened_at),
                "created_at": _iso(item.created_at),
                "status": "Active" if item.id in active_ids else "Inactive",
                "cited": item.memory_ref in cited_refs,
            }
            for item in sorted(
                members.values(),
                key=lambda item: (item.memory_ref is None, item.memory_ref or 0, item.id),
            )
        ]
        return node

    def graph_memory(self, item_id: str, *, where: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        store = self._get_database()
        item_id = str(item_id or "")
        kind, _, raw_id = item_id.partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind == "category":
            category = store.memory_category_repo.list_categories(where).get(raw_id)
            if category is None:
                return None
            return self._category_detail_node(
                category,
                where=where,
                active_category_ids=self._graph_active_category_ids(where),
            )
        if kind == "entity":
            entity = next((entity for entity in store.entity_repo.list_all(where) if entity.id == raw_id), None)
            return self._entity_node(entity) if entity is not None else None

        item = store.memory_item_repo.list_items_by_ids({raw_id}, where).get(raw_id)
        if item is None:
            return None

        categories = store.memory_category_repo.list_categories(where)
        relation_scope = dict(where or {})
        relation_scope["item_id"] = item.id
        relations = store.category_item_repo.list_relations(relation_scope)
        category_names = [
            categories[rel.category_id].name
            for rel in relations
            if rel.item_id == item.id and rel.category_id in categories
        ]
        category_ids = [
            rel.category_id
            for rel in relations
            if rel.item_id == item.id and rel.category_id in categories
        ]
        entity_pairs = self._entity_pairs_by_memory([item.id], where or {}).get(item.id, [])
        return self._memory_node(
            item,
            category_names=category_names,
            category_ids=category_ids,
            entity_pairs=entity_pairs,
        )

    def graph_atomic_atoms(
        self,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
        category_id: str | None = None,
        cursor: str | None = None,
        cursor_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._get_database()
        limit = max(1, min(int(limit or 50), 200))
        offset = max(0, int(offset or 0))
        categories = store.memory_category_repo.list_categories(where)
        active_category_ids = self._graph_active_category_ids(where) if categories else set()
        relations = store.category_item_repo.list_relations(where)
        raw_category_id = str(category_id or "").removeprefix("category:")
        item_category_ids: dict[str, list[str]] = {}
        for rel in relations:
            if rel.category_id in categories:
                item_category_ids.setdefault(rel.item_id, []).append(rel.category_id)

        items = list(store.memory_item_repo.list_items(where).values())
        if raw_category_id:
            items = [item for item in items if raw_category_id in item_category_ids.get(item.id, [])]
        item_nodes = [
            self._memory_node(
                item,
                category_names=[categories[cat_id].name for cat_id in item_category_ids.get(item.id, [])],
                category_ids=item_category_ids.get(item.id, []),
            )
            for item in items
        ]
        category_nodes = [
            self._scoped_category_node(
                category,
                where=where,
                active_category_ids=active_category_ids,
            )
            for category in categories.values()
            if not raw_category_id or category.id == raw_category_id
        ]
        nodes = sorted(item_nodes + category_nodes, key=lambda node: (node.get("updated_at") or node.get("created_at") or "", node["id"]), reverse=True)
        start = offset
        if cursor and cursor_id:
            for idx, node in enumerate(nodes):
                node_cursor = node.get("updated_at") or node.get("created_at") or ""
                if node_cursor == cursor and node["id"] == cursor_id:
                    start = idx + 1
                    break
        page = nodes[start : start + limit]
        next_node = page[-1] if start + limit < len(nodes) and page else None
        next_cursor = None
        if next_node:
            next_cursor = next_node.get("updated_at") or next_node.get("created_at") or ""
        return {
            "atoms": [self._atomic_atom(node) for node in page],
            "total_count": len(nodes),
            "limit": limit,
            "offset": start,
            "next_cursor": next_cursor,
            "next_cursor_id": next_node["id"] if next_node else None,
        }

    def graph_atomic_tags(self, *, where: Mapping[str, Any] | None = None, min_count: int = 0) -> list[dict[str, Any]]:
        store = self._get_database()
        min_count = max(0, int(min_count or 0))
        categories = store.memory_category_repo.list_categories(where)
        active_category_ids = self._graph_active_category_ids(where) if categories else set()
        counts = dict.fromkeys(categories, 0)
        for rel in store.category_item_repo.list_relations(where):
            if rel.category_id in counts:
                counts[rel.category_id] += 1
        return [
            self._atomic_tag(category, counts[category.id], active=category.id in active_category_ids)
            for category in sorted(categories.values(), key=lambda category: category.name.lower())
            if counts[category.id] >= min_count
        ]

    def _atomic_entity_rows(
        self,
        where: Mapping[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
        # ponytail: loads all items + edges to avoid N+1; use one bounded aggregate query if this becomes slow
        store = self._get_database()
        entities = store.entity_repo.list_all(where)
        items = store.memory_item_repo.list_items(where, include_embeddings=False)
        links: dict[str, list[Any]] = {entity.id: [] for entity in entities}
        for edge in store.triple_repo.list_edges_for_memories(
            items,
            {"mentions"},
            where,
        ):
            if edge.subject_kind == "memory" and edge.object_kind == "entity" and edge.subject_id in items:
                if edge.object_id in links:
                    links[edge.object_id].append(items[edge.subject_id])

        rows = []
        for entity in entities:
            linked = links[entity.id]
            last_mentioned = max(
                (_utc(item.happened_at or item.created_at) for item in linked),
                default=None,
            )
            properties = entity.properties if isinstance(entity.properties, dict) else {}
            is_relationship = properties.get("origin") == "user_declared"
            visible_property_keys = ENTITY_PROPERTY_KEYS.copy()
            if not is_relationship:
                visible_property_keys -= {"active", "relationship"}
            rows.append({
                "id": entity.id,
                "atom_id": f"entity:{entity.id}",
                "name": entity.name,
                "normalized": entity.normalized,
                "entity_type": entity.entity_type,
                "properties": {key: properties[key] for key in visible_property_keys if key in properties},
                # Legacy inactive rows stay visible here so the Entity Manager can remove them.
                "is_relationship": is_relationship,
                "ignored": properties.get("ignored") is True,
                "linked_memory_count": len(linked),
                "orphan": not linked,
                "last_mentioned_at": _iso(last_mentioned),
                "created_at": _iso(entity.created_at),
                "updated_at": _iso(entity.updated_at),
            })
        rows.sort(key=lambda row: (row["name"].casefold(), row["id"]))
        return rows, links

    def graph_atomic_entities(self, *, where: Mapping[str, Any] | None = None) -> dict[str, Any]:
        rows, _links = self._atomic_entity_rows(where)
        return {"entities": rows, "total_count": len(rows)}

    def graph_atomic_entity(
        self,
        entity_id: str,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        # ponytail: rebuilds full aggregate for one entity; dedicated query if detail latency matters
        rows, links = self._atomic_entity_rows(where)
        entity = next((row for row in rows if row["id"] == entity_id), None)
        if entity is None:
            return None
        memories = sorted(
            links[entity_id],
            key=lambda item: (
                _utc(item.happened_at or item.created_at) or datetime.min.replace(tzinfo=UTC),
                item.memory_ref if item.memory_ref is not None else float("inf"),
                item.id,
            ),
        )
        return {
            **entity,
            "memories": [
                {
                    "id": f"memory:{item.id}",
                    "memory_id": item.id,
                    "memory_ref": self.format_memory_ref(item.memory_ref) if item.memory_ref is not None else None,
                    "memory_type": item.memory_type,
                    "summary": item.summary,
                    "happened_at": _iso(item.happened_at),
                    "created_at": _iso(item.created_at),
                }
                for item in memories
            ],
        }

    def graph_create_entity(
        self,
        name: str,
        entity_type: str,
        *,
        aliases: list[str] | None = None,
        where: Mapping[str, Any],
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        clean_type = str(entity_type or "").strip()
        if not clean_name:
            raise ValueError("entity name is required")
        if not clean_type:
            raise ValueError("entity_type is required")
        properties = {"aliases": aliases} if aliases else None
        entity = self._get_database().entity_repo.create(
            clean_name,
            clean_type,
            where,
            properties=properties,
        )
        return self.graph_atomic_entity(entity.id, where=where) or {}

    def graph_update_entity(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        entity_type: str | None = None,
        aliases: list[str] | None = None,
        where: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        clean_type = str(entity_type or "").strip() if entity_type is not None else None
        if clean_type is not None and not clean_type:
            raise ValueError("entity_type is required")
        store = self._get_database()
        with store._sessions.session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            with store.entity_repo.write_lock():
                matches = store.entity_repo.list_by_ids({entity_id}, where, session=session)
                if not matches:
                    return None
                entity = store.entity_repo.update(
                    entity_id,
                    where=where,
                    name=name,
                    entity_type=clean_type,
                    aliases=aliases,
                    session=session,
                )
                session.commit()
        return self.graph_atomic_entity(entity.id, where=where)

    @staticmethod
    def _entity_scope(where: Mapping[str, Any]) -> dict[str, Any]:
        scope = {key: str(where.get(key) or "").strip() for key in ("user_id", "soul_id")}
        if not all(scope.values()):
            raise ValueError("entity action requires user_id and soul_id")
        return scope

    def graph_set_entity_ignored(
        self,
        entity_id: str,
        *,
        ignored: bool,
        where: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = self._entity_scope(where)
        store = self._get_database()
        with store._sessions.session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            with store.entity_repo.write_lock():
                matches = store.entity_repo.list_by_ids({entity_id}, scope, session=session)
                if not matches:
                    raise KeyError(f"entity not found in scope: {entity_id}")
                entity = matches[0]
                if (entity.properties or {}).get("origin") == "user_declared":
                    raise EntityActionConflictError(["Remove the Relationship before ignoring this entity"])
                store.entity_repo.update(
                    entity_id,
                    where=scope,
                    property_updates={"ignored": True} if ignored else None,
                    property_removals=None if ignored else {"ignored"},
                    session=session,
                )
                session.commit()
        return self.graph_atomic_entity(entity_id, where=scope) or {}

    def graph_delete_entity(
        self,
        entity_id: str,
        *,
        where: Mapping[str, Any],
    ) -> None:
        scope = self._entity_scope(where)
        store = self._get_database()
        with store._sessions.session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            with store.entity_repo.write_lock():
                matches = store.entity_repo.list_by_ids({entity_id}, scope, session=session)
                if not matches:
                    raise KeyError(f"entity not found in scope: {entity_id}")
                entity = matches[0]
                properties = dict(entity.properties or {})
                conflicts: list[str] = []
                if properties.get("origin") == "user_declared":
                    conflicts.append("Relationships cannot be deleted")
                if properties.get("source_refs"):
                    conflicts.append("Entity has a source identity")
                categories = [
                    category.name
                    for category in store.memory_category_repo.list_categories(scope, session=session).values()
                    if category.entity_id == entity_id
                ]
                if categories:
                    conflicts.append(f"Dossiers: {', '.join(sorted(categories, key=str.casefold))}")
                speaker_items = store.memory_item_repo.list_by_speaker_id(
                    f"entity:{entity_id}", scope, session
                )
                if speaker_items:
                    refs = [
                        self.format_memory_ref(item.memory_ref) if item.memory_ref is not None else item.id
                        for item in speaker_items
                    ]
                    conflicts.append(f"Speaker memories: {', '.join(refs)}")
                references = store.triple_repo.list_entity_references(entity_id, scope, session)
                if references:
                    conflicts.append(f"Graph references: {len(references)}")
                if conflicts:
                    raise EntityActionConflictError(conflicts)
                store.entity_repo.delete(entity_id, where=scope, session=session)
                session.commit()

    @staticmethod
    def _entity_merge_state(canonical: Any, duplicate: Any, entities: list[Any]) -> dict[str, Any]:
        canonical_props = dict(canonical.properties or {})
        duplicate_props = dict(duplicate.properties or {})
        canonical_rel = canonical_props.get("origin") == "user_declared"
        duplicate_rel = duplicate_props.get("origin") == "user_declared"
        conflicts: list[str] = []
        warnings: list[str] = []

        if (canonical_rel and canonical_props.get("active") is False) or (
            duplicate_rel and duplicate_props.get("active") is False
        ):
            conflicts.append("Remove the inactive Relationship before merging")
        if canonical_rel and duplicate_rel:
            if normalize_entity_name(canonical.name) != normalize_entity_name(duplicate.name):
                conflicts.append("Relationship names differ")
            if canonical_props.get("relationship") != duplicate_props.get("relationship"):
                conflicts.append("Relationship descriptions differ")
        if canonical_props.get("ignored") != duplicate_props.get("ignored"):
            conflicts.append("Ignore states differ")

        known = ENTITY_PROPERTY_KEYS | {"deleted_at"}
        for key in (set(canonical_props) | set(duplicate_props)) - known:
            if key in duplicate_props and (
                key not in canonical_props or canonical_props[key] != duplicate_props[key]
            ):
                conflicts.append(f"Unknown property differs: {key}")

        aliases = _stable_entity_values(
            [*(canonical_props.get("aliases") or []), duplicate.name, *(duplicate_props.get("aliases") or [])],
            exclude=canonical.name,
        )
        source_refs = _stable_text_values(
            [*(canonical_props.get("source_refs") or []), *(duplicate_props.get("source_refs") or [])]
        )
        absorbed_names = {normalize_entity_name(value) for value in [duplicate.name, *aliases] if value}
        for entity in entities:
            if entity.id in {canonical.id, duplicate.id}:
                continue
            other_names = [entity.name, *((entity.properties or {}).get("aliases") or [])]
            if absorbed_names & {normalize_entity_name(str(value or "")) for value in other_names}:
                warnings.append(f"Name or alias also belongs to {entity.name}")
            other_refs = set((entity.properties or {}).get("source_refs") or [])
            overlap = other_refs & set(source_refs)
            if overlap:
                conflicts.append(f"Source reference belongs to another entity: {sorted(overlap)[0]}")

        updates: dict[str, Any] = {"source_refs": source_refs}
        relationship = canonical if canonical_rel else duplicate if duplicate_rel else None
        if relationship is not None:
            updates["origin"] = "user_declared"
            if (relationship.properties or {}).get("relationship") is not None:
                updates["relationship"] = relationship.properties["relationship"]
        return {"aliases": aliases, "property_updates": updates, "conflicts": conflicts, "warnings": warnings}

    def _load_entity_merge(self, canonical_id: str, duplicate_id: str, scope: dict[str, Any], *, session: Any = None):
        if canonical_id == duplicate_id:
            raise ValueError("canonical and duplicate entities must differ")
        entities = self._get_database().entity_repo.list_all(scope, session=session)
        by_id = {entity.id: entity for entity in entities}
        if canonical_id not in by_id:
            raise KeyError(f"entity not found in scope: {canonical_id}")
        if duplicate_id not in by_id:
            raise KeyError(f"entity not found in scope: {duplicate_id}")
        return by_id[canonical_id], by_id[duplicate_id], entities

    def graph_preview_entity_merge(
        self,
        canonical_id: str,
        duplicate_id: str,
        *,
        where: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = self._entity_scope(where)
        canonical, duplicate, entities = self._load_entity_merge(canonical_id, duplicate_id, scope)
        state = self._entity_merge_state(canonical, duplicate, entities)
        store = self._get_database()
        categories = store.memory_category_repo.list_categories(scope)
        items = store.memory_item_repo.list_items(
            scope, include_superseded=True, include_merged=True, include_embeddings=False
        )
        triples = {
            edge.id: edge
            for edge in [
                *store.triple_repo.get_edges_from(duplicate_id, current_only=False, where=scope),
                *store.triple_repo.get_edges_to(duplicate_id, current_only=False, where=scope),
            ]
        }
        duplicate_detail = self.graph_atomic_entity(duplicate_id, where=scope) or {}
        return {
            "canonical": self.graph_atomic_entity(canonical_id, where=scope),
            "duplicate": duplicate_detail,
            "impact": {
                "memory_count": len(duplicate_detail.get("memories", [])),
                "category_titles": sorted(
                    category.name for category in categories.values() if category.entity_id == duplicate_id
                ),
                "current_triple_count": sum(edge.valid_to is None for edge in triples.values()),
                "historical_triple_count": sum(edge.valid_to is not None for edge in triples.values()),
                "speaker_memory_count": sum(
                    item.speaker_id == f"entity:{duplicate_id}" for item in items.values()
                ),
                "aliases": state["aliases"],
                "source_refs": state["property_updates"]["source_refs"],
            },
            "conflicts": state["conflicts"],
            "warnings": state["warnings"],
            "can_merge": not state["conflicts"],
        }

    def graph_merge_entities(
        self,
        canonical_id: str,
        duplicate_id: str,
        *,
        where: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = self._entity_scope(where)
        store = self._get_database()
        with store._sessions.session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            with store.entity_repo.write_lock():
                canonical, duplicate, entities = self._load_entity_merge(
                    canonical_id, duplicate_id, scope, session=session
                )
                state = self._entity_merge_state(canonical, duplicate, entities)
                if state["conflicts"]:
                    raise EntityMergeConflictError(
                        _entity_conflict_payload(canonical),
                        _entity_conflict_payload(duplicate),
                        state["conflicts"],
                    )
                store.entity_repo.update(
                    canonical_id,
                    where=scope,
                    aliases=state["aliases"],
                    property_updates=state["property_updates"],
                    session=session,
                )
                store.triple_repo.replace_entity_id(duplicate_id, canonical_id, scope, session)
                for category in store.memory_category_repo.list_categories(scope, session=session).values():
                    if category.entity_id == duplicate_id:
                        store.memory_category_repo.update_category(
                            category_id=category.id, entity_id=canonical_id, where=scope, session=session
                        )
                store.memory_item_repo.replace_speaker_id(
                    f"entity:{duplicate_id}", f"entity:{canonical_id}", scope, session
                )
                store.entity_repo.delete(duplicate_id, where=scope, session=session)
                session.commit()
        return self.graph_atomic_entity(canonical_id, where=scope) or {}

    def _graph_set_entity_mention(
        self,
        memory_id: str,
        entity_id: str,
        *,
        attached: bool,
        where: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        store = self._get_database()
        with store._sessions.session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            with store.entity_repo.write_lock():
                if memory_id not in store.memory_item_repo.list_items_by_ids({memory_id}, where, session=session):
                    return None
                entities = store.entity_repo.list_by_ids({entity_id}, where, session=session)
                if not entities:
                    raise KeyError(f"entity not found in scope: {entity_id}")
                if attached and entity_is_ignored(entities[0]):
                    raise EntityActionConflictError(["Entity is ignored"])
                if attached:
                    store.triple_repo.add(
                        Triple(
                            subject_id=memory_id,
                            subject_kind="memory",
                            predicate="mentions",
                            object_id=entity_id,
                            object_kind="entity",
                        ),
                        user_data=where,
                        session=session,
                    )
                else:
                    store.triple_repo.invalidate(
                        memory_id,
                        "mentions",
                        entity_id,
                        where,
                        session=session,
                    )
                session.commit()
        return self.graph_memory(f"memory:{memory_id}", where=where)

    def graph_attach_entity(
        self,
        memory_id: str,
        entity_id: str,
        *,
        where: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        return self._graph_set_entity_mention(memory_id, entity_id, attached=True, where=where)

    def graph_detach_entity(
        self,
        memory_id: str,
        entity_id: str,
        *,
        where: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        return self._graph_set_entity_mention(memory_id, entity_id, attached=False, where=where)

    def graph_set_category_membership(
        self,
        category_id: str,
        memory_id: str,
        *,
        attached: bool,
        expected_displayed_summary: str,
        where: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = {key: str(where.get(key) or "").strip() for key in ("user_id", "soul_id")}
        if not all(scope.values()):
            raise ValueError("category membership requires user_id and soul_id")
        store = self._get_database()
        with store._sessions.session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            category = store.memory_category_repo.list_categories(scope, session=session).get(category_id)
            if category is None:
                raise KeyError(f"category not found in scope: {category_id}")
            if category.kind not in DOSSIER_KINDS:
                raise ValueError("category is not a current dossier")
            if category.anchor_role is not None:
                raise DossierMembershipConflictError("anchor dossier membership is consolidation-owned")
            if str(category.summary or category.description or "") != expected_displayed_summary:
                raise DossierRevisionStaleError("summary_snapshot_stale")

            all_items = store.memory_item_repo.list_items_by_ids(
                {memory_id},
                scope,
                include_superseded=True,
                include_merged=True,
                session=session,
            )
            item = all_items.get(memory_id)
            if item is None:
                raise KeyError(f"memory not found in scope: {memory_id}")
            if attached and memory_id not in store.memory_item_repo.list_items_by_ids(
                {memory_id}, scope, session=session
            ):
                raise ValueError("only active memories may be attached")

            relation_scope = {**scope, "category_id": category_id}
            relations = store.category_item_repo.list_relations(relation_scope, session=session)
            linked = any(
                relation.category_id == category_id and relation.item_id == memory_id
                for relation in relations
            )
            if attached == linked:
                session.commit()
            else:
                if not attached and item.memory_ref in self.extract_memory_refs(category.summary or ""):
                    raise DossierMembershipConflictError(
                        f"remove {self.format_memory_ref(item.memory_ref)} from dossier text first"
                    )
                if attached:
                    store.category_item_repo.link_item_category(memory_id, category_id, scope, session=session)
                else:
                    store.category_item_repo.unlink_item_category(
                        memory_id, category_id, scope, session=session
                    )
                remaining_ids = {
                    relation.item_id
                    for relation in store.category_item_repo.list_relations(
                        relation_scope, session=session
                    )
                }
                active_items = store.memory_item_repo.list_items_by_ids(
                    remaining_ids, scope, session=session
                )
                last_evidence_at = max(
                    (
                        when
                        for item in active_items.values()
                        if (when := _utc(item.happened_at or item.created_at)) is not None
                    ),
                    default=None,
                )
                store.memory_category_repo.update_category(
                    category_id=category_id,
                    last_evidence_at=last_evidence_at,
                    where=scope,
                    session=session,
                )
                session.commit()

        store.memory_category_repo.list_categories(scope)
        store.category_item_repo.refresh_category_relations(category_id, scope)
        detail = self.graph_memory(f"category:{category_id}", where=scope)
        if detail is None:
            raise KeyError(f"category not found in scope: {category_id}")
        return detail

    def graph_atomic_canvas_source(
        self,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int = 500,
        atom_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        store = self._get_database()
        store_done = perf_counter()
        limit = max(1, min(int(limit or 500), 1000))
        requested_memory_ids: set[str] | None = None
        requested_category_ids: set[str] | None = None
        if atom_ids is not None:
            requested_memory_ids = {
                atom_id.removeprefix("memory:")
                for atom_id in atom_ids
                if atom_id.startswith("memory:")
            }
            requested_category_ids = {
                atom_id.removeprefix("category:")
                for atom_id in atom_ids
                if atom_id.startswith("category:")
            }
        categories = store.memory_category_repo.list_categories(where)
        relations = store.category_item_repo.list_relations(where)
        item_category_ids: dict[str, list[str]] = {}
        for rel in relations:
            if requested_memory_ids is not None and rel.item_id not in requested_memory_ids:
                continue
            if rel.category_id in categories:
                item_category_ids.setdefault(rel.item_id, []).append(rel.category_id)
        taxonomy_done = perf_counter()

        atoms: list[dict[str, Any]] = []
        if requested_memory_ids is not None:
            items = store.memory_item_repo.list_items_by_ids(requested_memory_ids, where, include_embeddings=True)
            total_memory_count = len(items)
        else:
            items, total_memory_count = store.memory_item_repo.list_canvas_items(where, limit=limit)
        for item in items.values():
            if not item.embedding:
                continue
            category_pairs = sorted(
                ((cat_id, categories[cat_id].name) for cat_id in item_category_ids.get(item.id, [])),
                key=lambda pair: pair[1].lower(),
            )
            atoms.append({
                "id": f"memory:{item.id}",
                "title": _label(item.summary),
                "embedding": item.embedding,
                "primary_tag": category_pairs[0][1] if category_pairs else None,
                "tag_count": len(category_pairs),
                "tag_ids": [f"category:{cat_id}" for cat_id, _name in category_pairs],
                "source_url": None,
                "updated_at": _iso(item.updated_at),
            })

        category_values = list(
            (categories[category_id] for category_id in requested_category_ids if category_id in categories)
            if requested_category_ids is not None
            else categories.values()
        )
        for category in category_values:
            if not category.embedding:
                continue
            atoms.append({
                "id": f"category:{category.id}",
                "title": category.name,
                "embedding": category.embedding,
                "primary_tag": category.name,
                "tag_count": 0,
                "tag_ids": [f"category:{category.id}"],
                "source_url": None,
                "updated_at": _iso(category.updated_at),
            })

        atoms.sort(key=lambda atom: (atom.get("updated_at") or "", atom["id"]), reverse=True)
        page = atoms[:limit]
        memory_ids = {str(atom["id"]).removeprefix("memory:") for atom in page if str(atom["id"]).startswith("memory:")}
        atoms_done = perf_counter()
        canvas_triples = store.triple_repo.list_edges_for_memories(
            memory_ids,
            ["mentions", *SEMANTIC_PREDICATES],
            where=where,
        )
        entities_by_memory = self._entity_pairs_by_memory(memory_ids, where, triples=canvas_triples)
        for atom in page:
            memory_id = str(atom["id"]).removeprefix("memory:")
            pairs = entities_by_memory.get(memory_id, [])
            atom["entity_ids"] = [entity_id for entity_id, _name in pairs]
            atom["entity_names"] = [name for _entity_id, name in pairs]
        graph_done = perf_counter()

        edges: dict[str, dict[str, Any]] = {}
        for edge in _atomic_similarity_edges(page):
            edges[f"similarity:{edge['source']}:{edge['target']}"] = edge
        similarity_done = perf_counter()

        for triple in canvas_triples:
            if triple.predicate not in SEMANTIC_PREDICATES:
                continue
            if triple.subject_kind != "memory" or triple.object_kind != "memory":
                continue
            if triple.subject_id not in memory_ids or triple.object_id not in memory_ids:
                continue
            edge_id = f"semantic:{triple.subject_id}:{triple.predicate}:{triple.object_id}"
            confidence = getattr(triple, "confidence", None)
            edges[edge_id] = {
                "source": f"memory:{triple.subject_id}",
                "target": f"memory:{triple.object_id}",
                "weight": float(confidence) if confidence is not None else 0.7,
                "kind": "triple",
                "predicate": triple.predicate,
            }
        total_count = len(atoms) if atom_ids is not None else total_memory_count + sum(
            category.embedding is not None for category in category_values
        )
        for atom in page:
            atom["embedding_f32_le_b64"] = _pack_embedding(atom.pop("embedding"))
        finished = perf_counter()
        return {
            "atoms": page,
            "edges": list(edges.values()),
            "count": len(page),
            "total_count": total_count,
            "timing_ms": {
                "store": round((store_done - started) * 1000, 2),
                "taxonomy": round((taxonomy_done - store_done) * 1000, 2),
                "atoms": round((atoms_done - taxonomy_done) * 1000, 2),
                "graph": round((graph_done - atoms_done) * 1000, 2),
                "similarity": round((similarity_done - graph_done) * 1000, 2),
                "finalize": round((finished - similarity_done) * 1000, 2),
                "total": round((finished - started) * 1000, 2),
            },
        }

    def graph_atomic_neighborhood(
        self,
        item_id: str,
        *,
        where: Mapping[str, Any] | None = None,
        depth: int = 1,
        min_similarity: float = 0.5,
        similarity_limit: int = 5,
    ) -> dict[str, Any] | None:
        store = self._get_database()
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind != "memory":
            center = self.graph_memory(item_id, where=where)
            if center is None:
                return None
            return {"center_atom_id": center["id"], "nodes": [center | {"depth": 0}], "edges": []}

        depth = max(0, int(depth or 1))
        similarity_limit = max(1, min(int(similarity_limit or 5), 20))
        center_id = raw_id
        items = store.memory_item_repo.list_items(where)
        center_item = items.get(center_id)
        if center_item is None:
            return None

        categories = store.memory_category_repo.list_categories(where)
        item_category_ids: dict[str, list[str]] = {}
        for rel in store.category_item_repo.list_relations(where):
            if rel.category_id in categories:
                item_category_ids.setdefault(rel.item_id, []).append(rel.category_id)

        def memory_node(memory_id: str, node_depth: int) -> dict[str, Any] | None:
            item = items.get(memory_id)
            if item is None:
                return None
            category_ids = item_category_ids.get(memory_id, [])
            return self._memory_node(
                item,
                category_names=[categories[cat_id].name for cat_id in category_ids],
                category_ids=category_ids,
            ) | {"depth": node_depth}

        center = memory_node(center_id, 0)
        if center is None:
            return None
        if depth == 0:
            return {"center_atom_id": center["id"], "nodes": [center], "edges": []}

        neighbor_ids: set[str] = set()
        edges: list[dict[str, Any]] = []
        for predicate in SEMANTIC_PREDICATES:
            triples = store.triple_repo.get_edges_from(center_id, predicate=predicate, where=where)
            triples += store.triple_repo.get_edges_to(center_id, predicate=predicate, where=where)
            for triple in triples:
                if triple.subject_kind != "memory" or triple.object_kind != "memory":
                    continue
                if triple.subject_id == triple.object_id:
                    continue
                neighbor_ids.update({triple.subject_id, triple.object_id} - {center_id})
                edges.append({
                    "source_id": f"memory:{triple.subject_id}",
                    "target_id": f"memory:{triple.object_id}",
                    "edge_type": "semantic",
                    "strength": 0.7,
                    "shared_tag_count": 0,
                    "similarity_score": None,
                })

        nodes = [center]
        for raw_id in sorted(neighbor_ids):
            node = memory_node(raw_id, 1)
            if node is not None:
                nodes.append(node)
        center_embedding = getattr(center_item, "embedding", None)
        if center_embedding:
            seen = {node["id"] for node in nodes}
            for raw_id, score in cosine_topk(
                center_embedding,
                (
                    (item.id, getattr(item, "embedding", None))
                    for item in items.values()
                    if item.id != center_id
                ),
                k=similarity_limit,
            ):
                if score < min_similarity:
                    continue
                node = memory_node(raw_id, 1)
                if node is None or node["id"] in seen:
                    continue
                nodes.append(node)
                seen.add(node["id"])
                edges.append({
                    "source_id": center["id"],
                    "target_id": node["id"],
                    "edge_type": "semantic",
                    "strength": float(score),
                    "shared_tag_count": 0,
                    "similarity_score": float(score),
                })
        return {"center_atom_id": center["id"], "nodes": nodes, "edges": edges}

    def graph_atomic_similar(
        self,
        item_id: str,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> list[dict[str, Any]] | None:
        store = self._get_database()
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind != "memory":
            return []

        limit = max(1, min(int(limit or 5), 20))
        items = store.memory_item_repo.list_items(where)
        center_item = items.get(raw_id)
        if center_item is None:
            return None

        center_embedding = getattr(center_item, "embedding", None)
        if not center_embedding:
            return []

        categories = store.memory_category_repo.list_categories(where)
        item_category_ids: dict[str, list[str]] = {}
        for rel in store.category_item_repo.list_relations(where):
            if rel.category_id in categories:
                item_category_ids.setdefault(rel.item_id, []).append(rel.category_id)

        nodes = []
        for similar_id, score in cosine_topk(
            center_embedding,
            (
                (item.id, getattr(item, "embedding", None))
                for item in items.values()
                if item.id != raw_id
            ),
            k=limit,
        ):
            if score < min_similarity:
                continue
            category_ids = item_category_ids.get(similar_id, [])
            node = self._memory_node(
                items[similar_id],
                category_names=[categories[cat_id].name for cat_id in category_ids],
                category_ids=category_ids,
            )
            node["similarity_score"] = float(score)
            nodes.append(node)
        return nodes

    async def graph_update_memory_summary(
        self,
        item_id: str,
        *,
        summary: str,
        where: Mapping[str, Any] | None = None,
        edited_by: str | None = None,
        approved: bool = False,
    ) -> dict[str, Any] | None:
        store = self._get_database()
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind != "memory" or not raw_id:
            raise ValueError("only memory summaries are editable")

        summary = str(summary or "").strip()
        if not summary:
            raise ValueError("summary is required")

        current = store.memory_item_repo.list_items_by_ids({raw_id}, where=where).get(raw_id)
        if current is None:
            return None
        if current.summary.strip() == summary:
            if approved:
                store.memory_item_repo.approve_item(raw_id, where=where)
            return self.graph_memory(f"memory:{raw_id}", where=where)

        embedding = (await self._select_embedding_client(None).embed([summary]))[0]
        store.memory_item_repo.update_summary_with_history(
            item_id=raw_id,
            summary=summary,
            embedding=embedding,
            where=where,
            edited_by=edited_by,
            approved=approved,
        )
        return self.graph_memory(f"memory:{raw_id}", where=where)

    async def graph_update_category_summary(
        self,
        item_id: str,
        *,
        summary: str | None = None,
        title: str | None = None,
        description: str | None = None,
        where: Mapping[str, Any] | None = None,
        edited_by: str | None = None,
        approved: bool = False,
    ) -> dict[str, Any] | None:
        store = self._get_database()
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "category", kind
        if kind != "category" or not raw_id:
            raise ValueError("only category summaries are editable")

        clean = str(summary or "").strip() if summary is not None else None
        clean_title = str(title or "").strip() if title is not None else None
        clean_description = str(description or "").strip() if description is not None else None
        if summary is not None and not clean:
            raise ValueError("summary is required")
        if title is not None and not clean_title:
            raise ValueError("title is required")
        if description is not None and not clean_description:
            raise ValueError("description is required")
        if clean is None and clean_title is None and clean_description is None:
            raise ValueError("title, description, or summary is required")
        current = store.memory_category_repo.list_categories(where).get(raw_id)
        if current is None:
            msg = f"Category with id {raw_id} not found"
            raise KeyError(msg)
        summary_changed = clean is not None and str(current.summary or "").strip() != clean
        if clean_title is not None or clean_description is not None or summary_changed:
            await self.update_dossier(
                raw_id,
                where or {},
                name=clean_title,
                description=clean_description,
                summary=clean if summary_changed else None,
            )
        if summary_changed:
            try:
                append_category_summary_journal(
                    category_id=raw_id,
                    summary_before=str(current.summary or ""),
                    summary_after=clean,
                    scope=where,
                    edited_by=edited_by,
                )
            except Exception:
                logger.exception("Failed to journal committed category edit %s", raw_id)
        if approved:
            store.memory_category_repo.approve_category_summary(raw_id, where=where)
        return self.graph_memory(f"category:{raw_id}", where=where)

    def graph_list_pending(self, *, where: Mapping[str, Any] | None = None) -> dict[str, Any]:
        store = self._get_database()
        items = store.memory_item_repo.list_items(where)
        categories = store.memory_category_repo.list_categories(where)
        active_category_ids = self._graph_active_category_ids(where) if categories else set()
        relations = store.category_item_repo.list_relations(where)
        category_names_by_item: dict[str, list[str]] = {}
        category_ids_by_item: dict[str, list[str]] = {}
        for rel in relations:
            category = categories.get(rel.category_id)
            if category is not None:
                category_names_by_item.setdefault(rel.item_id, []).append(category.name)
                category_ids_by_item.setdefault(rel.item_id, []).append(rel.category_id)

        pending_source = [item for item in items.values() if getattr(item, "approved_at", None) is None]
        clusters = _cluster_by_embedding(pending_source)
        # Reorder so cluster members are adjacent; non-clustered items keep their place after clusters.
        original_order = {item.id: idx for idx, item in enumerate(pending_source)}
        pending_source.sort(
            key=lambda item: (
                item.id not in clusters,
                clusters.get(item.id, {}).get("_group_key", ""),
                original_order[item.id],
            )
        )
        pending_items = []
        for item in pending_source:
            node = self._memory_node(
                item,
                category_names=category_names_by_item.get(item.id, []),
                category_ids=category_ids_by_item.get(item.id, []),
            )
            cluster_info = clusters.get(item.id, {})
            if cluster_info:
                node["similar_to"] = cluster_info["similar_to"]
                node["similarity"] = cluster_info["similarity"]
            pending_items.append(node)
        pending_categories = [
            self._scoped_category_node(
                category,
                where=where,
                active_category_ids=active_category_ids,
            )
            for category in categories.values()
            if category.summary is not None
            and (
                category.summary != getattr(category, "approved_summary", None)
                or (
                    category.kind in DOSSIER_KINDS
                    and category.description != getattr(category, "approved_description", None)
                )
            )
        ]
        return {"items": pending_items, "categories": pending_categories}

    def graph_approve_memory(self, item_id: str, *, where: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind != "memory" or not raw_id:
            raise ValueError("only memory approvals are supported")
        try:
            self._get_database().memory_item_repo.approve_item(raw_id, where=where)
        except KeyError:
            return None
        return self.graph_memory(f"memory:{raw_id}", where=where)

    def graph_approve_category(self, item_id: str, *, where: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "category", kind
        if kind != "category" or not raw_id:
            raise ValueError("only category approvals are supported")
        try:
            self._get_database().memory_category_repo.approve_category_summary(raw_id, where=where)
        except KeyError:
            return None
        return self.graph_memory(f"category:{raw_id}", where=where)

    def graph_delete_memory(self, item_id: str, *, where: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        kind, _, raw_id = str(item_id or "").partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind != "memory" or not raw_id:
            raise ValueError("only memory deletion is supported")
        store = self._get_database()
        categories = store.memory_category_repo.list_categories(where)
        relations = store.category_item_repo.list_relations(where)
        try:
            deleted = store.memory_item_repo.hard_delete_item(raw_id, where=where)
        except KeyError:
            return None
        category_names = [
            categories[rel.category_id].name
            for rel in relations
            if rel.item_id == deleted.id and rel.category_id in categories
        ]
        category_ids = [
            rel.category_id
            for rel in relations
            if rel.item_id == deleted.id and rel.category_id in categories
        ]
        return self._memory_node(deleted, category_names=category_names, category_ids=category_ids)

    async def graph_search(
        self,
        query: str,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int = 5,
        mode: str = "hybrid",
        since_days: int | None = None,
        memory_only: bool = False,
        exclude_entity_id: str | None = None,
        exclude_category_id: str | None = None,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")
        mode = str(mode or "hybrid").strip().lower()
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("mode must be keyword, semantic, or hybrid")
        store = self._get_database()
        limit = max(1, min(int(limit or 5), 20))
        scope = dict(where or {})
        exclude_entity_id = exclude_entity_id.removeprefix("entity:") if exclude_entity_id else None
        exclude_category_id = (
            exclude_category_id.removeprefix("category:") if exclude_category_id else None
        )

        if memory_ref := _normalize_search_memory_ref(query):
            if exclude_entity_id and not store.entity_repo.list_by_ids({exclude_entity_id}, scope):
                raise ValueError("excluded entity not found in scope")
            if (
                exclude_category_id
                and exclude_category_id not in store.memory_category_repo.list_categories(scope)
            ):
                raise ValueError("excluded category not found in scope")
            try:
                resolved = self.resolve_memory_ref(memory_ref, scope)
            except KeyError:
                return {"nodes": [], "limit": limit, "count": 0}
            node = self.graph_memory(f"memory:{resolved.id}", where=scope)
            if node is None:
                return {"nodes": [], "limit": limit, "count": 0}
            if since_days is not None:
                cutoff = datetime.now(UTC) - timedelta(days=max(1, int(since_days)))
                when = _utc(resolved.happened_at or resolved.created_at)
                if when is None or when < cutoff:
                    return {"nodes": [], "limit": limit, "count": 0}
            if exclude_entity_id and any(
                edge.object_kind == "entity" and edge.object_id == exclude_entity_id
                for edge in store.triple_repo.get_edges_from(
                    resolved.id, predicate="mentions", where=scope
                )
            ):
                return {"nodes": [], "limit": limit, "count": 0}
            if exclude_category_id and exclude_category_id in node.get("category_ids", []):
                return {"nodes": [], "limit": limit, "count": 0}
            node["score"] = 1.0
            return {"nodes": [node], "limit": limit, "count": 1}

        pool = store.memory_item_repo.list_items(scope)
        categories = store.memory_category_repo.list_categories(scope)
        relations = store.category_item_repo.list_relations(scope)
        excluded_ids: set[str] = set()
        if exclude_entity_id:
            if not store.entity_repo.list_by_ids({exclude_entity_id}, scope):
                raise ValueError("excluded entity not found in scope")
            excluded_ids.update(
                edge.subject_id
                for edge in store.triple_repo.list_edges_for_memories(pool, {"mentions"}, scope)
                if edge.subject_kind == "memory"
                and edge.object_kind == "entity"
                and edge.object_id == exclude_entity_id
            )
        if exclude_category_id:
            if exclude_category_id not in categories:
                raise ValueError("excluded category not found in scope")
            excluded_ids.update(rel.item_id for rel in relations if rel.category_id == exclude_category_id)
        if excluded_ids:
            pool = {item_id: item for item_id, item in pool.items() if item_id not in excluded_ids}

        if since_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=max(1, int(since_days)))
            pool = {
                item_id: item
                for item_id, item in pool.items()
                if (when := _utc(item.happened_at or item.created_at)) is not None and when >= cutoff
            }
        search_categories = {} if memory_only else categories
        active_category_ids = self._graph_active_category_ids(scope) if search_categories else set()
        scores: dict[str, float] = {}
        category_scores: dict[str, float] = {}
        if mode in {"keyword", "hybrid"}:
            for rank, (item_id, _score) in enumerate(
                store.memory_item_repo.fts_search_items(query, limit, pool_ids=set(pool)),
                start=1,
            ):
                scores[item_id] = max(scores.get(item_id, 0.0), 1.0 / rank)

        if mode in {"semantic", "hybrid"}:
            query_vec = (await self._select_embedding_client(None).embed([query]))[0]
            for item_id, score in cosine_topk(query_vec, ((item.id, item.embedding) for item in pool.values()), k=limit):
                if score <= 0:
                    continue
                scores[item_id] = max(scores.get(item_id, 0.0), float(score))
            for category_id, score in cosine_topk(
                query_vec,
                ((category.id, category.embedding) for category in search_categories.values()),
                k=limit,
            ):
                if score <= 0:
                    continue
                category_scores[category_id] = max(category_scores.get(category_id, 0.0), float(score))

        category_names_by_item: dict[str, list[str]] = {item_id: [] for item_id in scores}
        category_ids_by_item: dict[str, list[str]] = {item_id: [] for item_id in scores}
        for rel in relations:
            if rel.item_id in category_names_by_item and rel.category_id in categories:
                category_names_by_item[rel.item_id].append(categories[rel.category_id].name)
                category_ids_by_item[rel.item_id].append(rel.category_id)

        nodes = []
        for item_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]:
            node = self._memory_node(
                pool[item_id],
                category_names=category_names_by_item.get(item_id, []),
                category_ids=category_ids_by_item.get(item_id, []),
            )
            node["score"] = score
            nodes.append(node)
        if mode in {"keyword", "hybrid"}:
            query_lower = query.lower()
            for category in search_categories.values():
                text = " ".join([category.name or "", category.summary or "", category.description or ""]).lower()
                if query_lower in text:
                    category_scores[category.id] = max(category_scores.get(category.id, 0.0), 1.0)
        for category_id, score in category_scores.items():
            if category_id in categories:
                node = self._scoped_category_node(
                    categories[category_id],
                    where=scope,
                    active_category_ids=active_category_ids,
                )
                node["score"] = score
                nodes.append(node)
        nodes.sort(key=lambda node: node.get("score", 0.0), reverse=True)
        nodes = nodes[:limit]
        return {"nodes": nodes, "limit": limit, "count": len(nodes)}

    def graph_recent(
        self,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int = 200,
        semantic_edges_per_memory: int = 3,
    ) -> dict[str, Any]:
        store = self._get_database()
        scope = dict(where or {})
        limit = max(1, min(int(limit or 200), 500))
        per_memory = max(0, min(int(semantic_edges_per_memory or 0), 10))

        recent_items = store.memory_item_repo.list_recent_items(scope, limit=limit)
        recent = list(recent_items.values())
        selected_ids = {item.id for item in recent}

        edges: dict[str, dict[str, Any]] = {}
        if per_memory:
            for seed_id in list(selected_ids):
                count = 0
                for predicate in SEMANTIC_PREDICATES:
                    triples = store.triple_repo.get_edges_from(seed_id, predicate=predicate, where=scope)
                    triples += store.triple_repo.get_edges_to(seed_id, predicate=predicate, where=scope)
                    for triple in triples:
                        if triple.subject_kind != "memory" or triple.object_kind != "memory":
                            continue
                        if triple.subject_id == triple.object_id:
                            continue
                        selected_ids.update({triple.subject_id, triple.object_id})
                        edge_id = f"semantic:{triple.subject_id}:{triple.predicate}:{triple.object_id}"
                        edges[edge_id] = {
                            "id": edge_id,
                            "source": f"memory:{triple.subject_id}",
                            "target": f"memory:{triple.object_id}",
                            "kind": "semantic",
                            "predicate": triple.predicate,
                            "weight": 0.7,
                        }
                        count += 1
                        if count >= per_memory:
                            break
                    if count >= per_memory:
                        break

        all_items = recent_items | store.memory_item_repo.list_items_by_ids(selected_ids - set(recent_items), scope)
        selected_ids = set(all_items)
        edges = {
            edge_id: edge
            for edge_id, edge in edges.items()
            if edge["source"].removeprefix("memory:") in selected_ids
            and edge["target"].removeprefix("memory:") in selected_ids
        }

        categories = store.memory_category_repo.list_categories(scope)
        active_category_ids = self._graph_active_category_ids(scope) if categories else set()
        relations = store.category_item_repo.list_relations(scope)
        category_names_by_item: dict[str, list[str]] = {item_id: [] for item_id in selected_ids}
        category_ids: set[str] = set()
        for rel in relations:
            if rel.item_id not in selected_ids or rel.category_id not in categories:
                continue
            category = categories[rel.category_id]
            category_ids.add(category.id)
            category_names_by_item.setdefault(rel.item_id, []).append(category.name)
            edge_id = f"category:{rel.item_id}:{category.id}"
            edges[edge_id] = {
                "id": edge_id,
                "source": f"memory:{rel.item_id}",
                "target": f"category:{category.id}",
                "kind": "category",
                "predicate": "in_category",
                "weight": 0.45,
            }

        entities = {
            entity.id: entity
            for entity in store.entity_repo.list_all(scope)
            if not entity_is_ignored(entity)
        }
        entity_ids: set[str] = set()
        for memory_id in list(selected_ids):
            for triple in store.triple_repo.get_edges_from(memory_id, predicate="mentions", where=scope):
                if triple.object_kind != "entity" or triple.object_id not in entities:
                    continue
                entity_ids.add(triple.object_id)
                edge_id = f"mentions:{memory_id}:{triple.object_id}"
                edges[edge_id] = {
                    "id": edge_id,
                    "source": f"memory:{memory_id}",
                    "target": f"entity:{triple.object_id}",
                    "kind": "entity",
                    "predicate": "mentions",
                    "weight": 0.35,
                }

        nodes = [
            self._memory_node(
                all_items[item_id],
                category_names=category_names_by_item.get(item_id, []),
                category_ids=[
                    rel.category_id
                    for rel in relations
                    if rel.item_id == item_id and rel.category_id in categories
                ],
            )
            for item_id in selected_ids
            if item_id in all_items
        ]
        nodes.extend(
            self._scoped_category_node(
                categories[cat_id],
                where=scope,
                active_category_ids=active_category_ids,
            )
            for cat_id in category_ids
            if cat_id in categories
        )
        nodes.extend(self._entity_node(entities[entity_id]) for entity_id in entity_ids if entity_id in entities)

        nodes.sort(key=lambda node: (node["kind"], node.get("happened_at") or "", node["id"]))
        return {"nodes": nodes, "edges": list(edges.values()), "limit": limit, "count": len(nodes)}

    def _entity_pairs_by_memory(
        self,
        memory_ids: Iterable[str],
        where: Mapping[str, Any] | None,
        *,
        triples: Iterable[Any] | None = None,
    ) -> dict[str, list[tuple[str, str]]]:
        store = self._get_database()
        ids = set(memory_ids)
        mention_triples: dict[str, list[Any]] = {}
        source = (
            triples
            if triples is not None
            else store.triple_repo.list_edges_for_memories(ids, ["mentions"], where=where)
        )
        for triple in source:
            if triple.predicate == "mentions" and triple.subject_id in ids and triple.object_kind == "entity":
                mention_triples.setdefault(triple.subject_id, []).append(triple)
        entities = {
            entity.id: entity
            for entity in store.entity_repo.list_by_ids(
                {triple.object_id for rows in mention_triples.values() for triple in rows},
                where,
            )
            if not entity_is_ignored(entity)
        }
        return {
            memory_id: sorted(
                {(f"entity:{entities[row.object_id].id}", entities[row.object_id].name) for row in rows if row.object_id in entities},
                key=lambda pair: pair[1].lower(),
            )
            for memory_id, rows in mention_triples.items()
        }

    @staticmethod
    def _memory_node(
        item: Any,
        *,
        category_names: list[str],
        category_ids: list[str] | None = None,
        entity_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        salience = max(
            [v for v in (item.reflection_salience, item.emotional_intensity) if isinstance(v, int | float)],
            default=None,
        )
        category_pairs = sorted(set(zip(category_ids or [], category_names, strict=False)), key=lambda pair: pair[1].lower())
        node = {
            "id": f"memory:{item.id}",
            "kind": "memory",
            "memory_id": item.id,
            "label": _label(item.summary),
            "summary": item.summary,
            "memory_type": item.memory_type,
            "happened_at": _iso(item.happened_at),
            "created_at": _iso(item.created_at),
            "updated_at": _iso(item.updated_at),
            "approved_at": _iso(getattr(item, "approved_at", None)),
            "salience": salience,
            "category_ids": [pair[0] for pair in category_pairs],
            "category_names": [pair[1] for pair in category_pairs],
        }
        if entity_pairs is not None:
            node["entity_ids"] = [pair[0] for pair in entity_pairs]
            node["entity_names"] = [pair[1] for pair in entity_pairs]
        return node

    @staticmethod
    def _category_node(
        category: Any,
        *,
        active: bool,
        citations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": f"category:{category.id}",
            "kind": "category",
            "category_id": category.id,
            "label": category.name,
            "summary": category.summary or category.description,
            "description": category.description,
            "previous_description": getattr(category, "previous_description", None),
            "approved_description": getattr(category, "approved_description", None),
            "previous_summary": getattr(category, "previous_summary", None),
            "approved_summary": getattr(category, "approved_summary", None),
            "category_kind": getattr(category, "kind", None),
            "lore_subtype": getattr(category, "lore_subtype", None),
            "anchor_role": getattr(category, "anchor_role", None),
            "active": active,
            "last_evidence_at": _iso(getattr(category, "last_evidence_at", None)),
            "last_revised_at": _iso(getattr(category, "last_revised_at", None)),
            "citations": citations,
            "created_at": _iso(category.created_at),
            "updated_at": _iso(category.updated_at),
            "category_names": [],
        }

    @staticmethod
    def _atomic_atom(node: dict[str, Any]) -> dict[str, Any]:
        text = str(node.get("summary") or "")
        timestamp = node.get("updated_at") or node.get("happened_at") or node.get("created_at") or datetime.now(UTC).isoformat()
        tags = [
            {
                "id": f"category:{cat_id}",
                "name": name,
                "parent_id": None,
                "created_at": "1970-01-01T00:00:00Z",
                "is_autotag_target": False,
                "autotag_description": "",
            }
            for cat_id, name in zip(node.get("category_ids", []), node.get("category_names", []), strict=False)
        ]
        atom = {
            "id": node["id"],
            "title": node.get("label") or node["id"],
            "snippet": text,
            "source_url": None,
            "source": node.get("memory_type") or node.get("kind"),
            "published_at": node.get("happened_at"),
            "created_at": node.get("created_at") or timestamp,
            "updated_at": timestamp,
            "embedding_status": "complete",
            "tagging_status": "skipped",
            "embedding_error": None,
            "tagging_error": None,
            "tags": tags,
        }
        for field in (
            "description",
            "approved_description",
            "approved_summary",
            "category_kind",
            "lore_subtype",
            "anchor_role",
            "active",
            "last_evidence_at",
            "last_revised_at",
            "citations",
        ):
            if field in node:
                atom[field] = node[field]
        return atom

    @staticmethod
    def _atomic_tag(category: Any, count: int, *, active: bool) -> dict[str, Any]:
        return {
            "id": f"category:{category.id}",
            "name": category.name,
            "parent_id": None,
            "created_at": _iso(category.created_at) or "1970-01-01T00:00:00Z",
            "is_autotag_target": False,
            "autotag_description": "",
            "atom_count": count,
            "children_total": 0,
            "children": [],
            "category_kind": getattr(category, "kind", None),
            "anchor_role": getattr(category, "anchor_role", None),
            "active": active,
        }

    @staticmethod
    def _entity_node(entity: Any) -> dict[str, Any]:
        return {
            "id": f"entity:{entity.id}",
            "kind": "entity",
            "entity_id": entity.id,
            "label": entity.name,
            "summary": entity.entity_type,
            "created_at": _iso(entity.created_at),
            "updated_at": _iso(entity.updated_at),
            "category_names": [],
        }
