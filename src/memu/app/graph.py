from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from memu.app.category_summary_journal import update_category_summary_with_journal
from memu.database.vector import cosine_topk

SEMANTIC_PREDICATES = ["caused_by", "evokes", "conflicts_with", "parallels", "shaped_by"]


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _label(text: str, limit: int = 80) -> str:
    clean = " ".join(str(text or "").split())
    return clean[: limit - 1] + "..." if len(clean) > limit else clean


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


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


class GraphMixin:
    def graph_memory(self, item_id: str, *, where: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        store = self._get_database()
        item_id = str(item_id or "")
        kind, _, raw_id = item_id.partition(":")
        if not raw_id:
            kind, raw_id = "memory", kind
        if kind == "category":
            category = store.memory_category_repo.list_categories(where).get(raw_id)
            return self._category_node(category) if category is not None else None
        if kind == "entity":
            entity = next((entity for entity in store.entity_repo.list_all(where) if entity.id == raw_id), None)
            return self._entity_node(entity) if entity is not None else None

        items = store.memory_item_repo.list_items(where)
        item = items.get(raw_id)
        if item is None:
            return None

        categories = store.memory_category_repo.list_categories(where)
        relations = store.category_item_repo.list_relations(where)
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
        return self._memory_node(item, category_names=category_names, category_ids=category_ids)

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
            self._category_node(category)
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
        counts = dict.fromkeys(categories, 0)
        for rel in store.category_item_repo.list_relations(where):
            if rel.category_id in counts:
                counts[rel.category_id] += 1
        return [
            self._atomic_tag(category, counts[category.id])
            for category in sorted(categories.values(), key=lambda category: category.name.lower())
            if counts[category.id] >= min_count
        ]

    def graph_atomic_canvas_source(
        self,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int = 500,
        atom_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        store = self._get_database()
        limit = max(1, min(int(limit or 500), 1000))
        categories = store.memory_category_repo.list_categories(where)
        relations = store.category_item_repo.list_relations(where)
        item_category_ids: dict[str, list[str]] = {}
        for rel in relations:
            if rel.category_id in categories:
                item_category_ids.setdefault(rel.item_id, []).append(rel.category_id)

        atoms: list[dict[str, Any]] = []
        for item in store.memory_item_repo.list_items(where).values():
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

        for category in categories.values():
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

        if atom_ids is not None:
            atoms = [atom for atom in atoms if str(atom["id"]) in atom_ids]
        atoms.sort(key=lambda atom: (atom.get("updated_at") or "", atom["id"]), reverse=True)
        page = atoms[:limit]
        memory_ids = {str(atom["id"]).removeprefix("memory:") for atom in page if str(atom["id"]).startswith("memory:")}
        entities = {entity.id: entity for entity in store.entity_repo.list_all(where)}
        entities_by_memory: dict[str, list[tuple[str, str]]] = {}
        for memory_id in memory_ids:
            pairs = []
            for triple in store.triple_repo.get_edges_from(memory_id, predicate="mentions", where=where):
                if triple.object_kind != "entity" or triple.object_id not in entities:
                    continue
                entity = entities[triple.object_id]
                pairs.append((f"entity:{entity.id}", entity.name))
            entities_by_memory[memory_id] = sorted(set(pairs), key=lambda pair: pair[1].lower())
        for atom in page:
            memory_id = str(atom["id"]).removeprefix("memory:")
            pairs = entities_by_memory.get(memory_id, [])
            atom["entity_ids"] = [entity_id for entity_id, _name in pairs]
            atom["entity_names"] = [name for _entity_id, name in pairs]

        edges: dict[str, dict[str, Any]] = {}
        for edge in _atomic_similarity_edges(page):
            edges[f"similarity:{edge['source']}:{edge['target']}"] = edge

        for memory_id in memory_ids:
            for predicate in SEMANTIC_PREDICATES:
                triples = store.triple_repo.get_edges_from(memory_id, predicate=predicate, where=where)
                triples += store.triple_repo.get_edges_to(memory_id, predicate=predicate, where=where)
                for triple in triples:
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
        return {"atoms": page, "edges": list(edges.values()), "count": len(page), "total_count": len(atoms)}

    def graph_atomic_neighborhood(
        self,
        item_id: str,
        *,
        where: Mapping[str, Any] | None = None,
        depth: int = 1,
        min_similarity: float = 0.5,
    ) -> dict[str, Any] | None:
        store = self._get_database()
        center = self.graph_memory(item_id, where=where)
        if center is None:
            return None
        depth = max(0, int(depth or 1))
        if center["kind"] != "memory":
            return {"center_atom_id": center["id"], "nodes": [center | {"depth": 0}], "edges": []}
        if depth == 0:
            return {"center_atom_id": center["id"], "nodes": [center | {"depth": 0}], "edges": []}

        center_id = center["memory_id"]
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

        nodes = [center | {"depth": 0}]
        for raw_id in sorted(neighbor_ids):
            node = self.graph_memory(f"memory:{raw_id}", where=where)
            if node is not None:
                nodes.append(node | {"depth": 1})
        return {"center_atom_id": center["id"], "nodes": nodes, "edges": edges}

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
        summary: str,
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

        clean = str(summary or "").strip()
        if not clean:
            raise ValueError("summary is required")
        current = store.memory_category_repo.list_categories(where).get(raw_id)
        if current is None:
            msg = f"Category with id {raw_id} not found"
            raise KeyError(msg)
        if str(current.summary or "").strip() == clean:
            if approved:
                store.memory_category_repo.approve_category_summary(raw_id, where=where)
            return self.graph_memory(f"category:{raw_id}", where=where)

        update_category_summary_with_journal(
            store,
            category_id=raw_id,
            summary=clean,
            where=where,
            edited_by=edited_by,
        )
        if approved:
            store.memory_category_repo.approve_category_summary(raw_id, where=where)
        return self.graph_memory(f"category:{raw_id}", where=where)

    def graph_list_pending(self, *, where: Mapping[str, Any] | None = None) -> dict[str, Any]:
        store = self._get_database()
        items = store.memory_item_repo.list_items(where)
        categories = store.memory_category_repo.list_categories(where)
        relations = store.category_item_repo.list_relations(where)
        category_names_by_item: dict[str, list[str]] = {}
        category_ids_by_item: dict[str, list[str]] = {}
        for rel in relations:
            category = categories.get(rel.category_id)
            if category is not None:
                category_names_by_item.setdefault(rel.item_id, []).append(category.name)
                category_ids_by_item.setdefault(rel.item_id, []).append(rel.category_id)

        pending_items = [
            self._memory_node(
                item,
                category_names=category_names_by_item.get(item.id, []),
                category_ids=category_ids_by_item.get(item.id, []),
            )
            for item in items.values()
            if getattr(item, "approved_at", None) is None
        ]
        pending_categories = [
            self._category_node(category)
            for category in categories.values()
            if category.summary is not None and category.summary != getattr(category, "approved_summary", None)
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
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")
        mode = str(mode or "hybrid").strip().lower()
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("mode must be keyword, semantic, or hybrid")
        store = self._get_database()
        limit = max(1, min(int(limit or 5), 20))

        pool = store.memory_item_repo.list_items(where)
        categories = store.memory_category_repo.list_categories(where)
        if since_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=max(1, int(since_days)))
            pool = {
                item_id: item
                for item_id, item in pool.items()
                if (when := _utc(item.happened_at or item.created_at)) is not None and when >= cutoff
            }
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
                ((category.id, category.embedding) for category in categories.values()),
                k=limit,
            ):
                if score <= 0:
                    continue
                category_scores[category_id] = max(category_scores.get(category_id, 0.0), float(score))

        relations = store.category_item_repo.list_relations(where)
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
            for category in categories.values():
                text = " ".join([category.name or "", category.summary or "", category.description or ""]).lower()
                if query_lower in text:
                    category_scores[category.id] = max(category_scores.get(category.id, 0.0), 1.0)
        for category_id, score in category_scores.items():
            if category_id in categories:
                node = self._category_node(categories[category_id])
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

        entities = {entity.id: entity for entity in store.entity_repo.list_all(scope)}
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
        nodes.extend(self._category_node(categories[cat_id]) for cat_id in category_ids if cat_id in categories)
        nodes.extend(self._entity_node(entities[entity_id]) for entity_id in entity_ids if entity_id in entities)

        nodes.sort(key=lambda node: (node["kind"], node.get("happened_at") or "", node["id"]))
        return {"nodes": nodes, "edges": list(edges.values()), "limit": limit, "count": len(nodes)}

    @staticmethod
    def _memory_node(item: Any, *, category_names: list[str], category_ids: list[str] | None = None) -> dict[str, Any]:
        salience = max(
            [v for v in (item.reflection_salience, item.emotional_intensity) if isinstance(v, int | float)],
            default=None,
        )
        category_pairs = sorted(set(zip(category_ids or [], category_names, strict=False)), key=lambda pair: pair[1].lower())
        return {
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

    @staticmethod
    def _category_node(category: Any) -> dict[str, Any]:
        return {
            "id": f"category:{category.id}",
            "kind": "category",
            "category_id": category.id,
            "label": category.name,
            "summary": category.summary or category.description,
            "previous_summary": getattr(category, "previous_summary", None),
            "approved_summary": getattr(category, "approved_summary", None),
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
        return {
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

    @staticmethod
    def _atomic_tag(category: Any, count: int) -> dict[str, Any]:
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
