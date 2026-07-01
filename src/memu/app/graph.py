from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

SEMANTIC_PREDICATES = ["caused_by", "evokes", "conflicts_with", "parallels", "shaped_by"]


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _label(text: str, limit: int = 80) -> str:
    clean = " ".join(str(text or "").split())
    return clean[: limit - 1] + "..." if len(clean) > limit else clean


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
        return self._memory_node(item, category_names=category_names)

    async def graph_update_memory_summary(
        self,
        item_id: str,
        *,
        summary: str,
        where: Mapping[str, Any] | None = None,
        edited_by: str | None = None,
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
        if current.summary == summary:
            return self.graph_memory(f"memory:{raw_id}", where=where)

        embedding = (await self._select_embedding_client(None).embed([summary]))[0]
        store.memory_item_repo.update_summary_with_history(
            item_id=raw_id,
            summary=summary,
            embedding=embedding,
            where=where,
            edited_by=edited_by,
        )
        return self.graph_memory(f"memory:{raw_id}", where=where)

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
            self._memory_node(all_items[item_id], category_names=category_names_by_item.get(item_id, []))
            for item_id in selected_ids
            if item_id in all_items
        ]
        nodes.extend(self._category_node(categories[cat_id]) for cat_id in category_ids if cat_id in categories)
        nodes.extend(self._entity_node(entities[entity_id]) for entity_id in entity_ids if entity_id in entities)

        nodes.sort(key=lambda node: (node["kind"], node.get("happened_at") or "", node["id"]))
        return {"nodes": nodes, "edges": list(edges.values()), "limit": limit, "count": len(nodes)}

    @staticmethod
    def _memory_node(item: Any, *, category_names: list[str]) -> dict[str, Any]:
        salience = max(
            [v for v in (item.reflection_salience, item.emotional_intensity) if isinstance(v, int | float)],
            default=None,
        )
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
            "salience": salience,
            "category_names": sorted(set(category_names)),
        }

    @staticmethod
    def _category_node(category: Any) -> dict[str, Any]:
        return {
            "id": f"category:{category.id}",
            "kind": "category",
            "category_id": category.id,
            "label": category.name,
            "summary": category.summary or category.description,
            "created_at": _iso(category.created_at),
            "updated_at": _iso(category.updated_at),
            "category_names": [],
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
