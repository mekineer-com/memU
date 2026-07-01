from types import SimpleNamespace
from datetime import datetime, UTC, timedelta

from memu.app.graph import GraphMixin
from memu.app.service import MemoryService
from memu.database.models import Triple
from pydantic import BaseModel


class GraphScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


class _Repo:
    def __init__(self, value):
        self.value = value

    def list_items(self, where=None):
        return self.value

    def list_recent_items(self, where=None, *, limit, include_superseded=False):
        return dict(list(self.value.items())[:limit])

    def list_items_by_ids(self, item_ids, where=None, *, include_superseded=False):
        return {item_id: self.value[item_id] for item_id in item_ids if item_id in self.value}

    def list_categories(self, where=None):
        return self.value

    def list_relations(self, where=None):
        return self.value

    def list_all(self, where=None):
        return self.value


class _Triples:
    def get_edges_from(self, subject_id, predicate=None, where=None):
        if subject_id == "m1" and predicate == "mentions":
            return [SimpleNamespace(object_kind="entity", object_id="e1")]
        if subject_id == "m2" and predicate == "caused_by":
            return [
                SimpleNamespace(
                    subject_id="m2",
                    subject_kind="memory",
                    predicate="caused_by",
                    object_id="m1",
                    object_kind="memory",
                )
            ]
        return []

    def get_edges_to(self, object_id, predicate=None, where=None):
        if object_id == "m1" and predicate == "caused_by":
            return [
                SimpleNamespace(
                    subject_id="m2",
                    subject_kind="memory",
                    predicate="caused_by",
                    object_id="m1",
                    object_kind="memory",
                )
            ]
        return []


class _Service(GraphMixin):
    def __init__(self, db):
        self.db = db

    def _get_database(self):
        return self.db


def _item(id, summary, when):
    return SimpleNamespace(
        id=id,
        summary=summary,
        memory_type="episode",
        happened_at=when,
        created_at=when,
        updated_at=when,
        reflection_salience=None,
        emotional_intensity=None,
    )


def test_graph_recent_returns_bounded_memory_graph():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m1": _item("m1", "First memory", now),
            "m2": _item("m2", "Linked memory", now - timedelta(days=1)),
        }),
        memory_category_repo=_Repo({"c1": SimpleNamespace(id="c1", name="People", description="", summary=None, created_at=now, updated_at=now)}),
        category_item_repo=_Repo([SimpleNamespace(item_id="m1", category_id="c1")]),
        entity_repo=_Repo([SimpleNamespace(id="e1", name="Annie", entity_type="person", created_at=now, updated_at=now)]),
        triple_repo=_Triples(),
    )

    graph = _Service(db).graph_recent(where={"user_id": "u", "soul_id": "s"}, limit=1)

    node_ids = {node["id"] for node in graph["nodes"]}
    edge_ids = {edge["id"] for edge in graph["edges"]}
    assert {"memory:m1", "memory:m2", "category:c1", "entity:e1"} <= node_ids
    assert "semantic:m2:caused_by:m1" in edge_ids
    assert "category:m1:c1" in edge_ids
    assert "mentions:m1:e1" in edge_ids


def test_graph_recent_uses_real_bounded_sqlite_reads():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_sql", "soul_id": "s"}
    older = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="older linked memory",
        embedding=[0.1],
        user_data=scope,
    )
    newer = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="newer visible memory",
        embedding=[0.2],
        user_data=scope,
    )
    store.triple_repo.add(
        Triple(
            subject_id=older.id,
            subject_kind="memory",
            predicate="caused_by",
            object_id=newer.id,
            object_kind="memory",
        ),
        user_data=scope,
    )

    graph = service.graph_recent(where=scope, limit=1)

    edge_ids = {edge["id"] for edge in graph["edges"]}
    assert f"semantic:{older.id}:caused_by:{newer.id}" in edge_ids
