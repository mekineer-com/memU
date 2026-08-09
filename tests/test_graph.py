import asyncio
import base64
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel

from memu.app import category_summary_journal
from memu.app.graph import GraphMixin
from memu.app.memorize_categories import _update_category_summaries
from memu.app.service import MemoryService
from memu.database.models import Triple


class GraphScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


class _Repo:
    def __init__(self, value):
        self.value = value
        self.list_items_calls = 0
        self.list_canvas_items_calls = 0
        self.list_items_by_ids_calls = 0
        self.list_all_calls = 0
        self.list_by_ids_calls = 0

    def list_items(self, where=None, *, include_superseded=False, include_embeddings=True):
        self.list_items_calls += 1
        return self.value

    def list_recent_items(self, where=None, *, limit, include_superseded=False):
        return dict(list(self.value.items())[:limit])

    def list_canvas_items(self, where=None, *, limit):
        self.list_canvas_items_calls += 1
        items = [item for item in self.value.values() if getattr(item, "embedding", None)]
        items.sort(key=lambda item: (item.updated_at, f"memory:{item.id}"), reverse=True)
        return {item.id: item for item in items[:limit]}, len(items)

    def list_items_by_ids(
        self,
        item_ids,
        where=None,
        *,
        include_superseded=False,
        include_merged=False,
        include_embeddings=False,
        session=None,
    ):
        self.list_items_by_ids_calls += 1
        return {item_id: self.value[item_id] for item_id in item_ids if item_id in self.value}

    def list_categories(self, where=None, *, session=None):
        return self.value

    def list_relations(self, where=None):
        return self.value

    def list_all(self, where=None):
        self.list_all_calls += 1
        return self.value

    def list_by_ids(self, entity_ids, where=None):
        self.list_by_ids_calls += 1
        if isinstance(self.value, dict):
            return [self.value[entity_id] for entity_id in entity_ids if entity_id in self.value]
        return [entity for entity in self.value if entity.id in entity_ids]


class _Triples:
    def __init__(self):
        self.list_edges_for_memories_calls = 0

    def get_edges_from(self, subject_id, predicate=None, where=None):
        if subject_id == "m1" and predicate == "mentions":
            return [
                SimpleNamespace(
                    subject_id="m1",
                    subject_kind="memory",
                    predicate="mentions",
                    object_id="e1",
                    object_kind="entity",
                )
            ]
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

    def list_edges_for_memories(self, memory_ids, predicates, where=None, *, current_only=True):
        self.list_edges_for_memories_calls += 1
        triples = []
        for memory_id in memory_ids:
            for predicate in predicates:
                triples.extend(self.get_edges_from(memory_id, predicate=predicate, where=where))
                triples.extend(self.get_edges_to(memory_id, predicate=predicate, where=where))
        return triples


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


def test_graph_atomic_atoms_pages_with_cursor():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m1": _item("m1", "First memory", now),
            "m2": _item("m2", "Second memory", now - timedelta(minutes=1)),
            "m3": _item("m3", "Third memory", now - timedelta(minutes=2)),
        }),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
    )
    service = _Service(db)

    page1 = service.graph_atomic_atoms(limit=2)
    page2 = service.graph_atomic_atoms(
        limit=2,
        cursor=page1["next_cursor"],
        cursor_id=page1["next_cursor_id"],
    )

    assert [atom["id"] for atom in page1["atoms"]] == ["memory:m1", "memory:m2"]
    assert page1["next_cursor_id"] == "memory:m2"
    assert [atom["id"] for atom in page2["atoms"]] == ["memory:m3"]
    assert page2["next_cursor"] is None


def test_graph_atomic_canvas_source_includes_embeddings_and_category_tags():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m1": _item("m1", "First memory", now),
            "m2": _item("m2", "Linked memory", now),
        }),
        memory_category_repo=_Repo({
            "c1": SimpleNamespace(
                id="c1",
                name="Core",
                description="",
                summary="Core summary",
                embedding=[0.0, 1.0],
                created_at=now,
                updated_at=now,
            ),
            "c2": SimpleNamespace(
                id="c2",
                name="Alpha",
                description="",
                summary="Alpha summary",
                embedding=None,
                created_at=now,
                updated_at=now,
            ),
        }),
        category_item_repo=_Repo([
            SimpleNamespace(item_id="m1", category_id="c1"),
            SimpleNamespace(item_id="m1", category_id="c2"),
        ]),
        entity_repo=_Repo([SimpleNamespace(id="e1", name="Annie")]),
        triple_repo=_Triples(),
    )
    db.memory_item_repo.value["m1"].embedding = [1.0, 0.0]
    db.memory_item_repo.value["m2"].embedding = [0.9, 0.1]

    out = _Service(db).graph_atomic_canvas_source(limit=10)

    atoms = {atom["id"]: atom for atom in out["atoms"]}
    assert atoms["memory:m1"]["primary_tag"] == "Alpha"
    assert atoms["memory:m1"]["tag_ids"] == ["category:c2", "category:c1"]
    assert atoms["memory:m1"]["entity_ids"] == ["entity:e1"]
    assert atoms["memory:m1"]["entity_names"] == ["Annie"]
    assert "embedding" not in atoms["memory:m1"]
    assert np.frombuffer(
        base64.b64decode(atoms["memory:m1"]["embedding_f32_le_b64"]), dtype="<f4"
    ).tolist() == [1.0, 0.0]
    assert atoms["category:c1"]["primary_tag"] == "Core"
    assert atoms["category:c1"]["tag_ids"] == ["category:c1"]
    assert "embedding" not in atoms["category:c1"]
    assert np.frombuffer(
        base64.b64decode(atoms["category:c1"]["embedding_f32_le_b64"]), dtype="<f4"
    ).tolist() == [0.0, 1.0]
    assert atoms["category:c1"]["entity_ids"] == []
    assert atoms["category:c1"]["entity_names"] == []
    edges = {(edge["source"], edge["target"], edge["predicate"]): edge for edge in out["edges"]}
    assert edges[("memory:m2", "memory:m1", "caused_by")]["weight"] == 0.7
    assert edges[("memory:m1", "memory:m2", "similarity")]["weight"] == pytest.approx(0.9938837)
    assert db.memory_item_repo.list_canvas_items_calls == 1
    assert db.triple_repo.list_edges_for_memories_calls == 1
    assert set(out["timing_ms"]) == {"store", "taxonomy", "atoms", "graph", "similarity", "finalize", "total"}
    assert all(value >= 0 for value in out["timing_ms"].values())


def test_canvas_item_read_applies_scope_active_filter_order_and_limit():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "canvas_page", "soul_id": "s"}
    first = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="first",
        embedding=[1.0, 0.0],
        user_data=scope,
    )
    second = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="second",
        embedding=[0.0, 1.0],
        user_data=scope,
    )
    store.memory_item_repo.create_item(
        memory_type="episode",
        summary="other scope",
        embedding=[1.0, 1.0],
        user_data={"user_id": "canvas_page_other", "soul_id": "s"},
    )
    store.memory_item_repo.update_item(item_id=first.id, summary="first, updated")

    items, total = store.memory_item_repo.list_canvas_items(scope, limit=1)

    assert list(items) == [first.id]
    assert total == 2

    store.triple_repo.add(
        Triple(
            subject_id=first.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=second.id,
            object_kind="memory",
        ),
        user_data=scope,
    )
    items, total = store.memory_item_repo.list_canvas_items(scope, limit=1)
    assert list(items) == [second.id]
    assert total == 1


def test_graph_atomic_canvas_source_includes_category_similarity_edges():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({"m1": _item("m1", "First memory", now)}),
        memory_category_repo=_Repo({
            "c1": SimpleNamespace(
                id="c1",
                name="Core",
                description="",
                summary="Core summary",
                embedding=[0.99, 0.01],
                created_at=now,
                updated_at=now,
            ),
        }),
        category_item_repo=_Repo([]),
        entity_repo=_Repo([]),
        triple_repo=_Triples(),
    )
    db.memory_item_repo.value["m1"].embedding = [1.0, 0.0]

    out = _Service(db).graph_atomic_canvas_source(limit=10)

    assert any(
        {edge["source"], edge["target"]} == {"memory:m1", "category:c1"}
        and edge["predicate"] == "similarity"
        for edge in out["edges"]
    )


def test_graph_atomic_canvas_source_caps_similarity_edges_per_atom():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            f"m{i}": _item(f"m{i}", f"Memory {i}", now)
            for i in range(5)
        }),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        entity_repo=_Repo([]),
        triple_repo=_Triples(),
    )
    for i, item in enumerate(db.memory_item_repo.value.values()):
        item.embedding = [1.0, i / 100.0]

    out = _Service(db).graph_atomic_canvas_source(limit=10)

    counts: dict[str, int] = {}
    for edge in out["edges"]:
        if edge["predicate"] != "similarity":
            continue
        counts[edge["source"]] = counts.get(edge["source"], 0) + 1
        counts[edge["target"]] = counts.get(edge["target"], 0) + 1
    assert max(counts.values()) <= 3


def test_graph_atomic_canvas_source_filters_before_item_scan():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m1": _item("m1", "Visible memory", now),
            "m2": _item("m2", "Hidden memory", now),
            "m3": _item("m3", "Second visible memory", now),
        }),
        memory_category_repo=_Repo({
            "c1": SimpleNamespace(
                id="c1",
                name="Health",
                description="",
                summary="Health summary",
                embedding=[0.0, 1.0],
                created_at=now,
                updated_at=now,
            ),
            "c2": SimpleNamespace(
                id="c2",
                name="Hidden",
                description="",
                summary="Hidden summary",
                embedding=[1.0, 0.0],
                created_at=now,
                updated_at=now,
            ),
        }),
        category_item_repo=_Repo([
            SimpleNamespace(item_id="m1", category_id="c1"),
            SimpleNamespace(item_id="m2", category_id="c2"),
            SimpleNamespace(item_id="m3", category_id="c1"),
        ]),
        entity_repo=_Repo([]),
        triple_repo=_Triples(),
    )
    db.memory_item_repo.value["m1"].embedding = [1.0, 0.0]
    db.memory_item_repo.value["m2"].embedding = [0.9, 0.1]
    db.memory_item_repo.value["m3"].embedding = [0.95, 0.05]
    db.triple_repo.get_edges_from = lambda subject_id, predicate=None, where=None: [  # type: ignore[method-assign]
        SimpleNamespace(
            subject_id="m1",
            subject_kind="memory",
            predicate="mentions",
            object_kind="entity",
            object_id="e1",
        )
    ] if subject_id == "m1" and predicate == "mentions" else []
    db.entity_repo.value.append(SimpleNamespace(id="e1", name="Visible entity"))

    out = _Service(db).graph_atomic_canvas_source(
        limit=10,
        atom_ids={"memory:m1", "memory:m3", "category:c1", "memory:missing"},
    )

    assert {atom["id"] for atom in out["atoms"]} == {"memory:m1", "memory:m3", "category:c1"}
    assert all("memory:m2" not in {edge["source"], edge["target"]} for edge in out["edges"])
    assert any({edge["source"], edge["target"]} == {"memory:m1", "memory:m3"} for edge in out["edges"])
    assert db.memory_item_repo.list_items_calls == 0
    assert db.memory_item_repo.list_items_by_ids_calls == 1
    assert db.entity_repo.list_all_calls == 0
    assert db.entity_repo.list_by_ids_calls == 1
    assert next(atom for atom in out["atoms"] if atom["id"] == "memory:m1")["entity_names"] == ["Visible entity"]


def test_graph_atomic_canvas_source_rejects_cross_scope_atom_ids():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    own_scope = {"user_id": "canvas_scope_owner", "soul_id": "s"}
    other_scope = {"user_id": "canvas_scope_other", "soul_id": "s"}
    other = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="other scope memory",
        embedding=[1.0, 0.0],
        user_data=other_scope,
    )

    out = service.graph_atomic_canvas_source(
        where=own_scope,
        atom_ids={f"memory:{other.id}"},
    )

    assert out["atoms"] == []
    assert out["edges"] == []


def test_graph_atomic_neighborhood_is_seeded_by_memory():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m1": _item("m1", "First memory", now),
            "m2": _item("m2", "Linked memory", now),
        }),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        triple_repo=_Triples(),
    )

    graph = _Service(db).graph_atomic_neighborhood("memory:m1")

    assert graph is not None
    assert {node["id"] for node in graph["nodes"]} == {"memory:m1", "memory:m2"}
    assert graph["center_atom_id"] == "memory:m1"
    assert graph["edges"][0]["edge_type"] == "semantic"
    assert db.memory_item_repo.list_items_calls == 1
    assert db.memory_item_repo.list_items_by_ids_calls == 0


def test_graph_memory_fetches_single_memory_by_id():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({"m1": _item("m1", "First memory", now)}),
        memory_category_repo=_Repo({
            "c1": SimpleNamespace(id="c1", name="Core", description="", summary=None, created_at=now, updated_at=now),
        }),
        category_item_repo=_Repo([SimpleNamespace(item_id="m1", category_id="c1")]),
        entity_repo=_Repo([]),
        triple_repo=_Triples(),
    )

    node = _Service(db).graph_memory("memory:m1", where={"user_id": "u", "soul_id": "s"})

    assert node is not None
    assert node["id"] == "memory:m1"
    assert node["category_names"] == ["Core"]
    assert db.memory_item_repo.list_items_calls == 0
    assert db.memory_item_repo.list_items_by_ids_calls == 1


def test_graph_atomic_neighborhood_includes_similarity_neighbors():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m10": _item("m10", "First memory", now),
            "m11": _item("m11", "Similar memory", now),
            "m12": _item("m12", "Different memory", now),
        }),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        triple_repo=_Triples(),
    )
    db.memory_item_repo.value["m10"].embedding = [1.0, 0.0]
    db.memory_item_repo.value["m11"].embedding = [0.9, 0.1]
    db.memory_item_repo.value["m12"].embedding = [0.0, 1.0]

    graph = _Service(db).graph_atomic_neighborhood("memory:m10", min_similarity=0.7)

    assert graph is not None
    assert {node["id"] for node in graph["nodes"]} == {"memory:m10", "memory:m11"}
    assert graph["edges"] == [{
        "source_id": "memory:m10",
        "target_id": "memory:m11",
        "edge_type": "semantic",
        "strength": pytest.approx(0.9938837),
        "shared_tag_count": 0,
        "similarity_score": pytest.approx(0.9938837),
    }]
    assert db.memory_item_repo.list_items_calls == 1
    assert db.memory_item_repo.list_items_by_ids_calls == 0


def test_graph_atomic_neighborhood_honors_similarity_limit():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    items = {"m10": _item("m10", "Center memory", now)}
    items.update({f"m{idx}": _item(f"m{idx}", f"Similar memory {idx}", now) for idx in range(11, 18)})
    db = SimpleNamespace(
        memory_item_repo=_Repo(items),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        triple_repo=_Triples(),
    )
    for item in db.memory_item_repo.value.values():
        item.embedding = [1.0, 0.0]

    graph = _Service(db).graph_atomic_neighborhood("memory:m10", min_similarity=0.7, similarity_limit=6)

    assert graph is not None
    assert len(graph["nodes"]) == 7


def test_graph_atomic_similar_is_pure_similarity():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({
            "m1": _item("m1", "Center memory", now),
            "m2": _item("m2", "Triple only memory", now),
            "m3": _item("m3", "Similar memory", now),
            "m4": _item("m4", "Different memory", now),
        }),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        triple_repo=_Triples(),
    )
    db.memory_item_repo.value["m1"].embedding = [1.0, 0.0]
    db.memory_item_repo.value["m2"].embedding = [0.0, 1.0]
    db.memory_item_repo.value["m3"].embedding = [0.9, 0.1]
    db.memory_item_repo.value["m4"].embedding = [0.2, 0.8]

    nodes = _Service(db).graph_atomic_similar("memory:m1", min_similarity=0.7)

    assert nodes is not None
    assert [node["id"] for node in nodes] == ["memory:m3"]
    assert nodes[0]["similarity_score"] == pytest.approx(0.9938837)


def test_graph_atomic_similar_threads_scope_to_item_repo():
    now = datetime(2026, 7, 1, tzinfo=UTC)

    class _ScopedRepo(_Repo):
        def list_items(self, where=None, *, include_superseded=False, include_embeddings=True):
            self.list_items_calls += 1
            return {
                item_id: item
                for item_id, item in self.value.items()
                if item.user_id == where["user_id"] and item.soul_id == where["soul_id"]
            }

    items = {
        "m1": _item("m1", "Center memory", now),
        "m2": _item("m2", "In scope similar", now),
        "m3": _item("m3", "Out of scope similar", now),
    }
    for item in items.values():
        item.embedding = [1.0, 0.0]
        item.user_id = "u1"
        item.soul_id = "s1"
    items["m3"].user_id = "u2"
    db = SimpleNamespace(
        memory_item_repo=_ScopedRepo(items),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        triple_repo=_Triples(),
    )

    nodes = _Service(db).graph_atomic_similar("memory:m1", where={"user_id": "u1", "soul_id": "s1"})

    assert nodes is not None
    assert [node["id"] for node in nodes] == ["memory:m2"]


def test_graph_atomic_similar_handles_non_memory_and_missing():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    db = SimpleNamespace(
        memory_item_repo=_Repo({"m1": _item("m1", "Center memory", now)}),
        memory_category_repo=_Repo({}),
        category_item_repo=_Repo([]),
        triple_repo=_Triples(),
    )

    assert _Service(db).graph_atomic_similar("category:c1") == []
    assert _Service(db).graph_atomic_similar("entity:e1") == []
    assert _Service(db).graph_atomic_similar("memory:missing") is None


def test_graph_search_is_scoped_and_honors_since_days():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_search", "soul_id": "s"}
    store.memory_item_repo.create_item(
        memory_type="episode",
        summary="recent sushi memory",
        embedding=[1.0, 0.0],
        happened_at=datetime.now(UTC) - timedelta(days=1),
        user_data=scope,
    )
    old = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="old sushi memory",
        embedding=[1.0, 0.0],
        happened_at=datetime.now(UTC) - timedelta(days=30),
        user_data=scope,
    )
    other = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="other soul sushi memory",
        embedding=[1.0, 0.0],
        user_data={"user_id": "graph_search", "soul_id": "other"},
    )

    class _Embedder:
        async def embed(self, texts):
            assert texts == ["sushi"]
            return [[1.0, 0.0]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    out = asyncio.run(service.graph_search("sushi", where=scope, limit=10, since_days=7))
    ids = {node["memory_id"] for node in out["nodes"]}

    assert old.id not in ids
    assert other.id not in ids
    assert len(ids) == 1


def test_graph_search_requires_query():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )

    with pytest.raises(ValueError, match="query is required"):
        asyncio.run(service.graph_search(" ", where={"user_id": "u", "soul_id": "s"}))


def test_graph_search_drops_unrelated_vector_only_hits():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    scope = {"user_id": "graph_search", "soul_id": "s"}
    service._get_database().memory_item_repo.create_item(
        memory_type="episode",
        summary="totally unrelated",
        embedding=[-1.0, 0.0],
        user_data=scope,
    )

    class _Embedder:
        async def embed(self, texts):
            return [[1.0, 0.0]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    out = asyncio.run(service.graph_search("sushi", where=scope, limit=5))
    assert out["nodes"] == []


def test_graph_search_mode_controls_keyword_vs_semantic():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    item = _item("m1", "sushi keyword", now)
    item.embedding = [-1.0, 0.0]

    class _MemoryRepo(_Repo):
        def fts_search_items(self, query, limit, pool_ids=None):
            return [("m1", 1.0)]

    service = _Service(
        SimpleNamespace(
            memory_item_repo=_MemoryRepo({"m1": item}),
            memory_category_repo=_Repo({}),
            category_item_repo=_Repo([]),
        )
    )

    class _Embedder:
        async def embed(self, texts):
            assert texts == ["sushi"]
            return [[1.0, 0.0]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    keyword = asyncio.run(service.graph_search("sushi", mode="keyword"))
    semantic = asyncio.run(service.graph_search("sushi", mode="semantic"))

    assert [node["summary"] for node in keyword["nodes"]] == ["sushi keyword"]
    assert semantic["nodes"] == []


def test_graph_search_semantic_includes_categories():
    now = datetime(2026, 7, 1, tzinfo=UTC)

    class _MemoryRepo(_Repo):
        def fts_search_items(self, query, limit, pool_ids=None):
            return []

    service = _Service(
        SimpleNamespace(
            memory_item_repo=_MemoryRepo({}),
            memory_category_repo=_Repo({
                "c1": SimpleNamespace(
                    id="c1",
                    name="Sushi",
                    description="",
                    summary="fish rice",
                    embedding=[1.0, 0.0],
                    approved_summary="fish rice",
                    created_at=now,
                    updated_at=now,
                )
            }),
            category_item_repo=_Repo([]),
        )
    )

    class _Embedder:
        async def embed(self, texts):
            assert texts == ["sushi"]
            return [[1.0, 0.0]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    out = asyncio.run(service.graph_search("sushi", mode="semantic"))
    assert [node["id"] for node in out["nodes"]] == ["category:c1"]


def test_graph_update_memory_summary_embeds_before_history_update(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_edit", "soul_id": "s"}
    item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="old summary",
        embedding=[0.1],
        user_data=scope,
    )

    class _Embedder:
        async def embed(self, texts):
            assert texts == ["new summary"]
            return [[0.9, 0.8]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    updated = asyncio.run(
        service.graph_update_memory_summary(f"memory:{item.id}", summary=" new summary ", where=scope)
    )

    saved = store.memory_item_repo.get_item(item.id)
    assert updated["summary"] == "new summary"
    assert saved.summary == "new summary"
    assert saved.embedding == pytest.approx([0.9, 0.8])
    with store._sessions.engine.connect() as conn:
        row = conn.exec_driver_sql(
            "SELECT summary_before, summary_after, scope_json FROM memory_item_edit_history"
        ).fetchone()
    assert row == ("old summary", "new summary", '{"soul_id": "s", "user_id": "graph_edit"}')
    assert list(tmp_path.iterdir()) == []


def test_graph_update_memory_summary_embed_failure_leaves_memory_unchanged():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_edit_fail", "soul_id": "s"}
    item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="old summary",
        embedding=[0.1],
        user_data=scope,
    )

    class _Embedder:
        async def embed(self, texts):
            raise RuntimeError("embed failed")

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        asyncio.run(service.graph_update_memory_summary(item.id, summary="new summary", where=scope))

    saved = store.memory_item_repo.get_item(item.id)
    assert saved.summary == "old summary"
    assert saved.embedding == pytest.approx([0.1])
    with store._sessions.engine.connect() as conn:
        count = conn.exec_driver_sql("SELECT COUNT(*) FROM memory_item_edit_history").scalar()
    assert count == 0


def test_graph_update_memory_summary_stripped_noop_skips_embed_and_history():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_edit_noop", "soul_id": "s"}
    item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="same summary ",
        embedding=[0.1],
        user_data=scope,
    )

    class _Embedder:
        async def embed(self, texts):
            raise AssertionError("noop edit should not embed")

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    updated = asyncio.run(service.graph_update_memory_summary(item.id, summary="same summary", where=scope))

    assert updated["summary"] == "same summary "
    with store._sessions.engine.connect() as conn:
        count = conn.exec_driver_sql("SELECT COUNT(*) FROM memory_item_edit_history").scalar()
    assert count == 0


def test_graph_pending_and_memory_approval_semantics():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pending_memory", "soul_id": "s"}
    item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="approved memory",
        embedding=[0.1],
        user_data=scope,
    )
    original_updated_at = item.updated_at
    store.memory_item_repo.approve_item(item.id, where=scope)

    assert service.graph_list_pending(where=scope)["items"] == []
    approved = store.memory_item_repo.list_items_by_ids({item.id}, scope)[item.id]
    assert approved.approved_at is not None
    assert approved.updated_at == original_updated_at
    approved_again = store.memory_item_repo.approve_item(item.id, where=scope)
    assert approved_again.approved_at == approved.approved_at

    class _Embedder:
        async def embed(self, texts):
            return [[0.2]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]

    # Soul edit (approved=False) re-opens approval.
    asyncio.run(service.graph_update_memory_summary(item.id, summary="siri edit", where=scope))
    pending = service.graph_list_pending(where=scope)["items"]
    assert [node["memory_id"] for node in pending] == [item.id]
    assert pending[0]["approved_at"] is None

    service.graph_approve_memory(item.id, where=scope)
    assert service.graph_list_pending(where=scope)["items"] == []
    reapproved = store.memory_item_repo.list_items_by_ids({item.id}, scope)[item.id]
    assert reapproved.approved_at is not None

    # Human "Save" on an approved item keeps the original approval stamp.
    asyncio.run(
        service.graph_update_memory_summary(item.id, summary="human edit", where=scope, approved=True)
    )
    edited = store.memory_item_repo.list_items_by_ids({item.id}, scope)[item.id]
    assert edited.approved_at == reapproved.approved_at
    assert service.graph_list_pending(where=scope)["items"] == []

    # Human "Save + approve" on a pending item stamps it.
    asyncio.run(service.graph_update_memory_summary(item.id, summary="soul edit 2", where=scope))
    assert service.graph_list_pending(where=scope)["items"] != []
    asyncio.run(
        service.graph_update_memory_summary(item.id, summary="human fix", where=scope, approved=True)
    )
    fixed = store.memory_item_repo.list_items_by_ids({item.id}, scope)[item.id]
    assert fixed.approved_at is not None
    assert service.graph_list_pending(where=scope)["items"] == []


def test_category_approval_does_not_touch_updated_at():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pending_category", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="Identity",
        description="",
        embedding=[0.1],
        user_data=scope,
        kind="topic",
    )
    category = store.memory_category_repo.update_category(category_id=category.id, summary="new summary")
    original_updated_at = category.updated_at

    approved = store.memory_category_repo.approve_category_summary(category.id, where=scope)

    assert approved.approved_summary == "new summary"
    assert approved.approved_description == ""
    assert approved.updated_at == original_updated_at


def test_pending_dossier_requires_prose_and_tracks_description():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pending_pair", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="Daily Life",
        description="A personal brief.",
        embedding=[0.1],
        user_data=scope,
        kind="topic",
    )
    assert service.graph_list_pending(where=scope)["categories"] == []

    category = store.memory_category_repo.update_category(
        category_id=category.id,
        summary="A fuller account.",
    )
    pending = service.graph_list_pending(where=scope)["categories"]
    assert pending[0]["description"] == "A personal brief."
    assert pending[0]["approved_description"] is None

    approved = store.memory_category_repo.approve_category_summary(category.id, where=scope)
    assert approved.approved_description == approved.description
    assert service.graph_list_pending(where=scope)["categories"] == []


def test_graph_pending_excludes_superseded_memories():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pending_superseded", "soul_id": "s"}
    old = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="old pending",
        embedding=[0.1],
        user_data=scope,
    )
    new = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="replacement",
        embedding=[0.2],
        user_data=scope,
    )
    store.triple_repo.add(
        Triple(
            subject_id=old.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=new.id,
            object_kind="memory",
        ),
        user_data=scope,
    )

    pending_ids = {node["memory_id"] for node in service.graph_list_pending(where=scope)["items"]}
    assert old.id not in pending_ids
    assert service.graph_approve_memory(old.id, where=scope) is None
    assert service.graph_delete_memory(old.id, where=scope) is None
    assert store.memory_item_repo.get_item(old.id, include_superseded=True) is not None
    assert store.triple_repo.get_edges_from(old.id, predicate="evolved_into", where=scope)


def test_graph_pending_groups_near_duplicate_embeddings():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pending_dupe", "soul_id": "s"}
    dupe_a = store.memory_item_repo.create_item(
        memory_type="episode", summary="likes coffee in the morning", embedding=[1.0, 0.0], user_data=scope
    )
    dupe_b = store.memory_item_repo.create_item(
        memory_type="episode", summary="enjoys morning coffee", embedding=[0.99, 0.01], user_data=scope
    )
    distinct = store.memory_item_repo.create_item(
        memory_type="episode", summary="lives in Lisbon", embedding=[0.0, 1.0], user_data=scope
    )

    items = service.graph_list_pending(where=scope)["items"]
    by_id = {node["memory_id"]: node for node in items}

    assert by_id[dupe_a.id]["similar_to"] == [f"memory:{dupe_b.id}"]
    assert by_id[dupe_b.id]["similar_to"] == [f"memory:{dupe_a.id}"]
    assert by_id[dupe_a.id]["similarity"] > 0.99
    assert by_id[dupe_b.id]["similarity"] > 0.99
    assert "similar_to" not in by_id[distinct.id]
    assert "similarity" not in by_id[distinct.id]

    ids_in_order = [node["memory_id"] for node in items]
    assert abs(ids_in_order.index(dupe_a.id) - ids_in_order.index(dupe_b.id)) == 1


def test_graph_delete_memory_removes_dependents():
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "delete_memory", "soul_id": "s"}
    item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="delete me",
        embedding=[0.1],
        user_data=scope,
    )
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data=scope,
    )
    store.category_item_repo.link_item_category(item.id, category.id, scope)
    store.triple_repo.add(
        Triple(
            subject_id=item.id,
            subject_kind="memory",
            predicate="mentions",
            object_id="entity1",
            object_kind="entity",
            source_memory_id=item.id,
        ),
        user_data=scope,
    )

    class _Embedder:
        async def embed(self, texts):
            return [[0.2]]

    service._select_embedding_client = lambda _ctx: _Embedder()  # type: ignore[method-assign]
    asyncio.run(service.graph_update_memory_summary(item.id, summary="updated delete me", where=scope))

    deleted = service.graph_delete_memory(item.id, where=scope)

    assert deleted["memory_id"] == item.id
    assert store.memory_item_repo.get_item(item.id) is None
    with store._sessions.engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM memory_items_fts WHERE item_id = ?", (item.id,)).scalar() == 0
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM memory_item_edit_history WHERE memory_item_id = ?", (item.id,)).scalar() == 0
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM category_items WHERE item_id = ?", (item.id,)).scalar() == 0
        assert conn.exec_driver_sql(
            "SELECT COUNT(*) FROM triples WHERE subject_id = ? OR object_id = ? OR source_memory_id = ?",
            (item.id, item.id, item.id),
        ).scalar() == 0


def test_sqlite_approval_backfill_runs_only_when_column_is_added(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
CREATE TABLE memory_items (
  id TEXT PRIMARY KEY,
  created_at DATETIME,
  updated_at DATETIME,
  resource_id TEXT,
  memory_type TEXT NOT NULL,
  summary TEXT NOT NULL,
  embedding BLOB,
  happened_at DATETIME,
  source_role TEXT,
  speaker_id TEXT,
  speaker_label TEXT,
  confidence FLOAT,
  source_message_ids JSON,
  reflection_salience FLOAT,
  emotional_intensity FLOAT,
  conversation_id TEXT,
  segment_id TEXT,
  unresolved TEXT,
  merged_into TEXT,
  extra JSON,
  user_id TEXT,
  soul_id TEXT
);
INSERT INTO memory_items (id, created_at, updated_at, memory_type, summary, embedding, extra, user_id, soul_id)
VALUES ('old', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'episode', 'old approved', X'CDCCCC3D', '{}', 'backfill', 's');
"""
    )
    conn.close()
    dsn = f"sqlite:///{db_path}"
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": dsn}},
        user_config={"model": GraphScope},
    )
    scope = {"user_id": "backfill", "soul_id": "s"}
    old = service._get_database().memory_item_repo.get_item("old")
    assert old.approved_at is not None

    new = service._get_database().memory_item_repo.create_item(
        memory_type="episode",
        summary="new pending",
        embedding=[0.2],
        user_data=scope,
    )
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": dsn}},
        user_config={"model": GraphScope},
    )
    pending_ids = {node["memory_id"] for node in service.graph_list_pending(where=scope)["items"]}
    assert pending_ids == {new.id}


def test_graph_update_category_summary_journals_before_db_update(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_cat_edit", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data=scope,
    )
    store.memory_category_repo.update_category(category_id=category.id, summary="old category summary")

    updated = asyncio.run(
        service.graph_update_category_summary(
            f"category:{category.id}",
            summary=" new category summary ",
            where=scope,
            edited_by="surfer",
        )
    )

    saved = store.memory_category_repo.list_categories(scope)[category.id]
    journal = (tmp_path / "s.summary_journal.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(journal[0])
    assert updated["summary"] == "new category summary"
    assert updated["previous_summary"] == "old category summary"
    assert saved.summary == "new category summary"
    assert saved.previous_summary == "old category summary"
    assert saved.embedding == pytest.approx([0.1])
    assert entry["summary_before"] == "old category summary"
    assert entry["summary_after"] == "new category summary"
    assert entry["edited_by"] == "surfer"
    assert entry["scope"] == scope


def test_graph_update_category_summary_approved_noop_blesses_without_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_cat_noop_approve", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data=scope,
    )
    store.memory_category_repo.update_category(category_id=category.id, summary="same")

    updated = asyncio.run(
        service.graph_update_category_summary(
            category.id,
            summary=" same ",
            where=scope,
            approved=True,
        )
    )

    saved = store.memory_category_repo.list_categories(scope)[category.id]
    assert updated["approved_summary"] == "same"
    assert saved.approved_summary == "same"
    assert list(tmp_path.iterdir()) == []


def test_graph_update_category_summary_out_of_scope_does_not_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data={"user_id": "right", "soul_id": "s"},
    )
    store.memory_category_repo.update_category(category_id=category.id, summary="old")

    with pytest.raises(KeyError):
        asyncio.run(
            service.graph_update_category_summary(
                category.id,
                summary="new",
                where={"user_id": "wrong", "soul_id": "s"},
            )
        )

    assert store.memory_category_repo.list_categories({"user_id": "right", "soul_id": "s"})[category.id].summary == "old"
    assert list(tmp_path.iterdir()) == []


def test_graph_update_category_summary_journal_failure_leaves_db_unchanged(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("", encoding="utf-8")
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", blocked)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "graph_cat_fail", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data=scope,
    )
    store.memory_category_repo.update_category(category_id=category.id, summary="old")

    with pytest.raises(FileExistsError):
        asyncio.run(service.graph_update_category_summary(category.id, summary="new", where=scope))

    saved = store.memory_category_repo.list_categories(scope)[category.id]
    assert saved.summary == "old"
    assert saved.previous_summary is None


def test_update_category_summaries_journals_pipeline_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pipeline", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data=scope,
    )
    store.memory_category_repo.update_category(category_id=category.id, summary="old pipeline summary")

    class _LLM:
        async def chat(self, _prompt):
            return "pipeline summary"

    updated = asyncio.run(
        _update_category_summaries(
            {category.id: ["new memory"]},
            store=store,
            llm_client=_LLM(),
            user=scope,
            build_category_summary_prompt=lambda *_args: "prompt",
            summary_user_name=lambda _user: "",
        )
    )

    saved = store.memory_category_repo.list_categories(scope)[category.id]
    entry = json.loads((tmp_path / "s.summary_journal.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert updated == {category.id: "pipeline summary"}
    assert saved.summary == "pipeline summary"
    assert saved.previous_summary == "old pipeline summary"
    assert entry["summary_before"] == "old pipeline summary"
    assert entry["summary_after"] == "pipeline summary"
    assert entry["edited_by"] == "pipeline"


def test_append_soul_summary_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)

    category_summary_journal.append_summary_journal(
        kind="narrative_self",
        summary_id="soul-summary:narrative_self",
        summary_before="before",
        summary_after="after",
        scope={"user_id": "u", "soul_id": "s"},
        edited_by="consolidation",
    )

    entry = json.loads((tmp_path / "s.summary_journal.jsonl").read_text(encoding="utf-8"))
    assert entry["kind"] == "narrative_self"
    assert entry["summary_id"] == "soul-summary:narrative_self"
    assert entry["summary_before"] == "before"
    assert entry["summary_after"] == "after"


def test_update_category_summaries_empty_pipeline_output_keeps_old_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": GraphScope},
    )
    store = service._get_database()
    scope = {"user_id": "pipeline_empty", "soul_id": "s"}
    category = store.memory_category_repo.get_or_create_category(
        name="People",
        description="",
        embedding=[0.1],
        user_data=scope,
    )
    store.memory_category_repo.update_category(category_id=category.id, summary="old summary")

    class _LLM:
        async def chat(self, _prompt):
            return "```markdown\n   \n```"

    updated = asyncio.run(
        _update_category_summaries(
            {category.id: ["new memory"]},
            store=store,
            llm_client=_LLM(),
            user=scope,
            build_category_summary_prompt=lambda *_args: "prompt",
            summary_user_name=lambda _user: "",
        )
    )

    saved = store.memory_category_repo.list_categories(scope)[category.id]
    assert updated == {}
    assert saved.summary == "old summary"
    assert saved.previous_summary is None
    assert list(tmp_path.iterdir()) == []
