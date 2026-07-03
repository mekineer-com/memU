import asyncio
import json
from types import SimpleNamespace
from datetime import datetime, UTC, timedelta

import pytest
from memu.app import category_summary_journal
from memu.app.graph import GraphMixin
from memu.app.memorize_categories import _update_category_summaries
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
    assert saved.embedding == [0.9, 0.8]
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
    assert saved.embedding == [0.1]
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
    assert saved.embedding == [0.1]
    assert entry["summary_before"] == "old category summary"
    assert entry["summary_after"] == "new category summary"
    assert entry["edited_by"] == "surfer"
    assert entry["scope"] == scope


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
