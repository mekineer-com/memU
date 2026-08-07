from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from memu.app.service import MemoryService
from memu.database.models import Triple


class DossierScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


SCOPE = {"user_id": "test-user", "soul_id": "test-soul"}
OTHER_SCOPE = {"user_id": "other-user", "soul_id": "other-soul"}


class FakeEmbedClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.casefold()
            if "health" in lowered:
                vectors.append([1.0, 0.0])
            elif "work" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def _service(
    tmp_path,
    name: str = "dossier.db",
    *,
    retrieve_config=None,
    **memorize_config,
) -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{tmp_path / name}"}},
        memorize_config=memorize_config,
        retrieve_config=retrieve_config,
        user_config={"model": DossierScope},
    )


def _category(
    service: MemoryService,
    name: str,
    *,
    scope: dict[str, str] = SCOPE,
    kind: str = "topic",
    evidence: datetime | None = None,
    embedding: list[float] | None = None,
    summary: str | None = None,
):
    category = service.database.memory_category_repo.get_or_create_category(
        name=name,
        description=f"{name} description",
        embedding=embedding or [1.0, 0.0],
        user_data=scope,
        kind=kind,
        last_evidence_at=evidence,
    )
    if summary is not None:
        category = service.database.memory_category_repo.update_category(category_id=category.id, summary=summary)
    return category


@pytest.mark.asyncio
async def test_anchor_seeding_is_scoped_idempotent_and_immutable(tmp_path) -> None:
    service = _service(tmp_path)
    client = FakeEmbedClient()

    anchors = await service.ensure_dossier_anchors(SCOPE, embedding_client=client)
    again = await service.ensure_dossier_anchors(SCOPE, embedding_client=client)

    assert set(anchors) == {"soul", "user"}
    assert {role: row.id for role, row in anchors.items()} == {role: row.id for role, row in again.items()}
    assert client.calls == [[
        "test-soul: Identity, history, relationships, and lived experience of test-soul.",
        "test-user: Identity, history, relationships, and lived experience of test-user.",
    ]]
    assert all(row.kind == "lore" and row.lore_subtype == "person" for row in anchors.values())

    other = await service.ensure_dossier_anchors(OTHER_SCOPE, embedding_client=client)
    assert {row.id for row in other.values()}.isdisjoint({row.id for row in anchors.values()})
    with pytest.raises(ValueError, match="cannot be renamed"):
        await service.update_dossier(anchors["soul"].id, SCOPE, name="renamed", embedding_client=client)
    with pytest.raises(ValueError, match="remain Lore"):
        await service.update_dossier(anchors["soul"].id, SCOPE, kind="topic")
    with pytest.raises(ValueError, match="must differ"):
        await service.ensure_dossier_anchors(
            {"user_id": "same", "soul_id": "SAME"}, embedding_client=client
        )


@pytest.mark.asyncio
async def test_anchor_collision_and_corruption_fail_without_replacement(tmp_path) -> None:
    collision = _service(tmp_path, "collision.db")
    _category(collision, "test-soul", kind="lore")
    with pytest.raises(ValueError, match="title already exists"):
        await collision.ensure_dossier_anchors(SCOPE, embedding_client=FakeEmbedClient())
    assert collision.database.memory_category_repo.list_anchor_categories(SCOPE) == {}

    corrupt = _service(tmp_path, "corrupt.db")
    anchors = await corrupt.ensure_dossier_anchors(SCOPE, embedding_client=FakeEmbedClient())
    corrupt.database.memory_category_repo.update_category(category_id=anchors["soul"].id, name="wrong")
    with pytest.raises(ValueError, match="Invalid soul"):
        await corrupt.ensure_dossier_anchors(SCOPE, embedding_client=FakeEmbedClient())
    assert len(corrupt.database.memory_category_repo.list_anchor_categories(SCOPE)) == 2


@pytest.mark.asyncio
async def test_activity_quota_and_compact_index_are_deterministic(tmp_path) -> None:
    service = _service(tmp_path)
    anchors = await service.ensure_dossier_anchors(SCOPE, embedding_client=FakeEmbedClient())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    oldest_ids: set[str] = set()
    newest_names: list[str] = []
    for kind in ("lore", "topic", "goal"):
        for index in range(31):
            category = _category(
                service,
                f"{kind}-{index:02d}",
                kind=kind,
                evidence=start + timedelta(days=index),
            )
            if index == 0:
                oldest_ids.add(category.id)
            if index == 30:
                newest_names.append(category.name)
    dormant = _category(service, "dormant", kind="topic")

    active = service.list_active_dossiers(SCOPE)
    active_ids = {category.id for category in active}
    assert len(active) == 92
    assert {row.id for row in anchors.values()} <= active_ids
    assert not oldest_ids & active_ids
    assert dormant.id not in active_ids
    assert dormant.id in {category.id for category in service.list_inactive_dossiers(SCOPE)}

    index_lines = service.build_dossier_index(SCOPE).splitlines()
    assert len(index_lines) == 20
    assert index_lines[:3] == [
        "- goal-30: goal-30 description",
        "- lore-30: lore-30 description",
        "- topic-30: topic-30 description",
    ]
    assert not any(row.name in "\n".join(index_lines) for row in anchors.values())
    assert set(newest_names) <= {line.split(":", 1)[0].removeprefix("- ") for line in index_lines}


@pytest.mark.asyncio
async def test_update_dossier_refreshes_identity_and_derived_index(tmp_path) -> None:
    service = _service(tmp_path, active_dossiers_per_kind=1)
    client = FakeEmbedClient()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    alpha = _category(service, "Alpha", kind="topic", evidence=start + timedelta(days=2))
    beta = _category(service, "Beta", kind="topic", evidence=start + timedelta(days=1))
    assert "Beta" not in service.build_dossier_index(SCOPE)

    updated = await service.update_dossier(
        alpha.id,
        SCOPE,
        name="Health",
        description="A current health dossier",
        kind="goal",
        last_evidence_at=start,
        embedding_client=client,
    )

    rendered = service.build_dossier_index(SCOPE)
    assert updated.embedding == [1.0, 0.0]
    assert "- Beta: Beta description" in rendered
    assert "- Health: A current health dossier" in rendered
    assert "Alpha" not in rendered
    assert client.calls == [["Health: A current health dossier"]]


@pytest.mark.asyncio
async def test_identity_and_content_search_boundaries_and_cache(tmp_path) -> None:
    service = _service(tmp_path)
    client = FakeEmbedClient()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    health = _category(
        service,
        "Health",
        evidence=now,
        embedding=[1.0, 0.0],
        summary="Health facts",
    )
    work = _category(
        service,
        "Work",
        evidence=now,
        embedding=[0.0, 1.0],
        summary="Work facts",
    )
    dormant = _category(service, "Dormant", embedding=[0.8, 0.2], summary="Inactive private prose")

    identity = await service.search_dossiers(
        [1.0, 0.0], where=SCOPE, view="identity", activity="active", limit=1, min_score=0.5
    )
    assert [(row.id, score) for row, score in identity][0][0] == health.id
    all_hits = await service.search_dossiers(
        [1.0, 0.0], where=SCOPE, view="identity", activity="all", limit=3, min_score=-1.0
    )
    assert {row.id for row, _score in all_hits} == {health.id, work.id, dormant.id}
    assert all(row.summary is None for row, _score in all_hits)
    assert service.database.memory_category_repo.list_categories(SCOPE)[dormant.id].summary == "Inactive private prose"

    first = await service.search_dossiers(
        [1.0, 0.0],
        where=SCOPE,
        view="content",
        activity="active",
        limit=2,
        min_score=-1.0,
        embedding_client=client,
    )
    second = await service.search_dossiers(
        [1.0, 0.0],
        where=SCOPE,
        view="content",
        activity="active",
        limit=2,
        min_score=-1.0,
        embedding_client=client,
    )
    assert [row.id for row, _score in first] == [row.id for row, _score in second]
    assert client.calls == [["Health: Health description\nHealth facts", "Work: Work description\nWork facts"]]

    service.database.memory_category_repo.update_category(category_id=work.id, summary="Health at work")
    await service.search_dossiers(
        [1.0, 0.0],
        where=SCOPE,
        view="content",
        activity="active",
        limit=2,
        min_score=-1.0,
        embedding_client=client,
    )
    assert client.calls[-1] == ["Work: Work description\nHealth at work"]
    with pytest.raises(ValueError, match="limited to active"):
        await service.search_dossiers(
            [1.0, 0.0],
            where=SCOPE,
            view="content",
            activity="inactive",
            limit=1,
            min_score=0.0,
            embedding_client=client,
        )


@pytest.mark.asyncio
async def test_identity_search_fails_loud_on_missing_or_mismatched_vectors(tmp_path) -> None:
    missing = _service(tmp_path, "missing.db")
    category = _category(missing, "Missing", evidence=datetime(2026, 1, 1, tzinfo=UTC))
    with missing.database._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE categories SET embedding = NULL WHERE id = ?", (category.id,))
    with pytest.raises(ValueError, match="missing its identity embedding"):
        await missing.search_dossiers(
            [1.0, 0.0], where=SCOPE, view="identity", activity="active", limit=1, min_score=0.0
        )

    mismatch = _service(tmp_path, "mismatch.db")
    _category(
        mismatch,
        "Mismatch",
        evidence=datetime(2026, 1, 1, tzinfo=UTC),
        embedding=[1.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        await mismatch.search_dossiers(
            [1.0, 0.0], where=SCOPE, view="identity", activity="active", limit=1, min_score=0.0
        )


@pytest.mark.asyncio
async def test_sparse_memorize_context_reuses_bounded_category_sets(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path)
    client = FakeEmbedClient()
    anchors = await service.ensure_dossier_anchors(SCOPE, embedding_client=client)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for index, name in enumerate(("Health", "Work", "Travel", "Art")):
        _category(
            service,
            name,
            evidence=now + timedelta(days=index),
            embedding=[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0],
            summary=f"{name} complete active prose",
        )
    for index in range(12):
        _category(
            service,
            f"Dormant {index:02d}",
            embedding=[1.0, 0.0],
            summary=f"private inactive prose {index}",
        )

    expected_active = service.list_active_dossiers(SCOPE)
    active_calls = 0
    original_active = service.list_active_dossiers
    category_reads = 0
    original_categories = service.database.memory_category_repo.list_categories

    def traced_active(where):
        nonlocal active_calls
        active_calls += 1
        return original_active(where)

    def traced_categories(where=None):
        nonlocal category_reads
        category_reads += 1
        return original_categories(where)

    monkeypatch.setattr(service, "list_active_dossiers", traced_active)
    monkeypatch.setattr(service.database.memory_category_repo, "list_categories", traced_categories)
    monkeypatch.setattr(
        service,
        "list_inactive_dossiers",
        lambda _where: pytest.fail("selector must derive inactive rows from the preloaded category set"),
    )
    supplied_category_ids: list[set[str]] = []
    original_search = service.search_dossiers

    async def traced_search(query_embedding, **kwargs):
        supplied = kwargs.get("categories")
        assert supplied is not None
        supplied_category_ids.append({category.id for category in supplied})
        return await original_search(query_embedding, **kwargs)

    monkeypatch.setattr(service, "search_dossiers", traced_search)

    client.calls.clear()
    episodes = [
        {"title": "Health day", "summary": "A full health summary"},
        {"title": "Work day", "summary": "A full work summary"},
        {"title": "Mixed day", "summary": "A full mixed summary"},
    ]
    result = await service.select_memorize_dossier_context(
        episodes,
        SCOPE,
        narrative_self="  current self narrative  ",
        embedding_client=client,
    )

    anchor_ids = {row.id for row in anchors.values()}
    relevant = result["relevant_dossiers"]
    assert active_calls == 1
    assert category_reads == 2  # Existing anchor validation plus the selector's one full read.
    assert len(supplied_category_ids) == 9
    assert all(not anchor_ids & category_ids for category_ids in supplied_category_ids)
    assert len(relevant) <= 9
    assert len({row.id for row in relevant}) == len(relevant)
    assert not anchor_ids & {row.id for row in relevant}
    assert all(row.summary and "complete active prose" in row.summary for row in relevant)
    assert [row.id for row in result["anchor_dossiers"]] == [anchors["soul"].id, anchors["user"].id]
    assert result["narrative_self"] == "current self narrative"
    assert client.calls[0] == [
        "Health day: A full health summary",
        "Work day: A full work summary",
        "Mixed day: A full mixed summary",
    ]

    choice_lines = result["categories_str"].splitlines()
    assert choice_lines[: len(expected_active)] == [f"- {row.name}" for row in expected_active]
    assert len(choice_lines) == len(expected_active) + 10
    assert all(": Dormant " in line for line in choice_lines[len(expected_active) :])
    assert "private inactive prose" not in str(result)
    anchor_names = {row.name for row in anchors.values()}
    assert not anchor_names & {
        line.split(":", 1)[0].removeprefix("- ")
        for line in result["dossier_index"].splitlines()
    }
    assert len(result["dossier_index"].splitlines()) == 4

    single = await service.select_memorize_dossier_context(
        episodes[:1],
        SCOPE,
        narrative_self=None,
        embedding_client=client,
    )
    repeated = await service.select_memorize_dossier_context(
        episodes[:1],
        SCOPE,
        narrative_self=None,
        embedding_client=client,
    )
    assert single["narrative_self"] is None
    assert len(single["relevant_dossiers"]) <= 3
    assert [row.id for row in single["relevant_dossiers"]] == [
        row.id for row in repeated["relevant_dossiers"]
    ]


@pytest.mark.asyncio
async def test_sparse_memorize_context_validates_before_database_work(tmp_path) -> None:
    service = _service(tmp_path)
    client = FakeEmbedClient()
    invalid = (
        [],
        [{"title": "", "summary": "summary"}],
        [{"title": "title", "summary": "summary"}] * 4,
    )
    for episodes in invalid:
        with pytest.raises(ValueError):
            await service.select_memorize_dossier_context(
                episodes,
                SCOPE,
                narrative_self=None,
                embedding_client=client,
            )
    assert client.calls == []
    assert service.database.memory_category_repo.list_categories(SCOPE) == {}


def test_memory_refs_are_strict_scoped_and_resolve_merged_rows(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    first = store.memory_item_repo.create_item(
        memory_type="episode", summary="first", embedding=[1.0, 0.0], user_data=SCOPE
    )
    merged = store.memory_item_repo.create_item(
        memory_type="episode", summary="merged", embedding=[1.0, 0.0], user_data=SCOPE
    )
    other = store.memory_item_repo.create_item(
        memory_type="episode", summary="other", embedding=[1.0, 0.0], user_data=OTHER_SCOPE
    )
    refs = store.memory_item_repo.backfill_memory_refs(SCOPE)
    other_refs = store.memory_item_repo.backfill_memory_refs(OTHER_SCOPE)
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET merged_into = ? WHERE id = ?", (first.id, merged.id))

    merged_ref = refs[merged.id]
    assert service.format_memory_ref(merged_ref) == f"[M{merged_ref}]"
    assert service.parse_memory_ref(f"[M{merged_ref}]") == merged_ref
    assert service.resolve_memory_ref(f"[M{merged_ref}]", SCOPE).id == merged.id
    assert service.resolve_memory_ref(other_refs[other.id], OTHER_SCOPE).id == other.id
    with pytest.raises(KeyError):
        service.resolve_memory_ref(merged_ref, OTHER_SCOPE)
    for invalid in ("[m1]", "[M0]", "M1", " [M1]", "[M1] "):
        with pytest.raises(ValueError):
            service.parse_memory_ref(invalid)
    with pytest.raises(ValueError):
        service.format_memory_ref(0)


def test_relation_caller_session_does_not_mutate_cache(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    category = _category(service, "Health")
    item = store.memory_item_repo.create_item(
        memory_type="episode", summary="health", embedding=[1.0, 0.0], user_data=SCOPE
    )
    relation = store.category_item_repo.link_item_category(item.id, category.id, SCOPE)
    assert [cached.id for cached in store.category_item_repo.relations] == [relation.id]
    store.category_item_repo.relations.clear()

    with store._sessions.session() as session:
        assert len(store.category_item_repo.list_relations(SCOPE, session=session)) == 1
        assert store.category_item_repo.relations == []

    assert len(store.category_item_repo.list_relations(SCOPE)) == 1
    assert len(store.category_item_repo.relations) == 1

    rolled_back = store.memory_item_repo.create_item(
        memory_type="episode", summary="rolled back", embedding=[1.0, 0.0], user_data=SCOPE
    )
    with store._sessions.session() as session:
        store.category_item_repo.link_item_category(
            rolled_back.id, category.id, SCOPE, session=session
        )
        assert all(rel.item_id != rolled_back.id for rel in store.category_item_repo.relations)
        session.rollback()
    assert store.category_item_repo.list_relations({"item_id": rolled_back.id}) == []


def test_due_dossiers_cover_first_revision_and_watermark(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    due = _category(service, "Due")
    clean = _category(service, "Clean")
    due_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="due", embedding=[1.0, 0.0], user_data=SCOPE
    )
    clean_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="clean", embedding=[1.0, 0.0], user_data=SCOPE
    )
    store.category_item_repo.link_item_category(due_item.id, due.id, SCOPE)
    clean_relation = store.category_item_repo.link_item_category(clean_item.id, clean.id, SCOPE)
    revised_at = clean_relation.created_at + timedelta(seconds=1)
    store.memory_category_repo.update_category(
        category_id=clean.id, last_revised_at=revised_at
    )

    assert [category.id for category in service.list_due_dossiers(SCOPE)] == [due.id]
    assert [
        category.id
        for category in service.list_dossiers_revised_since(SCOPE, revised_after=None)
    ] == [clean.id]
    with pytest.raises(ValueError, match="not due for revision"):
        service.prepare_dossier_revision(clean.id, SCOPE)


def test_due_dossiers_include_linked_merge_and_supersession(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    merged_category = _category(service, "Merged")
    superseded_category = _category(service, "Superseded")
    merged_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="merged", embedding=[1.0, 0.0], user_data=SCOPE
    )
    superseded_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="superseded", embedding=[1.0, 0.0], user_data=SCOPE
    )
    survivor = store.memory_item_repo.create_item(
        memory_type="episode", summary="survivor", embedding=[1.0, 0.0], user_data=SCOPE
    )
    merged_rel = store.category_item_repo.link_item_category(
        merged_item.id, merged_category.id, SCOPE
    )
    superseded_rel = store.category_item_repo.link_item_category(
        superseded_item.id, superseded_category.id, SCOPE
    )
    revised_at = max(merged_rel.created_at, superseded_rel.created_at) + timedelta(seconds=1)
    for category in (merged_category, superseded_category):
        store.memory_category_repo.update_category(
            category_id=category.id,
            last_revised_at=revised_at,
        )
    assert service.list_due_dossiers(SCOPE) == []

    store.memory_item_repo.update_item(item_id=merged_item.id, merged_into=survivor.id)
    store.triple_repo.add(
        Triple(
            subject_id=superseded_item.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=survivor.id,
            object_kind="memory",
            source_memory_id=survivor.id,
        ),
        user_data=SCOPE,
    )

    assert {category.id for category in service.list_due_dossiers(SCOPE)} == {
        merged_category.id,
        superseded_category.id,
    }
    assert {
        category.id
        for category in service.list_dossiers_revised_since(
            SCOPE,
            revised_after=revised_at - timedelta(seconds=1),
        )
    } == {merged_category.id, superseded_category.id}


def test_prepare_dossier_revision_bounds_actionable_evidence(tmp_path) -> None:
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 1}})
    store = service.database
    category = _category(service, "Health", kind="goal", embedding=[1.0, 0.0])
    cited = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="cited", embedding=[0.0, 1.0], user_data=SCOPE
    )
    cited_unlinked = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="cited unlinked", embedding=[0.0, 1.0], user_data=SCOPE
    )
    untouched = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="untouched", embedding=[0.0, 1.0], user_data=SCOPE
    )
    cleanup = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="cleanup", embedding=[0.0, 1.0], user_data=SCOPE
    )
    pending = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="pending", embedding=[0.0, 1.0], user_data=SCOPE
    )
    candidate = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="candidate", embedding=[1.0, 0.0], user_data=SCOPE
    )
    refs = store.memory_item_repo.backfill_memory_refs(SCOPE)

    old_relations = [
        store.category_item_repo.link_item_category(item.id, category.id, SCOPE)
        for item in (cited, untouched, cleanup)
    ]
    revised_at = max(relation.created_at for relation in old_relations) + timedelta(microseconds=1)
    store.memory_category_repo.update_category(
        category_id=category.id,
        summary=(
            f"A cited fact [M{refs[cited.id]}] and unlinked fact "
            f"[M{refs[cited_unlinked.id]}] with an ordinary [Speaker] label."
        ),
        last_revised_at=revised_at,
    )
    store.category_item_repo.link_item_category(pending.id, category.id, SCOPE)
    store.memory_item_repo.update_item(item_id=cleanup.id, merged_into=candidate.id)

    bundle = service.prepare_dossier_revision(
        category.id,
        SCOPE,
        active_life_goals=["  active goal  "],
        removed_life_goals=["removed goal"],
    )

    assert service.extract_memory_refs("[Speaker] [M] [M2] [M2]") == [2]
    with pytest.raises(ValueError, match="Invalid memory reference"):
        service.extract_memory_refs("bad [M0]")
    assert {item.id for item in bundle["cited_items"]} == {cited.id, cited_unlinked.id}
    assert bundle["cited_unlinked_item_ids"] == [cited_unlinked.id]
    assert [item.id for item in bundle["pending_items"]] == [pending.id]
    assert bundle["cleanup_memberships"] == [{
        "item_id": cleanup.id,
        "memory_ref": refs[cleanup.id],
        "lineage_state": "merged",
        "merged_into": candidate.id,
    }]
    assert [item.id for item in bundle["candidate_items"]] == [candidate.id]
    assert bundle["untouched_item_ids"] == [untouched.id]
    assert bundle["active_life_goals"] == ["active goal"]
    assert bundle["removed_life_goals"] == ["removed goal"]


def test_prepare_dossier_revision_requires_refs_and_goal_kind(tmp_path) -> None:
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 0}})
    store = service.database
    category = _category(service, "Work", kind="topic")
    item = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="work", embedding=[1.0, 0.0], user_data=SCOPE
    )
    store.category_item_repo.link_item_category(item.id, category.id, SCOPE)

    with pytest.raises(ValueError, match="lack stable references"):
        service.prepare_dossier_revision(category.id, SCOPE)

    store.memory_item_repo.backfill_memory_refs(SCOPE)
    bundle = service.prepare_dossier_revision(
        category.id,
        SCOPE,
        active_life_goals=["active goal"],
        removed_life_goals=["removed goal"],
    )
    assert bundle["active_life_goals"] == []
    assert bundle["removed_life_goals"] == []
