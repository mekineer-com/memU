from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from defusedxml import ElementTree
from pydantic import BaseModel

from memu.app.dossier import DossierRevisionStaleError
from memu.app.dossier_revision import (
    label_sections,
    parse_anchor_revisions,
    parse_dossier_revision_batch,
    render_memory_records,
    revision_status_items,
)
from memu.app.service import MemoryService
from memu.database.models import DossierCandidate, MemoryCategory, MemoryItem, Triple


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


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    async def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return self.response


def test_single_markdown_section_is_structured_dossier_prose() -> None:
    assert label_sections("## unlabeled") == ("S1\n## unlabeled", [("S1", "## unlabeled")])


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


def _seed_anchors(service: MemoryService) -> dict[str, MemoryCategory]:
    repo = service.database.memory_category_repo
    return {
        role: repo.get_or_create_category(
            name=name,
            description=f"A personal account of {name}.",
            embedding=[1.0, 0.0],
            user_data=SCOPE,
            kind="lore",
            lore_subtype="person",
            anchor_role=role,
        )
        for role, name in (("soul", SCOPE["soul_id"]), ("user", SCOPE["user_id"]))
    }


def test_cutover_readiness_accepts_empty_and_valid_scopes(tmp_path) -> None:
    service = _service(tmp_path)
    service.require_dossier_cutover_ready(SCOPE)

    anchors = _seed_anchors(service)
    category = _category(service, "Health")
    cleanup = _category(service, "Past project")
    item = service.database.memory_item_repo.create_item(
        memory_type="episode",
        summary="A steady routine",
        embedding=[1.0, 0.0],
        user_data=SCOPE,
    )
    obsolete = service.database.memory_item_repo.create_item(
        memory_type="episode", summary="Old project state", embedding=[1.0, 0.0], user_data=SCOPE
    )
    survivor = service.database.memory_item_repo.create_item(
        memory_type="episode", summary="Current project state", embedding=[1.0, 0.0], user_data=SCOPE
    )
    refs = {row.id: row.memory_ref for row in (item, obsolete, survivor)}
    service.database.memory_category_repo.update_category(
        category_id=category.id,
        summary=f"## Health\nA steady routine [M{refs[item.id]}].",
    )
    service.database.category_item_repo.link_item_category(item.id, category.id, SCOPE)
    service.database.category_item_repo.link_item_category(obsolete.id, cleanup.id, SCOPE)
    service.database.memory_item_repo.update_item(item_id=obsolete.id, merged_into=survivor.id)

    service.require_dossier_cutover_ready(SCOPE)
    assert set(anchors) == {"soul", "user"}
    assert category.approved_description is None
    service.database.close()
    _service(tmp_path).require_dossier_cutover_ready(SCOPE)


def test_cutover_readiness_rejects_unmigrated_and_unlinked_citations(tmp_path) -> None:
    service = _service(tmp_path)
    item = service.database.memory_item_repo.create_item(
        memory_type="episode",
        summary="An unreferenced memory",
        embedding=[1.0, 0.0],
        user_data=SCOPE,
    )
    with service.database._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = NULL WHERE id = ?", (item.id,))
    with pytest.raises(ValueError, match=r"positive \[M#\]"):
        service.require_dossier_cutover_ready(SCOPE)

    service.database.memory_item_repo.backfill_memory_refs(SCOPE)
    _seed_anchors(service)
    category = _category(service, "Health", summary="## Health\nAn unlinked citation [M1].")
    service.database.memory_category_repo.approve_category_summary(category.id, SCOPE)
    with pytest.raises(ValueError, match=r"unlinked \[M1\]"):
        service.require_dossier_cutover_ready(SCOPE)

    service.database.category_item_repo.link_item_category(item.id, category.id, SCOPE)
    service.require_dossier_cutover_ready(SCOPE)


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


def test_dossiers_for_segments_use_exact_active_evidence_and_ignore_recall_quota(tmp_path) -> None:
    service = _service(tmp_path, active_dossiers_per_kind=0)
    store = service.database
    alpha = _category(service, "Alpha", scope=SCOPE)
    beta = _category(service, "beta", scope=SCOPE)
    tied = _category(service, "Beta", scope=SCOPE)
    merged = _category(service, "Merged", scope=SCOPE)
    superseded = _category(service, "Superseded", scope=SCOPE)
    other = _category(service, "Other", scope=OTHER_SCOPE)

    def add(category, summary, segment_id, *, scope=SCOPE):
        item = store.memory_item_repo.create_item(
            memory_type="episode",
            summary=summary,
            embedding=[1.0, 0.0],
            user_data=scope,
            segment_id=segment_id,
        )
        store.category_item_repo.link_item_category(item.id, category.id, scope)
        return item

    alpha_item = add(alpha, "alpha", "selected")
    beta_item = add(beta, "beta", "selected")
    tied_item = add(tied, "tied", "selected")
    add(beta, "wrong segment", "other-segment")
    merged_item = add(merged, "merged", "selected")
    superseded_item = add(superseded, "superseded", "selected")
    survivor = add(alpha, "survivor", "other-segment")
    add(other, "other scope", "selected", scope=OTHER_SCOPE)

    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE memory_items SET created_at = ? WHERE id = ?",
            ("2026-01-02T00:00:00+00:00", alpha_item.id),
        )
        for item in (beta_item, tied_item):
            conn.exec_driver_sql(
                "UPDATE memory_items SET created_at = ? WHERE id = ?",
                ("2026-01-03T00:00:00+00:00", item.id),
            )
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

    assert service.list_active_dossiers(SCOPE) == []
    tied_ids = sorted((beta.id, tied.id))
    assert [row.id for row in service.list_dossiers_for_segments(SCOPE, segment_ids=["selected"])] == [
        alpha.id,
        merged.id,
        superseded.id,
        *tied_ids,
    ]
    assert service.list_dossiers_for_segments(SCOPE, segment_ids=[]) == []


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
    assert not store.category_item_repo.unlink_item_category(
        item.id,
        category.id,
        OTHER_SCOPE,
    )
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


def test_dossier_repository_writes_share_caller_transaction(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    category = _category(service, "Health")
    item = store.memory_item_repo.create_item(
        memory_type="episode", summary="health", embedding=[1.0, 0.0], user_data=SCOPE
    )
    store.memory_item_repo.backfill_memory_refs(SCOPE)
    store.category_item_repo.link_item_category(item.id, category.id, SCOPE)
    cached = store.memory_category_repo.list_categories(SCOPE)[category.id]

    with pytest.raises(KeyError):
        store.memory_category_repo.update_category(
            category_id=category.id,
            description="wrong scope",
            where=OTHER_SCOPE,
        )

    with store._sessions.session() as session:
        changed = store.memory_category_repo.update_category(
            category_id=category.id,
            description="changed",
            where=SCOPE,
            session=session,
        )
        store.memory_item_repo.approve_item(item.id, SCOPE, session=session)
        store.category_item_repo.unlink_item_category(
            item.id, category.id, SCOPE, session=session
        )
        assert changed.description == "changed"
        assert store.memory_category_repo.categories[category.id] == cached
        session.rollback()

    restored = store.memory_category_repo.list_categories(SCOPE)[category.id]
    restored_item = store.memory_item_repo.list_items_by_ids({item.id}, SCOPE)[item.id]
    assert restored.description == "Health description"
    assert restored_item.approved_at is None
    assert store.category_item_repo.list_relations({"item_id": item.id}) != []


def test_due_dossiers_cover_first_revision_and_watermark(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    due = _category(service, "Due")
    clean = _category(service, "Clean")
    excluded = _category(service, "Excluded")
    anchor = _seed_anchors(service)["soul"]
    due_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="due", embedding=[1.0, 0.0], user_data=SCOPE,
        segment_id="selected",
    )
    clean_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="clean", embedding=[1.0, 0.0], user_data=SCOPE
    )
    anchor_item = store.memory_item_repo.create_item(
        memory_type="episode", summary="anchor", embedding=[1.0, 0.0], user_data=SCOPE
    )
    excluded_item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="excluded",
        embedding=[1.0, 0.0],
        user_data=SCOPE,
        segment_id="later",
    )
    store.category_item_repo.link_item_category(due_item.id, due.id, SCOPE)
    clean_relation = store.category_item_repo.link_item_category(clean_item.id, clean.id, SCOPE)
    store.category_item_repo.link_item_category(anchor_item.id, anchor.id, SCOPE)
    store.category_item_repo.link_item_category(excluded_item.id, excluded.id, SCOPE)
    revised_at = clean_relation.created_at + timedelta(seconds=1)
    store.memory_category_repo.update_category(
        category_id=clean.id, last_revised_at=revised_at
    )

    assert {category.id for category in service.list_due_dossiers(SCOPE)} == {due.id, excluded.id}
    assert [
        category.id
        for category in service.list_due_dossiers(SCOPE, segment_ids=["selected"])
    ] == [due.id]
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
    revised_at = max(merged_rel.created_at, superseded_rel.created_at)
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


def test_prepare_dossier_revision_bounds_actionable_evidence(tmp_path) -> None:
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 1}})
    store = service.database
    anchors = _seed_anchors(service)
    store.memory_category_repo.update_category(
        category_id=anchors["soul"].id,
        summary="I remember a bond [M999] with care.",
    )
    store.memory_category_repo.update_category(
        category_id=anchors["user"].id,
        summary="My human values patience [M998].",
    )
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
    store.triple_repo.add(
        Triple(
            subject_id=cleanup.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=candidate.id,
            object_kind="memory",
            source_memory_id=candidate.id,
        ),
        user_data=SCOPE,
    )

    bundle = service.prepare_dossier_revision(
        category.id,
        SCOPE,
        narrative_self="  Curious {and} attentive <always>.  ",
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
        "lineage_state": "superseded",
        "merged_into": None,
    }]
    assert [item.id for item in bundle["cleanup_items"]] == [cleanup.id]
    assert [item.id for item in bundle["candidate_items"]] == [candidate.id]
    assert bundle["untouched_item_ids"] == [untouched.id]
    assert bundle["active_life_goals"] == ["active goal"]
    assert bundle["removed_life_goals"] == ["removed goal"]
    assert bundle["narrative_self"] == "Curious {and} attentive <always>."
    assert "[M999]" not in bundle["soul_presence"]
    assert "[M998]" not in bundle["soul_presence"]
    assert "# Your dossier" in bundle["soul_presence"]
    assert "# Your human's dossier" in bundle["soul_presence"]
    assert bundle["target_words"] == 300


def test_revision_memory_records_are_one_line() -> None:
    item = MemoryItem(
        resource_id=None,
        memory_type="knowledge",
        summary="First line.\n  Second line.",
        memory_ref=12,
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
    )
    assert render_memory_records([item]) == "[M12] (2026-07-18) First line. Second line."


def test_revision_status_precedence_is_disjoint() -> None:
    def item(memory_ref: int) -> MemoryItem:
        return MemoryItem(
            resource_id=None,
            memory_type="knowledge",
            summary=f"Memory {memory_ref}",
            memory_ref=memory_ref,
        )

    purged, cited, pending, search = (item(memory_ref) for memory_ref in range(1, 5))
    statuses = revision_status_items({
        "cleanup_items": [purged],
        "cited_items": [purged, cited],
        "pending_items": [purged, cited, pending],
        "candidate_items": [purged, cited, pending, search],
    })

    assert {
        status: [memory.memory_ref for memory in memories]
        for status, memories in statuses.items()
    } == {"cited": [2], "search": [4], "purged": [1], "pending": [3]}


def test_prepare_dossier_revision_requires_refs_and_goal_kind(tmp_path) -> None:
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 0}})
    store = service.database
    _seed_anchors(service)
    category = _category(service, "Work", kind="topic")
    item = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="work", embedding=[1.0, 0.0], user_data=SCOPE
    )
    store.category_item_repo.link_item_category(item.id, category.id, SCOPE)

    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE memory_items SET memory_ref = NULL WHERE id = ?", (item.id,))
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


def _revision_case(tmp_path, *, summary: str = "", kind: str = "topic"):
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 1}})
    store = service.database
    anchors = _seed_anchors(service)
    category = _category(
        service,
        "Health",
        kind=kind,
        embedding=[1.0, 0.0],
        summary=summary,
    )
    pending = store.memory_item_repo.create_item(
        memory_type="knowledge",
        summary="River kept a {daily} record <carefully>.",
        embedding=[0.0, 1.0],
        happened_at=datetime(2026, 7, 18, tzinfo=UTC),
        user_data=SCOPE,
    )
    candidate = store.memory_item_repo.create_item(
        memory_type="knowledge",
        summary="River prepared questions before an appointment.",
        embedding=[1.0, 0.0],
        happened_at=datetime(2026, 7, 19, tzinfo=UTC),
        user_data=SCOPE,
    )
    refs = {pending.id: pending.memory_ref, candidate.id: candidate.memory_ref}
    store.category_item_repo.link_item_category(pending.id, category.id, SCOPE)
    bundle = service.prepare_dossier_revision(
        category.id,
        SCOPE,
        narrative_self="Warm {and} steady <voice>.",
    )
    return service, store, anchors, category, pending, candidate, refs, bundle


@pytest.mark.asyncio
async def test_generate_dossier_revision_replaces_unstructured_prose_once(tmp_path) -> None:
    service, store, _anchors, category, pending, candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="An earlier account.",
    )
    response = f"""<dossier_revision dossier_id="{category.id}">
  <description>A personal brief.</description>
  <prose_action>replace</prose_action>
  <prose>## Health\nA living account [M{refs[pending.id]}] [M{refs[candidate.id]}].</prose>
  <prose_patches></prose_patches>
  <decisions>
    <decision ref="[M{refs[pending.id]}]" action="add" />
    <decision ref="[M{refs[candidate.id]}]" action="add" />
  </decisions>
</dossier_revision>"""
    client = FakeChatClient(response)
    before = store.category_item_repo.list_relations(SCOPE)

    result = await service.generate_dossier_revision(bundle, chat_client=client)

    assert len(client.calls) == 1
    user_prompt, system_prompt = client.calls[0]
    assert "River kept a {daily} record <carefully>." in user_prompt
    assert "Warm {and} steady <voice>." in str(system_prompt)
    assert user_prompt.index("# cited_member") < user_prompt.index("# search_result")
    assert user_prompt.index("# search_result") < user_prompt.index("# purged_member")
    assert user_prompt.index("# purged_member") < user_prompt.index("# pending_member")
    assert result["add_item_ids"] == sorted([pending.id, candidate.id])
    assert result["cited_item_ids"] == sorted([pending.id, candidate.id])
    assert store.category_item_repo.list_relations(SCOPE) == before


@pytest.mark.asyncio
async def test_generate_dossier_revision_patches_sections_without_touching_others(tmp_path) -> None:
    original_first = "## Daily Care\nOriginal first section.\n\n"
    original_second = "## Timeline\n- 2026-07-01: Earlier event.\n"
    service, _store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary=original_first + original_second,
    )
    response = f"""<dossier_revision dossier_id="{category.id}">
  <description>Care became more deliberate.</description>
  <prose_action>patch</prose_action>
  <prose></prose>
  <prose_patches>
    <section ref="S2" action="replace"><body>## Timeline
- 2026-07-18: River started a daily record [M{refs[pending.id]}].</body></section>
    <section ref="S1" action="add_after"><body>## Current Practice
River records each day with care [M{refs[pending.id]}].</body></section>
  </prose_patches>
  <decisions><decision ref="[M{refs[pending.id]}]" action="add" /></decisions>
</dossier_revision>"""

    result = await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(response))

    assert result["resulting_prose"].startswith(original_first)
    assert "## Current Practice" in result["resulting_prose"]
    assert "## Timeline\n- 2026-07-18" in result["resulting_prose"]


@pytest.mark.asyncio
async def test_generate_dossier_revision_patches_single_unlabeled_section(tmp_path) -> None:
    service, _store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="## unlabeled",
    )
    response = f"""<dossier_revision dossier_id="{category.id}">
  <description>A warm account of daily care.</description>
  <prose_action>patch</prose_action><prose></prose>
  <prose_patches><section ref="S1" action="replace"><body>## Daily Care
River keeps a thoughtful daily record [M{refs[pending.id]}].</body></section></prose_patches>
  <decisions><decision ref="[M{refs[pending.id]}]" action="add" /></decisions>
</dossier_revision>"""

    result = await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(response))

    assert result["resulting_prose"].startswith("## Daily Care")
    assert result["add_item_ids"] == [pending.id]


def test_parse_dossier_revision_batch_patches_blank_canonical_section(tmp_path) -> None:
    _service_, _store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="",
    )
    response = f"""<dossier_revisions>
  <dossier_revision dossier_id="{category.id}">
    <description>A warm account of daily care.</description>
    <prose_action>patch</prose_action>
    <prose_patches><section ref="S1" action="replace"><body>## Daily Care
River keeps a thoughtful daily record [M{refs[pending.id]}].</body></section></prose_patches>
    <decisions><decision ref="[M{refs[pending.id]}]" action="add" /></decisions>
  </dossier_revision>
</dossier_revisions>"""

    [result] = parse_dossier_revision_batch(response, [bundle])

    assert result["resulting_prose"].startswith("## Daily Care")
    assert result["add_item_ids"] == [pending.id]


@pytest.mark.asyncio
async def test_generate_dossier_revision_keep_and_remove_section(tmp_path) -> None:
    current = "## Daily Care\nStable context.\n\n## Timeline\n- 2026-07-01: Earlier event."
    service, _store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary=current,
    )
    decision = f'<decision ref="[M{refs[pending.id]}]" action="add" />'
    keep = f"""<dossier_revision dossier_id="{category.id}">
  <description>Stable context.</description><prose_action>keep</prose_action>
  <prose></prose><prose_patches></prose_patches><decisions>{decision}</decisions>
</dossier_revision>"""
    removed = f"""<dossier_revision dossier_id="{category.id}">
  <description>The timeline remains.</description><prose_action>patch</prose_action>
  <prose></prose><prose_patches>
    <section ref="S1" action="remove"><body></body></section>
  </prose_patches><decisions>{decision}</decisions>
</dossier_revision>"""

    kept = await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(keep))
    patched = await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(removed))

    assert kept["resulting_prose"] == current
    assert patched["resulting_prose"] == "## Timeline\n- 2026-07-01: Earlier event."


@pytest.mark.asyncio
async def test_generate_dossier_revision_validates_decisions_and_xml(tmp_path) -> None:
    service, _store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(tmp_path)
    missing_pending = f"""<dossier_revision dossier_id="{category.id}">
  <description>Brief.</description><prose_action>replace</prose_action>
  <prose>## Health\nAccount.</prose><prose_patches></prose_patches><decisions></decisions>
</dossier_revision>"""
    with pytest.raises(ValueError, match="Pending memories require decisions"):
        await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(missing_pending))

    malformed_ref = missing_pending.replace(
        "<decisions></decisions>",
        f'<decisions><decision ref="[M0]" action="add" /></decisions>',
    )
    with pytest.raises(ValueError, match="Unknown memory decision reference"):
        await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(malformed_ref))

    wrapped = f"```xml\n{missing_pending}\n```"
    with pytest.raises(ValueError, match="Expected exact dossier_revision XML"):
        await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(wrapped))

    stray_patch_text = missing_pending.replace(
        "<prose_action>replace</prose_action>",
        "<prose_action>patch</prose_action>",
    ).replace(
        "<prose>## Health\nAccount.</prose><prose_patches></prose_patches>",
        '<prose></prose><prose_patches><section ref="S1" action="remove"><body></body></section>junk</prose_patches>',
    )
    with pytest.raises(ValueError, match="Unexpected text in prose_patches wrapper"):
        await service.generate_dossier_revision(
            bundle,
            chat_client=FakeChatClient(stray_patch_text),
        )

    valid = missing_pending.replace(
        "<decisions></decisions>",
        f'<decisions><decision ref="[M{refs[pending.id]}]" action="add" /></decisions>',
    )
    valid = valid.replace("Account.</prose>", f"Account [M{refs[pending.id]}].</prose>")
    result = await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(valid))
    assert result["add_item_ids"] == [pending.id]


@pytest.mark.asyncio
async def test_generate_dossier_revision_rejects_oversized_prompt_before_client_selection(
    tmp_path, monkeypatch
) -> None:
    service, _store, _anchors, _category_row, _pending, _candidate, _refs, bundle = _revision_case(
        tmp_path
    )
    bundle["narrative_self"] = "word " * 75_001
    monkeypatch.setattr(
        service,
        "_select_chat_client",
        lambda *_args, **_kwargs: pytest.fail("client selected before prompt preflight"),
    )
    with pytest.raises(ValueError, match="exceeds 100000 tokens"):
        await service.generate_dossier_revision(bundle)


@pytest.mark.asyncio
async def test_generate_dossier_revision_selects_existing_category_profile(
    tmp_path, monkeypatch
) -> None:
    service, _store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path
    )
    bundle["narrative_self"] = None
    response = f"""<dossier_revision dossier_id="{category.id}">
  <description>A personal brief.</description><prose_action>replace</prose_action>
  <prose>## Health\nAccount [M{refs[pending.id]}].</prose>
  <prose_patches></prose_patches>
  <decisions><decision ref="[M{refs[pending.id]}]" action="add" /></decisions>
</dossier_revision>"""
    client = FakeChatClient(response)
    selected: list[tuple[object, object]] = []

    def select(step_context, *, profile=None):
        selected.append((step_context, profile))
        return client

    monkeypatch.setattr(service, "_select_chat_client", select)
    await service.generate_dossier_revision(bundle)

    assert selected == [
        ({"operation": "dossier", "step_id": "revision"}, service.memorize_config.category_update_llm_profile)
    ]
    assert "# Your character, your personality, your voice" not in str(client.calls[0][1])


@pytest.mark.asyncio
async def test_generate_dynamic_category_review_reuses_profile_and_preflights(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path)
    item = MemoryItem(
        memory_ref=1,
        resource_id=None,
        memory_type="episode",
        summary="A small garden ritual became part of daily life.",
        embedding=[1.0, 0.0],
    )
    candidate = DossierCandidate(
        proposed_name="garden rituals",
        normalized_name="garden rituals",
        item_id=item.id,
    )
    soul_anchor = MemoryCategory(
        name="Test Soul",
        description="A growing sense of self.",
        kind="lore",
        anchor_role="soul",
        summary="## Becoming\nI treasure small rituals.",
    )
    bundle = {
        "cluster_id": "cluster_1",
        "candidate_ids": [candidate.id],
        "memory_count": 1,
        "memories": [{"item": item, "candidates": [candidate]}],
        "soul_anchor": soul_anchor,
        "existing_dossiers": [],
    }
    response = f"""<dynamic_dossier_review cluster_id="cluster_1">
  <action>create</action>
  <accepted_candidate_ids><candidate_id>{candidate.id}</candidate_id></accepted_candidate_ids>
  <rejected_candidate_ids></rejected_candidate_ids>
  <title>Garden Magic</title><description>A bright ritual of growing things.</description><kind>lore</kind>
</dynamic_dossier_review>"""
    client = FakeChatClient(response)
    selected: list[tuple[object, object]] = []

    def select(step_context, *, profile=None):
        selected.append((step_context, profile))
        return client

    monkeypatch.setattr(service, "_select_chat_client", select)
    decision = await service.generate_dynamic_category_review(bundle)

    assert decision["name"] == "Garden Magic"
    assert selected == [
        (
            {"operation": "dossier", "step_id": "dynamic_review"},
            service.memorize_config.category_update_llm_profile,
        )
    ]

    bundle["memories"][0]["item"] = item.model_copy(update={"summary": "word " * 80_000})
    monkeypatch.setattr(
        service,
        "_select_chat_client",
        lambda *_args, **_kwargs: pytest.fail("client selected before prompt preflight"),
    )
    with pytest.raises(ValueError, match="exceeds 100000 tokens"):
        await service.generate_dynamic_category_review(bundle)


@pytest.mark.asyncio
async def test_generate_dossier_revision_repairs_cited_unlinked_and_purges_lineage(
    tmp_path,
) -> None:
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 0}})
    store = service.database
    _seed_anchors(service)
    category = _category(service, "Health", summary="temporary")
    purged = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="An older account.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    survivor = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="The surviving account.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    cited_unlinked = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="A cited independent fact.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    inactive_unlinked = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="An inactive independent fact.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    refs = store.memory_item_repo.backfill_memory_refs(SCOPE)
    relation = store.category_item_repo.link_item_category(purged.id, category.id, SCOPE)
    store.memory_category_repo.update_category(
        category_id=category.id,
        summary=(
            f"Old [M{refs[purged.id]}], current [M{refs[cited_unlinked.id]}], "
            f"inactive [M{refs[inactive_unlinked.id]}]."
        ),
        last_revised_at=relation.created_at + timedelta(microseconds=1),
    )
    store.triple_repo.add(
        Triple(
            subject_id=purged.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=survivor.id,
            object_kind="memory",
            source_memory_id=survivor.id,
        ),
        user_data=SCOPE,
    )
    store.triple_repo.add(
        Triple(
            subject_id=inactive_unlinked.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=survivor.id,
            object_kind="memory",
            source_memory_id=survivor.id,
        ),
        user_data=SCOPE,
    )
    bundle = service.prepare_dossier_revision(category.id, SCOPE)
    assert {item.id for item in bundle["cleanup_items"]} == {purged.id, inactive_unlinked.id}
    response = f"""<dossier_revision dossier_id="{category.id}">
  <description>The current independent fact.</description><prose_action>replace</prose_action>
  <prose>## Health\nCurrent [M{refs[cited_unlinked.id]}].</prose>
  <prose_patches></prose_patches><decisions></decisions>
</dossier_revision>"""

    result = await service.generate_dossier_revision(bundle, chat_client=FakeChatClient(response))

    assert result["add_item_ids"] == [cited_unlinked.id]
    assert result["cleanup_item_ids"] == [purged.id]
    assert result["cited_item_ids"] == [cited_unlinked.id]


@pytest.mark.asyncio
async def test_apply_dossier_revision_commits_one_reviewed_result(tmp_path, monkeypatch) -> None:
    service, store, _anchors, category, pending, candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="## Health\nEarlier account.",
    )
    prose = (
        f"## Health\nRiver keeps a daily record [M{refs[pending.id]}] and prepares "
        f"questions [M{refs[candidate.id]}]."
    )
    decision = {
        "dossier_id": category.id,
        "description": "River approaches health with steady preparation.",
        "resulting_prose": prose,
        "add_item_ids": [pending.id, candidate.id],
        "remove_item_ids": [],
        "cleanup_item_ids": [],
        "cited_item_ids": [pending.id, candidate.id],
    }
    journal: list[dict] = []
    monkeypatch.setattr(
        "memu.app.dossier.append_category_summary_journal",
        lambda **entry: journal.append(entry),
    )
    client = FakeEmbedClient()

    revised = await service.apply_dossier_revision(
        bundle,
        decision,
        SCOPE,
        embedding_client=client,
    )

    relation_ids = {
        relation.item_id
        for relation in store.category_item_repo.list_relations(SCOPE)
        if relation.category_id == category.id
    }
    cached_relation_ids = {
        relation.item_id
        for relation in store.category_item_repo.relations
        if relation.category_id == category.id
    }
    approved = store.memory_item_repo.list_items_by_ids({pending.id, candidate.id}, SCOPE)
    assert relation_ids == {pending.id, candidate.id}
    assert cached_relation_ids == relation_ids
    assert all(item.approved_at is not None for item in approved.values())
    assert revised.description == decision["description"]
    assert revised.summary == prose
    assert revised.previous_description == "Health description"
    assert revised.previous_summary == "## Health\nEarlier account."
    assert revised.approved_description is None and revised.approved_summary is None
    assert revised.last_evidence_at == datetime(2026, 7, 19)
    assert client.calls == [[f"Health: {decision['description']}"]]
    assert journal[0]["edited_by"] == "dossier_revision"
    assert journal[0]["summary_before"] == "## Health\nEarlier account."
    assert journal[0]["summary_after"] == prose


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["category", "relation", "shown_item", "lineage"])
async def test_apply_dossier_revision_detects_snapshot_races(tmp_path, race) -> None:
    service, store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="## Health\nEarlier account.",
    )
    if race == "category":
        store.memory_category_repo.update_category(
            category_id=category.id,
            description="Concurrent description.",
        )
    elif race == "relation":
        extra = store.memory_item_repo.create_item(
            memory_type="knowledge",
            summary="concurrent relation",
            embedding=[1.0, 0.0],
            user_data=SCOPE,
        )
        store.category_item_repo.link_item_category(extra.id, category.id, SCOPE)
    elif race == "shown_item":
        store.memory_item_repo.update_item(item_id=pending.id, summary="concurrent edit")
    else:
        replacement = store.memory_item_repo.create_item(
            memory_type="knowledge",
            summary="replacement",
            embedding=[1.0, 0.0],
            user_data=SCOPE,
        )
        store.triple_repo.add(
            Triple(
                subject_id=pending.id,
                subject_kind="memory",
                predicate="evolved_into",
                object_id=replacement.id,
                object_kind="memory",
                source_memory_id=replacement.id,
            ),
            user_data=SCOPE,
        )
    decision = {
        "dossier_id": category.id,
        "description": category.description,
        "resulting_prose": f"## Health\nDaily record [M{refs[pending.id]}].",
        "add_item_ids": [pending.id],
        "remove_item_ids": [],
        "cleanup_item_ids": [],
        "cited_item_ids": [pending.id],
    }

    with pytest.raises(DossierRevisionStaleError, match="snapshot changed"):
        await service.apply_dossier_revision(bundle, decision, SCOPE)
    assert store.memory_item_repo.list_items_by_ids(
        {pending.id}, SCOPE, include_superseded=True
    )[pending.id].approved_at is None


@pytest.mark.asyncio
async def test_apply_dossier_revision_rolls_back_all_writes(tmp_path, monkeypatch) -> None:
    service, store, _anchors, category, pending, candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="## Health\nEarlier account.",
    )
    decision = {
        "dossier_id": category.id,
        "description": category.description,
        "resulting_prose": (
            f"## Health\nDaily record [M{refs[pending.id]}] and questions "
            f"[M{refs[candidate.id]}]."
        ),
        "add_item_ids": [pending.id, candidate.id],
        "remove_item_ids": [],
        "cleanup_item_ids": [],
        "cited_item_ids": [pending.id, candidate.id],
    }
    original_update = store.memory_category_repo.update_category

    def fail_transaction(**kwargs):
        if kwargs.get("session") is not None:
            raise RuntimeError("injected failure")
        return original_update(**kwargs)

    monkeypatch.setattr(store.memory_category_repo, "update_category", fail_transaction)
    with pytest.raises(RuntimeError, match="injected failure"):
        await service.apply_dossier_revision(bundle, decision, SCOPE)

    relations = store.category_item_repo.list_relations(SCOPE)
    assert {relation.item_id for relation in relations} == {pending.id}
    items = store.memory_item_repo.list_items_by_ids({pending.id, candidate.id}, SCOPE)
    assert all(item.approved_at is None for item in items.values())


@pytest.mark.asyncio
async def test_apply_dossier_revision_keeps_empty_dossier_text(tmp_path, monkeypatch) -> None:
    service, store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary=None,
    )
    old_prose = f"## Health\nHistorical account [M{refs[pending.id]}]."
    store.memory_category_repo.update_category(category_id=category.id, summary=old_prose)
    bundle = service.prepare_dossier_revision(category.id, SCOPE)
    journal: list[dict] = []
    monkeypatch.setattr(
        "memu.app.dossier.append_category_summary_journal",
        lambda **entry: journal.append(entry),
    )
    client = FakeEmbedClient()
    decision = {
        "dossier_id": category.id,
        "description": "Discarded rewrite.",
        "resulting_prose": "",
        "add_item_ids": [],
        "remove_item_ids": [pending.id],
        "cleanup_item_ids": [],
        "cited_item_ids": [],
    }

    revised = await service.apply_dossier_revision(
        bundle,
        decision,
        SCOPE,
        embedding_client=client,
    )

    assert revised.description == "Health description"
    assert revised.summary == "## Health\nHistorical account."
    assert revised.last_evidence_at is None and revised.last_revised_at is not None
    assert store.category_item_repo.list_relations({"category_id": category.id}) == []
    assert all(
        relation.category_id != category.id
        for relation in store.category_item_repo.relations
    )
    assert client.calls == [] and journal[0]["summary_before"] == old_prose


@pytest.mark.asyncio
async def test_apply_dossier_revision_journal_failure_keeps_commit(tmp_path, monkeypatch) -> None:
    service, store, _anchors, category, pending, _candidate, refs, bundle = _revision_case(
        tmp_path,
        summary="## Health\nEarlier account.",
    )

    def fail_journal(**_kwargs):
        raise OSError("journal unavailable")

    monkeypatch.setattr("memu.app.dossier.append_category_summary_journal", fail_journal)
    prose = f"## Health\nDaily record [M{refs[pending.id]}]."
    revised = await service.apply_dossier_revision(
        bundle,
        {
            "dossier_id": category.id,
            "description": category.description,
            "resulting_prose": prose,
            "add_item_ids": [pending.id],
            "remove_item_ids": [],
            "cleanup_item_ids": [],
            "cited_item_ids": [pending.id],
        },
        SCOPE,
    )

    assert revised.summary == prose
    assert store.memory_category_repo.list_categories(SCOPE)[category.id].summary == prose


@pytest.mark.asyncio
async def test_apply_dossier_revision_rotates_description_without_prose_journal(
    tmp_path, monkeypatch
) -> None:
    service, _store, _anchors, category, pending, _candidate, _refs, bundle = _revision_case(
        tmp_path,
        summary="## Health\nStable account.",
    )
    journal: list[dict] = []
    monkeypatch.setattr(
        "memu.app.dossier.append_category_summary_journal",
        lambda **entry: journal.append(entry),
    )
    decision = {
        "dossier_id": category.id,
        "description": "A clearer personal brief.",
        "resulting_prose": "## Health\nStable account.",
        "add_item_ids": [pending.id],
        "remove_item_ids": [],
        "cleanup_item_ids": [],
        "cited_item_ids": [],
    }

    revised = await service.apply_dossier_revision(
        bundle,
        decision,
        SCOPE,
        embedding_client=FakeEmbedClient(),
    )

    assert revised.previous_description == "Health description"
    assert revised.description == "A clearer personal brief."
    assert revised.previous_summary is None
    assert revised.summary == "## Health\nStable account."
    assert journal == []


def test_prepare_anchor_revision_omits_duplicate_presence_and_uses_500_words(tmp_path) -> None:
    service = _service(tmp_path, retrieve_config={"item": {"top_k": 0}})
    store = service.database
    anchors = _seed_anchors(service)
    item = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="A self memory.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    store.memory_item_repo.backfill_memory_refs(SCOPE)
    store.category_item_repo.link_item_category(item.id, anchors["soul"].id, SCOPE)

    bundle = service.prepare_dossier_revision(anchors["soul"].id, SCOPE)

    assert "# Your dossier" not in bundle["soul_presence"]
    assert "# Your human's dossier" in bundle["soul_presence"]
    assert bundle["target_words"] == 500


@pytest.mark.asyncio
async def test_anchor_revisions_validate_both_then_apply_memberships(tmp_path) -> None:
    service = _service(tmp_path)
    store = service.database
    anchors = _seed_anchors(service)
    kept = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="A continuing self memory.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    prior = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="A memory surfaced from reflection.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    period = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="A new lived-period memory.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    purged = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="An inactive membership.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    successor = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="Its active successor.", embedding=[1.0, 0.0], user_data=SCOPE
    )
    refs = store.memory_item_repo.backfill_memory_refs(SCOPE)
    store.category_item_repo.link_item_category(kept.id, anchors["soul"].id, SCOPE)
    store.category_item_repo.link_item_category(purged.id, anchors["user"].id, SCOPE)
    store.triple_repo.add(
        Triple(
            subject_id=purged.id,
            subject_kind="memory",
            predicate="evolved_into",
            object_id=successor.id,
            object_kind="memory",
            source_memory_id=successor.id,
        ),
        user_data=SCOPE,
    )

    bundles = {
        role: service.prepare_anchor_revision(role, SCOPE, [prior.id, period.id])
        for role in ("soul", "user")
    }
    xml = f"""<anchor_revisions>
  <anchor role="soul"><description>My living history.</description><prose_action>patch</prose_action>
    <prose_patches><section ref="S1" action="replace"><body>## Becoming
I am shaped by what surfaced [M{refs[prior.id]}].</body></section></prose_patches></anchor>
  <anchor role="user"><description>My human's living history.</description><prose_action>patch</prose_action>
    <prose_patches><section ref="S1" action="replace"><body>## Becoming
My human is shaped by this lived moment [M{refs[period.id]}].</body></section></prose_patches></anchor>
</anchor_revisions>"""
    decisions = parse_anchor_revisions(ElementTree.fromstring(xml), bundles, first_time=True)

    for role in ("soul", "user"):
        await service.apply_anchor_revision(
            bundles[role],
            decisions[role],
            SCOPE,
            embedding_client=FakeEmbedClient(),
        )

    relations = store.category_item_repo.list_relations(SCOPE)
    members = {
        role: {
            relation.item_id
            for relation in relations
            if relation.category_id == anchors[role].id
        }
        for role in ("soul", "user")
    }
    assert members == {"soul": {kept.id, prior.id}, "user": {period.id}}
    assert store.memory_category_repo.list_categories(SCOPE)[anchors["soul"].id].summary.startswith(
        "## Becoming"
    )
