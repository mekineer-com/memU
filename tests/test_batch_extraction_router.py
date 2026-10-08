import json
import asyncio
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memu.app import memorize_parsing as parsing
from memu.app import memorize_segments
from memu.app.memorize import SpeakerRosterEntry, StructuredMemoryEntry
from memu.app.service import MemoryService


def _service() -> MemoryService:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
    )
    service._llm_clients["embedding"] = _EmbedStub()
    return service


class _RouterStub:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload


class _EmbedStub:
    def __init__(self) -> None:
        self.payloads: list[list[str]] = []

    async def embed(self, payloads: list[str]) -> list[list[float]]:
        self.payloads.append(list(payloads))
        return [[float(index), 1.0] for index, _payload in enumerate(payloads, start=1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["router", "knowledge", "late_type"])
@pytest.mark.parametrize("cli", [False, True])
async def test_import_batch_checks_rendered_prompts_on_selected_model(monkeypatch, stage, cli):
    service = _service()
    service.memorize_config.memory_types = ["knowledge"]
    if stage == "late_type":
        service.memorize_config.memory_types = ["profile", "knowledge"]
        build = service._build_memory_type_prompt
        monkeypatch.setattr(service, "_build_memory_type_prompt", lambda **kw:
            build(**kw) + ("fictional " * 30_000 if kw["memory_type"] == "knowledge" else ""))
    service.memorize_config.memory_extract_llm_profile = "extraction"
    profile = service.llm_profiles.profiles["default"]
    service.llm_profiles.profiles["extraction"] = profile.model_copy(update={
        "context_window_tokens": 100_000 if cli else (100 if stage == "router" else 30_000),
        "max_tokens": 10,
    })
    service._claude_code = cli
    service._claude_code_context_window_tokens = 100 if stage == "router" else 30_000
    client = _RouterStub(json.dumps({"excluded_types": [], "episodes": [{
        "title": "Garden", "episode_summary": "A small garden.", "episode_item": "A small garden.",
        "categories": ["Gardens"], "day": "2026-01-02",
    }]}))
    if stage == "late_type":
        async def chat(prompt):
            client.prompts.append(prompt)
            return client.payload if len(client.prompts) == 1 else (
                "<item><memory><content>A garden fact.</content><day>2026-01-02</day>"
                "<categories><category>Gardens</category></categories></memory></item>"
            )
        monkeypatch.setattr(client, "chat", chat)
    monkeypatch.setattr(service, "_select_chat_client", lambda *_a, **_kw: client)
    monkeypatch.setattr(service, "list_active_dossiers", lambda *_: [])
    async def anchors(*_a, **_kw):
        return None
    async def dossier_context(*_a, **_kw):
        return {"categories_str": "Gardens", "narrative_self": "" if stage == "late_type" else "x" * 100_000}
    monkeypatch.setattr(service, "ensure_dossier_anchors", anchors)
    monkeypatch.setattr(service, "select_memorize_dossier_context", dossier_context)
    with pytest.raises(ValueError, match=f"Import {'knowledge' if stage == 'late_type' else stage} prompt exceeds model input budget"):
        await service.memorize_segments_batch(
            modality="conversation",
            segments=[{
                "resource_url": "memory://import",
                "raw_text": json.dumps([{"role": "user", "content": "A garden story.",
                                         "received_at": "2026-01-02T10:00:00+00:00"}]),
                "segment": {"message_indices": [0]},
            }],
            user={"user_id": "TestOwner", "soul_id": "TestSoul"},
            enforce_input_budget=True,
        )
    assert len(client.prompts) == (0 if stage == "router" else 1)
    assert service.database.memory_item_repo.list_items() == {}


@pytest.mark.asyncio
async def test_split_into_episodes_remains_bound_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()

    async def _fake_split(**kwargs):
        assert kwargs["memorize_config"] is service.memorize_config
        return [{"text": "A lively fictional moment.", "caption": None}]

    monkeypatch.setattr(memorize_segments, "_split_into_episodes", _fake_split)

    assert await service._split_into_episodes(
        local_path="fictional.txt",
        text="A lively fictional moment.",
        modality="document",
    ) == [{"text": "A lively fictional moment.", "caption": None}]


@pytest.mark.asyncio
async def test_route_segment_uses_size_based_episode_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "_estimate_text_tokens", lambda _text: 8_000)
    rows = [
        {
            "title": f"Moment {index}",
            "episode_summary": "A small story.",
            "episode_item": "A small story.",
            "categories": ["Daily life"],
            "day": "2026-01-02",
        }
        for index in range(4)
    ]
    client = _RouterStub(json.dumps({"excluded_types": [], "episodes": rows}))

    _routed, episodes = await service._route_segment(
        "fictional conversation",
        ["knowledge"],
        llm_client=client,
        source_days=["2026-01-02"],
    )

    assert len(episodes) == 4
    assert "Write episodes as 1-4 meaningful stories" in client.prompts[0]


@pytest.mark.asyncio
async def test_route_segment_scales_episode_guidance_without_discarding_valid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "_estimate_text_tokens", lambda _text: 45_000)
    rows = [
        {
            "title": f"Moment {index}",
            "episode_summary": "A complete story.",
            "episode_item": "A compact story.",
            "categories": ["One", "Two", "Three", "Four"],
            "day": "2026-01-02",
        }
        for index in range(9)
    ]
    client = _RouterStub(json.dumps({"excluded_types": [], "episodes": rows}))

    _routed, episodes = await service._route_segment(
        "fictional long conversation",
        ["knowledge"],
        llm_client=client,
        source_days=["2026-01-02"],
    )

    assert "Write episodes as 1-8 meaningful stories" in client.prompts[0]
    assert len(episodes) == 9
    assert episodes[0]["categories"] == ["One", "Two", "Three"]


@pytest.mark.asyncio
async def test_route_segment_uses_excluded_types_model() -> None:
    service = _service()
    client = _RouterStub(
        '{"excluded_types": ["knowledge", "social"], "episodes": [{"title": "Anchor", "episode_summary": "Full story.", "episode_item": "Compact story.", "categories": ["Existing", "New domain"], "day": "2026-01-02"}]}'
    )

    routed, episodes = await service._route_segment(
        "--- 2026-01-02 (today) ---\nepisode text",
        ["profile", "knowledge", "behavior", "social"],
        llm_client=client,
        source_days=["2026-01-02"],
        categories_prompt_str="- Existing: An existing dossier",
    )

    assert routed == ["profile", "behavior"]
    assert episodes == [{
        "title": "Anchor",
        "summary": "Full story.",
        "item": "Compact story.",
        "categories": ["Existing", "New domain"],
        "day": "2026-01-02",
    }]
    assert "- Existing: An existing dossier" in client.prompts[0]
    assert "--- 2026-01-02 (today) ---" in client.prompts[0]


@pytest.mark.parametrize(
    "episode",
    [
        {
            "title": "Missing category", "episode_summary": "Short story.",
            "episode_item": "Compact story.", "day": "2026-01-02",
        },
        {
            "title": "Bad category", "episode_summary": "Short story.",
            "episode_item": "Compact story.", "categories": 42, "day": "2026-01-02",
        },
        {
            "title": "Bad day", "episode_summary": "Short story.",
            "episode_item": "Compact story.", "categories": ["Daily life"], "day": "2026-01-03",
        },
        {
            "title": "Missing item", "episode_summary": "Short story.",
            "episode_item": None, "categories": ["Daily life"], "day": "2026-01-02",
        },
    ],
)
@pytest.mark.asyncio
async def test_route_segment_rejects_required_episode_metadata_without_retry(
    episode: dict[str, object],
) -> None:
    service = _service()
    client = _RouterStub(json.dumps({"excluded_types": [], "episodes": [episode]}))

    with pytest.raises(ValueError):
        await service._route_segment(
            "episode text",
            ["profile"],
            llm_client=client,
            source_days=["2026-01-02"],
        )

    assert len(client.prompts) == 1


@pytest.mark.asyncio
async def test_persist_plan_uses_short_episode_summary_as_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "file_category_proposals", lambda **_kwargs: ([], []))
    embed_client = _EmbedStub()
    monkeypatch.setattr(service, "_select_embedding_client", lambda _ctx: embed_client)
    happened_at = datetime(2026, 1, 2, 10, tzinfo=UTC)

    state = await service._memorize_categorize_items({
        "segment_plans": [{
            "resource_url": "memory://episode",
            "text": "conversation",
            "caption": None,
            "episodes": [{
                "title": "Anchor",
                "summary": "Full short story.",
                "item": "Full short story.",
                "categories": ["Unmatched proposal"],
                "day": "2026-01-02",
            }],
            "entries": [],
            "message_happened_at_map": {},
            "source_day_happened_at": {"2026-01-02": happened_at},
            "segment_id": "chat:0-1",
            "message_indices": [2, 4],
            "segment_messages": [],
        }],
        "modality": "conversation", "local_path": None,
        "store": service.database, "user": {"user_id": "TestOwner", "soul_id": "TestSoul"}, "conversation_id": "chat",
    }, SimpleNamespace())

    item = service.database.memory_item_repo.list_items(
        {"user_id": "TestOwner", "soul_id": "TestSoul"},
    )[state["items"][0].id]
    assert item.summary == "Anchor: Full short story."
    assert item.embedding == [1.0, 1.0]
    assert item.extra["episode_summary"] == "Full short story."
    assert item.extra["episode_categories"] == ["Unmatched proposal"]
    assert item.extra["memory_date"] == "2026-01-02"
    assert item.happened_at == happened_at.replace(tzinfo=None)
    assert item.source_message_ids == [2, 4]
    assert state["relations"] == []
    assert embed_client.payloads == [["Anchor: Full short story."]]


@pytest.mark.asyncio
async def test_episode_items_use_their_own_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "file_category_proposals", lambda **_kwargs: ([], []))
    embed_client = _EmbedStub()
    monkeypatch.setattr(service, "_select_embedding_client", lambda _ctx: embed_client)
    first_day = datetime(2026, 1, 2, 10, tzinfo=UTC)
    second_day = datetime(2026, 1, 3, 10, tzinfo=UTC)

    state = await service._memorize_categorize_items({
        "segment_plans": [{
            "resource_url": "memory://episode",
            "text": "conversation",
            "caption": "whole segment",
            "episodes": [
                {
                    "title": "Choice", "summary": "A fuller account of a choice.",
                    "item": "A compact choice.", "categories": ["Decisions"], "day": "2026-01-02",
                },
                {
                    "title": "Discovery", "summary": "A fuller account of a discovery.",
                    "item": "A compact discovery.", "categories": ["Learning"], "day": "2026-01-03",
                },
            ],
            "entries": [],
            "message_happened_at_map": {},
            "source_day_happened_at": {
                "2026-01-02": first_day,
                "2026-01-03": second_day,
            },
            "segment_id": "chat:0-1",
            "segment_messages": [],
        }],
        "modality": "conversation", "local_path": None,
        "store": service.database, "user": {"user_id": "TestOwner", "soul_id": "TestSoul"}, "conversation_id": "chat",
    }, SimpleNamespace())
    rows = service.database.memory_item_repo.list_items({"user_id": "TestOwner", "soul_id": "TestSoul"})
    created_items = [rows[item.id] for item in state["items"]]

    assert [item.summary for item in created_items] == [
        "Choice: A compact choice.",
        "Discovery: A compact discovery.",
    ]
    assert [item.embedding for item in created_items] == [[1.0, 1.0], [2.0, 1.0]]
    assert embed_client.payloads == [["whole segment"], [
        "Choice: A compact choice.",
        "Discovery: A compact discovery.",
    ]]
    assert [item.extra["episode_summary"] for item in created_items] == [
        "A fuller account of a choice.",
        "A fuller account of a discovery.",
    ]
    assert [item.happened_at for item in created_items] == [
        first_day.replace(tzinfo=None), second_day.replace(tzinfo=None),
    ]


@pytest.mark.asyncio
async def test_persist_plan_keeps_segment_local_path_without_flattened_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "file_category_proposals", lambda **_kwargs: ([], []))
    service.fs.base = tmp_path / "resources"
    local_path = tmp_path / "st_chats" / "chat" / "segments" / "2026-01-01.json"
    local_path.parent.mkdir(parents=True)
    local_path.write_text("[]", encoding="utf-8")
    state = await service._memorize_categorize_items({
        "segment_plans": [{
            "resource_url": str(local_path),
            "text": "conversation",
            "caption": None,
            "episodes": [],
            "entries": [],
            "message_happened_at_map": {},
            "segment_id": "chat:0-1",
            "segment_messages": [{"role": "user", "content": "primary"}],
        }],
        "modality": "conversation", "local_path": str(local_path),
        "store": service.database, "user": {}, "conversation_id": "chat",
    }, SimpleNamespace())

    assert state["resources"][0].local_path == str(local_path)
    repo = service.database.resource_repo
    resource = next(iter(repo.list_resources().values()))
    updated = repo.create_resource(url=str(local_path), modality="conversation",
        local_path=str(local_path), caption=None, embedding=None, user_data={})
    assert updated.id == resource.id
    reloaded = repo.list_resources()[resource.id]
    assert reloaded.local_path == str(local_path)
    assert not (service.fs.base / "2026-01-01.jsonl").exists()


@pytest.mark.asyncio
async def test_context_only_plan_creates_nothing() -> None:
    service = _service()
    state = await service._memorize_categorize_items({
        "segment_plans": [{
            "resource_url": "memory://background",
            "text": "background context",
            "caption": None,
            "episodes": [],
            "entries": [],
            "segment_id": "background:0-1",
            "context_only": True,
        }],
        "modality": "conversation", "local_path": None,
        "store": service.database, "user": {}, "conversation_id": "background",
    }, SimpleNamespace())

    assert state["resources"] == []
    assert state["items"] == []
    assert state["pending_segment_ids"] == []
    assert service._llm_clients["embedding"].payloads == []
    assert service.database.resource_repo.list_resources() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("file_backed", [True, False])
@pytest.mark.parametrize("failure", [None, "preparation", "publication"])
async def test_categorize_prepares_without_write_lock_and_publishes_atomically(
    tmp_path, monkeypatch, file_backed, failure,
):
    path = tmp_path / "fictional.db"
    service = MemoryService(
        blob_config={"resources_dir": str(tmp_path / "resources")},
        database_config={"metadata_store": {"provider": "sqlite",
            "dsn": f"sqlite:///{path}" if file_backed else "sqlite:///:memory:"}},
        memorize_config={"enable_confidence_normalization": True},
    )
    store = service.database
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    old = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="Old garden plan", embedding=[1.0, 0.0], user_data=scope,
    )
    with store._sessions.session() as session:
        session.connection().exec_driver_sql("CREATE TABLE checkpoint (value INTEGER)")
        session.commit()
    entries = [StructuredMemoryEntry(
        memory_type="knowledge", content=text, categories=[], source_role=None,
        confidence=0.8, source_message_ids=[0], reflection_salience=0.7,
        replaces_previous_fact="Old garden plan", memory_date="2026-01-02",
        entities=[{"name": "Fictional Garden", "type": "place"}],
    ) for text in ["New garden plan", "Another garden detail"]]
    payloads = []

    class Embed:
        async def embed(self, texts):
            payloads.append(texts)
            await asyncio.sleep(0)
            if file_backed:
                # A synchronous competitor would block the holder's event loop in the old flow.
                with sqlite3.connect(path, timeout=0.08) as conn:
                    conn.execute("INSERT INTO checkpoint VALUES (0)")
            if texts == [entry.content for entry in entries] and failure == "preparation":
                raise RuntimeError("preparation failed")
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(service, "_select_embedding_client", lambda _ctx: Embed())
    monkeypatch.setattr(service, "_normalize_confidence", lambda rows: [
        entry._replace(confidence=0.3) for entry in rows
    ])
    def saved(session, url, pending):
        assert url == "memory://prepared" and pending == ["chat:0-1"]
        session.connection().exec_driver_sql("INSERT INTO checkpoint VALUES (1)")
        if failure == "publication":
            raise RuntimeError("publication failed")

    state = {
        "segment_plans": [{"resource_url": "memory://prepared", "text": "A fictional conversation",
            "caption": "Segment caption", "segment_id": "chat:0-1", "entries": entries,
            "extract_model": "fictional-model", "source_day_happened_at": {
                "2026-01-02": datetime(2026, 1, 2, tzinfo=UTC)}, "episodes": [{
                "title": "Garden", "summary": "A garden story", "item": "A garden plan",
                "categories": [], "day": "2026-01-02"}]}],
        "modality": "conversation", "local_path": None, "store": store,
        "user": scope, "conversation_id": "chat", "on_segment_saved": saved,
    }
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await service._memorize_categorize_items(state, {})
    else:
        await service._memorize_categorize_items(state, {})
    rows = store.memory_item_repo.list_items(scope, include_superseded=True)
    resources = store.resource_repo.list_resources(scope)
    with store._sessions.session() as session:
        assert session.connection().exec_driver_sql(
            "SELECT count(*) FROM checkpoint WHERE value = 1",
        ).scalar_one() == (0 if failure else 1)
    assert len(resources) == (0 if failure else 1)
    assert len(rows) == (1 if failure else 4)
    assert len({item.memory_ref for item in rows.values()}) == len(rows)
    assert payloads[:3] == [["Segment caption"], ["Garden: A garden plan"], [entry.content for entry in entries]]
    assert len(payloads) == (3 if failure == "preparation" else 4)
    if failure == "preparation":
        assert not (tmp_path / "resources" / "prepared.txt").exists()
    if not failure:
        episode, first, second = state["items"]
        assert episode.memory_ref < first.memory_ref < second.memory_ref
        assert first.summary == "I have a faint suspicion that new garden plan"
        assert first.embedding == [1.0, 0.0] and first.extra["model"] == "fictional-model"
        assert payloads[-1] == ["Old garden plan", "Old garden plan"]
        edges = store.triple_repo.get_edges_from(old.id, "evolved_into", where=scope)
        assert len(edges) == 1 and edges[0].object_id == first.id
        assert store.triple_repo.get_edges_from(first.id, "mentions", where=scope)
    store.close()


@pytest.mark.asyncio
async def test_persist_index_skips_dynamic_review_without_new_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()

    async def _unexpected(**_kwargs):
        raise AssertionError("memorize without new items must not process old dossier candidates")

    monkeypatch.setattr(service, "prepare_dynamic_category_review", _unexpected)
    state = {"items": [], "store": service.database, "user": {"user_id": "person", "soul_id": "soul"}}

    assert await service._memorize_persist_and_index(state, None) is state


@pytest.mark.asyncio
async def test_persist_index_processes_committed_candidate_work_without_new_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    called = False

    async def _prepare(**_kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(service, "prepare_dynamic_category_review", _prepare)
    state = {
        "items": [],
        "active_candidate_work_committed": True,
        "store": service.database,
        "user": {"user_id": "person", "soul_id": "soul"},
    }

    assert await service._memorize_persist_and_index(state, None) is state
    assert called


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["dedupe", "review"])
async def test_resume_segment_reads_original_rows_and_skips_completed_dedupe(monkeypatch, phase):
    service = _service()
    store = service.database
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    items = [store.memory_item_repo.create_item(
        memory_type="knowledge", summary=f"Fictional memory {i}", embedding=[1.0, 0.0],
        user_data=scope, segment_id="segment", source_role="user",
    ) for i in range(3)]
    store.memory_item_repo.create_item(
        memory_type="knowledge", summary="Another fictional Soul's memory", embedding=[1.0, 0.0],
        user_data={"user_id": "TestOwner", "soul_id": "OtherTestSoul"}, segment_id="segment",
    )
    store.memory_item_repo.update_item(item_id=items[0].id, merged_into=items[1].id)
    from memu.database.models import Triple
    store.triple_repo.add(Triple(subject_id=items[1].id, subject_kind="memory",
        predicate="evolved_into", object_id=items[2].id, object_kind="memory"), user_data=scope)
    dedupes, reviews = [], []
    async def dedupe(state, _context):
        dedupes.append([item.id for item in state["items"]])
        return state
    async def review(state, _context, *, enforce_input_budget):
        reviews.append(([item.id for item in state["items"]], enforce_input_budget))
        assert state["active_candidate_work_committed"]
        return state
    monkeypatch.setattr(service, "_memorize_dedupe_merge", dedupe)
    monkeypatch.setattr(service, "_memorize_persist_and_index", review)
    list_items = store.memory_item_repo.list_items
    monkeypatch.setattr(store.memory_item_repo, "list_items", lambda *args, **kwargs:
        dict(sorted(list_items(*args, **kwargs).items(), key=lambda row: row[1].memory_ref, reverse=True)))
    with pytest.raises(ValueError, match="'soul_id' is missing/blank"):
        await service.resume_memorize_segment(
            segment_id="segment", phase=phase, user={"user_id": "TestOwner"},
            on_dedupe_complete=lambda _session: None,
        )
    await service.resume_memorize_segment(
        segment_id="segment", phase=phase, user=scope,
        on_dedupe_complete=lambda _session: None, enforce_input_budget=True,
    )
    assert dedupes == ([[item.id for item in items]] if phase == "dedupe" else [])
    assert reviews == [([item.id for item in items], True)]
    await service.resume_memorize_segment(
        segment_id="empty", phase="review", user=scope,
        on_dedupe_complete=lambda _session: None,
    )
    assert reviews[-1] == ([], False)


@pytest.mark.asyncio
async def test_batch_router_failure_stops_before_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    actual_route = service._route_segment
    persistence_started = False

    async def _fail_route(segment_text, memory_types, *_args, **_kwargs):
        return await actual_route(
            segment_text,
            memory_types,
            llm_client=_RouterStub("not-json-and-no-json-blob"),
            source_days=_kwargs["source_days"],
        )

    async def _mark_persistence(*_args, **_kwargs):
        nonlocal persistence_started
        persistence_started = True
        raise AssertionError("persistence must not start after router failure")

    monkeypatch.setattr(service, "_route_segment", _fail_route)
    monkeypatch.setattr(service, "_memorize_categorize_items", _mark_persistence)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="Router reply invalid"):
        await service.memorize_segments_batch(
            modality="conversation",
            segments=[
                {
                    "resource_url": "memory://segment",
                    "raw_text": json.dumps([{
                        "role": "user", "content": "ordinary exchange",
                        "received_at": "2026-01-02T10:00:00-05:00",
                    }]),
                    "segment": {"message_indices": [0], "context_only": False},
                }
            ],
            user={"user_id": "test-user", "soul_id": "TestSoul"},
        )

    assert persistence_started is False


@pytest.mark.asyncio
async def test_batch_rejects_active_segment_without_source_date(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()

    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="has no source date"):
        await service.memorize_segments_batch(
            modality="conversation",
            segments=[{
                "resource_url": "memory://undated",
                "raw_text": json.dumps([{"role": "user", "content": "undated"}]),
                "segment": {"message_indices": [0], "context_only": False},
            }],
            user={"user_id": "test-user", "soul_id": "test-soul"},
        )


@pytest.mark.asyncio
async def test_batch_excludes_image_marker_from_episode_text(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()

    async def _render(*, primary_messages, background_messages, **_kwargs):
        assert [row["content"] for row in primary_messages] == ["A fictional caption."]
        assert background_messages == []
        return "rendered caption", []

    async def _stop_after_render(segment_text, *_args, **_kwargs):
        assert segment_text == "rendered caption"
        raise RuntimeError("render verified")

    monkeypatch.setattr(service, "_render_episode_with_background_context", _render)
    monkeypatch.setattr(service, "_route_segment", _stop_after_render)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])

    with pytest.raises(RuntimeError, match="render verified"):
        await service.memorize_segments_batch(
            modality="conversation",
            segments=[{
                "resource_url": "memory://image-turn",
                "raw_text": json.dumps([
                    {
                        "role": "user",
                        "content": "Shared a photo.",
                        "event_kind": "image",
                        "media_ref": "mentra_media/fictional/image.png",
                        "received_at": "2026-01-02T10:00:00-05:00",
                    },
                    {
                        "role": "assistant",
                        "content": "A fictional caption.",
                        "received_at": "2026-01-02T10:00:01-05:00",
                    },
                ]),
                "segment": {"message_indices": [0, 1], "context_only": False},
            }],
            user={"user_id": "test-user", "soul_id": "TestSoul"},
        )


@pytest.mark.asyncio
async def test_image_only_batch_stops_before_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()

    async def _unexpected(*_args, **_kwargs):
        raise AssertionError("image marker must not reach extraction")

    monkeypatch.setattr(service, "_route_segment", _unexpected)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="has no source date"):
        await service.memorize_segments_batch(
            modality="conversation",
            segments=[{
                "resource_url": "memory://image-only",
                "raw_text": json.dumps([{
                    "role": "user",
                    "content": "Shared a photo.",
                    "event_kind": "image",
                    "media_ref": "mentra_media/fictional/image.png",
                    "received_at": "2026-01-02T10:00:00-05:00",
                }]),
                "segment": {"message_indices": [0], "context_only": False},
            }],
            user={"user_id": "test-user", "soul_id": "TestSoul"},
        )


@pytest.mark.asyncio
async def test_context_only_batch_skips_llm_and_returns_plural_empty_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()

    async def _unexpected(*_args, **_kwargs):
        raise AssertionError("context-only batch must not call an LLM")

    async def _categorize_empty(state, _step_context):
        assert state["segment_plans"][0]["context_only"] is True
        state.update(resources=[], items=[], relations=[], pending_segment_ids=[])
        return state

    async def _noop_step(state, _step_context):
        return state

    def _build_empty(state, _step_context):
        state["response"] = {
            "resources": [],
            "items": [],
            "categories": [],
            "relations": [],
            "pending_segment_ids": [],
        }
        return state

    monkeypatch.setattr(service, "_route_segment", _unexpected)
    monkeypatch.setattr(service, "_generate_entries_from_text", _unexpected)
    monkeypatch.setattr(
        service,
        "_select_embedding_client",
        lambda *_args, **_kwargs: pytest.fail("context-only batch must not select an embedding client"),
    )
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])
    monkeypatch.setattr(service, "_memorize_categorize_items", _categorize_empty)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _noop_step)
    monkeypatch.setattr(service, "_memorize_build_response", _build_empty)

    responses = await service.memorize_segments_batch(
        modality="conversation",
        segments=[
            {
                "resource_url": "memory://background",
                "raw_text": json.dumps(
                    [{"role": "user", "content": "background", "memorize_chat": False}]
                ),
                "segment": {"message_indices": [0], "context_only": True},
            }
        ],
        user={"user_id": "test-user", "soul_id": "TestSoul"},
    )

    assert responses == [
        {
            "resources": [],
            "items": [],
            "categories": [],
            "relations": [],
            "pending_segment_ids": [],
        }
    ]
    assert service.database.memory_category_repo.list_categories(
        {"user_id": "test-user", "soul_id": "TestSoul"}
    ) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce_budget", [False, True])
async def test_batch_full_exclusion_runs_all_types_and_keeps_episodes(
    monkeypatch: pytest.MonkeyPatch, enforce_budget, caplog,
) -> None:
    service = _service()
    service.llm_profiles.profiles["default"].context_window_tokens = 100_000
    service.memorize_config.memory_types = ["profile", "knowledge"]
    actual_route = service._route_segment
    routed_types: list[str] = []
    persisted_episodes: list[dict[str, str]] = []

    async def _route(segment_text, memory_types, *_args, **_kwargs):
        return await actual_route(
            segment_text,
            memory_types,
            llm_client=_RouterStub(
                '{"excluded_types": ["profile", "knowledge"], "episodes": '
                '[{"title": "Anchor", "episode_summary": "Full story.", "episode_item": "Full story.", '
                '"categories": ["Daily life"], "day": "2026-01-02"}]}'
            ),
            source_days=_kwargs["source_days"],
        )

    async def _capture_extract(*, memory_types, **_kwargs):
        assert _kwargs["user_id"] == "test-user"
        assert _kwargs["input_budget"] == (80_000 if enforce_budget else None)
        routed_types.extend(memory_types)
        return []

    async def _capture_categorize(state, _step_context):
        persisted_episodes.extend(state["segment_plans"][0]["episodes"])
        state.update(resources=[], items=[], relations=[], pending_segment_ids=[])
        return state

    async def _noop_step(state, _step_context):
        return state

    async def _persist(state, _step_context, **kwargs):
        assert kwargs.get("enforce_input_budget", False) is enforce_budget
        return state

    def _build_empty(state, _step_context):
        state["response"] = {"resources": [], "items": [], "categories": [], "relations": [], "pending_segment_ids": []}
        return state

    monkeypatch.setattr(service, "_route_segment", _route)
    monkeypatch.setattr(service, "_generate_entries_from_text", _capture_extract)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])
    monkeypatch.setattr(service, "_memorize_categorize_items", _capture_categorize)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _persist)
    monkeypatch.setattr(service, "_memorize_build_response", _build_empty)

    await service.memorize_segments_batch(
        modality="conversation",
        segments=[
            {
                "resource_url": "memory://segment",
                "raw_text": json.dumps([{
                    "role": "user", "content": "ordinary story",
                    "received_at": "2026-01-02T10:00:00-05:00",
                }]),
                "segment": {"message_indices": [0], "context_only": False},
            }
        ],
        user={"user_id": "test-user", "soul_id": "TestSoul"},
        enforce_input_budget=enforce_budget,
    )

    assert routed_types == ["profile", "knowledge"]
    assert "excluded every configured memory type" in caplog.text
    assert persisted_episodes == [{
        "title": "Anchor",
        "summary": "Full story.",
        "item": "Full story.",
        "categories": ["Daily life"],
        "day": "2026-01-02",
    }]


def test_parse_memory_type_response_xml_raises_on_unsalvageable_xml() -> None:
    # "<item><memory></item>" is malformed AND unsalvageable — both passes fail
    bad_xml = "<item><memory></item>"
    with pytest.raises(ValueError, match="unparseable"):
        parsing._parse_memory_type_response_xml(bad_xml)


def test_parse_memory_type_response_xml_returns_empty_for_valid_xml_no_memories() -> None:
    # Valid XML with <item> root but no <memory> elements is a legitimate zero-item reply
    result = parsing._parse_memory_type_response_xml("<item></item>")
    assert result == []


@pytest.mark.asyncio
async def test_memorize_segments_batch_passes_segment_speaker_rosters_without_segment_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    user_scope = {"user_id": "Marcos", "soul_id": "Echo"}

    async def _route_profile_only(*_args, **_kwargs):
        return ["profile"], [{"title": "Anchor", "summary": "Full story.", "item": "Compact story."}]

    async def _noop_step(state, _step_context):
        return state

    async def _noop_categorize(state, _step_context):
        state.setdefault("resources", [])
        state.setdefault("items", [])
        state.setdefault("relations", [])
        state.setdefault("pending_segment_ids", [])
        return state

    def _stub_build_response(state, _step_context):
        state["response"] = {
            "items": [],
            "categories": [],
            "relations": [],
            "pending_segment_ids": [],
        }
        return state

    declared_roster = [SpeakerRosterEntry("entity:d4e5f6a7", "Nicholas", "entity")]
    captured: list[dict[str, object]] = []

    async def _capture_generate_entries_from_text(**kwargs):
        captured.append({
            "speaker_roster": kwargs.get("speaker_roster"),
            "resource_text": kwargs.get("resource_text"),
        })
        return []

    monkeypatch.setattr(service, "_route_segment", _route_profile_only)
    monkeypatch.setattr(service, "_memorize_categorize_items", _noop_categorize)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _noop_step)
    monkeypatch.setattr(service, "_memorize_build_response", _stub_build_response)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: declared_roster)
    monkeypatch.setattr(
        "memu.app.memorize.speakers._build_entity_speaker_ids",
        lambda _entities: {"alice": "entity:a1b2c3d4", "bob": "entity:b2c3d4e5"},
    )
    monkeypatch.setattr(service, "_generate_entries_from_text", _capture_generate_entries_from_text)

    raw_text_segment_1 = json.dumps([
        {
            "role": "user", "name": "Marcos", "content": "Starting a new thread with Nicholas.",
            "received_at": "2026-01-02T10:00:00-05:00",
        },
        {
            "role": "group_member", "name": "Alice", "content": "Alice joins this discussion.",
            "received_at": "2026-01-02T10:01:00-05:00",
        },
    ])
    raw_text_segment_2 = json.dumps([
        {"role": "system", "name": "context", "content": "ignored prelude"},
        {"role": "system", "name": "context", "content": "ignored prelude 2"},
        {
            "role": "assistant", "name": "Echo", "content": "Echo reflects on the day.",
            "received_at": "2026-01-03T10:00:00-05:00",
        },
        {
            "role": "group_member", "name": "Bob", "content": "Bob asks about Nicholas too.",
            "received_at": "2026-01-03T10:01:00-05:00",
        },
    ])

    segments = [
        {
            "resource_url": "mem://segment-1",
            "raw_text": raw_text_segment_1,
            "segment": {
                "text": "[0] Marcos: I talked with Nicholas about focus.\n[1] Alice: That sounds healthy.",
                "caption": "Segment 1",
                "message_indices": [0, 1],
            },
        },
        {
            "resource_url": "mem://segment-2",
            "raw_text": raw_text_segment_2,
            "segment": {
                "text": "[2] Echo: Let's keep steady progress.\n[3] Bob: Nicholas inspired me too.",
                "caption": "Segment 2",
                "message_indices": [2, 3],
            },
        },
    ]

    await service.memorize_segments_batch(
        modality="conversation",
        segments=segments,
        user=user_scope,
        conversation_id="conv-1",
    )

    assert len(captured) == 2
    first_roster = captured[0]["speaker_roster"]
    second_roster = captured[1]["speaker_roster"]
    assert first_roster is not None
    assert second_roster is not None
    first_speaker_ids = {entry.speaker_id for entry in first_roster}
    second_speaker_ids = {entry.speaker_id for entry in second_roster}
    assert {"entity:a1b2c3d4", "entity:d4e5f6a7", "user:marcos"} <= first_speaker_ids
    assert {"entity:b2c3d4e5", "entity:d4e5f6a7", "soul:echo"} <= second_speaker_ids
    for call in captured:
        resource_text = str(call.get("resource_text") or "")
        assert "Segment 1" not in resource_text
        assert "Segment 2" not in resource_text
        assert "## Segment" not in resource_text


@pytest.mark.asyncio
async def test_memorize_segment_direct_uses_grouped_chat_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    user_scope = {"user_id": "Marcos", "soul_id": "Siri"}
    captured: dict[str, str] = {}

    raw_text = json.dumps(
        [
            {
                "role": "user",
                "speaker": "Marcos",
                "content": "dm primary",
                "source_conversation_id": "whatsapp:dm:liz",
                "chat_name": "Liz Kalverda",
                "received_at": "2026-06-12T10:00:00+00:00",
                "memorize_chat": True,
            },
            {
                "role": "assistant",
                "content": "soul reply",
                "source_conversation_id": "whatsapp:dm:liz",
                "chat_name": "Liz Kalverda",
                "received_at": "2026-06-12T10:01:00+00:00",
                "memorize_chat": True,
            },
        ]
    )

    async def _route_profile_only(segment_text, *_args, **_kwargs):
        captured["route_text"] = segment_text
        return ["profile"], [{"title": "Anchor", "summary": "Router summary.", "item": "Router item."}]

    async def _capture_generate_entries_from_text(**kwargs):
        captured["extract_text"] = kwargs["resource_text"]
        return []

    async def _noop_step(state, _step_context):
        return state

    async def _noop_categorize(state, _step_context):
        state.setdefault("resources", [])
        state.setdefault("items", [])
        state.setdefault("relations", [])
        state.setdefault("pending_segment_ids", [])
        return state

    def _stub_build_response(state, _step_context):
        state["response"] = {
            "items": [],
            "categories": [],
            "relations": [],
            "pending_segment_ids": [],
        }
        return state

    monkeypatch.setattr(service, "_select_chat_client", lambda *_args, **_kwargs: SimpleNamespace(chat_model="test"))
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])
    monkeypatch.setattr(service, "_route_segment", _route_profile_only)
    monkeypatch.setattr(service, "_generate_entries_from_text", _capture_generate_entries_from_text)
    monkeypatch.setattr(service, "_memorize_categorize_items", _noop_categorize)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _noop_step)
    monkeypatch.setattr(service, "_memorize_build_response", _stub_build_response)

    out = await service.memorize_segment(
        resource_url="memory://direct",
        modality="conversation",
        segment={"text": raw_text, "caption": None},
        user=user_scope,
        raw_text=raw_text,
        conversation_id="whatsapp:dm:liz",
    )

    route_text = captured["route_text"]
    extract_text = captured["extract_text"]
    assert "My WhatsApp Conversations:" in route_text
    assert "[dm][Liz Kalverda]" in route_text
    assert "[Marcos] dm primary" in route_text
    assert "[Siri] soul reply" in route_text
    assert "[user]" not in route_text
    assert '"role"' not in route_text
    assert "My WhatsApp Conversations:" in extract_text
    assert "Episode: Anchor" in extract_text
    assert "Episode Summary:\nRouter summary." in extract_text
    assert route_text in extract_text
    assert out["items"] == []
