import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memu.app import memorize_parsing as parsing
from memu.app import memorize_segments
from memu.app.memorize import SpeakerRosterEntry
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
async def test_route_segment_uses_configured_episode_limit() -> None:
    service = _service()
    service.memorize_config.episodes_per_segment = 4
    rows = [
        {
            "title": f"Moment {index}",
            "episode_summary": "A small story.",
            "episode_item": None,
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
    assert "Write 1-4 meaningful stories" in client.prompts[0]


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


@pytest.mark.asyncio
async def test_route_segment_ignores_full_exclusion(caplog: pytest.LogCaptureFixture) -> None:
    service = _service()
    client = _RouterStub(
        '{"excluded_types": ["profile", "knowledge"], "episodes": '
        '[{"title": "Anchor", "episode_summary": "Full story.", "episode_item": null, '
        '"categories": ["Daily life"], "day": "2026-01-02"}]}'
    )

    routed, episodes = await service._route_segment(
        "episode text",
        ["profile", "knowledge"],
        llm_client=client,
        source_days=["2026-01-02"],
    )

    assert routed == ["profile", "knowledge"]
    assert episodes[0]["item"] == "Full story."
    assert episodes[0]["categories"] == ["Daily life"]
    assert episodes[0]["day"] == "2026-01-02"
    assert "excluded every configured memory type" in caplog.text


@pytest.mark.parametrize(
    "episode",
    [
        {
            "title": "Missing category", "episode_summary": "Short story.",
            "episode_item": None, "day": "2026-01-02",
        },
        {
            "title": "Bad category", "episode_summary": "Short story.",
            "episode_item": None, "categories": 42, "day": "2026-01-02",
        },
        {
            "title": "Bad day", "episode_summary": "Short story.",
            "episode_item": None, "categories": ["Daily life"], "day": "2026-01-03",
        },
        {
            "title": "Verbose", "episode_summary": "First. Second. Third.",
            "episode_item": None, "categories": ["Daily life"], "day": "2026-01-02",
        },
    ],
)
@pytest.mark.asyncio
async def test_route_segment_retries_required_episode_metadata(
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

    assert len(client.prompts) == 2


@pytest.mark.asyncio
async def test_persist_plan_uses_short_episode_summary_as_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "file_category_proposals", lambda **_kwargs: ([], []))
    created_items: list[dict[str, object]] = []

    async def _resource(**_kwargs):
        return SimpleNamespace(id="res1", embedding=[0.1])

    class _MemoryItemRepo:
        def create_item(self, **kwargs):
            created_items.append(kwargs)
            return SimpleNamespace(id="episode1", summary=kwargs["summary"])

    monkeypatch.setattr(service, "_create_resource_with_caption", _resource)
    embed_client = _EmbedStub()
    happened_at = datetime(2026, 1, 2, 10, tzinfo=UTC)
    relations: list[object] = []

    await service._process_plan(
        {
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
        },
        modality="conversation",
        local_path=None,
        ctx=SimpleNamespace(),
        store=SimpleNamespace(
            memory_item_repo=_MemoryItemRepo(),
        ),
        embed_client=embed_client,
        user_scope={},
        conversation_id="chat",
        items=[],
        relations=relations,
        pending_segment_ids=[],
    )

    assert created_items
    assert created_items[0]["summary"] == "Anchor: Full short story."
    assert created_items[0]["embedding"] == [1.0, 1.0]
    assert created_items[0]["extra"]["episode_summary"] == "Full short story."
    assert created_items[0]["extra"]["episode_categories"] == ["Unmatched proposal"]
    assert created_items[0]["extra"]["memory_date"] == "2026-01-02"
    assert created_items[0]["happened_at"] == happened_at
    assert created_items[0]["source_message_ids"] == [2, 4]
    assert relations == []
    assert embed_client.payloads == [["Anchor: Full short story."]]


@pytest.mark.asyncio
async def test_episode_items_use_their_own_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "file_category_proposals", lambda **_kwargs: ([], []))
    created_items: list[dict[str, object]] = []

    async def _resource(**_kwargs):
        return SimpleNamespace(id="res1", embedding=[99.0, 99.0])

    class _MemoryItemRepo:
        def create_item(self, **kwargs):
            created_items.append(kwargs)
            return SimpleNamespace(id=f"episode-{len(created_items)}", summary=kwargs["summary"])

    monkeypatch.setattr(service, "_create_resource_with_caption", _resource)
    embed_client = _EmbedStub()
    first_day = datetime(2026, 1, 2, 10, tzinfo=UTC)
    second_day = datetime(2026, 1, 3, 10, tzinfo=UTC)

    await service._process_plan(
        {
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
        },
        modality="conversation",
        local_path=None,
        ctx=SimpleNamespace(),
        store=SimpleNamespace(
            memory_item_repo=_MemoryItemRepo(),
        ),
        embed_client=embed_client,
        user_scope={},
        conversation_id="chat",
        items=[],
        relations=[],
        pending_segment_ids=[],
    )

    assert [item["summary"] for item in created_items] == [
        "Choice: A compact choice.",
        "Discovery: A compact discovery.",
    ]
    assert [item["embedding"] for item in created_items] == [[1.0, 1.0], [2.0, 1.0]]
    assert embed_client.payloads == [[
        "Choice: A compact choice.",
        "Discovery: A compact discovery.",
    ]]
    assert [item["extra"]["episode_summary"] for item in created_items] == [
        "A fuller account of a choice.",
        "A fuller account of a discovery.",
    ]
    assert [item["happened_at"] for item in created_items] == [first_day, second_day]


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
    captured: dict[str, object] = {}

    async def _resource(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="res1", embedding=None)

    monkeypatch.setattr(service, "_create_resource_with_caption", _resource)

    await service._process_plan(
        {
            "resource_url": str(local_path),
            "text": "conversation",
            "caption": None,
            "episodes": [],
            "entries": [],
            "message_happened_at_map": {},
            "segment_id": "chat:0-1",
            "segment_messages": [{"role": "user", "content": "primary"}],
        },
        modality="conversation",
        local_path=str(local_path),
        ctx=SimpleNamespace(),
        store=SimpleNamespace(),
        embed_client=SimpleNamespace(),
        user_scope={},
        conversation_id="chat",
        items=[],
        relations=[],
        pending_segment_ids=[],
    )

    assert captured["local_path"] == str(local_path)
    assert not (service.fs.base / "2026-01-01.jsonl").exists()


@pytest.mark.asyncio
async def test_context_only_plan_creates_nothing() -> None:
    service = _service()
    pending_segment_ids: list[str] = []

    resources, item_count = await service._process_plan(
        {
            "resource_url": "memory://background",
            "text": "background context",
            "caption": None,
            "episodes": [],
            "entries": [],
            "segment_id": "background:0-1",
            "context_only": True,
        },
        modality="conversation",
        local_path=None,
        ctx=SimpleNamespace(),
        store=SimpleNamespace(),
        embed_client=SimpleNamespace(),
        user_scope={},
        conversation_id="background",
        items=[],
        relations=[],
        pending_segment_ids=pending_segment_ids,
    )

    assert resources == []
    assert item_count == 0
    assert pending_segment_ids == []


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
async def test_route_segment_raises_on_unparseable_router_response() -> None:
    service = _service()
    client = _RouterStub("not-json-and-no-json-blob")

    with pytest.raises(ValueError):
        await service._route_segment(
            "episode text",
            ["profile", "knowledge"],
            llm_client=client,
            source_days=["2026-01-02"],
        )


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

    with pytest.raises(ValueError, match="Router reply still invalid"):
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
async def test_batch_full_exclusion_runs_all_types_and_keeps_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
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
                '[{"title": "Anchor", "episode_summary": "Full story.", "episode_item": null, '
                '"categories": ["Daily life"], "day": "2026-01-02"}]}'
            ),
            source_days=_kwargs["source_days"],
        )

    async def _capture_extract(*, memory_types, **_kwargs):
        routed_types.extend(memory_types)
        return []

    async def _capture_categorize(state, _step_context):
        persisted_episodes.extend(state["segment_plans"][0]["episodes"])
        state.update(resources=[], items=[], relations=[], pending_segment_ids=[])
        return state

    async def _noop_step(state, _step_context):
        return state

    def _build_empty(state, _step_context):
        state["response"] = {"resources": [], "items": [], "categories": [], "relations": [], "pending_segment_ids": []}
        return state

    monkeypatch.setattr(service, "_route_segment", _route)
    monkeypatch.setattr(service, "_generate_entries_from_text", _capture_extract)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: [])
    monkeypatch.setattr(service, "_memorize_categorize_items", _capture_categorize)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _noop_step)
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
    )

    assert routed_types == ["profile", "knowledge"]
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

    declared_roster = [SpeakerRosterEntry("entity:nicholas", "Nicholas", "entity")]
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
    assert {"entity:alice", "entity:nicholas", "user:marcos"} <= first_speaker_ids
    assert {"entity:bob", "entity:nicholas", "soul:echo"} <= second_speaker_ids
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
