import json

import pytest

from memu.app import memorize_parsing as parsing
from memu.app.memorize import SpeakerRosterEntry
from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


class _RouterStub:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def chat(self, _prompt: str) -> str:
        return self.payload


@pytest.mark.asyncio
async def test_route_segment_uses_excluded_types_model() -> None:
    service = _service()
    client = _RouterStub(
        '{"excluded_types": ["knowledge", "social"], "episode_summary": "S", "episode_items": [{"title": "Story", "summary": "I"}]}'
    )

    routed, summary, items = await service._route_segment(
        "episode text",
        ["profile", "knowledge", "behavior", "social"],
        llm_client=client,
    )

    assert routed == ["profile", "behavior"]
    assert summary == "S"
    assert items == [{"title": "Story", "summary": "I"}]


@pytest.mark.asyncio
async def test_route_segment_raises_on_unparseable_router_response() -> None:
    service = _service()
    client = _RouterStub("not-json-and-no-json-blob")

    with pytest.raises(ValueError):
        await service._route_segment(
            "episode text",
            ["profile", "knowledge"],
            llm_client=client,
        )


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

    async def _noop_ensure_categories_ready(_ctx, _store, _user_scope=None) -> None:
        return None

    async def _route_profile_only(*_args, **_kwargs):
        return ["profile"], None, None

    async def _noop_step(state, _step_context):
        return state

    async def _noop_categorize(state, _step_context):
        state.setdefault("resources", [])
        state.setdefault("items", [])
        state.setdefault("relations", [])
        state.setdefault("category_updates", {})
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

    monkeypatch.setattr(service, "_ensure_categories_ready", _noop_ensure_categories_ready)
    monkeypatch.setattr(service, "_route_segment", _route_profile_only)
    monkeypatch.setattr(service, "_memorize_categorize_items", _noop_categorize)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _noop_step)
    monkeypatch.setattr(service, "_memorize_build_response", _stub_build_response)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: declared_roster)
    monkeypatch.setattr(service, "_generate_entries_from_text", _capture_generate_entries_from_text)

    raw_text_segment_1 = json.dumps([
        {"role": "user", "name": "Marcos", "content": "Starting a new thread with Nicholas."},
        {"role": "group_member", "name": "Alice", "content": "Alice joins this discussion."},
    ])
    raw_text_segment_2 = json.dumps([
        {"role": "system", "name": "context", "content": "ignored prelude"},
        {"role": "system", "name": "context", "content": "ignored prelude 2"},
        {"role": "assistant", "name": "Echo", "content": "Echo reflects on the day."},
        {"role": "group_member", "name": "Bob", "content": "Bob asks about Nicholas too."},
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
