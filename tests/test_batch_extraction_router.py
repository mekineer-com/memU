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
async def test_route_episode_uses_excluded_types_model() -> None:
    service = _service()
    client = _RouterStub(
        '{"excluded_types": ["knowledge", "social"], "episode_summary": "S", "episode_items": [{"title": "Story", "summary": "I"}]}'
    )

    routed, summary, items = await service._route_episode(
        "episode text",
        ["profile", "knowledge", "behavior", "social"],
        llm_client=client,
    )

    assert routed == ["profile", "behavior"]
    assert summary == "S"
    assert items == [{"title": "Story", "summary": "I"}]


@pytest.mark.asyncio
async def test_route_episode_raises_on_unparseable_router_response() -> None:
    service = _service()
    client = _RouterStub("not-json-and-no-json-blob")

    with pytest.raises(ValueError):
        await service._route_episode(
            "episode text",
            ["profile", "knowledge"],
            llm_client=client,
        )


@pytest.mark.asyncio
async def test_split_segment_into_episodes_raises_when_preprocessor_returns_no_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()

    monkeypatch.setattr(service, "_get_llm_client", lambda *_args, **_kwargs: object())

    async def _empty_split(**_kwargs):
        return []

    monkeypatch.setattr(service, "_split_into_episodes", _empty_split)

    with pytest.raises(RuntimeError, match="Preprocessor returned no episodes for segment"):
        await service.split_segment_into_episodes(
            local_path="mem://segment",
            raw_text='[{"role":"user","content":"hello"}]',
            modality="conversation",
        )


def test_parse_structured_entries_requires_episode_ref_when_requested() -> None:
    service = _service()
    missing_ref = """
<item>
  <memory>
    <source_role>user</source_role>
    <content>Marcos values consistency in system behavior</content>
    <categories><category>Identity</category></categories>
  </memory>
</item>
""".strip()
    with_ref = """
<item>
  <memory>
    <episode_ref>2</episode_ref>
    <source_role>user</source_role>
    <content>Marcos values consistency in system behavior</content>
    <categories><category>Identity</category></categories>
  </memory>
</item>
""".strip()

    dropped = service._parse_structured_entries(
        ["profile"],
        [missing_ref],
        require_episode_ref=True,
    )
    kept = service._parse_structured_entries(
        ["profile"],
        [with_ref],
        require_episode_ref=True,
    )

    assert dropped == []
    assert len(kept) == 1
    assert kept[0].episode_ref == 2


def test_parse_memory_type_response_xml_raises_extraction_parse_error() -> None:
    bad_xml = "<item><memory></item>"

    with pytest.raises(parsing.ExtractionParseError):
        parsing._parse_memory_type_response_xml(bad_xml)


@pytest.mark.asyncio
async def test_memorize_episodes_batch_passes_merged_speaker_roster_with_declared_entities(
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
        state.setdefault("pending_episode_ids", [])
        return state

    def _stub_build_response(state, _step_context):
        state["response"] = {
            "items": [],
            "categories": [],
            "relations": [],
            "pending_episode_ids": [],
        }
        return state

    declared_roster = [SpeakerRosterEntry("entity:nicholas", "Nicholas", "entity")]
    captured: dict[str, object] = {}

    async def _capture_generate_entries_from_text(**kwargs):
        captured["speaker_roster"] = kwargs.get("speaker_roster")
        captured["resource_text"] = kwargs.get("resource_text")
        return []

    monkeypatch.setattr(service, "_ensure_categories_ready", _noop_ensure_categories_ready)
    monkeypatch.setattr(service, "_route_episode", _route_profile_only)
    monkeypatch.setattr(service, "_memorize_categorize_items", _noop_categorize)
    monkeypatch.setattr(service, "_memorize_dedupe_merge", _noop_step)
    monkeypatch.setattr(service, "_memorize_persist_and_index", _noop_step)
    monkeypatch.setattr(service, "_memorize_build_response", _stub_build_response)
    monkeypatch.setattr(service, "_list_declared_relationship_roster", lambda **_kwargs: declared_roster)
    monkeypatch.setattr(service, "_generate_entries_from_text", _capture_generate_entries_from_text)

    raw_text_episode_1 = json.dumps([
        {"role": "user", "name": "Marcos", "content": "Starting a new thread with Nicholas."},
        {"role": "group_member", "name": "Alice", "content": "Alice joins this discussion."},
    ])
    raw_text_episode_2 = json.dumps([
        {"role": "system", "name": "context", "content": "ignored prelude"},
        {"role": "system", "name": "context", "content": "ignored prelude 2"},
        {"role": "assistant", "name": "Echo", "content": "Echo reflects on the day."},
        {"role": "group_member", "name": "Bob", "content": "Bob asks about Nicholas too."},
    ])

    episodes = [
        {
            "resource_url": "mem://episode-1",
            "raw_text": raw_text_episode_1,
            "episode": {
                "text": "[0] Marcos: I talked with Nicholas about focus.\n[1] Alice: That sounds healthy.",
                "caption": "Episode 1",
                "message_indices": [0, 1],
            },
        },
        {
            "resource_url": "mem://episode-2",
            "raw_text": raw_text_episode_2,
            "episode": {
                "text": "[2] Echo: Let's keep steady progress.\n[3] Bob: Nicholas inspired me too.",
                "caption": "Episode 2",
                "message_indices": [2, 3],
            },
        },
    ]

    await service.memorize_episodes_batch(
        modality="conversation",
        episodes=episodes,
        user=user_scope,
        conversation_id="conv-1",
    )

    roster = captured.get("speaker_roster")
    assert roster is not None
    speaker_ids = {entry.speaker_id for entry in roster}
    assert "entity:alice" in speaker_ids
    assert "entity:bob" in speaker_ids
    assert "entity:nicholas" in speaker_ids
    assert "user:marcos" in speaker_ids
    assert "soul:echo" in speaker_ids

    resource_text = str(captured.get("resource_text") or "")
    assert "Episode 1" in resource_text
    assert "Episode 2" in resource_text
