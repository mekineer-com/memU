"""End-to-end attribution pipeline test (without the LLM in the middle).

Covers the chain that broke in the Phase 5 live smoke:
  _build_speaker_map → _attribute_memory → _decorate_entries_with_plan_context → _persist_memory_items → DB
The LLM extraction step is skipped — we hand-construct StructuredMemoryEntry
objects as if the LLM emitted them, then verify the pipeline fills
speaker_id/speaker_label/full episode provenance correctly and the fields round-trip
through the sqlite reinforce-enabled write path.

Three real bugs this guards against in combination:
  - 30b8db2: empty provenance fallback
  - 39511ef: sqlite reinforce branch dropping source_message_ids
  - 55525ae: role=user always resolves to scope user, not entity:<name>

Any of these regressing alone would flip at least one assertion below.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from memu.app.memorize import SpeakerRosterEntry, StructuredMemoryEntry
from memu.app import memorize_dedupe as dedupe
from memu.app.service import MemoryService


class _Scope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


class _EmbedClient:
    async def embed(self, inputs: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in inputs]


def _service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        memorize_config={"enable_item_reinforcement": True},
        user_config={"model": _Scope},
    )


def test_attribution_pipeline_fills_user_and_soul_speakers_with_fallback_indices() -> None:
    service = _service()
    store = service._get_database()
    user_data = {"user_id": "testowner", "soul_id": "testsoul"}

    # Simulated episode. Note the display-name mismatch: message.name="TestOwnerDisplay"
    # is NOT case-equal to scope.user_id="testowner" — the bug was that this tripped
    # the old code into slugging the user as entity:testownerdisplay.
    segment_messages = [
        {"_message_index": 0, "role": "user", "name": "TestOwnerDisplay", "content": "Hi"},
        {"_message_index": 1, "role": "assistant", "name": "TestSoul", "content": "Hello"},
        {"_message_index": 2, "role": "user", "name": "TestOwnerDisplay", "content": "Thanks"},
    ]
    message_indices = [0, 1, 2]
    speaker_map = service._build_speaker_map(segment_messages, user_data)

    # Post-55525ae: user-role messages always resolve to the scope user slug,
    # with the display name as the label.
    assert speaker_map[0] == ("user:testowner", "TestOwnerDisplay")
    assert speaker_map[1] == ("soul:testsoul", "TestSoul")

    # Simulate generalized memories. The prompt does not ask the model to tie
    # each memory to individual messages.
    entries = [
        StructuredMemoryEntry(
            memory_type="behavior",
            content="TestOwner greeted TestSoul warmly",
            categories=[],
            source_role="user",
            confidence=0.8,
            source_message_ids=[],
            reflection_salience=None,
            emotional_intensity=0.7,
        ),
        StructuredMemoryEntry(
            memory_type="behavior",
            content="I greeted TestOwner back",
            categories=[],
            source_role="soul",
            confidence=0.8,
            source_message_ids=[],
            reflection_salience=None,
            emotional_intensity=0.2,
        ),
    ]

    # Provenance covers the full episode range; it is not a per-message claim.
    decorated = service._decorate_entries_with_plan_context(entries, message_indices=message_indices)
    assert decorated[0].source_message_ids == [0, 1, 2]
    assert decorated[1].source_message_ids == [0, 1, 2]

    # Attribute from speaker_map. Source_role=user + multi-speaker range →
    # role-based disambiguation picks user:testowner.
    attributed = [service._attribute_memory(entry, speaker_map) for entry in decorated]
    assert attributed[0].speaker_id == "user:testowner"
    assert attributed[0].speaker_label == "TestOwnerDisplay"
    assert attributed[1].speaker_id == "soul:testsoul"
    assert attributed[1].speaker_label == "TestSoul"

    # Persist. reinforce=True by default (memorize_config). Post-39511ef:
    # persisted provenance and reflection_salience survive the reinforce branch.
    async def _persist():
        service._llm_clients["embedding"] = _EmbedClient()
        state = await service._memorize_categorize_items({
            "segment_plans": [{"resource_url": "memory://attribution", "entries": attributed}],
            "modality": "conversation", "local_path": None,
            "store": store, "user": user_data,
        }, {})
        return state["items"]

    items = asyncio.run(_persist())
    assert len(items) == 2

    loaded_user = store.memory_item_repo.get_item(items[0].id)
    loaded_soul = store.memory_item_repo.get_item(items[1].id)
    assert loaded_user is not None and loaded_soul is not None

    assert loaded_user.speaker_id == "user:testowner"
    assert loaded_user.speaker_label == "TestOwnerDisplay"
    assert loaded_user.source_message_ids == [0, 1, 2]
    assert loaded_user.source_role == "user"
    assert loaded_user.emotional_intensity == 0.7

    assert loaded_soul.speaker_id == "soul:testsoul"
    assert loaded_soul.speaker_label == "TestSoul"
    assert loaded_soul.source_message_ids == [0, 1, 2]
    assert loaded_soul.source_role == "soul"
    assert loaded_soul.emotional_intensity == 0.2


def test_unresolved_import_guest_is_not_attributed_to_the_soul():
    service = _service()
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    messages = [
        {"_message_index": 0, "role": "user", "name": "GuestSpeaker", "source_label": "import"},
        {"_message_index": 1, "role": "assistant", "name": "CompanionDisplay", "source_label": "import"},
    ]
    speaker_map = service._build_speaker_map(messages, scope)
    responses = [f"<item><memory><source_role>{role}</source_role><content>{content}</content>"
                 "<day>2025-01-01</day><categories><category>Gardens</category></categories></memory></item>"
                 for role, content in [("entity", "GuestSpeaker grows orchids"), ("soul", "I enjoy the garden")]]
    entries = [service._parse_structured_entries(
        ["profile"], [response], default_source_message_ids=[0, 1], source_days=["2025-01-01"],
    )[0] for response in responses]
    attributed = [service._attribute_memory(entry, speaker_map) for entry in entries]
    service._llm_clients["embedding"] = _EmbedClient()
    state = asyncio.run(service._memorize_categorize_items({
        "segment_plans": [{"resource_url": "memory://guest", "entries": attributed}],
        "modality": "conversation", "local_path": None,
        "store": service.database, "user": scope,
    }, {}))
    items = state["items"]
    loaded = [service.database.memory_item_repo.get_item(item.id) for item in items]
    assert loaded[0].speaker_id is None and loaded[0].speaker_label is None
    assert loaded[1].speaker_id == "soul:testsoul" and loaded[1].speaker_label == "CompanionDisplay"
    assert all(item.source_message_ids == [0, 1] for item in loaded)


@pytest.mark.parametrize("role,ref,indices,expected", [
    ("peer", "", [0], "entity:guest-a"),
    ("peer", "", [0, 1], None),
    ("peer", "", [0, 4], None),
    ("peer", "", [0, 3], None),
    ("peer", "entity:guest-b", [0, 3], "entity:guest-b"),
    ("peer", "entity:guest-a", [0, 1], "entity:guest-a"),
    ("peer", "entity:missing", [0], None),
    ("entity", "", [0], None),
    ("entity", "entity:mentioned", [0], "entity:mentioned"),
    ("", "", [0], None),
    ("invalid", "", [0], None),
    ("user", "", [0, 1, 2, 4], "user:testowner"),
    ("soul", "", [0, 1, 5], "soul:testsoul"),
])
def test_guest_attribution_preserves_uncertainty_through_save(role, ref, indices, expected) -> None:
    service = _service()
    scope = {"user_id": "testowner", "soul_id": "testsoul"}
    messages = [
        {"role": "unknown", "name": "Same Label", "source_ref": "whatsapp:guest-a@lid"},
        {"role": "unknown", "name": "Unlinked Guest", "source_ref": "whatsapp:unlinked@lid"},
        {"role": "user", "name": "OwnerDisplay"},
        {"role": "unknown", "name": "Same Label", "source_ref": "whatsapp:guest-b@lid"},
        {"role": "user", "conversation_id": "import:dm:fictional"},
        {"role": "assistant", "name": "TestSoul"},
    ]
    speaker_map = service._build_speaker_map(messages, scope, entity_source_speaker_ids={
        "whatsapp:guest-a@lid": "entity:guest-a", "whatsapp:guest-b@lid": "entity:guest-b",
    })
    assert speaker_map[1] == ("", "Unlinked Guest") and speaker_map[4] == ("", "user")
    unresolved_roster = service._build_speaker_roster_for_segment(
        speaker_map={index: speaker_map[index] for index in (0, 1)}, declared_entities=[], segment_text="",
    )
    assert unresolved_roster == [SpeakerRosterEntry("entity:guest-a", "Same Label", "peer")]
    roster = service._build_speaker_roster_for_segment(
        speaker_map=speaker_map, declared_entities=[SpeakerRosterEntry("entity:mentioned", "Mentioned Guest", "entity")],
        segment_text="Mentioned Guest grows orchids.",
    )
    assert {row.coarse_role for row in roster if row.speaker_id in {"entity:guest-a", "entity:guest-b"}} == {"peer"}
    assert all(row.speaker_id for row in roster)
    xml = (f"<item><memory><source_role>{role}</source_role><speaker_ref>{ref}</speaker_ref>"
           "<content>A guest grows orchids.</content><categories><category>Gardens</category></categories>"
           "</memory></item>")
    entry = service._parse_structured_entries(["profile"], [xml], default_source_message_ids=indices, speaker_roster=roster)[0]
    attributed = service._attribute_memory(entry, speaker_map)
    assert attributed.speaker_id == expected
    service._llm_clients["embedding"] = _EmbedClient()
    state = asyncio.run(service._memorize_categorize_items({
        "segment_plans": [{"resource_url": "memory://guest-attribution", "entries": [attributed]}],
        "modality": "conversation", "local_path": None, "store": service.database, "user": scope,
    }, {}))
    item = service.database.memory_item_repo.get_item(state["items"][0].id)
    assert item.speaker_id == expected and item.speaker_id != ""
    assert item.source_message_ids == indices and item.source_role == entry.source_role


def test_invalid_speaker_ref_stays_distinct_from_omitted_ref_before_attribution() -> None:
    service = _service()
    responses = [f"<item><memory><source_role>peer</source_role><speaker_ref>{ref}</speaker_ref>"
                 "<content>A guest grows orchids.</content></memory></item>" for ref in ("", "entity:missing")]
    entries = service._parse_structured_entries(["profile", "profile"], responses, default_source_message_ids=[0])
    assert len(entries) == 2
    attributed = [service._attribute_memory(entry, {0: ("entity:guest-a", "Known Guest")}) for entry in entries]
    assert [entry.speaker_id for entry in attributed] == ["entity:guest-a", None]


def test_replacement_selection_keeps_speaker_identity_barrier() -> None:
    service = _service()
    scope = {"user_id": "testowner", "soul_id": "testsoul"}
    repo = service.database.memory_item_repo
    old = [repo.create_item(memory_type="profile", summary="I grow roses", embedding=[0.1, 0.2, 0.3],
                           user_data=scope, source_role="peer", speaker_id=identity)
           for identity in ("entity:guest-b", "entity:guest-a", None)]
    entry = StructuredMemoryEntry("profile", "I grow orchids", [], "peer", 0.8, [0], None,
                                  replaces_previous_fact="I grow roses")
    for identity, expected in (("entity:guest-a", old[1].id), (None, old[2].id), ("entity:missing", None)):
        targets = asyncio.run(dedupe._find_supersede_targets(
            structured_entries=[entry._replace(speaker_id=identity)], store=service.database,
            embed_client=_EmbedClient(), user=scope, threshold=0.8,
        ))
        assert targets == ({0: expected} if expected else {})
