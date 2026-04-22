from __future__ import annotations

import pytest
from pydantic import BaseModel

from memu.app.memorize import SpeakerRosterEntry
from memu.app.service import MemoryService


class SpeakerPhase4Scope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


@pytest.fixture(scope="module")
def service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": SpeakerPhase4Scope},
    )


def test_unambiguous_episode_skips_roster(service: MemoryService) -> None:
    speaker_map = {
        0: ("user:marcos", "Marcos"),
        1: ("soul:siri", "Siri"),
    }
    roster = service._build_speaker_roster_if_ambiguous(speaker_map)
    assert roster is None

    prompt = service._build_memory_type_prompt(
        memory_type="event",
        resource_text="[0] [Marcos] hi\n[1] [Siri] hello",
        categories_str="relationships",
        soul_context_str="## Relationships\nKnown context.",
        speaker_roster=roster,
    )
    assert "Speaker Roster (ambiguous episode fallback)" not in prompt
    assert "Allowed source_role schema for this episode" not in prompt
    assert "Only emit <speaker_ref>" not in prompt


def test_ambiguous_episode_attaches_roster_and_accepts_valid_speaker_ref(service: MemoryService) -> None:
    speaker_map = {
        10: ("entity:alice", "Alice"),
        11: ("entity:bob", "Bob"),
        12: ("soul:siri", "Siri"),
    }
    roster = service._build_speaker_roster_if_ambiguous(speaker_map)
    assert roster is not None
    assert len(roster) == 3

    prompt = service._build_memory_type_prompt(
        memory_type="event",
        resource_text="[10] [Alice] We should review this tomorrow.\n[11] [Bob] Agreed.",
        categories_str="relationships",
        soul_context_str="## Relationships\nKnown context.",
        speaker_roster=roster,
    )
    assert "Speaker Roster (ambiguous episode fallback)" in prompt
    assert "Allowed source_role schema for this episode: <source_role>user|soul|peer|entity|environment</source_role>." in prompt
    assert "Only emit <speaker_ref> if the speaker is in this roster. Never invent a slug." in prompt
    assert "- entity:alice | label=Alice | role=entity" in prompt
    assert "- entity:bob | label=Bob | role=entity" in prompt

    response = """
<item>
    <memory>
        <source_role>entity</source_role>
        <speaker_ref>entity:alice</speaker_ref>
        <content>Alice asked to review this tomorrow</content>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
""".strip()
    entries = service._parse_structured_entries(
        ["event"],
        [response],
        default_source_message_ids=[10, 11],
        speaker_roster=roster,
    )
    assert len(entries) == 1
    assert entries[0].speaker_id == "entity:alice"
    assert entries[0].speaker_label == "Alice"

    attributed = service._attribute_memory(entries[0], speaker_map)
    assert attributed.speaker_id == "entity:alice"
    assert attributed.speaker_label == "Alice"


def test_parser_rejects_hallucinated_speaker_ref_and_leaves_speaker_null(service: MemoryService) -> None:
    speaker_map = {
        20: ("entity:alice", "Alice"),
        21: ("entity:bob", "Bob"),
    }
    roster = service._build_speaker_roster_if_ambiguous(speaker_map)
    assert roster is not None

    response = """
<item>
    <memory>
        <source_role>entity</source_role>
        <speaker_ref>entity:ghost</speaker_ref>
        <content>The speaker made a request about tomorrow</content>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
""".strip()
    entries = service._parse_structured_entries(
        ["event"],
        [response],
        default_source_message_ids=[20, 21],
        speaker_roster=roster,
    )
    assert len(entries) == 1
    assert entries[0].speaker_id is None
    assert entries[0].speaker_label is None

    attributed = service._attribute_memory(entries[0], speaker_map)
    assert attributed.speaker_id is None
    assert attributed.speaker_label is None


def test_declared_entity_mention_triggers_roster_without_role_ambiguity(service: MemoryService) -> None:
    speaker_map = {
        30: ("user:marcos", "Marcos"),
        31: ("soul:siri", "Siri"),
    }
    roster = service._build_speaker_roster_for_episode(
        speaker_map=speaker_map,
        declared_entities=[SpeakerRosterEntry("entity:brother", "Brother", "entity")],
        episode_text="[30] [Marcos] My brother said he'll call tomorrow.",
    )
    assert roster is not None
    ids = {entry.speaker_id for entry in roster}
    assert "entity:brother" in ids
    assert "user:marcos" in ids


def test_attribute_memory_keeps_valid_parsed_speaker_ref_even_with_single_message_speaker(service: MemoryService) -> None:
    speaker_map = {40: ("user:marcos", "Marcos")}
    entry = service._parse_structured_entries(
        ["event"],
        [
            """
<item>
  <memory>
    <source_role>entity</source_role>
    <speaker_ref>entity:brother</speaker_ref>
    <content>Brother said he will call tomorrow</content>
    <categories><category>Relationships</category></categories>
  </memory>
</item>
""".strip()
        ],
        default_source_message_ids=[40],
        speaker_roster=[SpeakerRosterEntry("entity:brother", "Brother", "entity")],
    )[0]

    attributed = service._attribute_memory(entry, speaker_map)
    assert attributed.speaker_id == "entity:brother"
    assert attributed.speaker_label == "Brother"
