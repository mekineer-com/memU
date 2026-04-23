from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import BaseModel

from memu.app.memorize import StructuredMemoryEntry
from memu.app.service import MemoryService
from memu.database.models import MemoryItem


class SpeakerPhase2Scope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


@pytest.fixture(scope="module")
def service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": SpeakerPhase2Scope},
    )


def _entry(*, source_role: str | None, source_message_ids: list[int]) -> StructuredMemoryEntry:
    return StructuredMemoryEntry(
        memory_type="event",
        content="test memory",
        categories=["communication"],
        source_role=source_role,
        confidence=0.8,
        source_message_ids=source_message_ids,
        reflection_salience=None,
    )


def _item(*, item_id: str, summary: str, source_role: str | None, speaker_id: str | None) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        resource_id=None,
        memory_type="event",
        summary=summary,
        embedding=[0.2, 0.4, 0.6],
        source_role=source_role,
        speaker_id=speaker_id,
    )


def _build_token_index(summary_tokens: Mapping[str, set[str]]) -> tuple[dict[str, set[str]], dict[str, int]]:
    token_index: dict[str, set[str]] = {}
    token_freq: dict[str, int] = {}
    for item_id, tokens in summary_tokens.items():
        for token in tokens:
            token_index.setdefault(token, set()).add(item_id)
            token_freq[token] = token_freq.get(token, 0) + 1
    return token_index, token_freq


def test_build_speaker_map_resolves_user_soul_entity_and_environment(service: MemoryService) -> None:
    # role=user is always the scope user; message.name is a display label, not a
    # different identity. Third-party speakers come through role=entity or the
    # ambiguous-episode roster (Phase 4) — never as role=user+name=SomeoneElse.
    episode_messages = [
        {"_message_index": 0, "role": "user", "name": "Marcos"},
        {"_message_index": 1, "role": "assistant", "name": "Siri"},
        {"_message_index": 2, "role": "user", "name": "MarcosDisplay"},
        {"_message_index": 3, "role": "entity", "name": "Brother"},
        {"_message_index": 4, "role": "system"},
    ]
    speaker_map = service._build_speaker_map(episode_messages, {"user_id": "Marcos", "soul_id": "Siri"})

    assert speaker_map[0] == ("user:marcos", "Marcos")
    assert speaker_map[1] == ("soul:siri", "Siri")
    assert speaker_map[2] == ("user:marcos", "MarcosDisplay")
    assert speaker_map[3] == ("entity:brother", "Brother")
    assert speaker_map[4] == ("environment:system", "system")


def test_attribute_memory_fills_when_unambiguous(service: MemoryService) -> None:
    speaker_map = {7: ("user:marcos", "Marcos")}
    attributed = service._attribute_memory(_entry(source_role="user", source_message_ids=[7]), speaker_map)
    assert attributed.speaker_id == "user:marcos"
    assert attributed.speaker_label == "Marcos"


def test_attribute_memory_uses_source_role_to_disambiguate_mixed_speakers(service: MemoryService) -> None:
    speaker_map = {
        10: ("user:marcos", "Marcos"),
        11: ("entity:brother", "Brother"),
    }
    attributed = service._attribute_memory(_entry(source_role="user", source_message_ids=[10, 11]), speaker_map)
    assert attributed.speaker_id == "user:marcos"
    assert attributed.speaker_label == "Marcos"


def test_attribute_memory_leaves_null_when_role_stays_ambiguous(service: MemoryService) -> None:
    speaker_map = {
        20: ("entity:brother", "Brother"),
        21: ("entity:mother", "Mother"),
    }
    attributed = service._attribute_memory(_entry(source_role="entity", source_message_ids=[20, 21]), speaker_map)
    assert attributed.speaker_id is None
    assert attributed.speaker_label is None


def test_prefilter_dedupe_candidates_respects_source_role_and_speaker_id(service: MemoryService) -> None:
    anchor = _item(
        item_id="a",
        summary="Marcos said we should pause before replying to avoid confusion",
        source_role="user",
        speaker_id="user:marcos",
    )
    same_speaker = _item(
        item_id="b",
        summary="Marcos said we should pause before replying to avoid confusion",
        source_role="user",
        speaker_id="user:marcos",
    )
    other_speaker = _item(
        item_id="c",
        summary="Marcos said we should pause before replying to avoid confusion",
        source_role="user",
        speaker_id="user:brother",
    )

    active_pool = {
        "a": anchor,
        "b": same_speaker,
        "c": other_speaker,
    }
    summary_tokens = {
        item_id: service._dedupe_summary_tokens(item.summary)
        for item_id, item in active_pool.items()
    }
    token_index, token_freq = _build_token_index(summary_tokens)

    candidates = service._prefilter_dedupe_candidate_ids(
        anchor_id="a",
        anchor=anchor,
        active_pool=active_pool,
        merged_map={},
        summary_tokens=summary_tokens,
        token_index=token_index,
        token_freq=token_freq,
    )

    assert "b" in candidates
    assert "c" not in candidates


def test_retrieve_materialized_item_includes_speaker_fields(service: MemoryService) -> None:
    item = _item(
        item_id="m1",
        summary="Marcos and Siri agreed to revisit this tomorrow",
        source_role="user",
        speaker_id="user:marcos",
    )
    item.speaker_label = "Marcos"

    rendered = service._materialize_hits([("m1", 0.88)], {"m1": item})

    assert len(rendered) == 1
    assert rendered[0]["speaker_id"] == "user:marcos"
    assert rendered[0]["speaker_label"] == "Marcos"
