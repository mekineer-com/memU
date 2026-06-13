"""End-to-end attribution pipeline test (without the LLM in the middle).

Covers the chain that broke in the Phase 5 live smoke:
  _build_speaker_map → _attribute_memory → _decorate_entries_with_plan_context → _persist_memory_items → DB
The LLM extraction step is skipped — we hand-construct StructuredMemoryEntry
objects as if the LLM emitted them, then verify the pipeline fills
speaker_id/speaker_label/source_message_ids correctly and the fields round-trip
through the sqlite reinforce-enabled write path.

Three real bugs this guards against in combination:
  - 30b8db2: _resolve_source_message_ids empty fallback
  - 39511ef: sqlite reinforce branch dropping source_message_ids
  - 55525ae: role=user always resolves to scope user, not entity:<name>

Any of these regressing alone would flip at least one assertion below.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from memu.app.memorize import StructuredMemoryEntry
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
    ctx = service._get_context()
    user_data = {"user_id": "marcos", "soul_id": "siri"}

    # Simulated episode. Note the display-name mismatch: message.name="MarcosDisplay"
    # is NOT case-equal to scope.user_id="marcos" — the bug was that this tripped
    # the old code into slugging the user as entity:marcosdisplay.
    segment_messages = [
        {"_message_index": 0, "role": "user", "name": "MarcosDisplay", "content": "Hi"},
        {"_message_index": 1, "role": "assistant", "name": "Siri", "content": "Hello"},
        {"_message_index": 2, "role": "user", "name": "MarcosDisplay", "content": "Thanks"},
    ]
    message_indices = [0, 1, 2]
    speaker_map = service._build_speaker_map(segment_messages, user_data)

    # Post-55525ae: user-role messages always resolve to the scope user slug,
    # with the display name as the label.
    assert speaker_map[0] == ("user:marcos", "MarcosDisplay")
    assert speaker_map[1] == ("soul:siri", "Siri")

    # Simulate the LLM emitting two entries without source_message_ids (the
    # field was removed from prompts in 02d8bde — prompts no longer request it).
    entries = [
        StructuredMemoryEntry(
            memory_type="behavior",
            content="Marcos greeted Siri warmly",
            categories=[],
            source_role="user",
            confidence=0.8,
            source_message_ids=[],
            reflection_salience=None,
            emotional_intensity=0.7,
        ),
        StructuredMemoryEntry(
            memory_type="behavior",
            content="I greeted Marcos back",
            categories=[],
            source_role="soul",
            confidence=0.8,
            source_message_ids=[],
            reflection_salience=None,
            emotional_intensity=0.2,
        ),
    ]

    # Post-30b8db2: decoration falls back to the full episode range when the
    # LLM emitted nothing.
    decorated = service._decorate_entries_with_plan_context(entries, message_indices=message_indices)
    assert decorated[0].source_message_ids == [0, 1, 2]
    assert decorated[1].source_message_ids == [0, 1, 2]

    # Attribute from speaker_map. Source_role=user + multi-speaker range →
    # role-based disambiguation picks user:marcos.
    attributed = [service._attribute_memory(entry, speaker_map) for entry in decorated]
    assert attributed[0].speaker_id == "user:marcos"
    assert attributed[0].speaker_label == "MarcosDisplay"
    assert attributed[1].speaker_id == "soul:siri"
    assert attributed[1].speaker_label == "Siri"

    # Persist. reinforce=True by default (memorize_config). Post-39511ef:
    # source_message_ids and reflection_salience survive the reinforce branch.
    async def _persist():
        return await service._persist_memory_items(
            resource_id="res-1",
            structured_entries=attributed,
            ctx=ctx,
            store=store,
            embed_client=_EmbedClient(),
            user=user_data,
        )

    items, _rels, _updates, _ = asyncio.run(_persist())
    assert len(items) == 2

    loaded_user = store.memory_item_repo.get_item(items[0].id)
    loaded_soul = store.memory_item_repo.get_item(items[1].id)
    assert loaded_user is not None and loaded_soul is not None

    assert loaded_user.speaker_id == "user:marcos"
    assert loaded_user.speaker_label == "MarcosDisplay"
    assert loaded_user.source_message_ids == [0, 1, 2]
    assert loaded_user.source_role == "user"
    assert loaded_user.emotional_intensity == 0.7

    assert loaded_soul.speaker_id == "soul:siri"
    assert loaded_soul.speaker_label == "Siri"
    assert loaded_soul.source_message_ids == [0, 1, 2]
    assert loaded_soul.source_role == "soul"
    assert loaded_soul.emotional_intensity == 0.2
