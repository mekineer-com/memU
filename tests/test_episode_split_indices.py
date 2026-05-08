import json

import pytest

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


class _DummyClient:
    async def chat(self, _prompt: str) -> str:
        return json.dumps(
            {
                "episodes": [
                    {
                        "message_indices": ["0", "x", 2, -1, "999"],
                        "caption": "mixed",
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_split_conversation_into_episodes_ignores_non_numeric_message_indices() -> None:
    service = _service()
    raw_text = json.dumps(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
    )

    episodes = await service._split_conversation_into_episodes(
        raw_text,
        "{conversation}\nlimit={episodes_per_segment}",
        llm_client=_DummyClient(),
    )

    assert len(episodes) == 1
    assert episodes[0]["message_indices"] == [0, 2]
    assert "[0]" in episodes[0]["text"]
    assert "[2]" in episodes[0]["text"]
