import json
from types import SimpleNamespace

import pytest

from memu.app.memorize import MemorizeMixin


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def chat(self, _prompt: str):
        return self._response


class _DummyMemorize(MemorizeMixin):
    memorize_config = SimpleNamespace(episodes_per_segment=4)

    @staticmethod
    def _escape_prompt_value(value: str) -> str:
        return value

    def _get_llm_client(self, *_args, **_kwargs):
        raise AssertionError("llm_client should be injected in this test")


@pytest.mark.asyncio
async def test_split_conversation_into_episodes_caps_and_merges_tail() -> None:
    conversation = json.dumps(
        [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "m1"},
            {"role": "user", "content": "m2"},
            {"role": "assistant", "content": "m3"},
            {"role": "user", "content": "m4"},
            {"role": "assistant", "content": "m5"},
        ]
    )
    llm_response = json.dumps(
        {
            "episodes": [
                {"start": 0, "end": 0, "caption": "e0"},
                {"start": 1, "end": 1, "caption": "e1"},
                {"start": 2, "end": 2, "caption": "e2"},
                {"start": 3, "end": 3, "caption": "e3"},
                {"start": 4, "end": 4, "caption": "e4"},
                {"start": 5, "end": 5, "caption": "e5"},
            ]
        }
    )

    service = _DummyMemorize()
    episodes = await service._split_conversation_into_episodes(
        conversation,
        "{conversation}",
        llm_client=_FakeLLM(llm_response),
    )

    assert len(episodes) == 4
    assert episodes[0]["message_indices"] == [0]
    assert episodes[1]["message_indices"] == [1]
    assert episodes[2]["message_indices"] == [2]
    assert episodes[3]["message_indices"] == [3, 4, 5]
