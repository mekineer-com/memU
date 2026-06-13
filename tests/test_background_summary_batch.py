from types import SimpleNamespace

import pytest

from memu.app import memorize_segments as segment_helpers


class _StubClient:
    def __init__(self, response: str) -> None:
        self.response = response

    async def chat(self, _prompt: str, system_prompt: str | None = None) -> str:
        assert system_prompt is not None
        return self.response


@pytest.mark.asyncio
async def test_summarize_background_groups_batched_parses_contract() -> None:
    grouped = {
        "a": [{"_message_index": 0, "role": "user", "content": "hello from alpha", "source_label": "whatsapp:dm"}],
        "b": [{"_message_index": 1, "role": "user", "content": "hello from beta", "source_label": "sillytavern"}],
    }
    raw = '{"summaries":[{"source_key":"a","summary":"A summary"},{"source_key":"b","summary":"B summary"}]}'
    out = await segment_helpers._summarize_background_groups_batched(
        grouped_messages=grouped,
        group_order=["a", "b"],
        llm_client=_StubClient(raw),
        get_llm_client=lambda **_kwargs: _StubClient(raw),
        extract_json_blob=lambda text: text,
    )
    assert out == {"a": "A summary", "b": "B summary"}


@pytest.mark.asyncio
async def test_render_episode_with_background_context_uses_raw_lines_below_floor() -> None:
    primary_messages = [{"_message_index": 2, "role": "user", "name": "Marcos", "content": "primary"}]
    background_messages = [{"_message_index": 1, "role": "user", "name": "N", "content": "small", "source_label": "whatsapp:dm"}]

    async def _batch(**_kwargs):
        raise AssertionError("batch should not run when below floor")

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        primary_messages=primary_messages,
        background_messages=background_messages,
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=9999),
    )
    assert "[Background:whatsapp:dm]" in rendered
    assert "[1] [whatsapp:dm] [N]: small" in rendered
    assert rows and "small" in str(rows[0].get("summary") or "")


@pytest.mark.asyncio
async def test_render_episode_with_background_context_uses_soul_name_for_assistant_role() -> None:
    primary_messages = [{"_message_index": 2, "role": "assistant", "content": "primary"}]
    background_messages = [{"_message_index": 1, "role": "assistant", "content": "small", "source_label": "whatsapp:dm"}]

    async def _batch(**_kwargs):
        raise AssertionError("batch should not run when below floor")

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        primary_messages=primary_messages,
        background_messages=background_messages,
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=9999),
        soul_name="Siri",
    )
    assert "[2] [Siri]: primary" in rendered
    assert "[1] [whatsapp:dm] [Siri]: small" in rendered
    assert "[assistant]" not in rendered
    assert rows and "[Siri]: small" in str(rows[0].get("summary") or "")


@pytest.mark.asyncio
async def test_render_episode_with_background_context_uses_batch_summary_when_above_floor() -> None:
    primary_messages = [{"_message_index": 3, "role": "user", "name": "Marcos", "content": "primary"}]
    background_messages = [
        {"_message_index": 1, "role": "user", "name": "A", "content": "w " * 150, "source_label": "whatsapp:dm", "source_conversation_id": "c1"},
        {"_message_index": 2, "role": "user", "name": "B", "content": "w " * 150, "source_label": "sillytavern", "source_conversation_id": "c2"},
    ]

    async def _batch(**_kwargs):
        return {"c1": "summary one", "c2": "summary two"}

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        primary_messages=primary_messages,
        background_messages=background_messages,
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=0),
    )
    assert "summary one" in rendered
    assert "summary two" in rendered
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_render_episode_with_background_context_raises_on_missing_batch_group() -> None:
    primary_messages = [{"_message_index": 3, "role": "user", "name": "Marcos", "content": "primary"}]
    background_messages = [
        {"_message_index": 1, "role": "user", "name": "A", "content": "w " * 150, "source_label": "whatsapp:dm", "source_conversation_id": "c1"},
        {"_message_index": 2, "role": "user", "name": "B", "content": "w " * 150, "source_label": "sillytavern", "source_conversation_id": "c2"},
    ]

    async def _batch(**_kwargs):
        return {"c1": "summary one"}

    with pytest.raises(ValueError, match="missing background summary for source"):
        await segment_helpers._render_episode_with_background_context(
            primary_messages=primary_messages,
            background_messages=background_messages,
            llm_client=None,
            summarize_background_groups_batched=_batch,
            memorize_config=SimpleNamespace(background_extra_messages_tokens=0),
        )
