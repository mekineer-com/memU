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
        extract_json_blob=lambda text: text,
    )
    assert out == {"a": "A summary", "b": "B summary"}


@pytest.mark.asyncio
async def test_render_episode_with_background_context_uses_raw_lines_below_floor() -> None:
    background_messages = [{"_message_index": 1, "role": "user", "name": "N", "content": "small",
                            "source_label": "whatsapp:dm", "source_conversation_id": "whatsapp:dm:fictional",
                            "chat_name": "Small chat"}]

    async def _batch(**_kwargs):
        raise AssertionError("batch should not run when below floor")

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        background_messages=background_messages,
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=9999),
    )
    assert "[Background:whatsapp:dm]" not in rendered
    assert "[whatsapp:dm]" not in rendered
    assert "[N] small" in rendered
    assert rows[0]["raw"] is True and rows[0]["content"] == "small"
    assert "summary" not in rows[0]


@pytest.mark.asyncio
async def test_render_episode_with_background_context_uses_soul_name_for_assistant_role() -> None:
    background_messages = [{"_message_index": 1, "role": "assistant", "content": "small",
                            "source_label": "whatsapp:dm", "source_conversation_id": "whatsapp:dm:fictional",
                            "chat_name": "Small chat"}]

    async def _batch(**_kwargs):
        raise AssertionError("batch should not run when below floor")

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        background_messages=background_messages,
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=9999),
        soul_name="TestSoul",
    )
    assert "[TestSoul] small" in rendered
    assert "[whatsapp:dm]" not in rendered
    assert "[assistant]" not in rendered
    assert rows[0]["raw"] is True and rows[0]["content"] == "small"


@pytest.mark.asyncio
async def test_render_episode_with_background_context_groups_primary_cross_chats() -> None:
    primary_messages = [
        {
            "_message_index": 1,
            "role": "user",
            "speaker": "TestGuest",
            "content": "dm primary",
            "source_conversation_id": "whatsapp:dm:liz",
            "chat_name": "Test Contact",
            "received_at": "2026-06-12T10:00:00+00:00",
            "memorize_chat": True,
        },
        {
            "_message_index": 2,
            "role": "user",
            "speaker": "TestOwner",
            "content": "group primary",
            "source_conversation_id": "whatsapp:group:fictional-group",
            "chat_name": "Workshop",
            "received_at": "2026-06-12T10:01:00+00:00",
            "memorize_chat": True,
        },
    ]

    async def _batch(**_kwargs):
        raise AssertionError("batch should not run without background messages")

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        background_messages=[],
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=100),
        soul_name="TestSoul",
    )
    assert rows == []
    assert rendered == ""
    rendered = segment_helpers._render_grouped_chat_messages(primary_messages, soul_name="TestSoul")
    assert "My WhatsApp Conversations:" in rendered
    assert "[dm][Test Contact]" in rendered
    assert "[group][Workshop]" in rendered
    assert "--- " in rendered
    assert "[TestGuest] dm primary" in rendered
    assert "[TestOwner] group primary" in rendered
    assert "[whatsapp:dm]" not in rendered
    assert "[whatsapp:group]" not in rendered
    assert "current chat" not in rendered
    original = [dict(row) for row in primary_messages]
    evidence = segment_helpers._render_grouped_chat_messages(
        list(reversed(primary_messages)),
        speaker_map={1: ("entity:known", "TestGuest"), 2: ("", "TestOwner")},
    )
    assert "[TestGuest; speaker_ref=entity:known] dm primary" in evidence
    assert "[TestOwner; unlinked] group primary" in evidence
    assert primary_messages == original


def test_render_grouped_chat_messages_places_activities_before_chats() -> None:
    rendered = segment_helpers._render_grouped_chat_messages(
        [
            {
                "_message_index": 1,
                "role": "user",
                "speaker": "User A",
                "content": "chat first",
                "source_conversation_id": "whatsapp:dm:contact-a",
                "received_at": "2026-06-12T10:00:00+00:00",
            },
            {
                "_message_index": 2,
                "role": "assistant",
                "speaker": "SoulA",
                "chat_name": "SoulA",
                "content": "I wrote a note to myself.",
                "source_conversation_id": "activity:dm:SoulA",
                "received_at": "2026-06-12T10:01:00+00:00",
            },
        ],
        soul_name="SoulA",
    )

    assert rendered.index("My Activities:") < rendered.index("My WhatsApp Conversations:")
    assert "[SoulA] I wrote a note to myself." in rendered


def test_render_grouped_chat_messages_shows_timestamp_fallback_date() -> None:
    rendered = segment_helpers._render_grouped_chat_messages([{
        "_message_index": 1,
        "role": "user",
        "speaker": "User A",
        "content": "dated message",
        "source_conversation_id": "whatsapp:dm:contact-a",
        "timestamp": "2026-01-02T10:00:00-05:00",
    }])

    assert "--- 2026-01-02 (" in rendered


@pytest.mark.asyncio
async def test_render_episode_with_background_context_uses_batch_summary_when_above_floor() -> None:
    background_messages = [
        {"_message_index": 1, "role": "user", "name": "A", "content": "w " * 150, "source_label": "whatsapp:dm", "source_conversation_id": "c1"},
        {"_message_index": 2, "role": "user", "name": "B", "content": "w " * 150, "source_label": "sillytavern", "source_conversation_id": "c2"},
    ]

    async def _batch(**_kwargs):
        return {"c1": "summary one", "c2": "summary two"}

    rendered, rows = await segment_helpers._render_episode_with_background_context(
        background_messages=background_messages,
        llm_client=None,
        summarize_background_groups_batched=_batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=0),
    )
    assert "summary one" in rendered
    assert "summary two" in rendered
    assert "[Background:" not in rendered
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_render_episode_with_background_context_raises_on_missing_batch_group() -> None:
    background_messages = [
        {"_message_index": 1, "role": "user", "name": "A", "content": "w " * 150, "source_label": "whatsapp:dm", "source_conversation_id": "c1"},
        {"_message_index": 2, "role": "user", "name": "B", "content": "w " * 150, "source_label": "sillytavern", "source_conversation_id": "c2"},
    ]

    async def _batch(**_kwargs):
        return {"c1": "summary one"}

    with pytest.raises(ValueError, match="missing background summary for source"):
        await segment_helpers._render_episode_with_background_context(
            background_messages=background_messages,
            llm_client=None,
            summarize_background_groups_batched=_batch,
            memorize_config=SimpleNamespace(background_extra_messages_tokens=0),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", [(99,), (100,), (101,), (60, 60), (99, 100, 101)])
async def test_background_threshold_is_per_chat_and_keeps_prior_prose(monkeypatch, sizes):
    monkeypatch.setattr(segment_helpers, "_estimate_text_tokens", lambda text: int(text))
    messages = [{
        "source_conversation_id": f"whatsapp:group:group-{index}",
        "chat_name": f"Workshop {index}", "source_label": "whatsapp:group",
        "role": "user", "speaker": "TestGuest", "content": str(size),
        "_message_index": index,
    } for index, size in enumerate(sizes)]
    prior = [{
        "source_conversation_id": row["source_conversation_id"],
        "source_label": row["source_label"], "chat_name": row["chat_name"],
        "summary": f"Prior {index}\n  unchanged paragraph.", "rolled_up": True,
    } for index, row in enumerate(messages)]
    calls = []

    async def batch(**kwargs):
        calls.append(kwargs)
        return {key: f"Follow-up {key}" for key in kwargs["group_order"]}

    block, context = await segment_helpers._render_episode_with_background_context(
        background_messages=messages, prior_rows=prior, llm_client=None,
        summarize_background_groups_batched=batch,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=100),
    )
    expected = [row["source_conversation_id"] for row, size in zip(messages, sizes) if size >= 100]
    assert len(calls) == bool(expected)
    if expected:
        assert calls[0]["group_order"] == expected
        assert list(calls[0]["grouped_messages"]) == expected
    assert context[:len(prior)] == prior
    for row, size, old in zip(messages, sizes, prior):
        assert block.count(f"[group][{row['chat_name']}]") == 1
        assert old["summary"] in block
        if size < 100:
            assert f"[TestGuest] {size}" in block
            raw = next(item for item in context if item.get("raw") and item["content"] == str(size))
            assert "summary" not in raw
        else:
            assert f"Follow-up {row['source_conversation_id']}" in block
            assert f"[TestGuest] {size}" not in block
    assert all("raw" not in row for row in messages)


@pytest.mark.asyncio
async def test_quiet_summary_renders_without_messages_or_paid_call():
    prior = [{"source_conversation_id": "whatsapp:dm:fictional", "chat_name": "Quiet chat",
              "summary": "A remembered decision.\n  Verbatim details."}]

    async def unexpected(**_kwargs):
        pytest.fail("quiet summaries must not be summarized again")

    block, rows = await segment_helpers._render_episode_with_background_context(
        background_messages=[], prior_rows=prior, llm_client=None,
        summarize_background_groups_batched=unexpected,
        memorize_config=SimpleNamespace(background_extra_messages_tokens=100),
    )
    assert rows == prior
    assert "[dm][Quiet chat]" in block and prior[0]["summary"] in block
    assert "[user]" not in block and "--- " not in block
