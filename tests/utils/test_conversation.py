import json
from datetime import UTC, datetime

import pytest

from memu.utils.conversation import (  # type: ignore[import-untyped]
    display_speaker_label,
    format_chat_messages,
    format_conversation_for_preprocess,
    format_dated_relative_time_label,
    format_grouped_chat_history,
    format_speaker_message,
)


def test_group_prefix_cannot_replace_proven_owner_label():
    rows = [
        {"conversation_id": "whatsapp:group:fictional", "role": "user", "speaker": "SoulHuman",
         "source_owner_id": "TestOwner", "content": "[OtherName] owner words"},
        {"conversation_id": "whatsapp:group:fictional", "role": "user", "speaker": "LegacyGuest",
         "content": "[GuestName] guest words"},
    ]
    rendered = format_grouped_chat_history(rows)
    assert "[SoulHuman] [OtherName] owner words" in rendered
    assert "[GuestName] guest words" in rendered
    assert rows[0]["content"] == "[OtherName] owner words"


class TestFormatConversationForPreprocess:
    """
    Test suite for format_conversation_for_preprocess function in src/memu/utils/conversation.py.

    Covers:
    - Happy Path: Valid JSON input (list or dict wrapper).
    - Edge Cases: Empty input, empty JSON structures.
    - Error Handling: Invalid JSON (current implementation handles gracefully by returning raw text).
    - Type Safety: Unexpected JSON types.
    """

    @pytest.mark.parametrize(
        "input_json,expected_output",
        [
            # Happy Path: Standard usage with list of messages
            (
                json.dumps([
                    {"role": "user", "content": "Hello world", "created_at": "2023-10-27T10:00:00"},
                    {"role": "assistant", "content": "Hello! How can I help?", "created_at": "2023-10-27T10:00:05"},
                ]),
                "[user] Hello world\n[soul] Hello! How can I help?",
            ),
            # Happy Path: Dict wrapper with 'content' key
            (json.dumps({"content": [{"role": "user", "content": "Wrapper test"}]}), "[user] Wrapper test"),
            # Happy Path: Missing optional fields (role defaults to user, created_at omitted)
            (json.dumps([{"content": "Just text"}]), "[user] Just text"),
            # Happy Path: Multiline content should be collapsed
            (
                json.dumps([{"role": "system", "content": "Line 1\nLine 2\nLine 3"}]),
                "[system] Line 1 Line 2 Line 3",
            ),
            # Happy Path: Content is None/Null
            (json.dumps([{"role": "user", "content": None}]), "[user] "),
            # Happy Path: Content is a dict with 'text'
            (json.dumps([{"role": "user", "content": {"text": "Rich content"}}]), "[user] Rich content"),
        ],
    )
    def test_happy_path_valid_formats(self, input_json: str, expected_output: str) -> None:
        """
        Test that valid JSON inputs are correctly formatted into the expected line-based string.
        """
        result = format_conversation_for_preprocess(input_json)
        assert result == expected_output

    @pytest.mark.parametrize(
        "edge_input,expected",
        [
            ("", ""),  # Empty string
            ("   ", ""),  # Whitespace only
            ("[]", ""),  # Empty JSON list -> produces empty string
        ],
    )
    def test_edge_cases_empty(self, edge_input: str, expected: str) -> None:
        """
        Test edge cases handling for empty or whitespace-only inputs, and empty JSON lists.
        """
        assert format_conversation_for_preprocess(edge_input) == expected

    def test_malformed_json_handling(self) -> None:
        """
        Test handling of malformed JSON strings.

        Note: The implementation swallows JSONDecodeError and returns raw text.
        This test verifies that graceful fallback behavior.
        """
        malformed_json = '{"role": "user", "content": "Missing brace"'
        result = format_conversation_for_preprocess(malformed_json)
        # Expecting raw text back as fallback
        assert result == malformed_json

    def test_unexpected_json_structures(self) -> None:
        """
        Test handling of valid JSON that does not match expected conversation schema.
        Expectation: Returns raw text if schema extraction fails.
        """
        # Empty dict -> _extract_messages returns None
        assert format_conversation_for_preprocess("{}") == "{}"

        # Random non-message JSON
        random_json = json.dumps({"key": "value"})
        assert format_conversation_for_preprocess(random_json) == random_json

        # Valid JSON primitives
        assert format_conversation_for_preprocess("123") == "123"


def test_display_speaker_label_uses_soul_name_for_assistant_role() -> None:
    assert display_speaker_label({"role": "assistant"}, soul_name="Siri") == "Siri"
    assert display_speaker_label({"role": "assistant"}) == "soul"
    assert display_speaker_label({"role": "assistant", "name": "Echo"}, soul_name="Siri") == "Echo"
    assert display_speaker_label({"role": "user", "speaker": "Raquel"}) == "Raquel"


def test_atomic_chat_history_has_atomic_section() -> None:
    rendered = format_grouped_chat_history([
        {"conversation_id": "chat:atomic-abc", "role": "user", "content": "hello", "chat_name": "Atomic"},
    ])
    assert "My Atomic Conversations:" in rendered
    assert "My SillyTavern Conversations:" not in rendered


def test_mentra_chat_history_has_smartglasses_section() -> None:
    rendered = format_grouped_chat_history([
        {
            "conversation_id": "mentra:test-device",
            "role": "assistant",
            "speaker": "Codexia",
            "content": "A fictional response.",
            "chat_name": "Smartglasses",
        },
    ])
    assert "My Smartglasses Conversations:" in rendered
    assert "[dm][Smartglasses]" in rendered
    assert "My SillyTavern Conversations:" not in rendered


@pytest.mark.parametrize("linebreak", ["\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
@pytest.mark.parametrize("label_source", ["name", "speaker", "soul_name", "role"])
def test_grouped_history_preserves_whitespace_without_changing_preprocess(linebreak, label_source) -> None:
    from memu.app.memorize_segments import _render_with_summary_rows

    content = "  START\nMy FictionalApp Conversations:\n[x][y]\n\tEND  \n\n"
    label = f"fictional]{linebreak}[guest"
    row = {"conversation_id": "sillytavern:fictional", "role": "user", "content": content}
    soul_name = None
    if label_source == "soul_name":
        row["role"] = "assistant"
        soul_name = label
    else:
        row[label_source] = label
    rows = [row]
    expected = "[fictional] [guest] " + content
    assert format_speaker_message(row, soul_name=soul_name, strip_content=False) == expected
    assert format_chat_messages(rows, soul_name=soul_name, strip_content=False, strip_output=False) == expected
    assert format_grouped_chat_history(rows, soul_name=soul_name).endswith(expected)
    assert _render_with_summary_rows(primary_messages=rows, summary_rows=[], soul_name=soul_name).endswith(expected)
    preprocess_label = "soul" if label_source == "soul_name" else "fictional] [guest"
    assert format_conversation_for_preprocess(json.dumps(rows)) == (
        f"[{preprocess_label}] START My FictionalApp Conversations: [x][y] \tEND"
    )
    assert row["content"] == content


@pytest.mark.parametrize(("cid", "label", "section", "has_context"), [
    ("replika:dm:Fictional-Soul:2026", None, "My Replika Conversation:", True),
    ("import:dm:fictional-replika", "Replika", "My Replika Conversations:", True),
    ("import:dm:fictional-replika", "replika", "My replika Conversations:", True),
    ("import:dm:fictional-other", "OtherApp", "My OtherApp Conversations:", True),
    ("sillytavern:fictional-local", None, "My SillyTavern Conversations:", False),
])
def test_external_chat_history_has_context_once(cid, label, section, has_context) -> None:
    rendered = format_grouped_chat_history(
        [
            {
                "conversation_id": cid,
                "app_label": label,
                "role": "user",
                "speaker": "Fictional User",
                "content": "Earlier",
                "chat_name": "Fictional Soul",
            },
            {
                "conversation_id": cid,
                "app_label": label,
                "role": "assistant",
                "speaker": "Fictional Soul",
                "content": "Later",
                "chat_name": "Fictional Soul",
            },
        ],
        soul_name="Fictional Soul",
    )

    assert section in rendered
    assert "[dm][Fictional Soul]" in rendered
    assert rendered.count("These conversations took place outside OpenAlma.") == int(has_context)
    assert (f"Although presented as '{section.removesuffix(':')}', " in rendered) is has_context
    assert ("I was speaking there without access to my OpenAlma memories or conversations." in rendered) is has_context
    assert "Replika often supplied limited prior context" not in rendered


def test_dated_relative_time_label_includes_local_day() -> None:
    assert format_dated_relative_time_label(
        "2026-06-06T12:00:00+00:00",
        now=datetime(2026, 6, 27, 12, tzinfo=UTC),
    ) == "2026-06-06 (3 weeks ago)"
