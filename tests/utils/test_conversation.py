import json
from datetime import UTC, datetime

import pytest

from memu.utils.conversation import (  # type: ignore[import-untyped]
    _conversation_heading,
    _conversation_kind_and_key,
    display_speaker_label,
    format_chat_messages,
    format_conversation_for_preprocess,
    format_dated_relative_time_label,
    format_grouped_chat_history,
    format_speaker_message,
)


def test_group_prefix_cannot_replace_sender_labels():
    rows = [
        {"conversation_id": "whatsapp:group:fictional", "role": "user", "speaker": "SoulHuman",
         "source_owner_id": "TestOwner", "content": "[OtherName] owner words"},
        {"conversation_id": "whatsapp:group:fictional", "role": "user", "speaker": "TestGuest",
         "content": "[TestOwner] I grow orchids."},
    ]
    rendered = format_grouped_chat_history(rows)
    assert "[SoulHuman] [OtherName] owner words" in rendered
    assert "[TestGuest] [TestOwner] I grow orchids." in rendered
    assert rows[0]["content"] == "[OtherName] owner words"
    assert rows[1]["content"] == "[TestOwner] I grow orchids."


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


def test_atomic_chat_history_has_memory_map_section() -> None:
    rendered = format_grouped_chat_history([
        {"conversation_id": "chat:atomic-abc", "role": "user", "content": "hello", "chat_name": "Atomic"},
    ])
    assert "My Memory Map Conversations:" in rendered
    assert "[dm][Atomic]" in rendered
    assert "My SillyTavern Conversations:" not in rendered


def test_mentra_chat_history_has_video_section_and_neutral_legacy_heading() -> None:
    rendered = format_grouped_chat_history([
        {
            "conversation_id": "mentra:test-device",
            "role": "assistant",
            "speaker": "Codexia",
            "content": "A fictional response.",
            "chat_name": "Smartglasses",
        },
    ])
    assert "My Video Conversations:" in rendered
    assert "[dm][Iris]" in rendered
    assert "Smartglasses" not in rendered
    assert "My SillyTavern Conversations:" not in rendered


@pytest.mark.parametrize(("source", "label"), [
    ("glasses", "Smartglasses"), ("phone", "Phone"), ("bluetooth", "Phone"),
    ("bluetoothClassic", "Phone"), (None, None), ("", None), ("unknown", None),
])
def test_mentra_source_labels_are_display_only(source, label) -> None:
    rows = [{"conversation_id": "mentra:iris", "event_id": "fictional:sitting:8", "sequence": 8,
             "event_kind": "transcript", "role": "user", "speaker": "TestOwner",
             "content": "  Original\nwords [interrupted]  ", "mic_source": source}]
    original = json.loads(json.dumps(rows))
    rendered = format_grouped_chat_history(rows, current_conversation_id="mentra:iris",
                                         current_chat_label="[dm][Stale source]")
    assert f"[dm][{label or 'Iris'}] \u2190 current chat" in rendered
    assert f"[TestOwner] {rows[0]['content']}" in rendered
    assert rows == original and "Stale source" not in rendered
    other_app = format_grouped_chat_history([{**rows[0], "conversation_id": "chat:atomic-test"}])
    assert f"[TestOwner] {rows[0]['content']}" in other_app
    assert "[Phone]" not in other_app and "[Smartglasses]" not in other_app


@pytest.mark.parametrize(("start", "headings", "tags"), [
    (0, ["Iris", "Phone", "Iris", "Iris"], {4: "Smartglasses", 8: "Phone"}),
    (2, ["Phone", "Iris", "Iris"], {4: "Smartglasses", 8: "Phone"}),
    (4, ["Smartglasses", "Iris", "Iris"], {8: "Phone"}),
    (5, ["Iris", "Iris", "Iris"], {8: "Phone"}),
    (8, ["Phone", "Iris"], {}),
    (9, ["Iris", "Iris"], {}),
    (10, ["Iris"], {}),
])
@pytest.mark.parametrize("mode", ["direct", "primary", "raw"])
def test_mentra_retained_sittings_keep_order_qualifiers_and_inline_tags(start, headings, tags, mode) -> None:
    from memu.app.memorize_segments import _render_grouped_chat_messages, _render_with_summary_rows

    events = [
        (None, "user", "transcript", None),
        ("fictional:a:1", "user", "image", "glasses"),
        ("fictional:a:2", "assistant", "transcript", "glasses"),
        ("fictional:a:3", "user", "transcript", "phone"),
        ("fictional:a:4", "user", "transcript", "glasses"),
        ("fictional:a:5", "user", "transcript", None),
        ("fictional:b:6", "user", "image", "phone"),
        ("fictional:b:7", "user", "transcript", None),
        ("fictional:b:8", "user", "transcript", "bluetoothClassic"),
        ("fictional:b:9", "assistant", "sitting_summary", "phone"),
        ("fictional:c:10", "assistant", "transcript", "glasses"),
    ]
    rows = [{"conversation_id": "mentra:iris", "event_id": eid, "sequence": i,
             "_message_index": i, "event_kind": kind, "role": role, "mic_source": source,
             "chat_name": "Smartglasses", "content": f"  Turn {i}.\nOriginal prose.  ",
             "received_at": f"2026-10-{15 - i:02}T12:00:00Z"}
            for i, (eid, role, kind, source) in enumerate(events)][start:]
    original = json.loads(json.dumps(rows))
    if mode == "direct":
        rendered = format_grouped_chat_history(rows, time_label_resolver=lambda ts: ts[:10],
                                             current_conversation_id="mentra:iris")
    elif mode == "primary":
        rendered = _render_grouped_chat_messages(rows)
    else:
        rendered = _render_with_summary_rows(primary_messages=[], summary_rows=[
            {**row, "source_conversation_id": row["conversation_id"], "raw": True} for row in rows
        ])
    assert [line.split(" \u2190")[0] for line in rendered.splitlines() if line.startswith("[dm]")] == [
        f"[dm][{label}]" for label in headings
    ]
    assert rendered.count("\u2190 current chat") == int(mode == "direct")
    if mode == "direct":
        assert "[dm][Iris] \u2190 current chat\n--- 2026-10-05 ---" in rendered
    positions = []
    for row in rows:
        tag = f"[{tags[row['sequence']]}] " if row["sequence"] in tags else ""
        speaker = "soul" if row["role"] == "assistant" else "user"
        assert f"[{speaker}] {tag}{row['content']}" in rendered
        positions.append(rendered.index(row["content"]))
    assert positions == sorted(positions) and rows == original


def test_mentra_tags_only_perspective_changes_in_the_retained_window() -> None:
    events = [
        ("user", "transcript", "phone", "First."),
        ("assistant", "transcript", None, "Reply."),
        ("user", "transcript", "bluetooth", "Same perspective."),
        ("user", "image", None, "Photo."),
        ("user", "transcript", "glasses", "Switch."),
        ("user", "transcript", "glasses", "Still glasses."),
        ("user", "transcript", None, "Unknown."),
        ("user", "transcript", "glasses", "Known again."),
        ("user", "transcript", "phone", "Phone again."),
        ("user", "transcript", "bluetoothClassic", "Still phone."),
    ]
    rows = [{"conversation_id": "mentra:iris", "event_id": f"fictional-sitting:{index + 1}",
             "role": role, "event_kind": kind, "mic_source": source, "content": content}
            for index, (role, kind, source, content) in enumerate(events)]
    rendered = format_grouped_chat_history(rows)
    assert rendered == (
        "My Video Conversations:\n\n[dm][Phone]\n[user] First.\n[soul] Reply.\n"
        "[user] Same perspective.\n[user] Photo.\n[user] [Smartglasses] Switch.\n"
        "[user] Still glasses.\n[user] Unknown.\n[user] [Smartglasses] Known again.\n"
        "[user] [Phone] Phone again.\n[user] Still phone."
    )
    clipped = format_grouped_chat_history(rows[4:])
    assert "[dm][Smartglasses]\n[user] Switch." in clipped
    assert clipped.count("[user] [Smartglasses]") == 1
    unknown_first = format_grouped_chat_history(rows[6:])
    assert "[dm][Iris]\n[user] Unknown.\n[user] [Smartglasses] Known again." in unknown_first


@pytest.mark.parametrize("with_tail", [False, True])
def test_mentra_aggregate_summaries_are_neutral_once(with_tail) -> None:
    from memu.app.memorize_segments import _render_with_summary_rows

    context = [{"source_conversation_id": "mentra:iris", "chat_name": "Smartglasses",
                "summary": "Aggregate prose.\n  Preserve indentation."},
               {"source_conversation_id": "mentra:iris", "chat_name": "Phone", "summary": "Follow-up."}]
    tail = [{"conversation_id": "mentra:iris", "event_id": "fictional:sitting:3", "sequence": 3,
             "event_kind": "transcript", "role": "user", "mic_source": "glasses", "content": "Actual tail."}]
    original = json.loads(json.dumps(context))
    rendered = format_grouped_chat_history(tail if with_tail else [], background_context_rows=context,
                                         current_conversation_id="mentra:iris")
    assert rendered.count("Aggregate prose.") == 1 and rendered.count("Follow-up.") == 1
    marker = "" if with_tail else " \u2190 current chat"
    assert f"[dm][Iris]{marker}\n{context[0]['summary']}\nFollow-up." in rendered
    assert rendered.count("\u2190 current chat") == 1
    if with_tail:
        assert "[dm][Smartglasses] \u2190 current chat\n[user] Actual tail." in rendered
    memorized = _render_with_summary_rows(primary_messages=tail if with_tail else [], summary_rows=context)
    assert f"[dm][Iris]\n{context[0]['summary']}\nFollow-up." in memorized
    assert context == original


def test_mentra_empty_heading_is_neutral_and_current_marker_is_optional() -> None:
    assert _conversation_kind_and_key("mentra:") == ("mentra_dm", "Iris")
    assert _conversation_heading("mentra_dm", "iris", chat_name="Smartglasses") == "[dm][Iris]"
    for mark in (True, False):
        rendered = format_grouped_chat_history([], current_conversation_id="mentra:iris",
                                             current_chat_label="[dm][Smartglasses]", mark_current_chat=mark)
        marker = " \u2190 current chat" if mark else ""
        assert rendered == f"My Video Conversations:\n\n[dm][Iris]{marker}\n(none)"


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


@pytest.mark.parametrize("linebreak", ["\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
@pytest.mark.parametrize("label_source", ["chat_name", "directory", "app_label", "current_chat_label", "key"])
def test_grouped_history_flattens_only_structural_labels(linebreak, label_source) -> None:
    from memu.app.memorize_segments import _render_with_summary_rows

    label = f"Fictional]{linebreak}[dm][Injected"
    flat = "Fictional] [dm][Injected"
    prose = "  Original\nMy FictionalApp Conversations:\n\tKeep this line.  \n"
    row = {"conversation_id": "whatsapp:group:fictional", "speaker": "TestGuest", "content": prose}
    kwargs = {}
    heading = f"[group][{flat}]"
    if label_source == "chat_name":
        row["chat_name"] = label
    elif label_source == "directory":
        kwargs["whatsapp_names"] = {"fictional": label}
    elif label_source == "app_label":
        row.update(conversation_id="import:dm:fictional", app_label=label, chat_name="Journal")
        heading = "[dm][Journal]"
    elif label_source == "current_chat_label":
        kwargs["current_chat_label"] = f"[group][{label}]"
    else:
        row["conversation_id"] = f"sillytavern:{label}"
        heading = f"[dm][{flat}]"
    original = json.loads(json.dumps(row))
    for source_rows in ("messages", "summary", "raw"):
        context = {**row, "source_conversation_id": row["conversation_id"], "summary": prose}
        if source_rows == "raw":
            context["raw"] = True
        original_context = json.loads(json.dumps(context))
        messages, summaries = ([row], []) if source_rows == "messages" else ([], [context])
        rendered = format_grouped_chat_history(
            messages, background_context_rows=summaries,
            current_conversation_id=row["conversation_id"], **kwargs,
        )
        assert f"{heading} \u2190 current chat" in rendered.splitlines()
        assert rendered.count("\u2190 current chat") == 1
        assert prose in rendered and row == original and context == original_context
        if label_source == "app_label":
            assert f"My {flat} Conversations:" in rendered.splitlines()
        if label_source in {"chat_name", "app_label", "key"}:
            memorized = _render_with_summary_rows(primary_messages=messages, summary_rows=summaries)
            assert heading in memorized.splitlines() and prose in memorized


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


def test_grouped_background_context_keeps_summary_and_actual_tail_under_one_heading():
    cid = "whatsapp:group:fictional-workshop"
    prior = "Prior prose.\n  Preserve this indentation."
    context = [
        {"source_conversation_id": cid, "chat_name": "Workshop", "summary": prior},
        {"source_conversation_id": cid, "chat_name": "Workshop", "summary": "Follow-up prose."},
        {"source_conversation_id": cid, "chat_name": "Workshop", "raw": True,
         "role": "user", "speaker": "TestGuest", "content": "A short actual tail."},
        {"source_conversation_id": "import:dm:fictional", "app_label": "ArchiveApp",
         "chat_name": "Quiet archive", "summary": "An imported summary."},
    ]
    rendered = format_grouped_chat_history([], background_context_rows=context,
                                         current_conversation_id=cid)
    assert rendered.count("[group][Workshop]") == 1
    assert prior in rendered and "Follow-up prose." in rendered
    assert "[TestGuest] A short actual tail." in rendered
    assert "My ArchiveApp Conversations:" in rendered
    assert "[dm][Quiet archive]" in rendered
    assert "(none)" not in rendered and "--- " not in rendered
    assert "raw" not in rendered and "[user] Prior" not in rendered
