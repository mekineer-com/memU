from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def format_conversation_for_preprocess(raw_text: str) -> str:
    """
    Normalize a conversation into a line-based format suitable for LLM preprocessing prompts.

    Supported input formats:
    - A JSON list of messages: [{"role": "...", "content": "...", "created_at": "..."}]
    - A JSON dict with a "content" list: {"content": [ ...messages... ]}

    Output format:
    - One message per line
    - The display speaker is included in square brackets: "[user]" / "[soul]" etc.

    Notes:
    - This function expects conversation data to be JSON.
    - Newlines in message content are collapsed to spaces to keep one message per line.
    """
    stripped = (raw_text or "").strip()
    if not stripped:
        return ""

    parsed = _try_parse_json(stripped)
    if parsed is None:
        # Conversation inputs are expected to be JSON. If invalid, return original text.
        return raw_text
    messages = _extract_messages(parsed)
    if messages is None:
        return raw_text
    return _format_messages(messages)


def _try_parse_json(text: str) -> Any | None:
    if not text:
        return None
    if not (text.startswith("[") or text.startswith("{")):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_messages(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            return [m for m in content if isinstance(m, dict)]
    return None


def _format_messages(messages: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for msg in messages:
        out.append(
            format_speaker_message(
                msg,
                default_role="user",
                collapse_newlines=True,
            )
        )
    return "\n".join(out)


def display_speaker_label(
    message: Mapping[str, Any],
    *,
    soul_name: str | None = None,
    default_role: str = "unknown",
) -> str:
    explicit = str(message.get("name") or message.get("speaker") or "").strip()
    if explicit:
        return explicit
    role_value = str(message.get("role") or default_role).strip().lower() or default_role
    if role_value == "assistant":
        return str(soul_name or "").strip() or "soul"
    return role_value or default_role


def extract_text_content(content: Any, *, collapse_newlines: bool = False) -> str:
    if isinstance(content, dict):
        text = content.get("text", "")
    elif isinstance(content, str):
        text = content
    else:
        text = "" if content is None else str(content)
    raw = str(text)
    if collapse_newlines:
        # Keep preprocessor inputs one-line so message index parsing stays stable.
        raw = " ".join(raw.splitlines())
    return raw.strip()


def format_speaker_message(
    message: Mapping[str, Any],
    *,
    soul_name: str | None = None,
    default_role: str = "unknown",
    separator: str = " ",
    collapse_newlines: bool = False,
) -> str:
    label = display_speaker_label(message, soul_name=soul_name, default_role=default_role)
    content = extract_text_content(message.get("content"), collapse_newlines=collapse_newlines)
    return f"[{label}]{separator}{content}"


def render_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    soul_name: str | None = None,
    default_role: str = "unknown",
    separator: str = " ",
    collapse_newlines: bool = False,
    time_label_resolver: Callable[[Mapping[str, Any]], str | None] | None = None,
    blank_line_before_time_label: bool = False,
) -> str:
    lines: list[str] = []
    last_time_label: str | None = None
    for message in messages:
        if time_label_resolver is not None:
            time_label = time_label_resolver(message)
            if time_label and time_label != last_time_label:
                if blank_line_before_time_label and lines:
                    lines.append("")
                lines.append(f"--- {time_label} ---")
                last_time_label = time_label
        lines.append(
            format_speaker_message(
                message,
                soul_name=soul_name,
                default_role=default_role,
                separator=separator,
                collapse_newlines=collapse_newlines,
            )
        )
    return "\n".join(lines).strip()


def _conversation_id(message: Mapping[str, Any], default_conversation_id: str | None) -> str:
    for key in ("source_conversation_id", "conversation_id"):
        value = str(message.get(key) or "").strip()
        if value:
            return value
    return str(default_conversation_id or "").strip() or "conversation"


def _conversation_kind_and_key(conversation_id: str, source_label: str) -> tuple[str, str]:
    cid = str(conversation_id or "").strip()
    label = str(source_label or "").strip()
    if cid.startswith("whatsapp:group:"):
        return "whatsapp_group", cid[len("whatsapp:group:"):].strip()
    if cid.startswith("whatsapp:dm:"):
        return "whatsapp_dm", cid[len("whatsapp:dm:"):].strip()
    if label == "whatsapp:group":
        return "whatsapp_group", cid
    if label == "whatsapp:dm":
        return "whatsapp_dm", cid
    if cid.startswith("sillytavern:"):
        return "sillytavern_dm", cid[len("sillytavern:"):].strip() or "sillytavern"
    if cid.startswith(("integrity:", "chat:")) or cid == "sillytavern" or label == "sillytavern":
        return "sillytavern_dm", cid or "sillytavern"
    return "sillytavern_dm", cid or "sillytavern"


def _section_title(kind: str) -> str:
    if kind.startswith("whatsapp_"):
        return "My WhatsApp Conversations:"
    return "My SillyTavern Conversations:"


def _chat_heading(kind: str, key: str, chat_name: str | None) -> str:
    pretty = str(chat_name or "").strip() or str(key or "").strip()
    if kind == "whatsapp_group":
        return f"[group][{pretty or 'group'}]"
    return f"[dm][{pretty or 'sillytavern'}]"


def render_grouped_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    soul_name: str | None = None,
    default_conversation_id: str | None = None,
    current_conversation_id: str | None = None,
    current_heading: str | None = None,
    include_sections: bool = True,
    current_marker: str = " \u2190 current chat",
    time_label_resolver: Callable[[Mapping[str, Any]], str | None] | None = None,
    blank_line_before_time_label: bool = False,
    default_role: str = "unknown",
    collapse_newlines: bool = False,
) -> str:
    by_conversation: dict[str, list[Mapping[str, Any]]] = {}
    for message in messages:
        cid = _conversation_id(message, default_conversation_id)
        by_conversation.setdefault(cid, []).append(message)

    current_id = str(current_conversation_id or default_conversation_id or "").strip()
    sections: dict[str, list[str]] = {}
    for cid, rows in by_conversation.items():
        if not rows:
            continue
        first = rows[0]
        source_label = str(first.get("source_label") or "").strip()
        kind, key = _conversation_kind_and_key(cid, source_label)
        section = _section_title(kind)
        chat_name = ""
        for row in reversed(rows):
            candidate = str(row.get("chat_name") or "").strip()
            if candidate:
                chat_name = candidate
                break
        heading = current_heading if current_id and cid == current_id and current_heading else _chat_heading(kind, key, chat_name or None)
        if current_id and cid == current_id and current_marker and not heading.endswith(current_marker):
            heading = f"{heading}{current_marker}"
        body = render_chat_messages(
            rows,
            soul_name=soul_name,
            default_role=default_role,
            collapse_newlines=collapse_newlines,
            time_label_resolver=time_label_resolver,
            blank_line_before_time_label=blank_line_before_time_label,
        )
        block = "\n".join(part for part in (heading, body) if part).strip()
        if block:
            sections.setdefault(section, []).append(block)

    lines: list[str] = []
    for section, blocks in sections.items():
        if lines:
            lines.append("")
        if include_sections:
            lines.extend([section, ""])
        lines.append("\n\n".join(blocks))
    return "\n".join(lines).strip()


def conversation_message_indices(raw_text: str) -> list[int]:
    parsed = _try_parse_json((raw_text or "").strip())
    messages = _extract_messages(parsed)
    if messages is None:
        return []
    return list(range(len(messages)))
