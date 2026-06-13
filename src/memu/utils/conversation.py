from __future__ import annotations

import json
from collections.abc import Mapping
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


def conversation_message_indices(raw_text: str) -> list[int]:
    parsed = _try_parse_json((raw_text or "").strip())
    messages = _extract_messages(parsed)
    if messages is None:
        return []
    return list(range(len(messages)))
