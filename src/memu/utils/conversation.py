from __future__ import annotations

import json
import re
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
    return format_chat_messages(
        messages,
        default_role="user",
        collapse_newlines=True,
        strip_output=False,
    )


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


_SHARED_GROUP_PREFIX_RE = re.compile(r"^\[([^\]]+)\]\s+(.+)$")
_NUMERIC_LIKE_RE = re.compile(r"^[0-9+\-() .]+$")


def normalize_whatsapp_identifier(value: str) -> str:
    normalized = (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )
    if not normalized or "/" in normalized or "\\" in normalized:
        return ""
    if normalized in {".", ".."}:
        return ""
    return normalized


def _parse_shared_group_sender_prefix(content: str) -> tuple[str, str] | None:
    match = _SHARED_GROUP_PREFIX_RE.match(content)
    if not match:
        return None
    sender = str(match.group(1) or "").strip()
    message = str(match.group(2) or "").strip()
    if not sender or not message:
        return None
    return sender, message


def _conversation_kind_and_key(conversation_id: str) -> tuple[str, str]:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:group:"):
        return ("whatsapp_group", cid[len("whatsapp:group:"):].strip())
    if cid.startswith("whatsapp:dm:"):
        return ("whatsapp_dm", cid[len("whatsapp:dm:"):].strip())
    if cid.startswith("sillytavern:"):
        return ("sillytavern_dm", cid[len("sillytavern:"):].strip() or "sillytavern")
    if cid.startswith("integrity:"):
        return ("sillytavern_dm", cid)
    if cid.startswith("chat:"):
        return ("sillytavern_dm", cid)
    if cid == "sillytavern":
        return ("sillytavern_dm", "sillytavern")
    return ("sillytavern_dm", cid or "sillytavern")


def _lookup_whatsapp_name(key: str, names: Mapping[str, str]) -> str:
    key_norm = normalize_whatsapp_identifier(key)
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(candidate_key: str) -> None:
        value = str(names.get(candidate_key) or "").strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    if key:
        _push(key)
    if key_norm:
        _push(key_norm)
        _push(f"{key_norm}@s.whatsapp.net")
        _push(f"{key_norm}@lid")

    if not candidates:
        return ""

    def _score(name: str) -> tuple[int, int, int]:
        normalized_name = normalize_whatsapp_identifier(name)
        same_as_key = int(bool(key_norm) and normalized_name == key_norm)
        numeric_like = int(bool(_NUMERIC_LIKE_RE.fullmatch(name)))
        return (same_as_key, numeric_like, len(name))

    return min(candidates, key=_score)


def _conversation_heading(
    kind: str,
    key: str,
    names: Mapping[str, str] | None = None,
    chat_name: str | None = None,
) -> str:
    names = names or {}
    if kind == "whatsapp_group":
        pretty = _lookup_whatsapp_name(key, names) or str(chat_name or "").strip() or key or "group"
        return f"[group][{pretty}]"
    if kind == "whatsapp_dm":
        pretty = _lookup_whatsapp_name(key, names) or str(chat_name or "").strip() or key or "contact"
        return f"[dm][{pretty}]"
    if kind == "sillytavern_dm":
        pretty = (chat_name or "").strip() or key or "sillytavern"
        return f"[dm][{pretty}]"
    return f"[dm][{key or 'sillytavern'}]"


def _conversation_section_title(kind: str) -> str:
    if kind.startswith("sillytavern_"):
        return "My SillyTavern Conversations:"
    if kind.startswith("whatsapp_"):
        return "My WhatsApp Conversations:"
    return "My SillyTavern Conversations:"


def format_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    soul_name: str | None = None,
    default_role: str = "unknown",
    separator: str = " ",
    collapse_newlines: bool = False,
    time_label_resolver: Callable[[Mapping[str, Any]], str | None] | None = None,
    blank_line_before_time_label: bool = False,
    strip_output: bool = True,
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
    rendered = "\n".join(lines)
    return rendered.strip() if strip_output else rendered


def format_grouped_chat_history(
    messages: Sequence[Mapping[str, Any]],
    *,
    whatsapp_names: Mapping[str, str] | None = None,
    time_label_resolver: Callable[[Any], str | None] | None = None,
    speaker_separator: str = ": ",
) -> str:
    by_conversation: dict[str, list[Mapping[str, Any]]] = {}
    for msg in messages:
        cid = str(msg.get("conversation_id") or "").strip() or "unknown"
        by_conversation.setdefault(cid, []).append(msg)

    sections: dict[str, list[tuple[str, str]]] = {}
    for cid, rows in by_conversation.items():
        kind, key = _conversation_kind_and_key(cid)
        section_key = _conversation_section_title(kind)
        entries = sections.setdefault(section_key, [])
        chat_name = ""
        for msg in reversed(rows):
            candidate = str(msg.get("chat_name") or "").strip()
            if candidate:
                chat_name = candidate
                break
        conv_lines: list[str] = [
            _conversation_heading(kind, key, whatsapp_names, chat_name or None)
        ]
        rendered_rows: list[dict[str, Any]] = []
        newest_ts = ""
        for msg in rows:
            ts = str(msg.get("received_at") or "")
            if ts > newest_ts:
                newest_ts = ts
            role = str(msg.get("role") or "").strip()
            speaker = str(msg.get("speaker") or "").strip()
            content = str(msg.get("content") or "")
            if role == "user" and kind == "whatsapp_group":
                parsed = _parse_shared_group_sender_prefix(content)
                if parsed is not None:
                    speaker, content = parsed
            rendered_rows.append(
                {
                    "role": role,
                    "speaker": speaker,
                    "content": content,
                    "received_at": msg.get("received_at"),
                }
            )
        rendered = format_chat_messages(
            rendered_rows,
            default_role="unknown",
            separator=speaker_separator,
            time_label_resolver=(
                (lambda item: time_label_resolver(item.get("received_at")))
                if time_label_resolver is not None
                else None
            ),
        )
        if rendered:
            conv_lines.append(rendered)
        entries.append((newest_ts, "\n".join(conv_lines)))

    lines: list[str] = []
    for section_title, entries in sections.items():
        if not entries:
            continue
        entries.sort(key=lambda e: e[0])
        blocks = [block for _, block in entries]
        if lines:
            lines.append("")
        lines.append(section_title)
        lines.append("")
        lines.append("\n\n".join(blocks))
    return "\n".join(lines).strip()


def conversation_message_indices(raw_text: str) -> list[int]:
    parsed = _try_parse_json((raw_text or "").strip())
    messages = _extract_messages(parsed)
    if messages is None:
        return []
    return list(range(len(messages)))
