from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
import pendulum

logger = logging.getLogger(__name__)


class ExtractionParseError(ValueError):
    """Raised when an extraction response cannot be parsed as valid XML."""


def _normalize_reflection_salience(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= parsed <= 1.0:
        return parsed
    return None


def _normalize_replaces_previous_fact(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _parse_episode_ref(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _dedupe_message_indices(values: Sequence[int | float | str]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            continue
        if candidate in seen or candidate < 0:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _coerce_to_iterable(values: Any) -> Sequence[Any]:
    if isinstance(values, (list, tuple)):
        return values
    return []


def _resolve_source_message_ids(
    values: Any,
    allowed_values: Any = None,
) -> list[int]:
    parsed = _dedupe_message_indices(_coerce_to_iterable(values))
    allowed = _dedupe_message_indices(_coerce_to_iterable(allowed_values))
    if not allowed:
        return parsed
    allowed_set = set(allowed)
    filtered = [candidate for candidate in parsed if candidate in allowed_set]
    return filtered if filtered else allowed


def _extract_message_indices(text: str | None) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        return []
    out: list[int] = []
    for line in text.splitlines():
        match = re.match(r"\[(\d+)\]\s", line)
        if match is None:
            continue
        try:
            out.append(int(match.group(1)))
        except (TypeError, ValueError):
            continue
    return out


def _extract_conversation_messages(raw_text: Any) -> list[tuple[int, dict[str, Any]]]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return []
    messages: list[dict[str, Any]] | None = None
    if isinstance(parsed, list):
        messages = [msg for msg in parsed if isinstance(msg, dict)]
    elif isinstance(parsed, dict) and isinstance(parsed.get("content"), list):
        messages = [msg for msg in parsed.get("content", []) if isinstance(msg, dict)]
    if not messages:
        return []
    return list(enumerate(messages))


def _parse_message_happened_at(raw: Any) -> Any | None:
    if isinstance(raw, (int, float)) and math.isfinite(raw):
        try:
            ts = datetime.fromtimestamp(float(raw) / 1000.0, tz=UTC)
            return pendulum.datetime(
                ts.year,
                ts.month,
                ts.day,
                ts.hour,
                ts.minute,
                ts.second,
                ts.microsecond,
                tz="UTC",
            )
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = pendulum.parse(raw, strict=False)
    except (ValueError, OverflowError):
        return None
    if not isinstance(parsed, pendulum.DateTime):
        return None
    if parsed.tzinfo is None:
        return pendulum.datetime(
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
            parsed.microsecond,
            tz="UTC",
        )
    return parsed


def _extract_message_happened_at_map(raw_text: Any) -> dict[int, Any]:
    messages = _extract_conversation_messages(raw_text)
    out: dict[int, Any] = {}
    for idx, msg in messages:
        happened_at = _parse_message_happened_at(msg.get("ts_ms"))
        if happened_at is None:
            happened_at = _parse_message_happened_at(msg.get("timestamp"))
        if happened_at is None:
            happened_at = _parse_message_happened_at(msg.get("created_at"))
        if happened_at is not None:
            out[idx] = happened_at
    return out


def _find_xml_boundaries(raw: str) -> tuple[int, int, str] | None:
    root_tags = ["item"]
    for tag in root_tags:
        opening = f"<{tag}>"
        closing = f"</{tag}>"
        start_idx = raw.find(opening)
        if start_idx != -1:
            end_idx = raw.rfind(closing)
            if end_idx != -1:
                return (start_idx, end_idx, closing)
    return None


def _parse_memory_element(memory_elem: Element) -> dict[str, Any] | None:
    memory_dict: dict[str, Any] = {}

    content_elem = memory_elem.find("content")
    if content_elem is not None and content_elem.text:
        memory_dict["content"] = content_elem.text.strip()

    categories_elem = memory_elem.find("categories")
    if categories_elem is not None:
        categories = [cat_elem.text.strip() for cat_elem in categories_elem.findall("category") if cat_elem.text]
        memory_dict["categories"] = categories

    source_role_elem = memory_elem.find("source_role")
    if source_role_elem is not None and source_role_elem.text:
        raw_role = source_role_elem.text.strip().lower()
        if raw_role in {"soul", "user", "peer", "entity", "environment"}:
            memory_dict["source_role"] = raw_role

    speaker_ref_elem = memory_elem.find("speaker_ref")
    if speaker_ref_elem is not None and speaker_ref_elem.text:
        memory_dict["speaker_ref"] = speaker_ref_elem.text.strip()

    episode_ref_elem = memory_elem.find("episode_ref")
    if episode_ref_elem is not None and episode_ref_elem.text:
        episode_ref = _parse_episode_ref(episode_ref_elem.text)
        if episode_ref is not None:
            memory_dict["episode_ref"] = episode_ref

    confidence_elem = memory_elem.find("confidence")
    if confidence_elem is not None and confidence_elem.text:
        try:
            confidence = float(confidence_elem.text.strip())
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and 0.0 <= confidence <= 1.0:
            memory_dict["confidence"] = confidence

    salience_elem = memory_elem.find("reflection_salience")
    if salience_elem is not None and salience_elem.text:
        reflection_salience = _normalize_reflection_salience(salience_elem.text)
        if reflection_salience is not None:
            memory_dict["reflection_salience"] = reflection_salience

    ei_elem = memory_elem.find("emotional_intensity")
    if ei_elem is not None and ei_elem.text:
        emotional_intensity = _normalize_reflection_salience(ei_elem.text)
        if emotional_intensity is not None:
            memory_dict["emotional_intensity"] = emotional_intensity

    replaces_previous_fact_elem = memory_elem.find("replaces_previous_fact")
    if replaces_previous_fact_elem is not None and replaces_previous_fact_elem.text:
        replaces_previous_fact = _normalize_replaces_previous_fact(replaces_previous_fact_elem.text)
        if replaces_previous_fact:
            memory_dict["replaces_previous_fact"] = replaces_previous_fact

    entities_elem = memory_elem.find("entities")
    if entities_elem is not None:
        entities = []
        for entity_el in entities_elem.findall("entity"):
            name_text = (entity_el.findtext("name") or "").strip()
            type_text = (entity_el.findtext("type") or "").strip()
            if name_text and type_text:
                entities.append({"name": name_text, "type": type_text})
        if entities:
            memory_dict["entities"] = entities

    if memory_dict.get("content") and memory_dict.get("categories"):
        return memory_dict
    return None


def _parse_memory_type_response_xml(raw: str) -> list[dict[str, Any]]:
    if not raw or not raw.strip():
        return []
    raw = raw.strip()

    try:
        boundaries = _find_xml_boundaries(raw)
        if boundaries is None:
            logger.warning("Could not find valid root tag in XML response")
            return []

        start_idx, end_idx, end_tag = boundaries
        xml_content = raw[start_idx : end_idx + len(end_tag)]
        xml_content = xml_content.replace("&", "&amp;")

        root = ET.fromstring(xml_content)
        result: list[dict[str, Any]] = []

        for memory_elem in root.findall("memory"):
            parsed = _parse_memory_element(memory_elem)
            if parsed:
                result.append(parsed)

    except ET.ParseError as exc:
        logger.exception("Failed to parse XML")
        raise ExtractionParseError("failed to parse extraction XML") from exc
    else:
        return result
