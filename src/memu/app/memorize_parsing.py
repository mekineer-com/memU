from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET

from memu.app.memorize_segments import grouped_chat_happened_at

logger = logging.getLogger(__name__)


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
    allowed = _dedupe_message_indices(_coerce_to_iterable(allowed_values))
    if allowed:
        return allowed
    # Extraction no longer asks the model to cite individual messages. If the
    # caller has no episode/segment provenance, do not trust stale emitted IDs.
    return []


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


def _extract_message_happened_at_map(raw_text: Any) -> dict[int, Any]:
    messages = _extract_conversation_messages(raw_text)
    out: dict[int, Any] = {}
    for idx, msg in messages:
        happened_at = grouped_chat_happened_at(msg)
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

    day_elem = memory_elem.find("day")
    if day_elem is not None and day_elem.text:
        memory_dict["day"] = day_elem.text.strip()

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
    """Parse XML extraction reply.  Returns [] for a valid reply with no items.
    Raises ValueError for an unparseable reply (caller should retry once)."""
    if not raw or not raw.strip():
        return []
    raw = raw.strip()

    try:
        boundaries = _find_xml_boundaries(raw)
        if boundaries is None:
            raise ValueError("Could not find valid root tag in XML response")

        start_idx, end_idx, end_tag = boundaries
        xml_content = raw[start_idx : end_idx + len(end_tag)]
        xml_content = xml_content.replace("&", "&amp;")

        root = ET.fromstring(xml_content)
        result: list[dict[str, Any]] = []

        for memory_elem in root.findall("memory"):
            parsed = _parse_memory_element(memory_elem)
            if parsed:
                result.append(parsed)

    except ET.ParseError:
        logger.warning("Malformed extraction XML — attempting salvage with synthetic root")
        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")
            result = []
            for memory_elem in root.iter("memory"):
                parsed = _parse_memory_element(memory_elem)
                if parsed:
                    result.append(parsed)
            if result:
                logger.info("Salvaged %d memories from malformed XML", len(result))
            return result
        except ET.ParseError:
            raise ValueError("Extraction XML salvage failed — reply is unparseable")
    else:
        return result
