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
    empty_root = re.search(r"<item\s*/>", raw)
    if empty_root is not None:
        return (empty_root.start(), empty_root.start(), empty_root.group())
    return None


def _xml_text(element: Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def _parse_memory_element(memory_elem: Element) -> dict[str, Any] | None:
    memory_dict: dict[str, Any] = {}

    content_elem = memory_elem.find("content")
    if content_elem is not None:
        memory_dict["content"] = _xml_text(content_elem)

    day_elem = memory_elem.find("day")
    if day_elem is not None and _xml_text(day_elem):
        memory_dict["day"] = _xml_text(day_elem)

    categories_elem = memory_elem.find("categories")
    if categories_elem is not None:
        categories = [text for cat_elem in categories_elem.findall("category") if (text := _xml_text(cat_elem))]
        memory_dict["categories"] = categories

    source_role_elem = memory_elem.find("source_role")
    if source_role_elem is not None:
        raw_role = _xml_text(source_role_elem).lower()
        if raw_role in {"soul", "user", "peer", "entity", "environment"}:
            memory_dict["source_role"] = raw_role

    speaker_ref_elem = memory_elem.find("speaker_ref")
    if speaker_ref_elem is not None and _xml_text(speaker_ref_elem):
        memory_dict["speaker_ref"] = _xml_text(speaker_ref_elem)

    confidence_elem = memory_elem.find("confidence")
    if confidence_elem is not None:
        try:
            confidence = float(_xml_text(confidence_elem))
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and 0.0 <= confidence <= 1.0:
            memory_dict["confidence"] = confidence

    salience_elem = memory_elem.find("reflection_salience")
    if salience_elem is not None:
        reflection_salience = _normalize_reflection_salience(_xml_text(salience_elem))
        if reflection_salience is not None:
            memory_dict["reflection_salience"] = reflection_salience

    ei_elem = memory_elem.find("emotional_intensity")
    if ei_elem is not None:
        emotional_intensity = _normalize_reflection_salience(_xml_text(ei_elem))
        if emotional_intensity is not None:
            memory_dict["emotional_intensity"] = emotional_intensity

    replaces_previous_fact_elem = memory_elem.find("replaces_previous_fact")
    if replaces_previous_fact_elem is not None:
        replaces_previous_fact = _normalize_replaces_previous_fact(_xml_text(replaces_previous_fact_elem))
        if replaces_previous_fact:
            memory_dict["replaces_previous_fact"] = replaces_previous_fact

    entities_elem = memory_elem.find("entities")
    if entities_elem is not None:
        entities = []
        for entity_el in entities_elem.findall("entity"):
            name_text = _xml_text(entity_el.find("name"))
            type_text = _xml_text(entity_el.find("type"))
            if name_text and type_text:
                entities.append({"name": name_text, "type": type_text})
        if entities:
            memory_dict["entities"] = entities

    if memory_dict.get("content"):
        return memory_dict
    logger.warning("Skipping extracted memory with missing or blank content")
    return None


def _parse_memory_type_response_xml(raw: str) -> list[dict[str, Any]]:
    """Parse XML extraction reply.  Returns [] for a valid reply with no items.
    Raises ValueError for an unparseable or unusable reply."""
    if not raw or not raw.strip():
        raise ValueError("Extraction reply is blank")
    raw = raw.strip()

    try:
        boundaries = _find_xml_boundaries(raw)
        if boundaries is None:
            raise ValueError("Could not find valid root tag in XML response")

        start_idx, end_idx, end_tag = boundaries
        xml_content = raw[start_idx : end_idx + len(end_tag)]
        content_depth = 0

        def escape_text(match: re.Match[str]) -> str:
            nonlocal content_depth
            token = match.group(0)
            if token.startswith("<content") and not token.endswith("/>"):
                content_depth += 1
            elif token.startswith("</content"):
                content_depth -= 1
            elif token == "&":
                return "&amp;"
            elif token == "<" and content_depth > 0:
                return "&lt;"
            return token

        # Track nested content without treating XML literals/comments as structure.
        xml_content = re.sub(
            r"<!\[CDATA\[.*?\]\]>|<!--.*?-->|<\?.*?\?>|<content(?:\s[^>]*)?>|</content\s*>|"
            r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)|<(?![/!?A-Za-z_:]|[^\x00-\x7f])",
            escape_text,
            xml_content,
            flags=re.DOTALL,
        )

        root = ET.fromstring(xml_content)
        memory_elements = root.findall("memory")

    except ET.ParseError:
        logger.warning("Malformed extraction XML — attempting salvage with synthetic root")
        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")
            memory_elements = root.findall("item/memory")
        except ET.ParseError:
            raise ValueError("Extraction XML salvage failed — reply is unparseable")

    result: list[dict[str, Any]] = []
    for memory_elem in memory_elements:
        parsed = _parse_memory_element(memory_elem)
        if parsed:
            result.append(parsed)
    if memory_elements and not result:
        raise ValueError("Extraction reply contains no usable memory content")
    return result
