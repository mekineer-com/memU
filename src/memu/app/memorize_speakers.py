from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from memu.app import memorize_parsing as parsing

TEntry = TypeVar("TEntry", bound="SpeakerLike")


class SpeakerLike(Protocol):
    speaker_id: str
    speaker_label: str
    coarse_role: str


def _normalize_speaker_slug(prefix: str, raw: Any) -> str:
    name = str(raw or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if not slug:
        slug = prefix
    return f"{prefix}:{slug}"


def _speaker_role_from_id(speaker_id: str | None) -> str | None:
    if not isinstance(speaker_id, str):
        return None
    value = speaker_id.strip().lower()
    if not value or ":" not in value:
        return None
    role, _sep, _rest = value.partition(":")
    return role or None


def _normalize_coarse_role(role: str | None) -> str:
    value = str(role or "").strip().lower()
    if value in {"user", "soul", "peer", "entity", "environment"}:
        return value
    return "environment"


def _build_speaker_roster(
    speaker_map: Mapping[int, tuple[str, str]] | None,
    *,
    roster_entry_factory: Callable[[str, str, str], TEntry],
) -> list[TEntry]:
    if not speaker_map:
        return []
    roster: list[TEntry] = []
    seen_ids: set[str] = set()
    for message_index in sorted(speaker_map):
        speaker_id, speaker_label = speaker_map[message_index]
        normalized_id = str(speaker_id or "").strip()
        if not normalized_id or normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        coarse_role = _normalize_coarse_role(_speaker_role_from_id(normalized_id))
        roster.append(roster_entry_factory(normalized_id, str(speaker_label or "").strip() or normalized_id, coarse_role))
    return roster


def _has_ambiguous_speaker_role(roster: Sequence[SpeakerLike]) -> bool:
    role_counts: dict[str, int] = {}
    for entry in roster:
        if entry.coarse_role == "environment":
            continue
        role_counts[entry.coarse_role] = role_counts.get(entry.coarse_role, 0) + 1
    return any(count > 1 for count in role_counts.values())


def _build_speaker_roster_if_ambiguous(
    speaker_map: Mapping[int, tuple[str, str]] | None,
    *,
    roster_entry_factory: Callable[[str, str, str], TEntry],
) -> list[TEntry] | None:
    roster = _build_speaker_roster(speaker_map, roster_entry_factory=roster_entry_factory)
    if not roster or not _has_ambiguous_speaker_role(roster):
        return None
    return roster


def _is_user_declared_relationship_entity(entity: Any) -> bool:
    props = getattr(entity, "properties", None)
    if not isinstance(props, Mapping):
        return False
    origin = str(props.get("origin") or "").strip()
    if origin != "user_declared":
        return False
    return props.get("active") is not False


def _list_declared_relationship_roster(
    *,
    store: Any,
    user: Mapping[str, Any] | None,
    roster_entry_factory: Callable[[str, str, str], TEntry],
) -> list[TEntry]:
    where = dict(user or {}) if isinstance(user, Mapping) else {}
    entities = store.entity_repo.list_all(where=where)
    roster: list[TEntry] = []
    seen_ids: set[str] = set()
    for entity in entities:
        if not _is_user_declared_relationship_entity(entity):
            continue
        normalized = str(getattr(entity, "normalized", "") or "").strip().lower()
        if not normalized:
            continue
        speaker_id = f"entity:{normalized}"
        if speaker_id in seen_ids:
            continue
        seen_ids.add(speaker_id)
        label = str(getattr(entity, "name", "") or "").strip() or normalized
        roster.append(roster_entry_factory(speaker_id, label, "entity"))
    return roster


def _episode_mentions_roster_entry(episode_text: Any, entry: SpeakerLike) -> bool:
    text = str(episode_text or "").strip().lower()
    if not text:
        return False
    label = str(entry.speaker_label or "").strip().lower()
    if label and label in text:
        return True
    speaker_tail = entry.speaker_id.split(":", 1)[1] if ":" in entry.speaker_id else entry.speaker_id
    speaker_tail = speaker_tail.replace("_", " ").strip().lower()
    if speaker_tail and speaker_tail in text:
        return True
    return False


def _build_speaker_roster_for_segment(
    *,
    speaker_map: Mapping[int, tuple[str, str]] | None,
    declared_entities: Sequence[TEntry] | None,
    segment_text: Any,
    roster_entry_factory: Callable[[str, str, str], TEntry],
) -> list[TEntry] | None:
    map_roster = _build_speaker_roster(speaker_map, roster_entry_factory=roster_entry_factory)
    map_has_ambiguity = _has_ambiguous_speaker_role(map_roster)
    mentioned_declared = [
        entry
        for entry in (declared_entities or [])
        if _episode_mentions_roster_entry(segment_text, entry)
    ]
    if not map_has_ambiguity and not mentioned_declared:
        return None

    merged: list[TEntry] = []
    seen_ids: set[str] = set()
    for entry in [*map_roster, *mentioned_declared]:
        if entry.speaker_id in seen_ids:
            continue
        seen_ids.add(entry.speaker_id)
        merged.append(entry)
    return merged or None


def _sanitize_prompt_label(label: str) -> str:
    cleaned = re.sub(r"[\n\r\t<>`]", " ", str(label or ""))
    return re.sub(r" +", " ", cleaned).strip()


def _format_speaker_roster_block_for_prompt(
    speaker_roster: Sequence[SpeakerLike] | None,
) -> str:
    if not speaker_roster:
        return ""
    lines = [
        "# Speaker Roster (ambiguous episode fallback)",
        "Allowed source_role schema for this episode: <source_role>user|soul|peer|entity|environment</source_role>.",
        "Only emit <speaker_ref> if the speaker is in this roster. Never invent a slug.",
        "Use source_role for coarse role; use speaker_ref only to disambiguate when multiple speakers share that role.",
    ]
    for entry in speaker_roster:
        safe_label = _sanitize_prompt_label(entry.speaker_label)
        lines.append(f"- {entry.speaker_id} | label={safe_label} | role={entry.coarse_role}")
    lines.append("When needed, add <speaker_ref>speaker_id_from_roster</speaker_ref> inside <memory>.")
    return "\n".join(lines)


def _parse_speaker_ref(
    raw: Any,
    roster: Sequence[SpeakerLike] | None,
) -> tuple[str | None, str | None]:
    if not isinstance(raw, str) or not roster:
        return None, None
    candidate = raw.strip()
    if not candidate:
        return None, None
    normalized = candidate.casefold()
    for entry in roster:
        if entry.speaker_id.casefold() == normalized:
            return entry.speaker_id, entry.speaker_label
    return None, None


def _build_speaker_map(
    episode_messages: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any] | None,
) -> dict[int, tuple[str, str]]:
    user_scope = dict(scope or {}) if isinstance(scope, Mapping) else {}
    user_name = str(user_scope.get("user_id") or "").strip()
    soul_name = str(user_scope.get("soul_id") or "").strip()
    user_label_default = user_name or "user"
    soul_label_default = soul_name or "soul"
    user_id_default = _normalize_speaker_slug("user", user_name or "user")

    speaker_map: dict[int, tuple[str, str]] = {}
    for idx, message in enumerate(episode_messages):
        raw_index = message.get("_message_index", idx)
        try:
            message_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if message_index < 0:
            continue

        role = str(message.get("role") or "").strip().lower()
        name = str(message.get("name") or "").strip()
        normalized_name = name or None

        speaker_id: str
        speaker_label: str
        if role in {"assistant", "soul"}:
            speaker_label = normalized_name or soul_label_default
            speaker_id = _normalize_speaker_slug("soul", soul_name or speaker_label)
        elif role in {"user", "human", "participant"}:
            speaker_label = normalized_name or user_label_default
            speaker_id = user_id_default
        elif normalized_name:
            speaker_label = normalized_name
            speaker_id = _normalize_speaker_slug("entity", normalized_name)
        else:
            speaker_label = role or "environment"
            speaker_id = _normalize_speaker_slug("environment", speaker_label)

        speaker_map[message_index] = (speaker_id, speaker_label)
    return speaker_map


def _attribute_memory(
    memory: Any,
    speaker_map: Mapping[int, tuple[str, str]] | None,
) -> Any:
    if memory.speaker_id and memory.speaker_label:
        return memory
    if not speaker_map:
        return memory
    candidates: dict[str, tuple[str, str]] = {}
    for message_idx in memory.source_message_ids or []:
        candidate = speaker_map.get(int(message_idx))
        if candidate is None:
            continue
        candidates[candidate[0]] = candidate
    if not candidates:
        return memory
    if len(candidates) == 1:
        speaker_id, speaker_label = next(iter(candidates.values()))
        return memory._replace(speaker_id=speaker_id, speaker_label=speaker_label)

    role = str(memory.source_role or "").strip().lower()
    if role:
        role_candidates = [
            candidate
            for candidate in candidates.values()
            if _speaker_role_from_id(candidate[0]) == role
        ]
        unique_role_candidates = {candidate[0]: candidate for candidate in role_candidates}
        if len(unique_role_candidates) == 1:
            speaker_id, speaker_label = next(iter(unique_role_candidates.values()))
            return memory._replace(speaker_id=speaker_id, speaker_label=speaker_label)
    return memory._replace(speaker_id=None, speaker_label=None)


def _decorate_entries_with_plan_context(
    entries: list[Any],
    *,
    message_indices: list[int],
) -> list[Any]:
    if not entries:
        return entries
    decorated: list[Any] = []
    default_ids = parsing._dedupe_message_indices(message_indices)
    for entry in entries:
        # Use the full episode/segment provenance set. Memory extraction is
        # generalized and must not be narrowed to model-emitted message IDs.
        resolved_ids = parsing._resolve_source_message_ids(entry.source_message_ids, default_ids)
        resolved_salience = entry.reflection_salience
        decorated.append(entry._replace(source_message_ids=resolved_ids, reflection_salience=resolved_salience))
    return decorated
