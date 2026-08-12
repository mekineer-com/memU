from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from xml.etree.ElementTree import Element, tostring

from defusedxml import ElementTree

from memu.database.models import MemoryItem
from memu.utils.conversation import format_relative_time_label, parse_happened_at

_HEADING = re.compile(r"(?m)^## [^\r\n]+\r?$")
# Model output is a trust boundary: these accept only exact [M#] tokens, unlike
# dossier.py's broader scanner that also discovers malformed prose references.
_MEMORY_REF = re.compile(r"^\[M([1-9][0-9]*)\]$")
_MEMORY_REFS = re.compile(r"\[M[1-9][0-9]*\]")
_MEMORY_TOKEN = re.compile(r"\[M[^\]\r\n]*\]")


def contains_memory_reference_token(text: str) -> bool:
    return _MEMORY_TOKEN.search(text) is not None


def estimate_prompt_tokens(text: str) -> int:
    return max(int(len(text.split()) / 0.75), (len(text) + 3) // 4)


def strip_memory_citations(text: str) -> str:
    stripped = _MEMORY_REFS.sub("", text)
    stripped = re.sub(r"[ \t]+([.,;:!?])", r"\1", stripped)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def label_sections(prose: str) -> tuple[str, list[tuple[str, str]]] | None:
    matches = list(_HEADING.finditer(prose))
    if not matches or matches[0].start() != 0:
        return None
    sections = [
        (f"S{index + 1}", prose[match.start() : matches[index + 1].start() if index + 1 < len(matches) else None])
        for index, match in enumerate(matches)
    ]
    headings = [_heading(section) for _label, section in sections]
    if len(headings) != len(set(headings)):
        raise ValueError("Dossier prose has duplicate section headings")
    return "\n".join(f"{label}\n{section}" for label, section in sections), sections


def render_memory_record(
    memory_ref: int,
    memory_type: str,
    happened_at: Any,
    summary: str,
    *,
    now: datetime | None = None,
) -> str:
    happened = parse_happened_at(happened_at)
    day = happened.date().isoformat() if happened is not None else "unknown date"
    relative = format_relative_time_label(happened, now=now)
    time_label = f"{day}, {relative}" if relative else day
    return f"[M{memory_ref}] [{memory_type}] ({time_label}) {' '.join(summary.split())}"


def render_memory_records(items: Sequence[MemoryItem], *, now: datetime | None = None) -> str:
    if not items:
        return "(none)"
    return "\n".join(
        render_memory_record(
            item.memory_ref,
            item.memory_type,
            item.happened_at or item.created_at,
            item.summary,
            now=now,
        )
        for item in items
    )


def revision_status_items(bundle: Mapping[str, Any]) -> dict[str, list[MemoryItem]]:
    purged = {item.id: item for item in bundle["cleanup_items"]}
    cited = {item.id: item for item in bundle["cited_items"] if item.id not in purged}
    pending = {
        item.id: item
        for item in bundle["pending_items"]
        if item.id not in purged and item.id not in cited
    }
    search = {
        item.id: item
        for item in bundle["candidate_items"]
        if item.id not in purged and item.id not in cited and item.id not in pending
    }
    return {
        "cited": list(cited.values()),
        "search": list(search.values()),
        "purged": list(purged.values()),
        "pending": list(pending.values()),
    }


def parse_dossier_revision(
    raw: str,
    bundle: Mapping[str, Any],
    *,
    batch: bool = False,
    normalize_blank: bool = False,
) -> dict[str, Any]:
    text = str(raw or "").strip()
    if (
        not text.startswith("<dossier_revision")
        or not text.endswith("</dossier_revision>")
        or "<!--" in text
        or "<?" in text
    ):
        raise ValueError("Expected exact dossier_revision XML")
    try:
        root = ElementTree.fromstring(text)
    except Exception as exc:
        raise ValueError("Invalid dossier revision XML") from exc
    if root.tag != "dossier_revision" or set(root.attrib) != {"dossier_id"}:
        raise ValueError("Expected exact dossier_revision root")

    dossier = bundle["dossier"]
    if root.attrib["dossier_id"] != dossier.id:
        raise ValueError("Dossier revision id does not match request")

    children = _singletons(
        root,
        {"description", "prose_action", "decisions"},
        optional={"prose_patches"} if batch else {"prose", "prose_patches"},
    )
    children.setdefault("prose_patches", Element("prose_patches"))
    description = _parse_description(children["description"])
    action, resulting_prose = parse_section_revision(
        children,
        str(dossier.summary or ""),
        allow_replace=not batch,
        normalize_blank=normalize_blank,
    )

    statuses = revision_status_items(bundle)
    items = {
        item.id: item
        for status_items in statuses.values()
        for item in status_items
    }
    by_ref = {f"[M{item.memory_ref}]": item for item in items.values()}
    decisions = _parse_decisions(children["decisions"], by_ref, statuses)
    purged_ids = {item.id for item in statuses["purged"]}
    cleanup_ids = set(bundle["linked_inactive_item_ids"])
    remove_ids = {by_ref[ref].id for ref, decision in decisions.items() if decision == "remove"}
    add_ids = {by_ref[ref].id for ref, decision in decisions.items() if decision == "add"}

    tokens = set(_MEMORY_TOKEN.findall(resulting_prose))
    invalid_tokens = {token for token in tokens if _MEMORY_REF.fullmatch(token) is None}
    if invalid_tokens:
        raise ValueError(f"Invalid memory citations: {sorted(invalid_tokens)}")
    resulting_refs = set(_MEMORY_REFS.findall(resulting_prose))
    unknown_refs = resulting_refs - by_ref.keys()
    if unknown_refs:
        raise ValueError(f"Resulting prose cites memories outside review context: {sorted(unknown_refs)}")
    inactive_refs = {
        ref for ref in resulting_refs if by_ref[ref].id in purged_ids
    }
    if inactive_refs:
        raise ValueError(f"Resulting prose cites purged memories: {sorted(inactive_refs)}")

    linked_ids = set(bundle["linked_item_ids"])
    cited_unlinked_ids = set(bundle["cited_unlinked_item_ids"])
    repaired_ids = {
        by_ref[ref].id
        for ref in resulting_refs
        if by_ref[ref].id in cited_unlinked_ids
    }
    resulting_members = (linked_ids | add_ids | repaired_ids) - cleanup_ids - remove_ids
    cited_ids = {by_ref[ref].id for ref in resulting_refs}
    if not cited_ids <= resulting_members:
        raise ValueError("Resulting prose cites a removed or unlinked memory")
    if resulting_members and not resulting_prose.strip():
        raise ValueError("A dossier with members requires nonblank prose")

    return {
        "dossier_id": dossier.id,
        "description": description,
        "prose_action": action,
        "resulting_prose": resulting_prose,
        "add_item_ids": sorted(add_ids | repaired_ids),
        "remove_item_ids": sorted(remove_ids),
        "cleanup_item_ids": sorted(cleanup_ids),
        "cited_item_ids": sorted(cited_ids),
    }


def parse_dossier_revision_batch(
    raw: str,
    bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    text = str(raw or "").strip()
    if (
        not text.startswith("<dossier_revisions>")
        or not text.endswith("</dossier_revisions>")
        or "<!--" in text
        or "<?" in text
    ):
        raise ValueError("Expected exact dossier_revisions XML")
    try:
        root = ElementTree.fromstring(text)
    except Exception as exc:
        raise ValueError("Invalid dossier revisions XML") from exc
    if root.tag != "dossier_revisions" or root.attrib or (root.text or "").strip():
        raise ValueError("Expected exact dossier_revisions root")

    by_id: dict[str, Mapping[str, Any]] = {}
    for bundle in bundles:
        dossier_id = str(bundle["dossier"].id)
        if dossier_id in by_id:
            raise ValueError(f"Duplicate requested dossier id: {dossier_id}")
        by_id[dossier_id] = bundle

    parsed: dict[str, dict[str, Any]] = {}
    for child in root:
        if child.tag != "dossier_revision" or (child.tail or "").strip():
            raise ValueError("Invalid dossier_revisions child")
        dossier_id = child.attrib.get("dossier_id")
        if dossier_id not in by_id:
            raise ValueError(f"Unexpected dossier revision id: {dossier_id}")
        if dossier_id in parsed:
            raise ValueError(f"Duplicate dossier revision id: {dossier_id}")
        child.tail = None
        parsed[dossier_id] = parse_dossier_revision(
            tostring(child, encoding="unicode"),
            by_id[dossier_id],
            batch=True,
            normalize_blank=True,
        )

    missing = by_id.keys() - parsed.keys()
    if missing:
        raise ValueError(f"Missing dossier revision ids: {sorted(missing)}")
    return [parsed[dossier_id] for dossier_id in by_id]


def parse_anchor_revisions(
    container: Element,
    bundles: Mapping[str, Mapping[str, Any]],
    *,
    first_time: bool,
) -> dict[str, dict[str, Any]]:
    if container.tag != "anchor_revisions" or container.attrib or (container.text or "").strip():
        raise ValueError("Expected exact anchor_revisions element")
    if set(bundles) != {"soul", "user"}:
        raise ValueError("Anchor revision requires soul and user bundles")

    parsed: dict[str, dict[str, Any]] = {}
    for child in container:
        if (
            child.tag != "anchor"
            or set(child.attrib) != {"role"}
            or (child.tail or "").strip()
        ):
            raise ValueError("Invalid anchor revision child")
        role = child.attrib["role"]
        if role not in bundles or role in parsed:
            raise ValueError(f"Invalid or duplicate anchor role: {role}")
        children = _singletons(
            child,
            {"description", "prose_action", "prose_patches"},
        )
        description = _parse_description(children["description"])
        action, resulting_prose = parse_section_revision(
            children,
            str(bundles[role]["dossier"].summary or ""),
            require_patch=first_time,
            normalize_blank=True,
        )

        bundle = bundles[role]
        evidence_items = {
            item.id: item
            for item in (*bundle["cited_items"], *bundle["candidate_items"])
        }
        by_ref = {f"[M{item.memory_ref}]": item for item in evidence_items.values()}
        tokens = set(_MEMORY_TOKEN.findall(resulting_prose))
        invalid_tokens = {token for token in tokens if _MEMORY_REF.fullmatch(token) is None}
        if invalid_tokens:
            raise ValueError(f"Invalid memory citations: {sorted(invalid_tokens)}")
        resulting_refs = set(_MEMORY_REFS.findall(resulting_prose))
        unknown_refs = resulting_refs - by_ref.keys()
        if unknown_refs:
            raise ValueError(
                f"Anchor prose cites memories outside reflection evidence: {sorted(unknown_refs)}"
            )
        cited_ids = {by_ref[ref].id for ref in resulting_refs}
        linked_ids = set(bundle["linked_item_ids"])
        cleanup_ids = set(bundle["linked_inactive_item_ids"])
        add_ids = cited_ids - linked_ids
        if not add_ids <= set(bundle["actionable_item_ids"]):
            raise ValueError("Anchor adds a memory outside actionable reflection evidence")
        if cited_ids & cleanup_ids:
            raise ValueError("Anchor prose cites an inactive memory")

        parsed[role] = {
            "anchor_role": role,
            "dossier_id": bundle["dossier"].id,
            "description": description,
            "prose_action": action,
            "resulting_prose": resulting_prose,
            "add_item_ids": sorted(add_ids),
            "remove_item_ids": [],
            "cleanup_item_ids": sorted(cleanup_ids),
            "cited_item_ids": sorted(cited_ids),
        }

    missing = bundles.keys() - parsed.keys()
    if missing:
        raise ValueError(f"Missing anchor revision roles: {sorted(missing)}")
    return parsed


def parse_section_revision(
    children: Mapping[str, Element],
    current_prose: str,
    *,
    allow_replace: bool = True,
    require_patch: bool = False,
    normalize_blank: bool = False,
) -> tuple[str, str]:
    action = _leaf_text(children["prose_action"]).strip()
    prose_element = children.get("prose")
    prose = _leaf_text(prose_element).strip() if prose_element is not None else ""
    if action not in {"keep", "patch", "replace"}:
        raise ValueError(f"Invalid dossier prose action: {action}")
    if not allow_replace and action == "replace":
        raise ValueError("Batch dossier revision does not allow replace")
    if require_patch and action != "patch":
        raise ValueError("First reflection requires an anchor patch")

    normalized_prose = current_prose or ("## unlabeled" if normalize_blank else "")
    inventory = label_sections(normalized_prose)
    patches = _parse_patches(children["prose_patches"])
    if action == "keep":
        if prose or patches:
            raise ValueError("Keep requires empty prose and no patches")
        return action, current_prose
    if action == "replace":
        if (
            (inventory is not None and normalized_prose.strip() != "## unlabeled")
            or not prose
            or patches
            or label_sections(prose) is None
        ):
            raise ValueError(
                "Replace requires unstructured current prose, structured full prose, and no patches"
            )
        return action, prose
    if inventory is None or prose or not patches:
        raise ValueError("Patch requires a section inventory, no prose, and at least one patch")
    return action, _apply_patches(inventory[1], patches)


def _parse_description(element: Element) -> str:
    description = _leaf_text(element).strip()
    if not description:
        raise ValueError("Dossier revision description is required")
    if contains_memory_reference_token(description):
        raise ValueError("Dossier revision description cannot contain memory citations")
    return description


def _singletons(
    root: Element,
    required: set[str],
    *,
    optional: set[str] | None = None,
) -> dict[str, Element]:
    allowed = required | (optional or set())
    if (root.text or "").strip():
        raise ValueError("Unexpected text inside dossier revision root")
    children: dict[str, Element] = {}
    for child in root:
        if child.tag not in allowed:
            raise ValueError(f"Unknown dossier revision element: {child.tag}")
        if child.tag in children:
            raise ValueError(f"Duplicate dossier revision element: {child.tag}")
        if (child.tail or "").strip():
            raise ValueError("Unexpected text inside dossier revision root")
        children[child.tag] = child
    missing = required - children.keys()
    if missing:
        raise ValueError(f"Missing dossier revision elements: {sorted(missing)}")
    return children


def _leaf_text(element: Element) -> str:
    if element.attrib or list(element):
        raise ValueError(f"Expected plain text in {element.tag}")
    return element.text or ""


def _parse_patches(container: Element) -> list[tuple[str, str, str]]:
    if container.attrib or (container.text or "").strip():
        raise ValueError("Invalid prose_patches wrapper")
    patches: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for section in container:
        if section.tag != "section" or set(section.attrib) != {"ref", "action"}:
            raise ValueError("Invalid prose patch section")
        if (section.tail or "").strip():
            raise ValueError("Unexpected text in prose_patches wrapper")
        ref = section.attrib["ref"]
        action = section.attrib["action"]
        if ref in seen:
            raise ValueError(f"Duplicate prose patch reference: {ref}")
        seen.add(ref)
        if action not in {"replace", "add_after", "remove"}:
            raise ValueError(f"Invalid prose patch action: {action}")
        if len(section) != 1 or section[0].tag != "body" or section[0].attrib:
            raise ValueError("Prose patch requires one body")
        if (section.text or "").strip() or (section[0].tail or "").strip():
            raise ValueError("Unexpected text in prose patch section")
        body = _leaf_text(section[0]).strip()
        if action == "remove":
            if body:
                raise ValueError("Remove patch body must be empty")
        elif not _is_single_section(body):
            raise ValueError("Replace and add_after bodies require one complete ## section")
        patches.append((ref, action, body))
    return patches


def _parse_decisions(
    container: Element,
    by_ref: Mapping[str, MemoryItem],
    statuses: Mapping[str, Sequence[MemoryItem]],
) -> dict[str, str]:
    if container.attrib or (container.text or "").strip():
        raise ValueError("Invalid decisions wrapper")
    status_by_ref = {
        f"[M{item.memory_ref}]": status
        for status, items in statuses.items()
        for item in items
    }
    decisions: dict[str, str] = {}
    for element in container:
        if (
            element.tag != "decision"
            or set(element.attrib) != {"ref", "action"}
            or list(element)
            or (element.text or "").strip()
            or (element.tail or "").strip()
        ):
            raise ValueError("Invalid memory decision")
        ref = element.attrib["ref"]
        if _MEMORY_REF.fullmatch(ref) is None or ref not in by_ref:
            raise ValueError(f"Unknown memory decision reference: {ref}")
        if ref in decisions:
            raise ValueError(f"Duplicate memory decision: {ref}")
        action = element.attrib["action"]
        status = status_by_ref[ref]
        if (
            action not in {"add", "remove"}
            or status == "purged"
            or (status == "cited" and action != "remove")
            or (status == "search" and action != "add")
        ):
            raise ValueError(f"Invalid {action} decision for {status} memory {ref}")
        decisions[ref] = action

    pending_refs = {
        f"[M{item.memory_ref}]" for item in statuses["pending"]
    }
    if pending_refs - decisions.keys():
        missing = pending_refs - decisions.keys()
        raise ValueError(f"Pending memories require decisions: {sorted(missing)}")
    return decisions


def _is_single_section(body: str) -> bool:
    matches = list(_HEADING.finditer(body))
    return len(matches) == 1 and matches[0].start() == 0


def _heading(section: str) -> str:
    return section.splitlines()[0].removeprefix("## ").strip().casefold()


def _apply_patches(
    sections: Sequence[tuple[str, str]], patches: Sequence[tuple[str, str, str]]
) -> str:
    section_by_ref = dict(sections)
    unknown = {ref for ref, _action, _body in patches} - section_by_ref.keys()
    if unknown:
        raise ValueError(f"Unknown prose section references: {sorted(unknown)}")
    operations = {ref: (action, body) for ref, action, body in patches}
    output: list[str] = []
    for ref, original in sections:
        operation = operations.get(ref)
        if operation is None:
            output.append(original)
            continue
        action, body = operation
        if action != "remove":
            output.append(body if action == "replace" else original)
        if action == "add_after":
            output.append(body)
    if not output:
        return ""
    headings = [_heading(section) for section in output]
    if len(headings) != len(set(headings)):
        raise ValueError("Resulting dossier has duplicate section headings")
    result = ""
    for section in output:
        if result and not result.endswith(("\n", "\r")):
            result += "\n\n"
        result += section
    return result
