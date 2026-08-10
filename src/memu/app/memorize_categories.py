from __future__ import annotations

import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, cast
from xml.etree.ElementTree import Element

from defusedxml import ElementTree

from memu.app.dossier_revision import strip_memory_citations
from memu.database.models import MemoryCategory, MemoryItem
from memu.database.vector import cosine_similarity, cosine_topk
from memu.prompts.dynamic_dossier_review import (
    SYSTEM_PROMPT as DYNAMIC_DOSSIER_REVIEW_SYSTEM_PROMPT,
)
from memu.prompts.dynamic_dossier_review import (
    USER_PROMPT as DYNAMIC_DOSSIER_REVIEW_USER_PROMPT,
)
from memu.utils.taxonomy import (
    DOSSIER_KINDS,
    category_identity_text,
    dossier_scope,
    embedding_vector,
    normalize_category_name,
)

logger = logging.getLogger(__name__)


def _partition_category_names(
    names: Sequence[str],
    known_names: set[str],
    normalize_name: Callable[[str], str | None],
    *,
    dedupe_unknown: bool = False,
) -> tuple[list[str], list[tuple[str, str]]]:
    known: list[str] = []
    unknown: list[tuple[str, str]] = []
    seen_known: set[str] = set()
    seen_unknown: set[str] = set()
    for raw in names:
        normalized = normalize_name(raw)
        if not normalized:
            continue
        if normalized in known_names:
            if normalized not in seen_known:
                known.append(normalized)
                seen_known.add(normalized)
        else:
            if dedupe_unknown and normalized in seen_unknown:
                continue
            unknown.append((raw.strip(), normalized))
            seen_unknown.add(normalized)
    return known, unknown


def _connected_components(
    candidate_indexes: Sequence[int],
    embeddings: Sequence[Sequence[float] | None],
    *,
    cosine_similarity: Callable[[Sequence[float], Sequence[float]], float],
    threshold: float,
) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in candidate_indexes}
    for pos, left_idx in enumerate(candidate_indexes):
        left = embeddings[left_idx]
        if left is None:
            continue
        for right_idx in candidate_indexes[pos + 1 :]:
            right = embeddings[right_idx]
            if right is not None and cosine_similarity(left, right) >= threshold:
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)

    components: list[list[int]] = []
    visited: set[int] = set()
    for idx in candidate_indexes:
        if idx in visited:
            continue
        stack = [idx]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(
                neighbor
                for neighbor in sorted(adjacency[current], reverse=True)
                if neighbor not in visited
            )
        components.append(sorted(component))
    return components


def _unit_vector(values: Sequence[float], *, label: str) -> list[float]:
    vector = embedding_vector(values, label=label)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError(f"{label} embedding has zero norm")
    return [value / norm for value in vector]


def _load_candidate_lineage(
    store: Any,
    item_ids: set[str],
    scope: Mapping[str, str],
    *,
    session: Any | None = None,
) -> tuple[dict[str, MemoryItem], dict[str, MemoryItem]]:
    loaded: dict[str, MemoryItem] = {}
    pending = set(item_ids)
    while pending:
        batch = store.memory_item_repo.list_items_by_ids(
            pending,
            scope,
            include_superseded=True,
            include_merged=True,
            include_embeddings=True,
            session=session,
        )
        missing = pending - batch.keys()
        if missing:
            raise KeyError(f"Candidate memory items not found in scope: {sorted(missing)}")
        loaded.update(batch)
        pending = {
            item.merged_into.strip()
            for item in batch.values()
            if item.merged_into and item.merged_into.strip() not in loaded
        }

    vectors = {
        item.id: embedding_vector(item.embedding, label=f"memory {item.id}")
        for item in loaded.values()
    }
    if len({len(vector) for vector in vectors.values()}) > 1:
        raise ValueError("Candidate memory embedding dimensions do not match")

    canonical: dict[str, MemoryItem] = {}
    for item_id in item_ids:
        current = loaded[item_id]
        seen: set[str] = set()
        while current.merged_into and current.merged_into.strip():
            if current.id in seen:
                raise ValueError(f"Candidate memory merge cycle at {current.id}")
            seen.add(current.id)
            target_id = current.merged_into.strip()
            if target_id not in loaded:
                raise KeyError(f"Candidate memory merge target not found in scope: {target_id}")
            current = loaded[target_id]
        if current.id in seen:
            raise ValueError(f"Candidate memory merge cycle at {current.id}")
        canonical[item_id] = current
    return loaded, canonical


def file_category_proposals(
    *,
    store: Any,
    item_proposals: Sequence[tuple[MemoryItem, Sequence[str]]],
    where: Mapping[str, Any],
    session: Any,
) -> tuple[list[Any], list[Any]]:
    scope = dossier_scope(where)
    categories = store.memory_category_repo.list_categories(scope, session=session)
    by_name: dict[str, MemoryCategory] = {}
    for category in categories.values():
        if category.anchor_role is not None:
            continue
        normalized = normalize_category_name(category.name)
        if normalized is None:
            raise ValueError(f"Dossier {category.id} title must contain letters or digits")
        if normalized in by_name:
            raise ValueError(f"Duplicate normalized dossier title in scope: {normalized}")
        by_name[normalized] = category

    proposals_by_item: dict[str, tuple[MemoryItem, list[str]]] = {}
    for item, proposals in item_proposals:
        saved_item, saved_proposals = proposals_by_item.setdefault(item.id, (item, []))
        if saved_item != item:
            raise ValueError(f"Conflicting records supplied for memory {item.id}")
        saved_proposals.extend(proposals)

    validated: list[tuple[MemoryItem, list[str], list[tuple[str, str]]]] = []
    for item, proposals in proposals_by_item.values():
        known, unknown = _partition_category_names(
            proposals,
            set(by_name),
            normalize_category_name,
            dedupe_unknown=True,
        )
        if len(known) + len(unknown) > 3:
            raise ValueError(f"Memory {item.id} has more than three category proposals")
        validated.append((item, known, unknown))
    item_ids = set(proposals_by_item)
    persisted = store.memory_item_repo.list_items_by_ids(
        item_ids,
        scope,
        include_superseded=True,
        include_merged=True,
        include_embeddings=False,
        session=session,
    )
    missing = item_ids - persisted.keys()
    if missing:
        raise KeyError(f"Memory items not found in scope: {sorted(missing)}")

    prepared: list[tuple[MemoryItem, list[str], list[tuple[str, str]], str | None]] = []
    for item, known, unknown in validated:
        saved = persisted[item.id]
        memory_day = str(saved.extra.get("memory_date") or "").strip() or None
        if memory_day is None and saved.happened_at is not None:
            memory_day = saved.happened_at.date().isoformat()
        if memory_day is not None:
            memory_day = date.fromisoformat(memory_day).isoformat()
        prepared.append((saved, known, unknown, memory_day))

    relations: list[Any] = []
    candidates: list[Any] = []
    for saved, known, unknown, memory_day in prepared:
        for normalized in known:
            relations.append(
                store.category_item_repo.link_item_category(
                    saved.id,
                    by_name[normalized].id,
                    scope,
                    session=session,
                )
            )
        for raw, _normalized in unknown:
            candidates.append(
                store.dossier_candidate_repo.add_candidate(
                    proposed_name=raw,
                    item_id=saved.id,
                    where=scope,
                    segment_id=saved.segment_id,
                    memory_day=memory_day,
                    session=session,
                )
            )
    return relations, candidates


async def prepare_dynamic_category_review(
    *,
    store: Any,
    where: Mapping[str, Any],
    cluster_size: int,
    search_dossiers: Callable[..., Awaitable[list[tuple[MemoryCategory, float]]]],
    cosine_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    scope = dossier_scope(where)
    if isinstance(cluster_size, bool) or not isinstance(cluster_size, int) or cluster_size < 2:
        raise ValueError("Dossier candidate cluster size must be at least two")
    if not math.isfinite(cosine_threshold) or not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError("Dossier candidate cosine threshold must be between -1 and 1")

    candidates = store.dossier_candidate_repo.list_candidates(scope)
    if len(candidates) < cluster_size:
        return []
    _, canonical_by_source = _load_candidate_lineage(
        store, {candidate.item_id for candidate in candidates}, scope
    )
    grouped: dict[str, list[Any]] = {}
    canonical_items: dict[str, MemoryItem] = {}
    for candidate in candidates:
        canonical = canonical_by_source[candidate.item_id]
        grouped.setdefault(canonical.id, []).append(candidate)
        canonical_items[canonical.id] = canonical

    def node_key(item_id: str) -> tuple[datetime, str, str]:
        first = min(grouped[item_id], key=lambda row: (row.created_at, row.id))
        return first.created_at, first.id, item_id

    node_ids = sorted(grouped, key=node_key)
    vectors = [
        _unit_vector(canonical_items[item_id].embedding or [], label=f"memory {item_id}")
        for item_id in node_ids
    ]
    if len({len(vector) for vector in vectors}) > 1:
        raise ValueError("Candidate memory embedding dimensions do not match")
    components = _connected_components(
        list(range(len(node_ids))),
        vectors,
        cosine_similarity=cosine_similarity,
        threshold=cosine_threshold,
    )
    components = [
        component
        for component in components
        if len(component) >= cluster_size
        and any(
            candidate.last_considered_at is None
            for index in component
            for candidate in grouped[node_ids[index]]
        )
    ]
    if not components:
        return []

    all_categories = list(store.memory_category_repo.list_categories(scope).values())
    soul_anchors = [category for category in all_categories if category.anchor_role == "soul"]
    if len(soul_anchors) > 1:
        raise ValueError("Dossier review scope contains multiple soul anchors")
    soul_anchor = soul_anchors[0] if soul_anchors else None
    categories = [
        category
        for category in all_categories
        if category.kind in DOSSIER_KINDS and category.anchor_role is None
    ]
    categories_by_name: dict[str, MemoryCategory] = {}
    for category in categories:
        normalized = normalize_category_name(category.name)
        if normalized is None:
            raise ValueError(f"Dossier {category.id} title must contain letters or digits")
        if normalized in categories_by_name:
            raise ValueError(f"Duplicate normalized dossier title in scope: {normalized}")
        categories_by_name[normalized] = category

    bundles: list[dict[str, Any]] = []
    for component in components:
        member_ids = [node_ids[index] for index in component]
        component_candidates = sorted(
            [candidate for item_id in member_ids for candidate in grouped[item_id]],
            key=lambda candidate: (candidate.created_at, candidate.id),
        )
        centroid = [
            sum(vectors[index][dimension] for index in component) / len(component)
            for dimension in range(len(vectors[component[0]]))
        ]
        centroid = _unit_vector(centroid, label="dossier candidate centroid")
        semantic = await search_dossiers(
            centroid,
            where=scope,
            view="identity",
            activity="all",
            limit=10,
            min_score=-1.0,
            categories=categories,
        )
        exact: list[MemoryCategory] = []
        exact_ids: set[str] = set()
        for candidate in component_candidates:
            category = categories_by_name.get(candidate.normalized_name)
            if category is not None and category.id not in exact_ids:
                exact.append(
                    category.model_copy(
                        update={
                            "summary": None,
                            "previous_description": None,
                            "approved_description": None,
                            "previous_summary": None,
                            "approved_summary": None,
                        }
                    )
                )
                exact_ids.add(category.id)
        existing = exact + [
            category
            for category, _score in semantic[:10]
            if category.id not in exact_ids
        ]
        bundles.append(
            {
                "cluster_id": f"cluster_{len(bundles) + 1}",
                "candidate_ids": [candidate.id for candidate in component_candidates],
                "memory_count": len(member_ids),
                "memories": [
                    {"item": canonical_items[item_id], "candidates": grouped[item_id]}
                    for item_id in member_ids
                ],
                "soul_anchor": soul_anchor,
                "existing_dossiers": existing,
            }
        )
    return bundles


def _dynamic_review_candidate_ids(bundle: Mapping[str, Any]) -> list[str]:
    candidate_ids = bundle.get("candidate_ids")
    if not isinstance(candidate_ids, list) or any(
        not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids
    ):
        raise ValueError("Review bundle candidate IDs must be nonblank strings")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Review bundle contains duplicate candidate IDs")
    return candidate_ids


def _render_dynamic_category_review_bundle(bundle: Mapping[str, Any]) -> tuple[str, str, str, str]:
    cluster_id = bundle.get("cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise ValueError("Review bundle cluster ID is required")
    candidate_ids = _dynamic_review_candidate_ids(bundle)
    expected_ids = set(candidate_ids)
    rendered_ids: list[str] = []
    memory_blocks: list[str] = []
    memories = bundle.get("memories")
    if not isinstance(memories, list):
        raise ValueError("Review bundle memories must be a list")
    for entry in memories:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("item"), MemoryItem):
            raise ValueError("Review bundle contains an invalid memory entry")
        item = entry["item"]
        if not isinstance(item.memory_ref, int) or isinstance(item.memory_ref, bool) or item.memory_ref < 1:
            raise ValueError(f"Memory {item.id} has no valid [M#] reference")
        candidates = entry.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Memory {item.id} has no dossier candidates")
        lines = [
            f"[M{item.memory_ref}] type={item.memory_type}"
            + (f" date={item.happened_at.date().isoformat()}" if item.happened_at is not None else ""),
            f"Memory: {' '.join(item.summary.split())}",
            "Candidate labels:",
        ]
        for candidate in candidates:
            candidate_id = getattr(candidate, "id", None)
            proposed_name = getattr(candidate, "proposed_name", None)
            if candidate_id not in expected_ids or not isinstance(proposed_name, str) or not proposed_name.strip():
                raise ValueError(f"Memory {item.id} contains an invalid dossier candidate")
            rendered_ids.append(candidate_id)
            day = getattr(candidate, "memory_day", None)
            suffix = f" | day={day}" if isinstance(day, str) and day else ""
            lines.append(f"- candidate_id={candidate_id} | label={proposed_name.strip()}{suffix}")
        memory_blocks.append("\n".join(lines))
    if len(rendered_ids) != len(set(rendered_ids)) or set(rendered_ids) != expected_ids:
        raise ValueError("Review bundle candidate rows do not match candidate IDs")

    soul_anchor = bundle.get("soul_anchor")
    if (
        not isinstance(soul_anchor, MemoryCategory)
        or soul_anchor.anchor_role != "soul"
        or soul_anchor.kind != "lore"
    ):
        raise ValueError("Dynamic dossier review requires one soul anchor")
    soul_anchor_text = (
        f"Title: {soul_anchor.name}\n"
        f"Description: {' '.join(soul_anchor.description.split())}\n"
        f"Prose:\n{strip_memory_citations(soul_anchor.summary or '').strip() or '(none yet)'}"
    )

    dossier_lines: list[str] = []
    dossier_ids: set[str] = set()
    existing = bundle.get("existing_dossiers")
    if not isinstance(existing, list):
        raise ValueError("Review bundle existing dossiers must be a list")
    for dossier in existing:
        if not isinstance(dossier, MemoryCategory) or dossier.id in dossier_ids:
            raise ValueError("Review bundle contains an invalid existing dossier")
        dossier_ids.add(dossier.id)
        dossier_lines.append(
            f"- dossier_id={dossier.id} | kind={dossier.kind} | title={dossier.name}\n"
            f"  Description: {' '.join(dossier.description.split())}"
        )
    return (
        cluster_id,
        soul_anchor_text,
        "\n\n".join(memory_blocks),
        "\n".join(dossier_lines) or "(none)",
    )


def _dynamic_review_children(root: Element) -> dict[str, Element]:
    if (root.text or "").strip():
        raise ValueError("Unexpected text inside dynamic dossier review root")
    children: dict[str, Element] = {}
    for child in root:
        if child.tag in children:
            raise ValueError(f"Duplicate dynamic dossier review element: {child.tag}")
        if (child.tail or "").strip():
            raise ValueError("Unexpected text inside dynamic dossier review root")
        children[child.tag] = child
    return children


def _dynamic_review_leaf(element: Element) -> str:
    if element.attrib or list(element):
        raise ValueError(f"Expected plain text in {element.tag}")
    return element.text or ""


def _dynamic_review_id_list(element: Element) -> list[str]:
    if element.attrib or (element.text or "").strip():
        raise ValueError(f"Invalid {element.tag} wrapper")
    values: list[str] = []
    for child in element:
        if child.tag != "candidate_id" or (child.tail or "").strip():
            raise ValueError(f"Invalid candidate ID in {element.tag}")
        value = _dynamic_review_leaf(child).strip()
        if not value:
            raise ValueError(f"Blank candidate ID in {element.tag}")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate candidate ID in {element.tag}")
    return values


def parse_dynamic_category_review(raw: str, bundle: Mapping[str, Any]) -> dict[str, Any]:
    text = str(raw or "").strip()
    if (
        not text.startswith("<dynamic_dossier_review")
        or not text.endswith("</dynamic_dossier_review>")
        or "<!--" in text
        or "<?" in text
    ):
        raise ValueError("Expected exact dynamic_dossier_review XML")
    try:
        root = ElementTree.fromstring(text)
    except Exception as exc:
        raise ValueError("Invalid dynamic dossier review XML") from exc
    if root.tag != "dynamic_dossier_review" or set(root.attrib) != {"cluster_id"}:
        raise ValueError("Expected exact dynamic_dossier_review root")
    if root.attrib["cluster_id"] != bundle.get("cluster_id"):
        raise ValueError("Dynamic dossier review cluster ID does not match request")

    children = _dynamic_review_children(root)
    action_element = children.get("action")
    if action_element is None:
        raise ValueError("Missing dynamic dossier review element: action")
    action = _dynamic_review_leaf(action_element).strip()
    common = {"action", "accepted_candidate_ids", "rejected_candidate_ids"}
    action_fields = {
        "existing": {"existing_dossier_id"},
        "create": {"title", "description", "kind"},
        "defer": set(),
    }
    if action not in action_fields:
        raise ValueError(f"Invalid dynamic dossier review action: {action}")
    expected = common | action_fields[action]
    if set(children) != expected:
        raise ValueError(
            f"Dynamic dossier review {action} elements must be exactly: {sorted(expected)}"
        )

    accepted_ids = _dynamic_review_id_list(children["accepted_candidate_ids"])
    rejected_ids = _dynamic_review_id_list(children["rejected_candidate_ids"])
    bundle_ids = _dynamic_review_candidate_ids(bundle)
    if set(accepted_ids) & set(rejected_ids) or set(accepted_ids + rejected_ids) != set(bundle_ids):
        raise ValueError("Accepted and rejected candidate IDs must partition the bundle")
    if action in {"existing", "create"} and not accepted_ids:
        raise ValueError(f"Dynamic dossier review action {action} requires an accepted candidate")
    if action == "defer" and accepted_ids:
        raise ValueError("Deferred dynamic dossier review cannot accept candidates")

    decision: dict[str, Any] = {
        "cluster_id": root.attrib["cluster_id"],
        "action": action,
        "accepted_candidate_ids": accepted_ids,
        "rejected_candidate_ids": rejected_ids,
    }
    if action == "existing":
        target_id = _dynamic_review_leaf(children["existing_dossier_id"]).strip()
        allowed_ids = {
            dossier.id
            for dossier in bundle.get("existing_dossiers", [])
            if isinstance(dossier, MemoryCategory)
        }
        if not target_id or target_id not in allowed_ids:
            raise ValueError("Existing dossier target was not supplied in the review bundle")
        decision["existing_dossier_id"] = target_id
    elif action == "create":
        name = _dynamic_review_leaf(children["title"]).strip()
        description = _dynamic_review_leaf(children["description"]).strip()
        kind = _dynamic_review_leaf(children["kind"]).strip()
        if not name:
            raise ValueError("Created dossier title is required")
        if not description:
            raise ValueError("Created dossier description is required")
        if kind not in DOSSIER_KINDS:
            raise ValueError(f"Invalid dossier kind: {kind}")
        decision.update({"name": name, "description": description, "kind": kind})
    return decision


def render_dynamic_category_review_prompts(
    bundle: Mapping[str, Any],
) -> tuple[str, str]:
    cluster_id, soul_anchor, candidate_memories, existing_dossiers = (
        _render_dynamic_category_review_bundle(bundle)
    )
    user_prompt = DYNAMIC_DOSSIER_REVIEW_USER_PROMPT.format(
        soul_anchor=soul_anchor,
        cluster_id=cluster_id,
        candidate_memories=candidate_memories,
        existing_dossiers=existing_dossiers,
    )
    return DYNAMIC_DOSSIER_REVIEW_SYSTEM_PROMPT, user_prompt


async def generate_dynamic_category_review(
    *,
    bundle: Mapping[str, Any],
    select_chat_client: Callable[..., Any],
    profile: str,
    chat_client: Any | None = None,
) -> dict[str, Any]:
    system_prompt, user_prompt = render_dynamic_category_review_prompts(bundle)
    if len((system_prompt + "\n" + user_prompt).split()) / 0.75 > 100_000:
        raise ValueError("Dynamic dossier review prompt exceeds 100000 tokens")
    client = chat_client or select_chat_client(
        {"operation": "dossier", "step_id": "dynamic_review"},
        profile=profile,
    )
    raw = await client.chat(user_prompt, system_prompt=system_prompt)
    return parse_dynamic_category_review(str(raw or ""), bundle)


def apply_dynamic_category_review(
    *,
    store: Any,
    where: Mapping[str, Any],
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    session: Any,
    proposed_embedding: Sequence[float] | None = None,
    near_duplicate_threshold: float = 0.95,
) -> dict[str, Any]:
    scope = dossier_scope(where)
    if not math.isfinite(near_duplicate_threshold) or not -1.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("Near-duplicate threshold must be between -1 and 1")

    def ids(value: Any, label: str) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"{label} must be a list of candidate IDs")
        if len(value) != len(set(value)):
            raise ValueError(f"{label} contains duplicate candidate IDs")
        return value

    cluster_id = decision.get("cluster_id")
    if cluster_id != bundle.get("cluster_id"):
        raise ValueError("Decision cluster ID does not match the review bundle")
    bundle_ids = ids(bundle.get("candidate_ids"), "Bundle candidate IDs")
    accepted_ids = ids(decision.get("accepted_candidate_ids"), "Accepted candidate IDs")
    rejected_ids = ids(decision.get("rejected_candidate_ids"), "Rejected candidate IDs")
    if set(accepted_ids) & set(rejected_ids) or set(accepted_ids + rejected_ids) != set(bundle_ids):
        raise ValueError("Accepted and rejected candidate IDs must partition the bundle")

    action = decision.get("action")
    if action not in {"existing", "create", "defer"}:
        raise ValueError(f"Unknown dossier candidate action: {action}")
    target_id = decision.get("existing_dossier_id")
    name = decision.get("name")
    description = decision.get("description")
    kind = decision.get("kind")

    requested_ids = set(bundle_ids)
    current = {
        candidate.id: candidate
        for candidate in store.dossier_candidate_repo.list_candidates(
            scope, unresolved_only=False, session=session
        )
        if candidate.id in requested_ids
    }
    missing = requested_ids - current.keys()
    if missing:
        raise KeyError(f"Dossier candidates not found in scope: {sorted(missing)}")
    if any(candidate.resolved_category_id is not None for candidate in current.values()):
        raise ValueError("Review bundle contains already-resolved candidates")
    _, canonical = _load_candidate_lineage(
        store,
        {candidate.item_id for candidate in current.values()},
        scope,
        session=session,
    )
    all_categories = store.memory_category_repo.list_categories(
        scope, session=session
    )
    categories = {
        category_id: category
        for category_id, category in all_categories.items()
        if category.kind in DOSSIER_KINDS
    }
    bundle_targets = {
        category.id
        for category in bundle.get("existing_dossiers", [])
        if isinstance(category, MemoryCategory)
    }
    accepted = [current[candidate_id] for candidate_id in accepted_ids]
    considered_at = datetime.now(UTC)

    def considered() -> list[Any]:
        return cast(
            list[Any],
            store.dossier_candidate_repo.mark_candidates_considered(
                bundle_ids, considered_at, scope, session=session
            ),
        )

    if action == "defer":
        if accepted_ids or any(
            value is not None
            for value in (target_id, name, description, kind, proposed_embedding)
        ):
            raise ValueError("Deferred dossier decisions cannot contain a target or accepted candidates")
        return {
            "status": "deferred",
            "target_dossier": None,
            "relations": [],
            "resolved_candidates": [],
            "considered_candidates": considered(),
        }
    if not accepted:
        raise ValueError("A dossier decision requires at least one accepted candidate")

    if action == "existing":
        if not isinstance(target_id, str) or target_id not in bundle_targets:
            raise ValueError("Existing dossier target was not supplied in the bundle")
        if any(value is not None for value in (name, description, kind, proposed_embedding)):
            raise ValueError("Existing dossier decisions cannot contain new-dossier fields")
        target = categories.get(target_id)
        if target is None or target.anchor_role is not None:
            raise ValueError("Existing dossier target is invalid for category review")
        status = "existing"
    else:
        if target_id is not None:
            raise ValueError("Create dossier decisions cannot contain an existing target")
        if not isinstance(name, str):
            raise ValueError("Created dossier title must be a string")
        normalized_name = normalize_category_name(name)
        if normalized_name is None:
            raise ValueError("Created dossier title must contain letters or digits")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Created dossier description is required")
        if kind not in DOSSIER_KINDS:
            raise ValueError(f"Invalid dossier kind: {kind}")
        embedding = embedding_vector(proposed_embedding, label="proposed dossier")
        candidate_dimension = len(next(iter(canonical.values())).embedding or [])
        if len(embedding) != candidate_dimension:
            raise ValueError("Proposed dossier embedding dimension does not match candidate memories")

        normalized_categories: dict[str, MemoryCategory] = {}
        ordinary: list[MemoryCategory] = []
        for category in all_categories.values():
            normalized = normalize_category_name(category.name)
            if normalized is None:
                raise ValueError(f"Dossier {category.id} title must contain letters or digits")
            if normalized in normalized_categories:
                raise ValueError(f"Duplicate normalized dossier title in scope: {normalized}")
            normalized_categories[normalized] = category
            if category.kind in DOSSIER_KINDS and category.anchor_role is None:
                ordinary.append(category)
        collision = normalized_categories.get(normalized_name)
        if collision is None and ordinary:
            corpus: list[tuple[str, list[float]]] = []
            for category in ordinary:
                vector = embedding_vector(category.embedding, label=f"dossier {category.id}")
                if len(vector) != len(embedding):
                    raise ValueError(f"Dossier embedding dimension mismatch for: {category.id}")
                corpus.append((category.id, vector))
            hit_id, score = cosine_topk(embedding, corpus, k=1)[0]
            if score >= near_duplicate_threshold:
                collision = all_categories[hit_id]
        if collision is not None:
            return {
                "status": "collision",
                "target_dossier": collision,
                "relations": [],
                "resolved_candidates": [],
                "considered_candidates": considered(),
            }
        target = store.memory_category_repo.create_category_strict(
            name=name.strip(),
            description=description.strip(),
            embedding=embedding,
            user_data=scope,
            kind=kind,
            session=session,
        )
        status = "created"

    item_ids = sorted({canonical[candidate.item_id].id for candidate in accepted})
    relations = [
        store.category_item_repo.link_item_category(
            item_id, target.id, scope, session=session
        )
        for item_id in item_ids
    ]
    resolved = store.dossier_candidate_repo.resolve_candidates(
        accepted_ids, target.id, scope, session=session
    )
    return {
        "status": status,
        "target_dossier": target,
        "relations": relations,
        "resolved_candidates": resolved,
        "considered_candidates": considered(),
    }
