from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from memu.app.category_summary_journal import update_category_summary_with_journal
from memu.prompts.category_summary import (
    CUSTOM_PROMPT as CATEGORY_SUMMARY_CUSTOM_PROMPT,
)
from memu.prompts.category_summary import (
    PROMPT as CATEGORY_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


def _dynamic_category_cluster_threshold() -> float:
    return 0.75


def _dynamic_category_cluster_min_size(dynamic_category_cluster_size: Any) -> int:
    cluster_size = int(dynamic_category_cluster_size or 3)
    return max(2, cluster_size)


def _step_label(value: Any) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return label or "unknown"


def _cluster_homeless_entries(
    *,
    filtered_entries: list[Any],
    per_entry_unknowns: Sequence[list[str]],
    item_embeddings: Sequence[Any] | None = None,
    normalize_embedding_vector: Callable[[Any], list[float] | None],
    cosine_similarity: Callable[[Sequence[float], Sequence[float]], float],
    cluster_factory: Callable[..., Any],
    cluster_similarity_threshold: float,
    cluster_min_size: int,
) -> tuple[list[Any], dict[int, str]]:
    if not filtered_entries or not item_embeddings:
        return [], {}

    normalized_embeddings: list[list[float] | None] = [
        normalize_embedding_vector(raw) for raw in item_embeddings[: len(filtered_entries)]
    ]
    if len(normalized_embeddings) < len(filtered_entries):
        normalized_embeddings.extend([None] * (len(filtered_entries) - len(normalized_embeddings)))

    candidate_indexes = [
        idx
        for idx, (entry, unknowns, embedding) in enumerate(
            zip(filtered_entries, per_entry_unknowns, normalized_embeddings, strict=True)
        )
        if not entry.categories and unknowns and embedding is not None
    ]
    if len(candidate_indexes) < 2:
        return [], {}

    adjacency: dict[int, set[int]] = {idx: set() for idx in candidate_indexes}

    for pos, left_idx in enumerate(candidate_indexes):
        left_embedding = normalized_embeddings[left_idx]
        if left_embedding is None:
            continue
        for right_idx in candidate_indexes[pos + 1 :]:
            right_embedding = normalized_embeddings[right_idx]
            if right_embedding is None:
                continue
            if cosine_similarity(left_embedding, right_embedding) < cluster_similarity_threshold:
                continue
            adjacency[left_idx].add(right_idx)
            adjacency[right_idx].add(left_idx)

    visited: set[int] = set()
    clusters: list[Any] = []
    entry_cluster_ids: dict[int, str] = {}

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
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in visited:
                    stack.append(neighbor)

        component.sort()
        if len(component) < cluster_min_size:
            continue

        label_counts: dict[str, int] = {}
        saliences: list[float] = []
        examples: list[str] = []
        for entry_idx in component:
            for label in per_entry_unknowns[entry_idx]:
                label_counts[label] = label_counts.get(label, 0) + 1
            salience = filtered_entries[entry_idx].reflection_salience
            if salience is not None:
                saliences.append(salience)
            content = filtered_entries[entry_idx].content.strip()
            if content and len(examples) < 4:
                examples.append(content)

        cluster_id = f"cluster_{len(clusters) + 1}"
        cluster = cluster_factory(
            cluster_id=cluster_id,
            entry_indexes=component,
            label_counts=label_counts,
            average_salience=(sum(saliences) / len(saliences)) if saliences else None,
            max_salience=max(saliences) if saliences else None,
            examples=examples,
        )
        clusters.append(cluster)
        for entry_idx in component:
            entry_cluster_ids[entry_idx] = cluster_id

    return clusters, entry_cluster_ids


def _build_existing_category_block(
    *,
    ctx: Any,
    store: Any,
    fallback_category_configs: Sequence[Any],
) -> str:
    existing_lines: list[str] = []
    seen_existing: set[str] = set()
    for category_id in getattr(ctx, "category_ids", []) or []:
        category = store.memory_category_repo.categories.get(category_id)
        if category is None:
            continue
        nm = str(category.name or "").strip()
        if not nm:
            continue
        key = nm.casefold()
        if key in seen_existing:
            continue
        seen_existing.add(key)
        desc = str(getattr(category, "description", "") or "").strip()
        existing_lines.append(f"- {nm}: {desc}" if desc else f"- {nm}")
    if not existing_lines:
        for cfg in fallback_category_configs:
            nm = str(getattr(cfg, "name", "") or "").strip()
            if not nm:
                continue
            desc = str(getattr(cfg, "description", "") or "").strip()
            existing_lines.append(f"- {nm}: {desc}" if desc else f"- {nm}")
    return "\n".join(existing_lines) if existing_lines else "(none)"


def _build_cluster_prompt_block(strong_clusters: Sequence[Any]) -> str:
    cluster_lines: list[str] = []
    for cluster in strong_clusters[:12]:
        label_block = ", ".join(
            f"{label}({count})"
            for label, count in sorted(cluster.label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        )
        salience_bits: list[str] = []
        if cluster.average_salience is not None:
            salience_bits.append(f"avg_salience={cluster.average_salience:.2f}")
        if cluster.max_salience is not None:
            salience_bits.append(f"max_salience={cluster.max_salience:.2f}")
        salience_block = f" {' '.join(salience_bits)}" if salience_bits else ""
        examples_block = "\n".join(f"    - {example}" for example in cluster.examples)
        header = (
            f"- {cluster.cluster_id} size={len(cluster.entry_indexes)} labels={label_block or '(none)'}"
            f"{salience_block}"
        )
        cluster_lines.append(f"{header}\n{examples_block}" if examples_block else header)
    return "\n".join(cluster_lines) if cluster_lines else "(none)"


def _build_ungrouped_candidate_block(
    *,
    ungrouped_unknown_counts: Mapping[str, int],
    ungrouped_unknown_examples: Mapping[str, list[str]],
) -> str:
    candidates_sorted = sorted(ungrouped_unknown_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    cand_lines: list[str] = []
    for cand, cnt in candidates_sorted[:20]:
        examples = ungrouped_unknown_examples.get(cand, [])
        ex_block = "\n".join(f"    - {e}" for e in examples) if examples else ""
        cand_lines.append(f"- {cand} (count={cnt})\n{ex_block}" if ex_block else f"- {cand} (count={cnt})")
    return "\n".join(cand_lines) if cand_lines else "(none)"


async def _plan_dynamic_categories(
    *,
    ctx: Any,
    store: Any,
    strong_clusters: Sequence[Any],
    ungrouped_unknown_counts: Mapping[str, int],
    ungrouped_unknown_examples: Mapping[str, list[str]],
    min_mentions: int,
    policy: str,
    default_desc: str,
    fallback_category_configs: Sequence[Any],
    llm_client: Any,
    normalize_category_name: Callable[[str], str | None],
    dynamic_category_cluster_min_size: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    existing_block = _build_existing_category_block(
        ctx=ctx,
        store=store,
        fallback_category_configs=fallback_category_configs,
    )
    clusters_block = _build_cluster_prompt_block(strong_clusters)
    candidates_block = _build_ungrouped_candidate_block(
        ungrouped_unknown_counts=ungrouped_unknown_counts,
        ungrouped_unknown_examples=ungrouped_unknown_examples,
    )
    desc_line = default_desc or (
        "Categories are life domains and are thus broad by nature. "
        "Life domains are the core, interconnected areas of a being's existence—such as health, relationships, work, and finances."
    )
    policy_block = (
        f"""

Extra guidance (optional):
{policy}
""".strip()
        if policy
        else ""
    )
    system_prompt = f"""You are organizing memory categories.

{desc_line}

{policy_block}

Rules:
- Prefer mapping candidates into EXISTING categories.
- Homeless clusters are stronger evidence than raw labels. Use cluster meaning first and treat labels as hints.
- Only propose NEW categories if the topic is a broad life domain AND it is mentioned at least {min_mentions} times across the extracted memories OR it is clearly important.
- If multiple close candidates (e.g., health/medical) should be merged, merge them and accumulate the count.
- New category names must be 1-3 words, letters/spaces only (no underscores); we will normalize to snake_case later.
- Do not recreate an existing category under a new name.

Output ONLY valid JSON with this shape:
{{"create": [{{"cluster": "cluster_1", "name": "...", "description": "...", "important": true|false}}, {{"name": "...", "description": "...", "from": ["candidate1", "candidate2"], "important": true|false}}], "map": [{{"cluster": "cluster_2", "to": "existing_category"}}, {{"from": "candidate", "to": "existing_category"}}]}}
""".strip()
    user_prompt = f"""EXISTING CATEGORIES:
{existing_block}

HOMELESS CLUSTERS (embedding-similar memories that do not fit existing categories):
{clusters_block}

UNGROUPED UNKNOWN LABELS (homeless items not in a strong cluster):
{candidates_block}

Decide which clusters/candidates should map into existing categories, and which (if any) justify creating a NEW life-domain category.""".strip()

    cluster_mapping: dict[str, str] = {}
    label_mapping: dict[str, str] = {}
    new_defs: dict[str, str] = {}
    cluster_by_id = {cluster.cluster_id: cluster for cluster in strong_clusters}

    def _valid_new_name(raw: str) -> bool:
        raw = (raw or "").strip()
        if not raw:
            return False
        if len(raw.split()) > 3:
            return False
        return bool(re.fullmatch(r"[A-Za-z ]+", raw))

    try:
        resp = await llm_client.chat(user_prompt, system_prompt=system_prompt)
    except Exception:
        logger.warning("dynamic-category planner LLM call failed", exc_info=True)
        raise
    match = re.search(r"\{[\s\S]*\}", resp or "")
    if match is None:
        return cluster_mapping, label_mapping, new_defs
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("dynamic-category planner returned unparseable JSON: %.200s", resp)
        return cluster_mapping, label_mapping, new_defs

    if not isinstance(plan, dict):
        return cluster_mapping, label_mapping, new_defs

    for entry in plan.get("create", []) or []:
        if not isinstance(entry, dict):
            continue
        raw_name = str(entry.get("name", "") or "").strip()
        if not _valid_new_name(raw_name):
            continue
        desc = str(entry.get("description", "") or "").strip() or default_desc
        important = bool(entry.get("important", False))
        cluster_id = str(entry.get("cluster", "") or "").strip()
        norm_name = normalize_category_name(raw_name)
        if not norm_name:
            continue

        if cluster_id and cluster_id in cluster_by_id:
            cluster = cluster_by_id[cluster_id]
            if len(cluster.entry_indexes) < dynamic_category_cluster_min_size and not important:
                continue
            if norm_name not in ctx.category_name_to_id:
                new_defs.setdefault(norm_name, desc)
            cluster_mapping[cluster_id] = norm_name
            continue

        src = entry.get("from", []) or []
        if not isinstance(src, list):
            src = [src]
        src_norm = [
            sn
            for s in src
            if isinstance(s, str)
            for sn in [normalize_category_name(s)]
            if sn and sn in ungrouped_unknown_counts
        ]
        if not src_norm:
            continue
        total = sum(ungrouped_unknown_counts.get(s, 0) for s in src_norm)
        if total < min_mentions and not important:
            continue
        if norm_name not in ctx.category_name_to_id:
            new_defs.setdefault(norm_name, desc)
        for source_name in src_norm:
            label_mapping[source_name] = norm_name

    for entry in plan.get("map", []) or []:
        if not isinstance(entry, dict):
            continue
        cluster_id = str(entry.get("cluster", "") or "").strip()
        if cluster_id and cluster_id in cluster_by_id:
            tgt = normalize_category_name(str(entry.get("to", "") or ""))
            if tgt and (tgt in ctx.category_name_to_id or tgt in new_defs):
                cluster_mapping[cluster_id] = tgt
            continue
        src = normalize_category_name(str(entry.get("from", "") or ""))
        tgt = normalize_category_name(str(entry.get("to", "") or ""))
        if not src or src not in ungrouped_unknown_counts:
            continue
        if tgt and (tgt in ctx.category_name_to_id or tgt in new_defs):
            label_mapping[src] = tgt

    return cluster_mapping, label_mapping, new_defs


async def _maybe_create_dynamic_categories(
    *,
    structured_entries: list[Any],
    item_embeddings: Sequence[Any] | None,
    ctx: Any,
    store: Any,
    embed_client: Any,
    user: Mapping[str, Any] | None,
    session: Any | None,
    ensure_categories_ready: Callable[..., Awaitable[None]],
    normalize_category_name: Callable[[str], str | None],
    cluster_homeless_entries: Callable[..., tuple[list[Any], dict[int, str]]],
    plan_dynamic_categories: Callable[..., Awaitable[tuple[dict[str, str], dict[str, str], dict[str, str]]]],
    max_categories_total: int,
    dynamic_category_cluster_size: int,
    dynamic_category_policy: str,
    dynamic_category_description: str,
) -> list[Any]:
    await ensure_categories_ready(ctx, store, user)

    max_total = int(max_categories_total or 0)
    min_mentions = int(dynamic_category_cluster_size or 3)
    policy = str(dynamic_category_policy or "").strip()
    default_desc = str(dynamic_category_description or "").strip()

    cur_total = len(getattr(ctx, "category_ids", []) or [])
    remaining = (max_total - cur_total) if max_total else None
    if remaining is not None and remaining <= 0:
        return [
            entry._replace(categories=[c for c in (entry.categories or []) if c in ctx.category_name_to_id])
            for entry in structured_entries
        ]

    unknown_counts: dict[str, int] = {}
    per_entry_unknowns: list[list[str]] = []
    filtered_entries: list[Any] = []

    for entry in structured_entries:
        known: list[str] = []
        unknown: list[str] = []
        for c in entry.categories or []:
            n = normalize_category_name(c)
            if not n:
                continue
            if n in ctx.category_name_to_id:
                known.append(n)
            else:
                unknown.append(n)

        seen: set[str] = set()
        known_dedup: list[str] = []
        for k in known:
            if k not in seen:
                known_dedup.append(k)
                seen.add(k)
        homeless_unknowns = unknown if not known_dedup else []
        for n in homeless_unknowns:
            unknown_counts[n] = unknown_counts.get(n, 0) + 1
        filtered_entries.append(entry._replace(categories=known_dedup))
        per_entry_unknowns.append(homeless_unknowns)

    if not unknown_counts:
        return filtered_entries

    homeless_clusters, entry_cluster_ids = cluster_homeless_entries(
        filtered_entries=filtered_entries,
        per_entry_unknowns=per_entry_unknowns,
        item_embeddings=item_embeddings,
    )
    strong_clusters = [cluster for cluster in homeless_clusters if cluster.label_counts]
    clustered_indexes = set(entry_cluster_ids)

    ungrouped_unknown_counts: dict[str, int] = {}
    ungrouped_unknown_examples: dict[str, list[str]] = {}
    for idx, unknowns in enumerate(per_entry_unknowns):
        if idx in clustered_indexes:
            continue
        for label in unknowns:
            ungrouped_unknown_counts[label] = ungrouped_unknown_counts.get(label, 0) + 1
            ex_list = ungrouped_unknown_examples.setdefault(label, [])
            if len(ex_list) < 3:
                ex_list.append(filtered_entries[idx].content)

    cluster_mapping, label_mapping, new_defs = await plan_dynamic_categories(
        ctx=ctx,
        store=store,
        strong_clusters=strong_clusters,
        ungrouped_unknown_counts=ungrouped_unknown_counts,
        ungrouped_unknown_examples=ungrouped_unknown_examples,
        min_mentions=min_mentions,
        policy=policy,
        default_desc=default_desc,
    )

    to_create = [name for name in new_defs if name not in ctx.category_name_to_id]
    if remaining is not None:
        to_create = to_create[:remaining]

    if to_create:
        texts = [f"{name}: {new_defs.get(name, default_desc)}" for name in to_create]
        vecs = await embed_client.embed(texts)
        for name, vec in zip(to_create, vecs, strict=True):
            desc = new_defs.get(name, default_desc)
            cat = store.memory_category_repo.get_or_create_category(
                name=name,
                description=desc or "",
                embedding=vec,
                user_data=dict(user or {}),
                session=session,
            )
            ctx.category_ids.append(cat.id)
            ctx.category_name_to_id[name.lower()] = cat.id

    updated: list[Any] = []
    for idx, (entry, unk) in enumerate(zip(filtered_entries, per_entry_unknowns, strict=True)):
        cats = list(entry.categories)
        cluster_target = cluster_mapping.get(entry_cluster_ids.get(idx, ""))
        mapped_cluster_name = normalize_category_name(cluster_target) if cluster_target else None
        if mapped_cluster_name and mapped_cluster_name in ctx.category_name_to_id and mapped_cluster_name not in cats:
            cats.append(mapped_cluster_name)
        for unknown_name in unk or []:
            target_name = label_mapping.get(unknown_name)
            if not target_name:
                continue
            mapped_name = normalize_category_name(target_name)
            if mapped_name and mapped_name in ctx.category_name_to_id and mapped_name not in cats:
                cats.append(mapped_name)
        updated.append(entry._replace(categories=cats))

    return updated


def _category_scope_key(user_scope: Mapping[str, Any] | None) -> str:
    if not isinstance(user_scope, Mapping):
        return "__global__"
    user_id = str(user_scope.get("user_id") or "").strip()
    soul_id = str(user_scope.get("soul_id") or "").strip()
    if not user_id and not soul_id:
        return "__global__"
    return f"user={user_id}|soul={soul_id}"


async def _ensure_categories_ready(
    ctx: Any,
    store: Any,
    user_scope: Mapping[str, Any] | None,
    *,
    category_scope_key: Callable[[Mapping[str, Any] | None], str],
    initialize_categories: Callable[..., Awaitable[None]],
) -> None:
    scope_key = category_scope_key(user_scope)
    if ctx.categories_ready and ctx.category_scope_key == scope_key:
        return
    async with ctx._init_lock:
        if ctx.categories_ready and ctx.category_scope_key == scope_key:
            return
        await initialize_categories(ctx, store, user_scope, scope_key=scope_key)


async def _initialize_categories(
    ctx: Any,
    store: Any,
    user: Mapping[str, Any] | None,
    *,
    scope_key: str | None,
    category_scope_key: Callable[[Mapping[str, Any] | None], str],
    category_configs: Sequence[Any],
    embedding_client: Any | None,
    select_embedding_client: Callable[..., Any],
    category_embedding_text: Callable[[Any], str],
) -> None:
    resolved_scope_key = scope_key or category_scope_key(user)
    if ctx.categories_ready and ctx.category_scope_key == resolved_scope_key:
        return
    if not category_configs:
        ctx.categories_ready = True
        ctx.category_scope_key = resolved_scope_key
        return
    cat_texts = [category_embedding_text(cfg) for cfg in category_configs]
    client = embedding_client or select_embedding_client(
        {"operation": "memorize", "step_id": "initialize_categories"}
    )
    cat_vecs = await client.embed(cat_texts)
    ctx.category_ids = []
    ctx.category_name_to_id = {}
    for cfg, vec in zip(category_configs, cat_vecs, strict=True):
        name = str(getattr(cfg, "name", "") or "").strip() or "Untitled"
        description = str(getattr(cfg, "description", "") or "").strip()
        cat = store.memory_category_repo.get_or_create_category(
            name=name,
            description=description,
            embedding=vec,
            user_data=dict(user or {}),
        )
        ctx.category_ids.append(cat.id)
        ctx.category_name_to_id[name.lower()] = cat.id
    ctx.categories_ready = True
    ctx.category_scope_key = resolved_scope_key


def _category_embedding_text(cat: Any) -> str:
    name = str(getattr(cat, "name", "") or "").strip() or "Untitled"
    desc = str(getattr(cat, "description", "") or "").strip()
    return f"{name}: {desc}" if desc else name


def _map_category_names_to_ids(names: list[str], ctx: Any) -> list[str]:
    if not names:
        return []
    mapped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.strip().lower()
        cid = ctx.category_name_to_id.get(key)
        if cid and cid not in seen:
            mapped.append(cid)
            seen.add(cid)
    return mapped


def _build_category_summary_prompt(
    *,
    category: Any,
    new_memories: list[str] | list[tuple[str, str]],
    user: dict[str, Any] | None,
    memorize_config: Any,
    category_config_map: Mapping[str, Any],
    build_item_ref_id: Callable[[str], str],
    resolve_custom_prompt: Callable[[Any, Mapping[str, str]], str],
    summary_user_name: Callable[[Mapping[str, Any] | None], str],
    escape_prompt_value: Callable[[str], str],
) -> str:
    enable_refs = getattr(memorize_config, "enable_item_references", False)

    if enable_refs:
        from memu.prompts.category_summary import (
            CUSTOM_PROMPT_WITH_REFS as category_summary_custom_prompt,
        )
        from memu.prompts.category_summary import (
            PROMPT_WITH_REFS as category_summary_prompt,
        )

        tuple_memories = new_memories
        new_items_text = "\n".join(
            f"- [{build_item_ref_id(item_id)}] {summary}"
            for item_id, summary in tuple_memories
            if summary.strip()
        )
    else:
        category_summary_prompt = CATEGORY_SUMMARY_PROMPT
        category_summary_custom_prompt = CATEGORY_SUMMARY_CUSTOM_PROMPT

        if new_memories and isinstance(new_memories[0], tuple):
            tuple_memories = new_memories
            new_items_text = "\n".join(f"- {summary}" for _item_id, summary in tuple_memories if summary.strip())
        else:
            str_memories = new_memories
            new_items_text = "\n".join(f"- {m}" for m in str_memories if m.strip())

    original = category.summary or ""
    category_config = category_config_map.get(category.name)
    configured_prompt = (
        category_config and category_config.summary_prompt
    ) or memorize_config.default_category_summary_prompt
    if configured_prompt is None:
        prompt = category_summary_prompt
    elif isinstance(configured_prompt, str):
        prompt = configured_prompt
    else:
        prompt = resolve_custom_prompt(configured_prompt, category_summary_custom_prompt)
    target_length = (
        category_config and category_config.target_length
    ) or memorize_config.default_category_summary_target_length
    user_scope = user or {}
    user_name = summary_user_name(user_scope)
    raw_agent = (
        user_scope.get("soul_name")
        or user_scope.get("character_name")
        or user_scope.get("soul_id")
    )
    agent_name = str(raw_agent).strip() if raw_agent else "the soul"
    if " - " in agent_name:
        agent_name = agent_name.split(" - ", 1)[0].strip() or agent_name

    return prompt.format(
        category=escape_prompt_value(category.name),
        original_content=escape_prompt_value(original or ""),
        new_memory_items_text=escape_prompt_value(new_items_text or "No new memory items."),
        target_length=target_length,
        user_name=escape_prompt_value(user_name),
        agent_name=escape_prompt_value(agent_name),
    )


async def _update_category_summaries(
    updates: dict[str, list[tuple[str, str]]] | dict[str, list[str]],
    *,
    store: Any,
    llm_client: Any,
    user: dict[str, Any] | None,
    build_category_summary_prompt: Callable[[Any, Any, dict[str, Any] | None], str],
    summary_user_name: Callable[[Mapping[str, Any] | None], str],
) -> dict[str, str]:
    updated_summaries: dict[str, str] = {}
    if not updates:
        return updated_summaries

    tasks = []
    target_ids: list[str] = []
    for cid, memories in updates.items():
        cat = store.memory_category_repo.categories.get(cid)
        if not cat or not memories:
            continue
        prompt = build_category_summary_prompt(cat, memories, user)
        chat_client = llm_client
        with_metadata = getattr(chat_client, "with_metadata", None)
        if callable(with_metadata):
            chat_client = with_metadata(step_id=f"category_summary.{_step_label(getattr(cat, 'name', None) or cid)}")
        tasks.append(chat_client.chat(prompt))
        target_ids.append(cid)
    if not tasks:
        return updated_summaries

    summaries = await asyncio.gather(*tasks)
    for cid, summary in zip(target_ids, summaries, strict=True):
        cat = store.memory_category_repo.categories.get(cid)
        if not cat:
            continue
        cleaned_summary = summary.replace("```markdown", "").replace("```", "").strip()
        user_name = summary_user_name(user or {})
        if user_name and user_name.lower() not in ("user", "the user"):
            cleaned_summary = re.sub(
                r"(?m)^(\s*[-*]\s*)(?:The user|the user|User|user)\b",
                r"\1" + user_name,
                cleaned_summary,
            )
        if not cleaned_summary:
            continue

        scope = {key: user[key] for key in ("user_id", "soul_id") if isinstance(user, Mapping) and user.get(key)}
        update_category_summary_with_journal(
            store,
            category_id=cid,
            summary=cleaned_summary,
            where=scope or None,
            edited_by="pipeline",
        )
        updated_summaries[cid] = cleaned_summary
    return updated_summaries
