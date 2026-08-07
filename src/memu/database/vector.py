from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import cast

import numpy as np

log = logging.getLogger(__name__)


W_SIMILARITY = 0.5
W_RECENCY = 0.2
W_IMPORTANCE = 0.3


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_score_with_percentiles(value: float, params: Mapping[str, float]) -> float:
    p10 = float(params["p10"])
    p25 = float(params["p25"])
    p50 = float(params["p50"])
    p75 = float(params["p75"])
    p90 = float(params["p90"])
    if value <= p10:
        return 0.1
    if value >= p90:
        return 0.9

    ys_anchors = (0.1, 0.25, 0.5, 0.75, 0.9)
    xs = [p10, p25, p50, p75, p90]
    # For interior values, collapse duplicate x anchors by retaining the highest y
    # so repeated quantiles still interpolate from the last plateau.
    seen: dict[float, float] = {}
    for x, y in zip(xs, ys_anchors, strict=True):
        seen[x] = y
    xs_u, ys_u = zip(*sorted(seen.items()), strict=True)
    return float(np.clip(np.interp(value, xs_u, ys_u), 0.0, 1.0))


def salience_score(
    created_at: datetime,
    reflection_salience: float = 0.5,
    emotional_intensity: float = 0.0,
    recency_decay_days: float = 30.0,
) -> tuple[float, float]:
    """Return (importance, recency) for the tripartite salience formula.

    Park et al. 2023 — each signal independent, combined additively by caller.
    """
    importance = min(reflection_salience + 0.3 * emotional_intensity, 1.0)
    now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.utcnow()
    days_ago = (now - created_at).total_seconds() / 86400
    effective_half_life = recency_decay_days * (0.5 + reflection_salience)
    recency = math.exp(-0.693 * days_ago / effective_half_life)
    return importance, recency


def cosine_topk(
    query_vec: list[float],
    corpus: Iterable[tuple[str, list[float] | None]],
    k: int = 5,
) -> list[tuple[str, float]]:
    # Filter out None vectors and collect valid entries
    query_dim = len(query_vec)
    ids: list[str] = []
    vecs: list[list[float]] = []
    skipped_dim: list[int] = []
    for _id, vec in corpus:
        if vec is None:
            continue
        vec_list = cast(list[float], vec)
        if len(vec_list) != query_dim:
            skipped_dim.append(len(vec_list))
            continue
        ids.append(_id)
        vecs.append(vec_list)

    if skipped_dim:
        log.error(
            "cosine_topk: skipped %d vector(s) with mismatched dimension "
            "(expected %d, found dims: %s)",
            len(skipped_dim),
            query_dim,
            sorted(set(skipped_dim)),
        )

    if not vecs:
        return []

    # Vectorized computation: stack all vectors into a matrix
    q = np.array(query_vec, dtype=np.float32)
    matrix = np.array(vecs, dtype=np.float32)  # shape: (n, dim)

    # Compute all cosine similarities at once
    q_norm = np.linalg.norm(q)
    vec_norms = np.linalg.norm(matrix, axis=1)
    scores = matrix @ q / (vec_norms * q_norm + 1e-9)

    # Use argpartition for O(n) topk selection instead of O(n log n) sort
    n = len(scores)
    actual_k = min(k, n)
    if actual_k == n:
        topk_indices = np.argsort(scores)[::-1]
    else:
        # Get indices of top k elements (unordered), then sort only those
        topk_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]

    return [(ids[i], float(scores[i])) for i in topk_indices]


def rerank_by_salience(
    candidates: list[tuple[str, float, datetime, float | None, float | None, str | None]],
    recency_decay_days: float = 30.0,
    calibrations: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []

    for _id, sim, created_at, reflection_salience, emotional_intensity, model in candidates:
        sal = reflection_salience if reflection_salience is not None else 0.5
        emo = emotional_intensity if emotional_intensity is not None else 0.0
        if calibrations and model:
            if reflection_salience is not None:
                sal_params = calibrations.get((model, "reflection_salience"))
                if sal_params is not None:
                    sal = normalize_score_with_percentiles(sal, sal_params)
            if emotional_intensity is not None:
                emo_params = calibrations.get((model, "emotional_intensity"))
                if emo_params is not None:
                    emo = normalize_score_with_percentiles(emo, emo_params)
        importance, recency = salience_score(created_at, sal, emo, recency_decay_days)
        if reflection_salience is None:
            importance = sim
        score = W_SIMILARITY * sim + W_RECENCY * recency + W_IMPORTANCE * importance
        scored.append((_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked result lists using Reciprocal Rank Fusion (Cormack et al. 2009).

    RRF score for each document = sum over lists of 1/(k + rank_position).
    Scores are normalized to [0, 1] by dividing by the max possible score
    (num_lists / (k + 1)), so a document at rank 1 in all lists scores 1.0.
    """
    if not ranked_lists:
        return []
    num_lists = len(ranked_lists)
    max_score = num_lists / (k + 1)
    scores: dict[str, float] = {}
    for rlist in ranked_lists:
        for rank, (doc_id, _score) in enumerate(rlist):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    # Normalize
    if max_score > 0:
        scores = {doc_id: s / max_score for doc_id, s in scores.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
