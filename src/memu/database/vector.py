from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from typing import cast

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def salience_score(
    created_at: datetime,
    reflection_salience: float = 0.5,
    emotional_intensity: float = 0.0,
    recency_decay_days: float = 30.0,
) -> float:
    importance = reflection_salience + 0.3 * emotional_intensity
    now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.utcnow()
    days_ago = (now - created_at).total_seconds() / 86400
    effective_half_life = recency_decay_days * (0.5 + reflection_salience)
    recency = math.exp(-0.693 * days_ago / effective_half_life)
    return importance * recency


def cosine_topk(
    query_vec: list[float],
    corpus: Iterable[tuple[str, list[float] | None]],
    k: int = 5,
) -> list[tuple[str, float]]:
    # Filter out None vectors and collect valid entries
    query_dim = len(query_vec)
    ids: list[str] = []
    vecs: list[list[float]] = []
    for _id, vec in corpus:
        if vec is None:
            continue
        vec_list = cast(list[float], vec)
        if len(vec_list) != query_dim:
            continue
        ids.append(_id)
        vecs.append(vec_list)

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
    candidates: list[tuple[str, float, datetime, float, float]],
    recency_decay_days: float = 30.0,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []

    for _id, similarity, created_at, reflection_salience, emotional_intensity in candidates:
        score = similarity + salience_score(
            created_at,
            reflection_salience,
            emotional_intensity,
            recency_decay_days,
        )
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


def query_cosine(query_vec: list[float], vecs: list[list[float]]) -> list[tuple[int, float]]:
    res: list[tuple[int, float]] = []
    q = np.array(query_vec, dtype=np.float32)
    for i, v in enumerate(vecs):
        vec_array = np.array(v, dtype=np.float32)
        res.append((i, _cosine(q, vec_array)))
    res.sort(key=lambda x: x[1], reverse=True)
    return res
