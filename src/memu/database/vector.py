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
    reinforcement_count: int,
    last_reinforced_at: datetime | None,
    reflection_salience: float = 0.5,
    recency_decay_days: float = 30.0,
) -> float:
    """
    Compute salience factor from reinforcement, reflection salience, and recency.

    Formula: reinforcement_factor * recency_factor

    - reinforcement_factor: log(count + 1) + reflection_salience
      (Logarithmic scaling prevents runaway dominance by frequently repeated facts)
    - recency_factor: exponential decay based on days since last reinforcement,
      with half-life modulated by reflection_salience

    Args:
        reinforcement_count: Number of times this memory was reinforced
        last_reinforced_at: When the memory was last reinforced
        reflection_salience: Reflection salience weight for the memory
        recency_decay_days: Half-life for recency decay in days

    Returns:
        Salience factor (higher = more salient)
    """
    reinforcement_factor = math.log(reinforcement_count + 1) + reflection_salience

    if last_reinforced_at is None:
        recency_factor = 0.5
    else:
        now = datetime.now(last_reinforced_at.tzinfo) if last_reinforced_at.tzinfo else datetime.utcnow()
        days_ago = (now - last_reinforced_at).total_seconds() / 86400
        effective_half_life = recency_decay_days * (0.5 + reflection_salience)
        recency_factor = math.exp(-0.693 * days_ago / effective_half_life)

    return reinforcement_factor * recency_factor


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
    candidates: list[tuple[str, float, int, datetime | None, float]],
    recency_decay_days: float = 30.0,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []

    for _id, similarity, reinforcement_count, last_reinforced_at, reflection_salience in candidates:
        score = similarity + salience_score(
            reinforcement_count,
            last_reinforced_at,
            reflection_salience,
            recency_decay_days,
        )
        scored.append((_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def query_cosine(query_vec: list[float], vecs: list[list[float]]) -> list[tuple[int, float]]:
    res: list[tuple[int, float]] = []
    q = np.array(query_vec, dtype=np.float32)
    for i, v in enumerate(vecs):
        vec_array = np.array(v, dtype=np.float32)
        res.append((i, _cosine(q, vec_array)))
    res.sort(key=lambda x: x[1], reverse=True)
    return res
