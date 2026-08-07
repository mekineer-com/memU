from __future__ import annotations

import re
import math
from collections.abc import Mapping, Sequence
from typing import Any

from memu.database.models import DossierKind


DOSSIER_KINDS: tuple[DossierKind, ...] = ("lore", "topic", "goal")


def dossier_scope(where: Mapping[str, Any] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in ("user_id", "soul_id"):
        value = str((where or {}).get(field) or "").strip()
        if not value:
            raise ValueError(f"Complete dossier scope required; missing: {field}")
        values[field] = value
    return values


def embedding_vector(values: Sequence[float] | None, *, label: str) -> list[float]:
    try:
        vector = [float(value) for value in values or []]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} embedding is invalid") from exc
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} embedding must contain finite values")
    return vector


def category_identity_text(name: Any, description: Any) -> str:
    title = str(name or "").strip() or "Untitled"
    detail = str(description or "").strip()
    return f"{title}: {detail}" if detail else title


def normalize_category_name(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if not value:
        return None
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or None


__all__ = [
    "DOSSIER_KINDS",
    "category_identity_text",
    "dossier_scope",
    "embedding_vector",
    "normalize_category_name",
]
