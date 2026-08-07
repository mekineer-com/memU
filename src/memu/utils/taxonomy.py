from __future__ import annotations

import re
from typing import Any


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


__all__ = ["category_identity_text", "normalize_category_name"]
