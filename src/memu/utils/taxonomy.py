from __future__ import annotations

import re


def normalize_category_name(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if not value:
        return None
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or None


__all__ = ["normalize_category_name"]
