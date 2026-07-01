from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JOURNAL_DIR = Path(__file__).resolve().parents[3] / "journal"


def _safe_soul_id(scope: Mapping[str, Any] | None) -> str:
    raw = str((scope or {}).get("soul_id") or "unknown").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return safe or "unknown"


def append_category_summary_journal(
    *,
    category_id: str,
    summary_before: str,
    summary_after: str,
    scope: Mapping[str, Any] | None,
    edited_by: str | None = None,
) -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "category_id": category_id,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "edited_by": edited_by,
        "scope": dict(scope or {}),
    }
    path = JOURNAL_DIR / f"{_safe_soul_id(scope)}.summary_journal.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def update_category_summary_with_journal(
    store: Any,
    *,
    category_id: str,
    summary: str,
    where: Mapping[str, Any] | None = None,
    edited_by: str | None = None,
) -> Any:
    categories = store.memory_category_repo.list_categories(where)
    current = categories.get(category_id)
    if current is None:
        msg = f"Category with id {category_id} not found"
        raise KeyError(msg)

    clean = str(summary or "").strip()
    if not clean:
        raise ValueError("summary is required")

    before = str(current.summary or "")
    if before.strip() == clean:
        return current

    append_category_summary_journal(
        category_id=category_id,
        summary_before=before,
        summary_after=clean,
        scope=where,
        edited_by=edited_by,
    )
    return store.memory_category_repo.update_category(
        category_id=category_id,
        summary=clean,
        previous_summary=before,
    )
