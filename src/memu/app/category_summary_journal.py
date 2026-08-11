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


def append_summary_journal(
    *,
    kind: str,
    summary_id: str,
    summary_before: str,
    summary_after: str,
    scope: Mapping[str, Any] | None,
    edited_by: str | None = None,
    category_id: str | None = None,
) -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "kind": kind,
        "summary_id": summary_id,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "edited_by": edited_by,
        "scope": dict(scope or {}),
    }
    if category_id is not None:
        entry["category_id"] = category_id
    path = JOURNAL_DIR / f"{_safe_soul_id(scope)}.summary_journal.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def append_category_summary_journal(
    *,
    category_id: str,
    summary_before: str,
    summary_after: str,
    scope: Mapping[str, Any] | None,
    edited_by: str | None = None,
) -> None:
    append_summary_journal(
        kind="category",
        summary_id=f"category:{category_id}",
        category_id=category_id,
        summary_before=summary_before,
        summary_after=summary_after,
        scope=scope,
        edited_by=edited_by,
    )
