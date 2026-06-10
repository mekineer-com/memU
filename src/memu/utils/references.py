"""
Utilities for handling memory item references in category summaries.

References are inline citations in the format [ref:ITEM_ID] that link
specific statements in category summaries to their source memory items.
"""

from __future__ import annotations

import re

# Pattern to match references like [ref:abc123] or [ref:abc123,def456]
REFERENCE_PATTERN = re.compile(r"\[ref:([a-zA-Z0-9_,\-]+)\]")


def extract_references(text: str | None) -> list[str]:
    """
    Extract all item IDs referenced in a text.

    Example:
        >>> extract_references("User loves coffee [ref:abc123]. Also tea [ref:def456].")
        ['abc123', 'def456']
    """
    if not text:
        return []

    item_ids: list[str] = []
    seen: set[str] = set()

    for match in REFERENCE_PATTERN.finditer(text):
        ids_str = match.group(1)
        for item_id in ids_str.split(","):
            item_id = item_id.strip()
            if item_id and item_id not in seen:
                item_ids.append(item_id)
                seen.add(item_id)

    return item_ids
