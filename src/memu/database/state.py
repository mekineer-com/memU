from __future__ import annotations

from dataclasses import dataclass, field

from memu.database.models import MemoryCategory, Resource


@dataclass
class DatabaseState:
    resources: dict[str, Resource] = field(default_factory=dict)
    categories: dict[str, MemoryCategory] = field(default_factory=dict)


__all__ = ["DatabaseState"]
