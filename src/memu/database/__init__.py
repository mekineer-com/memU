"""Lightweight database interfaces and record aliases."""

from memu.database.interfaces import (
    Database,
    MemoryCategoryRecord,
    MemoryItemRecord,
    ResourceRecord,
)
from memu.database.repositories import (
    CategoryItemRepo,
    EntityRepo,
    MemoryCategoryRepo,
    MemoryItemRepo,
    ResourceRepo,
    TripleRepo,
)

__all__ = [
    "CategoryItemRepo",
    "Database",
    "EntityRepo",
    "MemoryCategoryRecord",
    "MemoryCategoryRepo",
    "MemoryItemRecord",
    "MemoryItemRepo",
    "ResourceRecord",
    "ResourceRepo",
    "TripleRepo",
]
