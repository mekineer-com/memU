from __future__ import annotations

from typing import Protocol, runtime_checkable

from memu.database.models import MemoryCategory as MemoryCategoryRecord
from memu.database.models import MemoryItem as MemoryItemRecord
from memu.database.models import Resource as ResourceRecord
from memu.database.repositories import (
    CategoryItemRepo,
    DossierCandidateRepo,
    EntityRepo,
    MemoryCategoryRepo,
    MemoryItemRepo,
    ResourceRepo,
    TripleRepo,
)


@runtime_checkable
class Database(Protocol):
    """Backend-agnostic database contract."""

    resource_repo: ResourceRepo
    memory_category_repo: MemoryCategoryRepo
    memory_item_repo: MemoryItemRepo
    category_item_repo: CategoryItemRepo
    dossier_candidate_repo: DossierCandidateRepo
    entity_repo: EntityRepo
    triple_repo: TripleRepo

    resources: dict[str, ResourceRecord]
    categories: dict[str, MemoryCategoryRecord]

    def close(self) -> None: ...


__all__ = [
    "Database",
    "MemoryCategoryRecord",
    "MemoryItemRecord",
    "ResourceRecord",
]
