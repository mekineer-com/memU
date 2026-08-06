from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from memu.database.models import DossierCandidate


@runtime_checkable
class DossierCandidateRepo(Protocol):
    def add_candidate(
        self,
        *,
        proposed_name: str,
        item_id: str,
        where: Mapping[str, Any],
        segment_id: str | None = None,
        memory_day: str | None = None,
        session: Any | None = None,
    ) -> DossierCandidate: ...

    def list_candidates(
        self,
        where: Mapping[str, Any],
        *,
        unresolved_only: bool = True,
    ) -> list[DossierCandidate]: ...

    def resolve_candidates(
        self,
        candidate_ids: Sequence[str],
        category_id: str,
        where: Mapping[str, Any],
        session: Any | None = None,
    ) -> list[DossierCandidate]: ...


__all__ = ["DossierCandidateRepo"]
