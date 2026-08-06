from __future__ import annotations

import secrets
from datetime import datetime
from typing import Any, Literal

import pendulum
from pydantic import BaseModel, Field

MemoryType = Literal["profile", "knowledge", "behavior", "social", "episode", "skill", "tool", "narrative_self", "subconscious", "reflection"]
DossierKind = Literal["lore", "topic", "goal"]

class BaseRecord(BaseModel):
    """Backend-agnostic record interface."""

    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    created_at: datetime = Field(default_factory=lambda: pendulum.now("UTC"))
    updated_at: datetime = Field(default_factory=lambda: pendulum.now("UTC"))


class Entity(BaseRecord):
    """A named entity: person, topic, place, project."""
    name: str
    entity_type: str
    normalized: str  # lowercase, underscores -- for fast lookup
    properties: dict[str, Any] = {}


class Triple(BaseRecord):
    """A directed edge between a memory and an entity, or two memories."""
    subject_id: str
    subject_kind: str  # "entity" or "memory"
    predicate: str
    object_id: str
    object_kind: str  # "entity" or "memory"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float | None = None
    source_memory_id: str | None = None
    properties: dict[str, Any] = {}


class Resource(BaseRecord):
    url: str
    modality: str
    local_path: str
    caption: str | None = None
    embedding: list[float] | None = None
    segment_id: str | None = None
    conversation_id: str | None = None
    memory_retrieve_history: list[str] | None = None
    memory_prior_context: list[str] | None = None


class MemoryItem(BaseRecord):
    memory_ref: int | None = None
    resource_id: str | None
    memory_type: str
    summary: str
    embedding: list[float] | None = None
    happened_at: datetime | None = None
    # Provenance fields for retrieval control and auditability.
    # source_role is a coarse speaker channel (for example soul/user/environment).
    source_role: str | None = None
    # Stable speaker identifier + display label (nullable for legacy rows).
    speaker_id: str | None = None
    speaker_label: str | None = None
    confidence: float | None = None
    source_message_ids: list[int] | None = None
    reflection_salience: float | None = None
    emotional_intensity: float | None = None
    # Conversation anchor.
    conversation_id: str | None = None
    segment_id: str | None = None
    unresolved: str | None = None
    # Soft-merge marker for conservative semantic dedupe.
    # When set, this item is treated as merged into another canonical item.
    merged_into: str | None = None
    approved_at: datetime | None = None
    extra: dict[str, Any] = {}
    # # Tool memory fields
    # - when_to_use: str - Hint for when this memory should be retrieved
    # - metadata: dict - Type-specific metadata (e.g., tool_name, avg_success_rate)
    # - tool_calls: list[dict] - Tool call history for tool memories


class MemoryCategory(BaseRecord):
    name: str
    description: str
    embedding: list[float] | None = None
    summary: str | None = None
    previous_summary: str | None = None
    approved_summary: str | None = None
    kind: DossierKind | None = None
    lore_subtype: str | None = None
    entity_id: str | None = None
    anchor_role: Literal["soul", "user"] | None = None
    last_evidence_at: datetime | None = None
    last_revised_at: datetime | None = None


class MemoryRefCounter(BaseRecord):
    counter_key: str = "memory"
    next_value: int


class DossierCandidate(BaseRecord):
    proposed_name: str
    normalized_name: str
    item_id: str
    segment_id: str | None = None
    memory_day: str | None = None
    resolved_category_id: str | None = None
    resolved_at: datetime | None = None


class CategoryItem(BaseRecord):
    item_id: str
    category_id: str


__all__ = [
    "BaseRecord",
    "CategoryItem",
    "DossierCandidate",
    "DossierKind",
    "Entity",
    "MemoryCategory",
    "MemoryItem",
    "MemoryRefCounter",
    "MemoryType",
    "Resource",
    "Triple",
]
