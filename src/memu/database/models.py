from __future__ import annotations

import secrets
from datetime import datetime
from typing import Any, Literal

import pendulum
from pydantic import BaseModel, ConfigDict, Field

MemoryType = Literal["profile", "knowledge", "behavior", "social", "episode", "skill", "tool", "narrative_self", "subconscious", "reflection"]

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


class CategoryItem(BaseRecord):
    item_id: str
    category_id: str


def merge_scope_model[TBaseRecord: BaseRecord](
    user_model: type[BaseModel], core_model: type[TBaseRecord], *, name_suffix: str
) -> type[TBaseRecord]:
    """Create a scoped model inheriting both the user scope model and the core model."""
    overlap = set(user_model.model_fields) & set(core_model.model_fields)
    if overlap:
        msg = f"Scope fields conflict with core model fields: {sorted(overlap)}"
        raise TypeError(msg)

    return type(
        f"{user_model.__name__}{core_model.__name__}{name_suffix}",
        (user_model, core_model),
        {"model_config": ConfigDict(extra="allow")},
    )


def build_scoped_models(
    user_model: type[BaseModel],
) -> tuple[type[Resource], type[MemoryCategory], type[MemoryItem], type[CategoryItem], type[Entity], type[Triple]]:
    """
    Build scoped interface models (Pydantic) that inherit from the base record models and user scope.
    """
    resource_model = merge_scope_model(user_model, Resource, name_suffix="Resource")
    memory_category_model = merge_scope_model(user_model, MemoryCategory, name_suffix="MemoryCategory")
    memory_item_model = merge_scope_model(user_model, MemoryItem, name_suffix="MemoryItem")
    category_item_model = merge_scope_model(user_model, CategoryItem, name_suffix="CategoryItem")
    entity_model = merge_scope_model(user_model, Entity, name_suffix="Entity")
    triple_model = merge_scope_model(user_model, Triple, name_suffix="Triple")
    return resource_model, memory_category_model, memory_item_model, category_item_model, entity_model, triple_model


__all__ = [
    "BaseRecord",
    "CategoryItem",
    "Entity",
    "MemoryCategory",
    "MemoryItem",
    "MemoryType",
    "Resource",
    "Triple",
    "build_scoped_models",
    "merge_scope_model",
]
