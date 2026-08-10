from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, cast

from memu.app.category_summary_journal import append_category_summary_journal
from memu.app.dossier_revision import (
    label_sections,
    parse_dossier_revision,
    render_memory_records,
    revision_status_items,
    strip_memory_citations,
)
from memu.database.models import CategoryItem, DossierKind, MemoryCategory, MemoryItem
from memu.database.vector import cosine_topk, reciprocal_rank_fusion
from memu.prompts.dossier_revision import NARRATIVE_SELF_BLOCK, SYSTEM_PROMPT, USER_PROMPT
from memu.utils.taxonomy import (
    DOSSIER_KINDS,
    category_identity_text,
    dossier_scope as _scope,
    embedding_vector as _embedding_vector,
)

if TYPE_CHECKING:
    from memu.app.settings import MemorizeConfig, RetrieveConfig
    from memu.database.interfaces import Database

DOSSIER_INDEX_LIMIT = 20
MEMORY_REF_PATTERN = re.compile(r"^\[M([1-9][0-9]*)\]$")
MEMORY_REF_SCAN_PATTERN = re.compile(r"\[M[0-9][^\]\r\n]*\]")
AnchorRole = Literal["soul", "user"]
logger = logging.getLogger(__name__)


class DossierRevisionStaleError(RuntimeError):
    pass


def render_dossier_revision_prompts(bundle: Mapping[str, Any]) -> tuple[str, str]:
    dossier = bundle["dossier"]
    sectioned = label_sections(str(dossier.summary or ""))
    current_prose = sectioned[0] if sectioned is not None else str(dossier.summary or "")
    statuses = revision_status_items(bundle)
    active_goals = list(bundle["active_life_goals"])
    removed_goals = list(bundle["removed_life_goals"])
    if active_goals or removed_goals:
        goal_context = "\n".join(
            [
                "Active:",
                *([f"- {goal}" for goal in active_goals] or ["(none)"]),
                "Removed:",
                *([f"- {goal}" for goal in removed_goals] or ["(none)"]),
            ]
        )
    else:
        goal_context = "(none)"

    system_prompt = SYSTEM_PROMPT.format(
        target_words=bundle["target_words"],
        dossier_id=dossier.id,
        soul_name=bundle["soul_name"],
        user_name=bundle["user_name"],
    )
    if bundle["narrative_self"]:
        system_prompt += "\n\n" + NARRATIVE_SELF_BLOCK.format(
            narrative_self=bundle["narrative_self"]
        )
    user_prompt = USER_PROMPT.format(
        soul_presence=bundle["soul_presence"],
        dossier_index=bundle["dossier_index"] or "(none)",
        goal_context_or_none=goal_context,
        dossier_id=dossier.id,
        dossier_kind=dossier.kind,
        dossier_title=dossier.name,
        dossier_description=dossier.description,
        current_prose=current_prose or "(none)",
        cited_memory_records=render_memory_records(statuses["cited"]),
        candidate_memory_records=render_memory_records(statuses["search"]),
        cleanup_memberships=render_memory_records(statuses["purged"]),
        required_memory_records=render_memory_records(statuses["pending"]),
    )
    return system_prompt, user_prompt


def _activity_key(category: MemoryCategory) -> tuple[int, float, str, str]:
    happened = category.last_evidence_at
    if happened is None:
        return (1, 0.0, category.name.casefold(), category.id)
    if happened.tzinfo is None:
        happened = happened.replace(tzinfo=UTC)
    return (0, -happened.timestamp(), category.name.casefold(), category.id)


def _timestamp(value: datetime) -> float:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).timestamp()


def _item_time(item: MemoryItem) -> datetime:
    return item.happened_at or item.created_at


def _item_sort_key(item: MemoryItem) -> tuple[float, int, str]:
    return (_timestamp(_item_time(item)), int(item.memory_ref or 0), item.id)


def _load_linked_items(
    store: Database,
    relations: Sequence[CategoryItem],
    scope: Mapping[str, str],
    *,
    session: Any | None = None,
) -> tuple[dict[str, MemoryItem], set[str]]:
    item_ids = {relation.item_id for relation in relations}
    if not item_ids:
        return {}, set()
    items = store.memory_item_repo.list_items_by_ids(
        item_ids,
        scope,
        include_superseded=True,
        include_merged=True,
        session=session,
    )
    missing = item_ids - items.keys()
    if missing:
        raise KeyError(f"Dossier memberships reference missing memories: {sorted(missing)}")
    active_ids = store.memory_item_repo.list_items_by_ids(item_ids, scope, session=session).keys()
    return items, item_ids - active_ids


def _content_text(category: MemoryCategory) -> str:
    identity = category_identity_text(category.name, category.description)
    summary = str(category.summary or "").strip()
    return f"{identity}\n{summary}" if summary else identity


def _render_index_field(value: str) -> str:
    return " ".join(value.split())


def _render_dossier_index_line(category: MemoryCategory) -> str:
    name = _render_index_field(category.name)
    description = _render_index_field(category.description)
    return f"- {name}: {description}" if description else f"- {name}"


def _validate_anchors(
    anchors: Mapping[str, MemoryCategory], scope: Mapping[str, str]
) -> None:
    expected: dict[AnchorRole, str] = {"soul": scope["soul_id"], "user": scope["user_id"]}
    for role, category in anchors.items():
        if (
            role not in expected
            or category.name != expected[cast(AnchorRole, role)]
            or category.kind != "lore"
        ):
            raise ValueError(f"Invalid {role} dossier anchor for scope")


class DossierMixin:
    if TYPE_CHECKING:
        memorize_config: MemorizeConfig
        retrieve_config: RetrieveConfig
        _dossier_content_embedding_cache: dict[str, tuple[str, list[float]]]
        _get_database: Callable[[], Database]
        _select_chat_client: Callable[..., Any]
        _select_embedding_client: Callable[[Mapping[str, Any] | None], Any]
        _sqlite_write_session: Callable[[Database], Any | None]

    async def ensure_dossier_anchors(
        self,
        where: Mapping[str, Any],
        *,
        embedding_client: Any | None = None,
    ) -> dict[str, MemoryCategory]:
        scope = _scope(where)
        if scope["user_id"].casefold() == scope["soul_id"].casefold():
            raise ValueError("Dossier soul and user identities must differ")

        store = self._get_database()
        categories = store.memory_category_repo.list_categories(scope)
        anchors = {
            cast(str, category.anchor_role): category
            for category in categories.values()
            if category.anchor_role is not None
        }
        _validate_anchors(anchors, scope)
        expected: dict[AnchorRole, str] = {"soul": scope["soul_id"], "user": scope["user_id"]}

        missing = [(role, name) for role, name in expected.items() if role not in anchors]
        for role, name in missing:
            collision = next((cat for cat in categories.values() if cat.name.casefold() == name.casefold()), None)
            if collision is not None:
                raise ValueError(f"Cannot seed {role} dossier anchor; title already exists in scope: {name}")

        if not missing:
            return anchors

        descriptions = {
            role: f"Identity, history, relationships, and lived experience of {name}."
            for role, name in missing
        }
        texts = [category_identity_text(name, descriptions[role]) for role, name in missing]
        client = embedding_client or self._select_embedding_client(
            {"operation": "dossier", "step_id": "seed_anchors"}
        )
        embeddings = await client.embed(texts)
        if len(embeddings) != len(missing):
            raise ValueError("Anchor embedding response count does not match input")

        for (role, name), raw_embedding in zip(missing, embeddings, strict=True):
            embedding = _embedding_vector(raw_embedding, label=f"{role} anchor")
            anchor = store.memory_category_repo.get_or_create_category(
                name=name,
                description=descriptions[role],
                embedding=embedding,
                user_data=scope,
                kind="lore",
                lore_subtype="person",
                anchor_role=role,
            )
            if anchor.anchor_role != role or anchor.kind != "lore":
                raise ValueError(f"Cannot seed {role} dossier anchor; title was claimed concurrently")
            anchors[role] = anchor
        return anchors

    async def update_dossier(
        self,
        category_id: str,
        where: Mapping[str, Any],
        *,
        name: str | None = None,
        description: str | None = None,
        kind: DossierKind | None | EllipsisType = ...,
        lore_subtype: str | None | EllipsisType = ...,
        entity_id: str | None | EllipsisType = ...,
        last_evidence_at: datetime | None | EllipsisType = ...,
        last_revised_at: datetime | None | EllipsisType = ...,
        embedding_client: Any | None = None,
    ) -> MemoryCategory:
        scope = _scope(where)
        store = self._get_database()
        current = store.memory_category_repo.list_categories(scope).get(category_id)
        if current is None:
            raise KeyError(f"Dossier with id {category_id} not found in scope")

        final_name = current.name if name is None else name.strip()
        if not final_name:
            raise ValueError("Dossier title is required")
        final_description = current.description if description is None else description.strip()
        final_kind = current.kind if kind is ... else kind
        if final_kind is not None and final_kind not in DOSSIER_KINDS:
            raise ValueError(f"Invalid dossier kind: {final_kind}")
        if current.anchor_role is not None:
            if final_name != current.name:
                raise ValueError("Dossier anchors cannot be renamed")
            if final_kind != "lore":
                raise ValueError("Dossier anchors must remain Lore dossiers")

        final_identity = category_identity_text(final_name, final_description)
        embedding = None
        if category_identity_text(current.name, current.description) != final_identity:
            client = embedding_client or self._select_embedding_client(
                {"operation": "dossier", "step_id": "update_identity"}
            )
            raw_embeddings = await client.embed([final_identity])
            if len(raw_embeddings) != 1:
                raise ValueError("Dossier embedding response count does not match input")
            embedding = _embedding_vector(raw_embeddings[0], label="dossier identity")

        return store.memory_category_repo.update_category(
            category_id=category_id,
            name=final_name if name is not None else None,
            description=final_description if description is not None else None,
            previous_description=(
                current.description
                if description is not None and final_description != current.description
                else None
            ),
            embedding=embedding,
            kind=kind,
            lore_subtype=lore_subtype,
            entity_id=entity_id,
            last_evidence_at=last_evidence_at,
            last_revised_at=last_revised_at,
        )

    def list_due_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        categories = store.memory_category_repo.list_categories(scope)
        relations = store.category_item_repo.list_relations(scope)
        _linked_items, inactive_ids = _load_linked_items(store, relations, scope)

        actionable: list[tuple[float, MemoryCategory]] = []
        for category in categories.values():
            if category.kind not in DOSSIER_KINDS or category.anchor_role is not None:
                continue
            category_relations = [
                relation for relation in relations if relation.category_id == category.id
            ]
            if not category_relations:
                continue
            if category.last_revised_at is None:
                due_relations = category_relations
            else:
                revised_at = _timestamp(category.last_revised_at)
                due_relations = [
                    relation
                    for relation in category_relations
                    if _timestamp(relation.created_at) > revised_at
                    or relation.item_id in inactive_ids
                ]
            if due_relations:
                actionable.append(
                    (min(_timestamp(relation.created_at) for relation in due_relations), category)
                )

        actionable.sort(
            key=lambda row: (
                row[0],
                str(row[1].kind),
                row[1].name.casefold(),
                row[1].id,
            )
        )
        return [category for _timestamp_value, category in actionable]

    @staticmethod
    def extract_memory_refs(text: str) -> list[int]:
        return list(
            dict.fromkeys(
                DossierMixin.parse_memory_ref(token)
                for token in MEMORY_REF_SCAN_PATTERN.findall(text or "")
            )
        )

    def prepare_dossier_revision(
        self,
        category_id: str,
        where: Mapping[str, Any],
        *,
        narrative_self: str | None = None,
        active_life_goals: Sequence[str] = (),
        removed_life_goals: Sequence[str] = (),
    ) -> dict[str, Any]:
        scope = _scope(where)
        store = self._get_database()
        category = store.memory_category_repo.list_categories(scope).get(category_id)
        if category is None or category.kind not in DOSSIER_KINDS:
            raise KeyError(f"Dossier with id {category_id} not found in scope")

        relations = [
            relation
            for relation in store.category_item_repo.list_relations(scope)
            if relation.category_id == category.id
        ]
        linked_items, linked_inactive_ids = _load_linked_items(store, relations, scope)
        revised_at = None if category.last_revised_at is None else _timestamp(category.last_revised_at)
        pending_relations = [
            relation
            for relation in relations
            if revised_at is None or _timestamp(relation.created_at) > revised_at
        ]
        pending_ids = {relation.item_id for relation in pending_relations}
        if not pending_ids and not linked_inactive_ids:
            raise ValueError(f"Dossier {category_id} is not due for revision")

        anchors = store.memory_category_repo.list_anchor_categories(scope)
        _validate_anchors(anchors, scope)
        if set(anchors) != {"soul", "user"}:
            raise ValueError("Dossier revision requires seeded soul and user anchors")

        cited_items: dict[str, MemoryItem] = {}
        for memory_ref in self.extract_memory_refs(category.summary or ""):
            item = store.memory_item_repo.get_item_by_memory_ref(memory_ref, scope)
            if item is None:
                raise KeyError(f"Memory reference [M{memory_ref}] not found in scope")
            cited_items[item.id] = item

        top_k = max(0, int(self.retrieve_config.item.top_k))
        candidate_items: dict[str, MemoryItem] = {}
        if top_k:
            query = _embedding_vector(category.embedding, label=f"dossier {category.id} identity")
            excluded_ids = set(linked_items) | set(cited_items)
            hits = store.memory_item_repo.vector_search_items(
                query,
                top_k + len(excluded_ids),
                scope,
                ranking="similarity",
            )
            candidate_ids = [item_id for item_id, _score in hits if item_id not in excluded_ids][
                :top_k
            ]
            candidate_items = store.memory_item_repo.list_items_by_ids(set(candidate_ids), scope)
            missing = set(candidate_ids) - candidate_items.keys()
            if missing:
                raise KeyError(f"Dossier candidate memories disappeared: {sorted(missing)}")

        prompt_items = {
            item_id: item
            for item_id, item in linked_items.items()
            if item_id in pending_ids
        }
        prompt_items.update(cited_items)
        prompt_items.update(candidate_items)
        all_shown_items = {
            **prompt_items,
            **{item_id: linked_items[item_id] for item_id in linked_inactive_ids},
        }
        missing_refs = sorted(
            item_id
            for item_id, item in all_shown_items.items()
            if item.memory_ref is None
        )
        if missing_refs:
            raise ValueError(f"Dossier revision memories lack stable references: {missing_refs}")

        active_shown_ids = store.memory_item_repo.list_items_by_ids(set(all_shown_items), scope).keys()
        shown_inactive_ids = set(all_shown_items) - active_shown_ids
        cleanup = [
            {
                "item_id": item_id,
                "memory_ref": linked_items[item_id].memory_ref,
                "lineage_state": "merged" if linked_items[item_id].merged_into else "superseded",
                "merged_into": linked_items[item_id].merged_into,
            }
            for item_id in sorted(linked_inactive_ids)
        ]
        return {
            "dossier": category,
            "category_updated_at": category.updated_at,
            "relation_tokens": sorted(
                (relation.id, relation.created_at, relation.updated_at) for relation in relations
            ),
            "linked_item_ids": sorted(linked_items),
            "linked_inactive_item_ids": sorted(linked_inactive_ids),
            "cited_unlinked_item_ids": sorted(set(cited_items) - set(linked_items)),
            "shown_inactive_item_ids": sorted(shown_inactive_ids),
            "shown_item_tokens": sorted(
                (
                    item.id,
                    item.updated_at,
                    item.memory_ref,
                    item.merged_into,
                )
                for item in all_shown_items.values()
            ),
            "cited_items": sorted(cited_items.values(), key=_item_sort_key),
            "pending_items": sorted(
                (linked_items[item_id] for item_id in pending_ids), key=_item_sort_key
            ),
            "cleanup_memberships": cleanup,
            "cleanup_items": sorted(
                (linked_items[item_id] for item_id in linked_inactive_ids), key=_item_sort_key
            ),
            "candidate_items": sorted(candidate_items.values(), key=_item_sort_key),
            "untouched_item_ids": sorted(
                set(linked_items) - pending_ids - set(cited_items) - linked_inactive_ids
            ),
            "active_life_goals": [
                goal.strip() for goal in active_life_goals if category.kind == "goal" and goal.strip()
            ],
            "removed_life_goals": [
                goal.strip() for goal in removed_life_goals if category.kind == "goal" and goal.strip()
            ],
            "soul_name": scope["soul_id"],
            "user_name": scope["user_id"],
            "narrative_self": narrative_self.strip() if narrative_self and narrative_self.strip() else None,
            "dossier_index": self.build_dossier_index(scope),
            "soul_presence": "\n\n".join(
                block
                for role, block in (
                    (
                        "soul",
                        "# Your dossier\n"
                        + strip_memory_citations(anchors["soul"].summary or anchors["soul"].description),
                    ),
                    (
                        "user",
                        "# Your human's dossier\n"
                        + strip_memory_citations(anchors["user"].summary or anchors["user"].description),
                    ),
                )
                if category.anchor_role != role
            ),
            "target_words": (
                500
                if category.anchor_role is not None
                else int(self.memorize_config.category_summary_target_words)
            ),
        }

    def prepare_anchor_revision(
        self,
        role: AnchorRole,
        where: Mapping[str, Any],
        actionable_item_ids: Sequence[str],
    ) -> dict[str, Any]:
        if role not in {"soul", "user"}:
            raise ValueError(f"Invalid anchor role: {role}")
        scope = _scope(where)
        store = self._get_database()
        anchors = store.memory_category_repo.list_anchor_categories(scope)
        _validate_anchors(anchors, scope)
        if set(anchors) != {"soul", "user"}:
            raise ValueError("Anchor revision requires seeded soul and user anchors")
        anchor = anchors[role]

        relations = [
            relation
            for relation in store.category_item_repo.list_relations(scope)
            if relation.category_id == anchor.id
        ]
        linked_items, linked_inactive_ids = _load_linked_items(store, relations, scope)
        actionable_ids = set(actionable_item_ids)
        if len(actionable_ids) != len(actionable_item_ids):
            raise ValueError("Anchor revision evidence contains duplicate memories")
        active_actionable = store.memory_item_repo.list_items_by_ids(actionable_ids, scope)
        if set(active_actionable) != actionable_ids:
            raise ValueError("Anchor revision evidence contains inactive or wrong-scope memories")

        cited_items: dict[str, MemoryItem] = {}
        for memory_ref in self.extract_memory_refs(anchor.summary or ""):
            item = store.memory_item_repo.get_item_by_memory_ref(memory_ref, scope)
            if item is None or item.id not in linked_items:
                raise ValueError(f"Anchor citation [M{memory_ref}] is not an anchor membership")
            cited_items[item.id] = item
        shown_items = {
            **cited_items,
            **active_actionable,
            **{item_id: linked_items[item_id] for item_id in linked_inactive_ids},
        }
        missing_refs = sorted(item.id for item in shown_items.values() if item.memory_ref is None)
        if missing_refs:
            raise ValueError(f"Anchor revision memories lack stable references: {missing_refs}")

        return {
            "dossier": anchor,
            "category_updated_at": anchor.updated_at,
            "relation_tokens": sorted(
                (relation.id, relation.created_at, relation.updated_at) for relation in relations
            ),
            "linked_item_ids": sorted(linked_items),
            "linked_inactive_item_ids": sorted(linked_inactive_ids),
            "cited_unlinked_item_ids": [],
            "shown_inactive_item_ids": sorted(linked_inactive_ids),
            "shown_item_tokens": sorted(
                (item.id, item.updated_at, item.memory_ref, item.merged_into)
                for item in shown_items.values()
            ),
            "cited_items": sorted(cited_items.values(), key=_item_sort_key),
            "pending_items": [],
            "cleanup_items": sorted(
                (linked_items[item_id] for item_id in linked_inactive_ids),
                key=_item_sort_key,
            ),
            "candidate_items": sorted(active_actionable.values(), key=_item_sort_key),
            "actionable_item_ids": sorted(actionable_ids),
        }

    async def generate_dossier_revision(
        self,
        bundle: Mapping[str, Any],
        *,
        chat_client: Any | None = None,
    ) -> dict[str, Any]:
        system_prompt, user_prompt = render_dossier_revision_prompts(bundle)
        if len((system_prompt + "\n" + user_prompt).split()) / 0.75 > 100_000:
            raise ValueError("Dossier revision prompt exceeds 100000 tokens")

        client = chat_client or self._select_chat_client(
            {"operation": "dossier", "step_id": "revision"},
            profile=self.memorize_config.category_update_llm_profile,
        )
        raw = await client.chat(user_prompt, system_prompt=system_prompt)
        return parse_dossier_revision(str(raw or ""), bundle)

    async def apply_dossier_revision(
        self,
        bundle: Mapping[str, Any],
        decision: Mapping[str, Any],
        where: Mapping[str, Any],
        *,
        embedding_client: Any | None = None,
        _allow_empty_text: bool = False,
        _journal_actor: str = "dossier_revision",
    ) -> MemoryCategory:
        scope = _scope(where)
        dossier = bundle["dossier"]
        if not isinstance(dossier, MemoryCategory):
            raise ValueError("Dossier revision bundle has no dossier")
        if decision.get("dossier_id") != dossier.id:
            raise ValueError("Dossier revision id does not match bundle")

        description = decision.get("description")
        prose = decision.get("resulting_prose")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Dossier revision description is required")
        if not isinstance(prose, str):
            raise ValueError("Dossier revision prose is required")
        description = description.strip()

        def decision_ids(key: str) -> set[str]:
            raw = decision.get(key)
            if not isinstance(raw, list) or any(not isinstance(value, str) or not value for value in raw):
                raise ValueError(f"Invalid dossier revision {key}")
            values = set(raw)
            if len(values) != len(raw):
                raise ValueError(f"Duplicate dossier revision {key}")
            return values

        add_ids = decision_ids("add_item_ids")
        remove_ids = decision_ids("remove_item_ids")
        cleanup_ids = decision_ids("cleanup_item_ids")
        cited_ids = decision_ids("cited_item_ids")
        snapshot_linked_ids = set(bundle["linked_item_ids"])
        provisional_members = (snapshot_linked_ids | add_ids) - remove_ids - cleanup_ids

        embedding: list[float] | None = None
        if (provisional_members or _allow_empty_text) and description != dossier.description:
            client = embedding_client or self._select_embedding_client(
                {"operation": "dossier", "step_id": "update_identity"}
            )
            raw_embeddings = await client.embed([category_identity_text(dossier.name, description)])
            if len(raw_embeddings) != 1:
                raise ValueError("Dossier embedding response count does not match input")
            embedding = _embedding_vector(raw_embeddings[0], label="dossier identity")

        store = self._get_database()
        session_cm = self._sqlite_write_session(store)
        if session_cm is None:
            raise RuntimeError("Dossier revision apply requires SQLite")

        committed: MemoryCategory | None = None
        prose_before = str(dossier.summary or "")
        prose_changed = False
        with session_cm as session:
            current = store.memory_category_repo.list_categories(scope, session=session).get(dossier.id)
            if current is None or current.kind not in DOSSIER_KINDS:
                raise DossierRevisionStaleError("Dossier revision target changed")
            relations = [
                relation
                for relation in store.category_item_repo.list_relations(scope, session=session)
                if relation.category_id == current.id
            ]
            linked_items, linked_inactive_ids = _load_linked_items(
                store,
                relations,
                scope,
                session=session,
            )

            shown_tokens = list(bundle["shown_item_tokens"])
            shown_ids = {token[0] for token in shown_tokens}
            shown_items = store.memory_item_repo.list_items_by_ids(
                shown_ids,
                scope,
                include_superseded=True,
                include_merged=True,
                session=session,
            )
            active_shown_ids = set(
                store.memory_item_repo.list_items_by_ids(shown_ids, scope, session=session)
            )
            current_shown_tokens = sorted(
                (item.id, item.updated_at, item.memory_ref, item.merged_into)
                for item in shown_items.values()
            )
            current_relation_tokens = sorted(
                (relation.id, relation.created_at, relation.updated_at) for relation in relations
            )
            stale = (
                current.updated_at != bundle["category_updated_at"]
                or current_relation_tokens != list(bundle["relation_tokens"])
                or linked_inactive_ids != set(bundle["linked_inactive_item_ids"])
                or current_shown_tokens != shown_tokens
                or shown_ids - active_shown_ids != set(bundle["shown_inactive_item_ids"])
            )
            if stale:
                raise DossierRevisionStaleError("Dossier revision snapshot changed")

            if cleanup_ids != linked_inactive_ids:
                raise ValueError("Dossier cleanup decisions do not match inactive memberships")
            linked_active_ids = set(linked_items) - linked_inactive_ids
            if not remove_ids <= (shown_ids & linked_active_ids):
                raise ValueError("Dossier removal includes an unshown or inactive membership")
            allowed_add_ids = {
                item.id for item in bundle["candidate_items"]
            } | set(bundle["cited_unlinked_item_ids"]) | (shown_ids & linked_active_ids)
            if not add_ids <= allowed_add_ids:
                raise ValueError("Dossier addition includes an unshown candidate")
            if add_ids & remove_ids:
                raise ValueError("Dossier memory cannot be both added and removed")

            resulting_members = (set(linked_items) | add_ids) - remove_ids - cleanup_ids
            shown_by_ref = {
                item.memory_ref: item.id
                for item in shown_items.values()
                if item.memory_ref is not None
            }
            try:
                prose_cited_ids = {
                    shown_by_ref[memory_ref] for memory_ref in self.extract_memory_refs(prose)
                }
            except KeyError as exc:
                raise ValueError("Dossier prose cites a memory outside review context") from exc
            if prose_cited_ids != cited_ids:
                raise ValueError("Dossier citations changed after generation")
            if not cited_ids <= resulting_members:
                raise ValueError("Dossier prose cites a removed or unlinked memory")
            active_members = store.memory_item_repo.list_items_by_ids(
                resulting_members,
                scope,
                session=session,
            )
            if set(active_members) != resulting_members:
                raise ValueError("Dossier result contains an inactive or wrong-scope memory")
            if resulting_members and not prose.strip():
                raise ValueError("A dossier with members requires nonblank prose")

            for item_id in remove_ids | cleanup_ids:
                store.category_item_repo.unlink_item_category(
                    item_id,
                    current.id,
                    scope,
                    session=session,
                )
            for item_id in add_ids:
                store.category_item_repo.link_item_category(
                    item_id,
                    current.id,
                    scope,
                    session=session,
                )

            reviewed_member_ids = (
                {
                    item.id
                    for item in (*bundle["pending_items"], *bundle["cited_items"])
                }
                & resulting_members
            ) | add_ids
            for item_id in reviewed_member_ids:
                store.memory_item_repo.approve_item(item_id, scope, session=session)

            final_items = store.memory_item_repo.list_items_by_ids(
                resulting_members,
                scope,
                session=session,
            )
            last_evidence_at = (
                max((_item_time(item) for item in final_items.values()), default=None)
            )
            completed_at = datetime.now(UTC)
            description_changed = (bool(resulting_members) or _allow_empty_text) and description != current.description
            prose_changed = (bool(resulting_members) or _allow_empty_text) and prose != str(current.summary or "")
            update: dict[str, Any] = {
                "category_id": current.id,
                "last_evidence_at": last_evidence_at,
                "last_revised_at": completed_at,
                "where": scope,
                "session": session,
            }
            if description_changed:
                update.update(
                    description=description,
                    previous_description=current.description,
                    embedding=embedding,
                )
            if prose_changed:
                update.update(summary=prose)
                if current.summary is not None:
                    update["previous_summary"] = current.summary
            committed = store.memory_category_repo.update_category(**update)
            session.commit()

        if committed is None:
            raise RuntimeError("Dossier revision did not commit")

        fresh = committed
        try:
            fresh = store.memory_category_repo.list_categories(scope)[committed.id]
            store.category_item_repo.refresh_category_relations(committed.id, scope)
        except Exception:
            logger.exception("Failed to refresh dossier revision caches for %s", committed.id)
        if prose_changed:
            try:
                append_category_summary_journal(
                    category_id=committed.id,
                    summary_before=prose_before,
                    summary_after=str(committed.summary or ""),
                    scope=scope,
                    edited_by=_journal_actor,
                )
            except Exception:
                logger.exception("Failed to journal committed dossier revision %s", committed.id)
        return fresh

    async def apply_anchor_revision(
        self,
        bundle: Mapping[str, Any],
        decision: Mapping[str, Any],
        where: Mapping[str, Any],
        *,
        embedding_client: Any | None = None,
    ) -> MemoryCategory:
        anchor = bundle.get("dossier")
        if not isinstance(anchor, MemoryCategory) or anchor.anchor_role not in {"soul", "user"}:
            raise ValueError("Anchor revision bundle has no dossier anchor")
        if decision.get("anchor_role") != anchor.anchor_role:
            raise ValueError("Anchor revision role does not match bundle")
        return await self.apply_dossier_revision(
            bundle,
            decision,
            where,
            embedding_client=embedding_client,
            _allow_empty_text=True,
            _journal_actor="anchor_revision",
        )

    def list_active_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        anchors = store.memory_category_repo.list_anchor_categories(scope)
        _validate_anchors(anchors, scope)
        active = list(anchors.values())

        limit = max(0, int(self.memorize_config.active_dossiers_per_kind))
        for kind in DOSSIER_KINDS:
            evidenced = [
                category
                for category in store.memory_category_repo.list_categories_by_activity(scope, kind=kind)
                if category.last_evidence_at is not None
            ]
            active.extend(evidenced[:limit])
        return sorted(active, key=_activity_key)

    def list_inactive_dossiers(self, where: Mapping[str, Any]) -> list[MemoryCategory]:
        scope = _scope(where)
        store = self._get_database()
        active_ids = {category.id for category in self.list_active_dossiers(scope)}
        inactive = [
            category
            for category in store.memory_category_repo.list_categories(scope).values()
            if category.kind in DOSSIER_KINDS and category.anchor_role is None and category.id not in active_ids
        ]
        return sorted(inactive, key=_activity_key)

    def list_dossiers_for_segments(
        self,
        where: Mapping[str, Any],
        *,
        segment_ids: Sequence[str],
    ) -> list[MemoryCategory]:
        scope = _scope(where)
        selected_ids = set(segment_ids)
        if not selected_ids:
            return []

        store = self._get_database()
        categories = store.memory_category_repo.list_categories(scope)
        relations = store.category_item_repo.list_relations(scope)
        items = store.memory_item_repo.list_items_by_ids(
            {relation.item_id for relation in relations},
            scope,
        )
        newest_by_category: dict[str, float] = {}
        for relation in relations:
            item = items.get(relation.item_id)
            if item is None or item.segment_id not in selected_ids:
                continue
            newest_by_category[relation.category_id] = max(
                newest_by_category.get(relation.category_id, float("-inf")),
                _timestamp(item.created_at),
            )

        relevant = [
            categories[category_id]
            for category_id in newest_by_category
            if category_id in categories and categories[category_id].anchor_role is None
        ]
        return sorted(
            relevant,
            key=lambda category: (
                -newest_by_category[category.id],
                str(category.kind),
                category.name.casefold(),
                category.id,
            ),
        )

    def build_dossier_index(
        self,
        where: Mapping[str, Any],
        *,
        dossiers: Sequence[MemoryCategory] | None = None,
    ) -> str:
        rows = list(dossiers) if dossiers is not None else self.list_active_dossiers(where)
        rows = [
            category
            for category in rows
            if category.anchor_role is None
        ][:DOSSIER_INDEX_LIMIT]
        return "\n".join(_render_dossier_index_line(category) for category in rows)

    def require_dossier_cutover_ready(self, where: Mapping[str, Any]) -> None:
        scope = _scope(where)
        store = self._get_database()
        memories = store.memory_item_repo.list_items(scope, include_embeddings=False)
        categories = store.memory_category_repo.list_categories(scope)
        relations = store.category_item_repo.list_relations(scope)
        if not memories and not categories and not relations:
            return

        refs = [item.memory_ref for item in memories.values()]
        if any(isinstance(ref, bool) or not isinstance(ref, int) or ref < 1 for ref in refs):
            raise ValueError("Dossier cutover requires a positive [M#] reference on every active memory")
        if len(refs) != len(set(refs)):
            raise ValueError("Dossier cutover requires unique [M#] references within the scope")

        invalid_kind = next((category for category in categories.values() if category.kind not in DOSSIER_KINDS), None)
        if invalid_kind is not None:
            raise ValueError(f"Dossier cutover found invalid kind on category: {invalid_kind.name}")
        anchor_rows = [category for category in categories.values() if category.anchor_role is not None]
        if sorted(category.anchor_role for category in anchor_rows) != ["soul", "user"]:
            raise ValueError("Dossier cutover requires exactly one soul anchor and one user anchor")
        anchors = {cast(str, category.anchor_role): category for category in anchor_rows}
        _validate_anchors(anchors, scope)

        ordinary = [category for category in categories.values() if category.anchor_role is None]
        unapproved = next(
            (
                category
                for category in ordinary
                if not str(category.approved_description or "").strip()
            ),
            None,
        )
        if unapproved is not None:
            raise ValueError(f"Dossier cutover requires an approved description for: {unapproved.name}")

        links_by_category: dict[str, set[str]] = {}
        related_items = store.memory_item_repo.list_items_by_ids(
            {relation.item_id for relation in relations},
            scope,
            include_superseded=True,
            include_merged=True,
        )
        for relation in relations:
            if relation.category_id not in categories or relation.item_id not in related_items:
                raise ValueError(f"Dossier cutover found dangling or cross-scope relation: {relation.id}")
            links_by_category.setdefault(relation.category_id, set()).add(relation.item_id)

        by_ref = {cast(int, item.memory_ref): item.id for item in memories.values()}
        for category in categories.values():
            linked = links_by_category.get(category.id, set())
            for memory_ref in self.extract_memory_refs(category.summary or ""):
                if by_ref.get(memory_ref) not in linked:
                    raise ValueError(
                        f"Dossier {category.name} cites inactive or unlinked [M{memory_ref}]"
                    )

    async def select_memorize_dossier_context(
        self,
        episodes: Sequence[Mapping[str, Any]],
        where: Mapping[str, Any],
        *,
        narrative_self: str | None,
        embedding_client: Any | None = None,
    ) -> dict[str, Any]:
        scope = _scope(where)
        if not 1 <= len(episodes) <= 3:
            raise ValueError("Memorize dossier context requires one to three episodes")

        episode_texts: list[str] = []
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise ValueError("Each episode must be a mapping")
            title = episode.get("title")
            summary = episode.get("summary")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Each episode requires a nonblank title")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Each episode requires a nonblank summary")
            episode_texts.append(category_identity_text(title, summary))

        anchors = await self.ensure_dossier_anchors(scope, embedding_client=embedding_client)
        client = embedding_client or self._select_embedding_client(
            {"operation": "dossier", "step_id": "memorize_context"}
        )
        raw_embeddings = await client.embed(episode_texts)
        if len(raw_embeddings) != len(episode_texts):
            raise ValueError("Episode embedding response count does not match input")
        query_embeddings = [
            _embedding_vector(raw, label=f"episode {index + 1}")
            for index, raw in enumerate(raw_embeddings)
        ]

        store = self._get_database()
        active = self.list_active_dossiers(scope)
        anchor_ids = {category.id for category in anchors.values()}
        active_candidates = [category for category in active if category.id not in anchor_ids]
        active_by_id = {category.id: category for category in active_candidates}
        active_ids = {category.id for category in active}
        inactive = sorted(
            [
                category
                for category in store.memory_category_repo.list_categories(scope).values()
                if category.kind in DOSSIER_KINDS
                and category.anchor_role is None
                and category.id not in active_ids
            ],
            key=_activity_key,
        )

        relevant: list[MemoryCategory] = []
        selected_ids: set[str] = set()
        for query in query_embeddings:
            identity = await self.search_dossiers(
                query,
                where=scope,
                view="identity",
                activity="active",
                limit=3,
                min_score=-1.0,
                categories=active_candidates,
            )
            content = await self.search_dossiers(
                query,
                where=scope,
                view="content",
                activity="active",
                limit=3,
                min_score=-1.0,
                embedding_client=client,
                categories=active_candidates,
            )
            ranked = reciprocal_rank_fusion(
                [(category.id, score) for category, score in identity],
                [(category.id, score) for category, score in content],
            )
            for category_id, _score in ranked[:3]:
                if category_id not in selected_ids:
                    relevant.append(active_by_id[category_id])
                    selected_ids.add(category_id)

        inactive_by_id = {category.id: category for category in inactive}
        inactive_rankings: list[list[tuple[str, float]]] = []
        for query in query_embeddings:
            hits = await self.search_dossiers(
                query,
                where=scope,
                view="identity",
                activity="inactive",
                limit=10,
                min_score=-1.0,
                categories=inactive,
            )
            inactive_rankings.append([(category.id, score) for category, score in hits])
        inactive_ids = [
            category_id
            for category_id, _score in reciprocal_rank_fusion(*inactive_rankings)[:10]
        ]

        category_lines = [f"- {_render_index_field(category.name)}" for category in active]
        category_lines.extend(
            _render_dossier_index_line(inactive_by_id[category_id]) for category_id in inactive_ids
        )
        narrative = (
            narrative_self.strip()
            if isinstance(narrative_self, str) and narrative_self.strip()
            else None
        )
        return {
            "categories_str": "\n".join(category_lines),
            "dossier_index": self.build_dossier_index(scope, dossiers=active),
            "narrative_self": narrative,
            "anchor_dossiers": [anchors[role] for role in ("soul", "user")],
            "relevant_dossiers": relevant[:9],
        }

    async def search_dossiers(
        self,
        query_embedding: Sequence[float],
        *,
        where: Mapping[str, Any],
        view: Literal["identity", "content"],
        activity: Literal["active", "inactive", "all"],
        limit: int,
        min_score: float,
        embedding_client: Any | None = None,
        categories: Sequence[MemoryCategory] | None = None,
    ) -> list[tuple[MemoryCategory, float]]:
        if view not in {"identity", "content"}:
            raise ValueError(f"Unknown dossier search view: {view}")
        if activity not in {"active", "inactive", "all"}:
            raise ValueError(f"Unknown dossier activity set: {activity}")
        if view == "content" and activity != "active":
            raise ValueError("Dossier content search is limited to active dossiers")
        if limit < 1:
            raise ValueError("Dossier search limit must be positive")
        if not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
            raise ValueError("Dossier search min_score must be between -1 and 1")

        query = _embedding_vector(query_embedding, label="query")
        scope = _scope(where)
        search_categories = list(categories) if categories is not None else None
        if search_categories is None:
            store = self._get_database()
            if activity == "active":
                search_categories = self.list_active_dossiers(scope)
            elif activity == "inactive":
                search_categories = self.list_inactive_dossiers(scope)
            else:
                search_categories = [
                    category
                    for category in store.memory_category_repo.list_categories(scope).values()
                    if category.kind in DOSSIER_KINDS
                ]
        if not search_categories:
            return []

        corpus: list[tuple[str, list[float]]] = []
        if view == "identity":
            for category in search_categories:
                if category.embedding is None:
                    raise ValueError(f"Dossier {category.id} is missing its identity embedding")
                corpus.append((category.id, _embedding_vector(category.embedding, label=f"dossier {category.id}")))
        else:
            cache = self._dossier_content_embedding_cache
            missing: list[tuple[str, str]] = []
            for category in search_categories:
                text = _content_text(category)
                cached = cache.get(category.id)
                if cached is None or cached[0] != text:
                    missing.append((category.id, text))
            if missing:
                client = embedding_client or self._select_embedding_client(
                    {"operation": "dossier", "step_id": "content_embedding"}
                )
                raw_embeddings = await client.embed([text for _, text in missing])
                if len(raw_embeddings) != len(missing):
                    raise ValueError("Dossier content embedding response count does not match input")
                for (category_id, text), raw_embedding in zip(missing, raw_embeddings, strict=True):
                    cache[category_id] = (
                        text,
                        _embedding_vector(raw_embedding, label=f"dossier {category_id} content"),
                    )
            corpus = [(category.id, cache[category.id][1]) for category in search_categories]

        wrong_dimension = [category_id for category_id, vector in corpus if len(vector) != len(query)]
        if wrong_dimension:
            raise ValueError(
                f"Dossier embedding dimension mismatch for: {', '.join(sorted(wrong_dimension))}"
            )

        category_by_id = {
            category.id: (
                category.model_copy(
                    update={
                        "summary": None,
                        "previous_description": None,
                        "approved_description": None,
                        "previous_summary": None,
                        "approved_summary": None,
                    }
                )
                if view == "identity"
                else category
            )
            for category in search_categories
        }
        scored = cosine_topk(query, corpus, k=len(corpus))
        results = [
            (category_by_id[category_id], score)
            for category_id, score in scored
            if score >= min_score
        ]
        results.sort(key=lambda hit: (-hit[1], hit[0].name.casefold(), hit[0].id))
        return results[:limit]

    @staticmethod
    def format_memory_ref(value: int | MemoryItem) -> str:
        memory_ref = value.memory_ref if isinstance(value, MemoryItem) else value
        if isinstance(memory_ref, bool) or not isinstance(memory_ref, int) or memory_ref < 1:
            raise ValueError("Memory reference must be a positive integer")
        return f"[M{memory_ref}]"

    @staticmethod
    def parse_memory_ref(value: str) -> int:
        match = MEMORY_REF_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"Invalid memory reference: {value}")
        return int(match.group(1))

    def resolve_memory_ref(self, value: int | str, where: Mapping[str, Any]) -> MemoryItem:
        if isinstance(value, str):
            memory_ref = self.parse_memory_ref(value)
        else:
            self.format_memory_ref(value)
            memory_ref = value
        scope = _scope(where)
        item = self._get_database().memory_item_repo.get_item_by_memory_ref(memory_ref, scope)
        if item is None:
            raise KeyError(f"Memory reference [M{memory_ref}] not found in scope")
        return item


__all__ = ["DossierMixin", "DossierRevisionStaleError"]
