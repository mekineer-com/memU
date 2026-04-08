import asyncio

from memu.app.service import MemoryService
from pydantic import BaseModel


class ScopedUserModel(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def _service(*, enable_item_reinforcement: bool) -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        memorize_config={
            "semantic_dedupe_enabled": True,
            "semantic_dedupe_similarity_threshold": 0.89,
            "enable_item_reinforcement": enable_item_reinforcement,
        },
        user_config={"model": ScopedUserModel},
    )


def _seed_dedupe_state(service: MemoryService) -> tuple[dict, dict]:
    store = service._get_database()
    scope = {"user_id": "CodexiaUser", "soul_id": "Codexia"}

    item_a = store.memory_item_repo.create_item(
        resource_id="resource-1",
        memory_type="profile",
        summary="Marcos withdraws when stressed at work",
        embedding=[0.11, 0.22, 0.33],
        user_data=scope,
        source_role="user",
    )
    item_b = store.memory_item_repo.create_item(
        resource_id="resource-1",
        memory_type="profile",
        summary="Marcos pulls back when stressed at work",
        embedding=[0.11, 0.22, 0.33],
        user_data=scope,
        source_role="user",
    )

    store.memory_item_repo.update_item(
        item_id=item_a.id,
        extra={
            "reinforcement_count": 3,
            "last_reinforced_at": "2026-01-01T00:00:00+00:00",
        },
    )
    store.memory_item_repo.update_item(
        item_id=item_b.id,
        extra={
            "reinforcement_count": 2,
            "last_reinforced_at": "2026-01-02T00:00:00+00:00",
        },
    )

    active_items = store.memory_item_repo.list_items(scope)
    state = {
        "items": [active_items[item_a.id], active_items[item_b.id]],
        "relations": [],
        "category_updates": {},
        "user": scope,
        "store": store,
    }
    return state, scope


def test_semantic_dedupe_rolls_up_reinforcement_count_when_enabled() -> None:
    service = _service(enable_item_reinforcement=True)
    state, scope = _seed_dedupe_state(service)
    out = asyncio.run(service._memorize_dedupe_merge(state, None))

    assert len(out["items"]) == 1
    survivor = out["items"][0]
    assert survivor.extra.get("reinforcement_count") == 5
    assert isinstance(survivor.extra.get("last_reinforced_at"), str)

    active_items = service._get_database().memory_item_repo.list_items(scope)
    assert len(active_items) == 1
    db_survivor = next(iter(active_items.values()))
    assert db_survivor.extra.get("reinforcement_count") == 5


def test_semantic_dedupe_does_not_roll_up_when_reinforcement_disabled() -> None:
    service = _service(enable_item_reinforcement=False)
    state, scope = _seed_dedupe_state(service)
    out = asyncio.run(service._memorize_dedupe_merge(state, None))

    assert len(out["items"]) == 1
    survivor = out["items"][0]
    assert survivor.extra.get("reinforcement_count") in {2, 3}

    active_items = service._get_database().memory_item_repo.list_items(scope)
    assert len(active_items) == 1
    db_survivor = next(iter(active_items.values()))
    assert db_survivor.extra.get("reinforcement_count") in {2, 3}
