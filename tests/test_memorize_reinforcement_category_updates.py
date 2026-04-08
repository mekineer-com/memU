import asyncio

from memu.app.memorize import StructuredMemoryEntry
from memu.app.service import MemoryService


class DummyEmbedClient:
    async def embed(self, inputs: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in inputs]


def _service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        memorize_config={"enable_item_reinforcement": True},
    )


def test_reinforced_repeat_updates_category_summary_context_without_new_relation() -> None:
    service = _service()
    store = service._get_database()
    ctx = service._get_context()
    embed_client = DummyEmbedClient()

    category = store.memory_category_repo.get_or_create_category(
        name="habits",
        description="",
        embedding=[0.0, 1.0],
        user_data={},
    )
    ctx.category_ids = [category.id]
    ctx.category_name_to_id = {"habits": category.id}

    entry = StructuredMemoryEntry(
        memory_type="profile",
        content="Marcos journals every night",
        categories=["habits"],
        source_role="user",
        confidence=0.9,
        source_message_ids=[1],
        reflection_salience=None,
        replaces_previous_fact=None,
    )

    async def run_once() -> tuple[tuple[list, list, dict, int], tuple[list, list, dict, int]]:
        first = await service._persist_memory_items(
            resource_id="resource-1",
            structured_entries=[entry],
            ctx=ctx,
            store=store,
            embed_client=embed_client,
            user={},
        )
        second = await service._persist_memory_items(
            resource_id="resource-1",
            structured_entries=[entry],
            ctx=ctx,
            store=store,
            embed_client=embed_client,
            user={},
        )
        return first, second

    first, second = asyncio.run(run_once())
    _items_1, rels_1, updates_1, _ = first
    items_2, rels_2, updates_2, _ = second

    assert len(rels_1) == 1
    assert updates_1[category.id][0][1] == "Marcos journals every night"

    assert len(rels_2) == 0
    assert category.id in updates_2
    item_id, summary = updates_2[category.id][0]
    assert item_id == items_2[0].id
    assert summary.startswith("[reinforced 2x] ")
