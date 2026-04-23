import pytest

from memu.app.service import MemoryService
from memu.database.models import MemoryCategory


class DummyEmbedClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        self.calls.append(list(inputs))
        vectors = []
        for text in inputs:
            lowered = text.lower()
            if "health" in lowered:
                vectors.append([1.0, 0.0])
            elif "work" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


@pytest.mark.asyncio
async def test_rank_categories_by_summary_reuses_cached_embeddings_until_summary_changes() -> None:
    service = _service()
    store = service._get_database()
    client = DummyEmbedClient()
    categories = {
        "cat_health": MemoryCategory(id="cat_health", name="health", description="", summary="Health concerns"),
        "cat_work": MemoryCategory(id="cat_work", name="work", description="", summary="Work projects"),
    }

    hits_1, lookup_1 = await service._rank_categories_by_summary(
        [1.0, 0.0], 2, store, embed_client=client, categories=categories
    )
    hits_2, lookup_2 = await service._rank_categories_by_summary(
        [1.0, 0.0], 2, store, embed_client=client, categories=categories
    )

    assert client.calls == [["Health concerns", "Work projects"]]
    assert lookup_1 == lookup_2 == {"cat_health": "Health concerns", "cat_work": "Work projects"}
    assert hits_1[0][0] == hits_2[0][0] == "cat_health"

    categories["cat_work"].summary = "Health-adjacent work"

    hits_3, lookup_3 = await service._rank_categories_by_summary(
        [1.0, 0.0], 2, store, embed_client=client, categories=categories
    )

    assert client.calls == [["Health concerns", "Work projects"], ["Health-adjacent work"]]
    assert lookup_3 == {"cat_health": "Health concerns", "cat_work": "Health-adjacent work"}
    assert {cid for cid, _ in hits_3} == {"cat_health", "cat_work"}
