from types import SimpleNamespace

import pytest

from memu.app.retrieve import RetrieveMixin


@pytest.mark.asyncio
async def test_resource_retrieval_fuses_visual_and_cached_caption_views() -> None:
    resources = {
        "visual": SimpleNamespace(embedding=[1.0, 0.0], caption="other"),
        "caption": SimpleNamespace(embedding=[0.0, 1.0], caption="target"),
        "both": SimpleNamespace(embedding=[0.8, 0.2], caption="related"),
    }
    embedded: list[list[str]] = []

    class Client:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            embedded.append(texts)
            vectors = {"other": [0.0, 1.0], "target": [1.0, 0.0], "related": [0.8, 0.2]}
            return [vectors[text] for text in texts]

    mixin = RetrieveMixin()
    mixin.retrieve_config = SimpleNamespace(resource=SimpleNamespace(top_k=3))
    mixin._resource_caption_embedding_cache = {}
    mixin._select_embedding_client = lambda _context: Client()
    state = {
        "needs_retrieval": True,
        "proceed_to_resources": True,
        "active_query": "target",
        "query_vector": [1.0, 0.0],
        "store": SimpleNamespace(
            resource_repo=SimpleNamespace(list_resources=lambda _where: resources)
        ),
        "where": {},
    }

    first = await mixin._rag_recall_resources(state, None)
    second = await mixin._rag_recall_resources(state, None)

    assert first["resource_hits"][0][0] == "both"
    assert {resource_id for resource_id, _score in first["resource_hits"]} == set(resources)
    assert second["resource_hits"] == first["resource_hits"]
    assert embedded == [["other", "target", "related"]]
