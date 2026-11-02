from types import SimpleNamespace

import pytest

from memu.app.retrieve import RetrieveMixin


@pytest.mark.asyncio
async def test_resource_retrieval_keeps_sensory_candidate_lanes_separate() -> None:
    resources = {
        "visual": SimpleNamespace(modality="image", embedding=[1.0, 0.0], caption="other"),
        "caption": SimpleNamespace(modality="image", embedding=[0.0, 1.0], caption="target"),
        "both": SimpleNamespace(modality="image", embedding=[0.8, 0.2], caption="related"),
        "conversation": SimpleNamespace(modality="conversation", embedding=[1.0, 0.0], caption="target"),
    }
    embedded: list[list[str]] = []
    resource_reads: list[dict] = []

    def list_resources(where):
        resource_reads.append(where)
        return resources

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
        "active_query": "ordinary text query",
        "visual_memory_query": "target",
        "new_message": "show me that picture",
        "temporal_start": None,
        "temporal_end": None,
        "query_vector": [1.0, 0.0],
        "store": SimpleNamespace(
            resource_repo=SimpleNamespace(list_resources=list_resources),
            memory_category_repo=SimpleNamespace(list_categories=lambda _where: {}),
            memory_item_repo=SimpleNamespace(list_items=lambda *_args, **_kwargs: {}),
        ),
        "where": {},
    }

    first = await mixin._rag_recall_resources(state, None)
    second = await mixin._rag_recall_resources(state, None)

    assert [resource_id for resource_id, _score in first["resource_candidate_lanes"]["media"]] == [
        "visual",
        "both",
        "caption",
    ]
    assert [resource_id for resource_id, _score in first["resource_candidate_lanes"]["caption"]] == [
        "caption",
        "both",
        "visual",
    ]
    assert second["resource_candidate_lanes"] == first["resource_candidate_lanes"]
    assert embedded == [["target"], ["other", "target", "related"], ["target"]]

    response = mixin._rag_build_context(first, None)["response"]
    assert response["visual_memory_query"] == "target"
    assert response["resource_candidates"]["media"][0] == {
        "id": "visual",
        "modality": "image",
        "caption": "other",
        "evidence": "media",
    }
    assert "local_path" not in response["resource_candidates"]["media"][0]
    assert "score" not in response["resource_candidates"]["media"][0]
    assert resource_reads == [{}, {}]


@pytest.mark.asyncio
@pytest.mark.parametrize("visual_query,top_k,empty_corpus,expected_reads", [
    (None, 5, False, 0),
    ("target", 0, False, 0),
    ("target", 5, True, 1),
])
async def test_empty_resource_lanes_skip_unused_work(visual_query, top_k, empty_corpus, expected_reads):
    resource_reads, embedded = [], []

    def list_resources(where):
        resource_reads.append(where)
        return {} if empty_corpus else {
            "photo": SimpleNamespace(modality="image", embedding=[1.0, 0.0], caption="target"),
        }

    class Client:
        async def embed(self, texts):
            embedded.append(texts)
            return [[1.0, 0.0] for _ in texts]

    mixin = RetrieveMixin()
    mixin.retrieve_config = SimpleNamespace(resource=SimpleNamespace(top_k=top_k))
    mixin._resource_caption_embedding_cache = {}
    mixin._select_embedding_client = lambda _context: Client()
    state = {
        "needs_retrieval": True,
        "visual_memory_query": visual_query,
        "new_message": "recall a fictional visit",
        "store": SimpleNamespace(
            resource_repo=SimpleNamespace(list_resources=list_resources),
            memory_category_repo=SimpleNamespace(list_categories=lambda _where: {}),
            memory_item_repo=SimpleNamespace(list_items=lambda *_args, **_kwargs: {}),
        ),
        "where": {},
    }
    await mixin._rag_item_sufficiency(state, None)
    await mixin._rag_recall_resources(state, None)
    response = mixin._rag_build_context(state, None)["response"]

    assert len(resource_reads) == expected_reads
    assert embedded == []
    assert response["resource_candidates"] == {}
    assert response["resources"] == []
    state["resource_candidate_lanes"] = {"media": [], "caption": []}
    assert mixin._rag_build_context(state, None)["response"]["resource_candidates"] == {"media": [], "caption": []}
    assert len(resource_reads) == expected_reads


@pytest.mark.asyncio
async def test_visual_query_alone_enables_resource_recall() -> None:
    mixin = RetrieveMixin()
    state = {"needs_retrieval": True, "visual_memory_query": "red doorway"}

    result = await mixin._rag_item_sufficiency(state, None)

    assert result["proceed_to_resources"] is True


@pytest.mark.asyncio
async def test_sensory_search_reuses_safe_resource_recall() -> None:
    mixin = RetrieveMixin()
    resource = SimpleNamespace(modality="image", embedding=[1.0], caption="red doorway")
    resource_reads, embedded = [], []

    def list_resources(where):
        resource_reads.append(where)
        return {"photo": resource}

    class Client:
        async def embed(self, texts):
            embedded.append(texts)
            return [[1.0] for _ in texts]

    mixin.retrieve_config = SimpleNamespace(resource=SimpleNamespace(top_k=5))
    mixin._resource_caption_embedding_cache = {}
    mixin._select_embedding_client = lambda _context: Client()
    mixin._get_database = lambda: SimpleNamespace(
        resource_repo=SimpleNamespace(list_resources=list_resources)
    )
    mixin._normalize_where = lambda where: dict(where or {})

    result = await mixin.sensory_search(" red doorway ", {"user_id": "u", "soul_id": "s"})

    assert result == {
        "visual_memory_query": "red doorway",
        "resource_candidates": {
            lane: [
                {
                    "id": "photo",
                    "modality": "image",
                    "caption": "red doorway",
                    "evidence": lane,
                }
            ] for lane in ("media", "caption")
        },
    }
    assert resource_reads == [{"user_id": "u", "soul_id": "s"}]
    assert embedded == [["red doorway"], ["red doorway"]]
