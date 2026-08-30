from pathlib import Path

import pytest
from pydantic import BaseModel

from memu.app import memorize_persistence
from memu.app.service import MemoryService


class Scope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(
        blob_config={"resources_dir": str(tmp_path)},
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": Scope},
    )


@pytest.mark.asyncio
async def test_supplied_image_caption_skips_preprocessor(tmp_path: Path) -> None:
    service = _service(tmp_path)

    async def fail(*_args, **_kwargs):
        raise AssertionError("image preprocessor must not run")

    service._split_into_episodes = fail
    state = {"supplied_caption": "A sitting-aware caption."}
    out = await service._memorize_split_episodes(state, None)
    assert out["episodes"] == [
        {"text": "A sitting-aware caption.", "caption": "A sitting-aware caption."}
    ]


@pytest.mark.asyncio
async def test_image_resource_uses_raw_bytes_and_preserves_path(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    captured = {}

    class Client:
        async def embed_media(self, data: bytes, mime_type: str) -> list[float]:
            captured["media"] = (data, mime_type)
            return [1.0, 0.0]

        async def embed(self, _texts):
            raise AssertionError("caption must not become the canonical Resource vector")

    class Repo:
        def create_resource(self, **kwargs):
            captured["resource"] = kwargs
            return kwargs

    await memorize_persistence._create_resource_with_caption(
        resource_url="mentra_media/test/image.png",
        modality="image",
        local_path=str(image),
        caption="A caption.",
        store=type("Store", (), {"resource_repo": Repo()})(),
        embed_client=Client(),
        select_embedding_client=lambda _context: Client(),
        user={"user_id": "u", "soul_id": "s"},
        segment_id=None,
        conversation_id="mentra:phone",
        memory_retrieve_history=None,
        memory_prior_context=None,
        session=None,
    )
    assert captured["media"] == (b"pixels", "image/png")
    assert captured["resource"]["local_path"] == str(image)
    assert captured["resource"]["embedding"] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_completed_image_resource_retry_creates_no_second_item(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope = {"user_id": "u", "soul_id": "s"}
    store = service.database
    resource = store.resource_repo.create_resource(
        url="mentra_media/test/image.png",
        modality="image",
        local_path=str(tmp_path / "image.png"),
        caption="A caption.",
        embedding=[1.0, 0.0],
        user_data=scope,
    )
    item = store.memory_item_repo.create_item(
        resource_id=resource.id,
        memory_type="episode",
        summary="A caption.",
        embedding=[1.0, 0.0],
        user_data=scope,
    )

    class FailClient:
        async def embed(self, _texts):
            raise AssertionError("retry must not embed or extract")

    async def fail_workflow(*_args, **_kwargs):
        raise AssertionError("completed retry must stop before the workflow")

    service._run_workflow = fail_workflow
    public = await service.memorize(
        resource_url=resource.url,
        modality="image",
        user=scope,
        caption="A caption.",
    )
    assert [row["id"] for row in public["items"]] == [item.id]

    items = []
    resources, homeless = await service._process_plan(
        {"resource_url": resource.url, "text": "A caption.", "caption": "A caption.", "entries": []},
        modality="image",
        local_path=resource.local_path,
        ctx=None,
        store=store,
        embed_client=FailClient(),
        user_scope=scope,
        conversation_id="mentra:phone",
        items=items,
        relations=[],
        pending_segment_ids=[],
    )
    assert resources[0].id == resource.id
    assert [row.id for row in items] == [item.id]
    assert homeless == 0

    with pytest.raises(ValueError, match="caption conflicts"):
        await service._process_plan(
            {"resource_url": resource.url, "text": "Different.", "caption": "Different.", "entries": []},
            modality="image",
            local_path=resource.local_path,
            ctx=None,
            store=store,
            embed_client=FailClient(),
            user_scope=scope,
            conversation_id="mentra:phone",
            items=[],
            relations=[],
            pending_segment_ids=[],
        )


@pytest.mark.asyncio
async def test_unlinked_image_resource_retry_reuses_resource(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope = {"user_id": "u", "soul_id": "s"}
    store = service.database
    resource = store.resource_repo.create_resource(
        url="mentra_media/test/image.png",
        modality="image",
        local_path=str(tmp_path / "image.png"),
        caption="A caption.",
        embedding=[1.0, 0.0],
        user_data=scope,
    )

    async def fail_create(**_kwargs):
        raise AssertionError("retry must reuse the unlinked Resource")

    service._create_resource_with_caption = fail_create
    resources, _ = await service._process_plan(
        {"resource_url": resource.url, "text": "A caption.", "caption": "A caption.", "entries": []},
        modality="image",
        local_path=resource.local_path,
        ctx=None,
        store=store,
        embed_client=object(),
        user_scope=scope,
        conversation_id="mentra:phone",
        items=[],
        relations=[],
        pending_segment_ids=[],
    )
    assert [row.id for row in resources] == [resource.id]
