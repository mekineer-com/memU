from pathlib import Path
from datetime import UTC, datetime
from types import SimpleNamespace
from itertools import count

import pytest
from pydantic import BaseModel

from memu.app import memorize_persistence
from memu.app import memorize_dedupe
from memu.database.models import Triple
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
@pytest.mark.parametrize("change", [None, "edit", "delete", "merge", "supersede", "conflict"])
async def test_dedupe_prepared_pairs_reject_concurrent_changes(tmp_path, monkeypatch, change):
    service = _service(tmp_path)
    store = service.database
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    def add(year):
        return store.memory_item_repo.create_item(
            memory_type="knowledge", summary="Fictional garden project", embedding=[1.0, 0.0],
            happened_at=datetime(year, 1, 1, tzinfo=UTC), user_data=scope,
        )
    old, anchor = add(2020), add(2021)
    state = {"items": [old, anchor], "relations": [SimpleNamespace(item_id=old.id)],
             "store": store, "user": scope}
    prepare = memorize_dedupe._prepare_dedupe_merges
    async def interleave(*args, **kwargs):
        pairs = await prepare(*args, **kwargs)
        assert len(pairs) == 1
        if change == "edit":
            store.memory_item_repo.update_summary_with_history(
                item_id=anchor.id, summary="A completely different fictional topic",
                embedding=[0.0, 1.0], where=scope,
            )
        elif change == "delete":
            service.graph_delete_memories([anchor.id], where=scope)
        elif change in {"merge", "supersede"}:
            successor = add(2022)
            if change == "merge":
                store.memory_item_repo.update_item(item_id=anchor.id, merged_into=successor.id)
            else:
                store.triple_repo.add(Triple(subject_id=anchor.id, subject_kind="memory",
                    predicate="evolved_into", object_id=successor.id, object_kind="memory"), user_data=scope)
        elif change == "conflict":
            for item, name in ((old, "First"), (anchor, "Second")):
                category = store.memory_category_repo.get_or_create_category(
                    name=name, description=name, embedding=[1.0, 0.0], user_data=scope,
                )
                candidate = store.dossier_candidate_repo.add_candidate(
                    proposed_name="Garden", item_id=item.id, where=scope,
                )
                with service._sqlite_write_session(store) as session:
                    session.connection().exec_driver_sql(
                        "UPDATE dossier_candidates SET resolved_category_id = ? WHERE id = ?",
                        (category.id, candidate.id),
                    )
                    session.commit()
        return pairs
    monkeypatch.setattr(memorize_dedupe, "_prepare_dedupe_merges", interleave)
    await service._memorize_dedupe_merge(state, None)
    rows = store.memory_item_repo.list_items(scope, include_merged=True, include_superseded=True)
    assert rows[old.id].merged_into == (anchor.id if change is None else None)
    assert {item.id for item in state["items"]} == ({anchor.id} if change is None else {old.id, anchor.id})
    assert bool(state["relations"]) == (change is not None)
    store.close()


@pytest.mark.asyncio
async def test_dedupe_chain_rechecks_candidates_after_preceding_merge(tmp_path, monkeypatch):
    service = _service(tmp_path)
    store = service.database
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    ids = count(1)
    with monkeypatch.context() as fixed_ids:
        fixed_ids.setattr("memu.database.sqlite.models.secrets.token_hex", lambda *_: f"{next(ids):08x}")
        items = [store.memory_item_repo.create_item(
            memory_type="knowledge", summary="Fictional garden project", embedding=[1.0, 0.0],
            happened_at=datetime(year, 1, 1, tzinfo=UTC), user_data=scope,
        ) for year in (2020, 2021, 2022)]
    for item, name in zip(items, ["First", None, "Second"], strict=True):
        candidate = store.dossier_candidate_repo.add_candidate(proposed_name="Garden", item_id=item.id, where=scope)
        if name:
            category = store.memory_category_repo.get_or_create_category(
                name=name, description=name, embedding=[1.0, 0.0], user_data=scope,
            )
            with service._sqlite_write_session(store) as session:
                session.connection().exec_driver_sql(
                    "UPDATE dossier_candidates SET resolved_category_id = ? WHERE id = ?", (category.id, candidate.id))
                session.commit()
    state = {"items": items[:2], "relations": [], "store": store, "user": scope}
    await service._memorize_dedupe_merge(state, None)
    rows = store.memory_item_repo.list_items(scope, include_merged=True)
    assert rows[items[0].id].merged_into == items[1].id
    assert rows[items[1].id].merged_into is None
    assert [item.id for item in state["items"]] == [items[1].id]
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["merge", "rollback", "disabled", "empty", "no_hook"])
async def test_dedupe_phase_and_merges_share_transaction(tmp_path, monkeypatch, mode):
    service = _service(tmp_path)
    store = service.database
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    items = [store.memory_item_repo.create_item(
        memory_type="knowledge", summary="Fictional garden project", embedding=[1.0, 0.0], user_data=scope,
    ) for _ in range(2)]
    with service._sqlite_write_session(store) as session:
        session.connection().exec_driver_sql("CREATE TABLE test_phase (phase TEXT)")
        session.connection().exec_driver_sql("INSERT INTO test_phase VALUES ('dedupe')")
        session.commit()
    def complete(session):
        session.connection().exec_driver_sql("UPDATE test_phase SET phase = 'review'")
        if mode == "rollback":
            raise RuntimeError("phase write failed")
    service.memorize_config.semantic_dedupe_enabled = mode != "disabled"
    state = {"store": store, "user": scope, "items": [] if mode in {"empty", "no_hook"} else [items[1]],
             "relations": [], "on_dedupe_complete": None if mode == "no_hook" else complete}
    if mode == "no_hook":
        with monkeypatch.context() as no_transaction:
            no_transaction.setattr(memorize_persistence, "_sqlite_write_session",
                lambda *_: pytest.fail("No pairs or phase hook should open no write transaction"))
            await service._memorize_dedupe_merge(state, None)
    elif mode == "rollback":
        with pytest.raises(RuntimeError, match="phase write failed"):
            await service._memorize_dedupe_merge(state, None)
    else:
        await service._memorize_dedupe_merge(state, None)
    rows = store.memory_item_repo.list_items(scope, include_merged=True)
    assert sum(bool(row.merged_into) for row in rows.values()) == (1 if mode == "merge" else 0)
    with service._sqlite_write_session(store) as session:
        assert session.connection().exec_driver_sql("SELECT phase FROM test_phase").scalar_one() == (
            "dedupe" if mode in {"rollback", "no_hook"} else "review")
    store.close()


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
    )
    assert captured["media"] == (b"pixels", "image/png")
    assert captured["resource"]["local_path"] == str(image)
    assert captured["resource"]["embedding"] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_image_resource_commits_before_extraction(tmp_path: Path) -> None:
    service = _service(tmp_path)
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")

    class Client:
        async def embed_media(self, _data: bytes, _mime_type: str) -> list[float]:
            return [1.0, 0.0]

    async def fail_workflow(*_args, **_kwargs):
        raise RuntimeError("fictional extraction failure")

    service._select_embedding_client = lambda _context: Client()
    service._run_workflow = fail_workflow
    scope = {"user_id": "u", "soul_id": "s"}

    with pytest.raises(RuntimeError, match="fictional extraction failure"):
        await service.memorize(
            resource_url="image.png",
            modality="image",
            user=scope,
            caption="A caption.",
        )

    resources = service.database.resource_repo.list_resources(scope)
    assert [(row.url, row.caption) for row in resources.values()] == [
        ("image.png", "A caption.")
    ]


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

    service._llm_clients["embedding"] = FailClient()
    state = await service._memorize_categorize_items({
        "segment_plans": [{"resource_url": resource.url, "text": "A caption.", "caption": "A caption.", "entries": []}],
        "modality": "image", "local_path": resource.local_path,
        "store": store, "user": scope, "conversation_id": "mentra:phone",
    }, {})
    assert state["resources"][0].id == resource.id
    assert [row.id for row in state["items"]] == [item.id]
    assert state["homeless_item_count"] == 0

    with pytest.raises(ValueError, match="caption conflicts"):
        await service._memorize_categorize_items({
            "segment_plans": [{"resource_url": resource.url, "text": "Different.", "caption": "Different.", "entries": []}],
            "modality": "image", "local_path": resource.local_path,
            "store": store, "user": scope, "conversation_id": "mentra:phone",
        }, {})


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

    class FailClient:
        async def embed_media(self, *_args):
            raise AssertionError("retry must not embed the saved media again")

    service._llm_clients["embedding"] = FailClient()
    state = await service._memorize_categorize_items({
        "segment_plans": [{"resource_url": resource.url, "text": "A caption.", "caption": "A caption.", "entries": []}],
        "modality": "image", "local_path": resource.local_path,
        "store": store, "user": scope, "conversation_id": "mentra:phone",
    }, {})
    assert [row.id for row in state["resources"]] == [resource.id]
    assert list(store.resource_repo.list_resources(where=scope)) == [resource.id]


def test_merged_only_image_lineage_is_not_complete(tmp_path: Path) -> None:
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
    old = store.memory_item_repo.create_item(
        resource_id=resource.id,
        memory_type="episode",
        summary="Old image memory.",
        embedding=[1.0, 0.0],
        user_data=scope,
    )
    survivor = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="Surviving memory.",
        embedding=[1.0, 0.0],
        user_data=scope,
    )
    store.memory_item_repo.update_item(item_id=old.id, merged_into=survivor.id)

    existing, linked = service._image_retry_state(
        store, scope, resource.url, "A caption."
    )
    assert existing is not None and existing.id == resource.id
    assert linked == {}
