from __future__ import annotations

import struct

import pytest
from pydantic import BaseModel
from sqlalchemy.schema import CreateTable

from memu.app.service import MemoryService
from memu.database.sqlite.schema import get_sqlite_sqlalchemy_models
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.sqlite.sqlite import SQLiteStore


class EmbeddingBlobScope(BaseModel):
    user_id: str | None = None


class LegacyEmbeddingScope(BaseModel):
    user_id: str | None = None


def _service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": EmbeddingBlobScope},
    )


def test_embedding_blob_round_trips_across_repositories() -> None:
    store = _service()._get_database()
    scope = {"user_id": "blob"}

    item = store.memory_item_repo.create_item(
        memory_type="episode",
        summary="before",
        embedding=[0.1, 0.2],
        user_data=scope,
    )
    updated_item = store.memory_item_repo.update_item(item_id=item.id, embedding=[0.3, 0.4])
    edited_item = store.memory_item_repo.update_summary_with_history(
        item_id=item.id,
        summary="after",
        embedding=[0.5, 0.6],
        where=scope,
    )

    category = store.memory_category_repo.get_or_create_category(
        name="blob category",
        description="blob category",
        embedding=[1.1, 1.2],
        user_data=scope,
    )
    updated_category = store.memory_category_repo.update_category(
        category_id=category.id,
        embedding=[1.3, 1.4],
    )

    resource = store.resource_repo.create_resource(
        url="blob-resource",
        modality="conversation",
        local_path="blob-resource",
        caption=None,
        embedding=[2.1, 2.2],
        user_data=scope,
    )
    updated_resource = store.resource_repo.create_resource(
        url="blob-resource",
        modality="conversation",
        local_path="blob-resource",
        caption=None,
        embedding=[2.3, 2.4],
        user_data=scope,
    )

    assert updated_item.embedding == pytest.approx([0.3, 0.4])
    assert edited_item.embedding == pytest.approx([0.5, 0.6])
    assert updated_category.embedding == pytest.approx([1.3, 1.4])
    assert updated_resource.id == resource.id
    assert updated_resource.embedding == pytest.approx([2.3, 2.4])

    with store._sessions.engine.connect() as conn:
        canonical = [
            conn.exec_driver_sql(
                "SELECT typeof(embedding), length(embedding) FROM memory_items WHERE id = ?", (item.id,)
            ).one(),
            conn.exec_driver_sql(
                "SELECT typeof(embedding), length(embedding) FROM categories WHERE id = ?", (category.id,)
            ).one(),
            conn.exec_driver_sql(
                "SELECT typeof(embedding), length(embedding) FROM resources WHERE id = ?", (resource.id,)
            ).one(),
        ]
        history_type, history_blob = conn.exec_driver_sql(
            "SELECT typeof(embedding_before), embedding_before FROM memory_item_edit_history WHERE memory_item_id = ?",
            (item.id,),
        ).one()

    assert canonical == [("blob", 8), ("blob", 8), ("blob", 8)]
    assert history_type == "blob"
    assert struct.unpack("2f", history_blob) == pytest.approx((0.3, 0.4))


def test_store_rejects_legacy_text_embeddings(tmp_path) -> None:
    dsn = f"sqlite:///{tmp_path / 'legacy.db'}"
    models = get_sqlite_sqlalchemy_models(scope_model=LegacyEmbeddingScope)
    manager = SQLiteSessionManager(dsn=dsn)
    try:
        current_ddl = str(CreateTable(models.Resource.__table__).compile(manager.engine))
        legacy_ddl = current_ddl.replace("embedding BLOB", "embedding TEXT")
        assert legacy_ddl != current_ddl
        with manager.engine.begin() as conn:
            conn.exec_driver_sql(legacy_ddl)
            conn.exec_driver_sql(
                "INSERT INTO resources "
                "(id, created_at, updated_at, url, modality, local_path, embedding, user_id) "
                "VALUES ('legacy', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'legacy', "
                "'conversation', 'legacy', ?, 'legacy')",
                ("[0.25, 0.75]",),
            )
    finally:
        manager.close()

    with pytest.raises(RuntimeError, match=r"resources: non_blob=1.*migrate-embeddings-to-blob"):
        SQLiteStore(dsn=dsn, scope_model=LegacyEmbeddingScope, sqla_models=models)


def test_store_rejects_zero_length_embedding_blob(tmp_path) -> None:
    dsn = f"sqlite:///{tmp_path / 'empty.db'}"
    store = SQLiteStore(dsn=dsn, scope_model=EmbeddingBlobScope)
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO resources "
            "(id, created_at, updated_at, url, modality, local_path, embedding, user_id) "
            "VALUES ('empty', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'empty', 'conversation', 'empty', ?, 'blob')",
            (b"",),
        )
    store.close()

    with pytest.raises(RuntimeError, match="resources: non_blob=0, empty_blob=1"):
        SQLiteStore(dsn=dsn, scope_model=EmbeddingBlobScope)


def test_malformed_embedding_blob_fails_loudly() -> None:
    store = _service()._get_database()
    resource = store.resource_repo.create_resource(
        url="malformed",
        modality="conversation",
        local_path="malformed",
        caption=None,
        embedding=[1.0],
        user_data={"user_id": "blob"},
    )
    with store._sessions.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE resources SET embedding = ? WHERE id = ?", (b"bad", resource.id))
    store.resources.clear()

    with pytest.raises(ValueError, match="Malformed embedding BLOB length: 3 bytes"):
        store.resource_repo.list_resources({"user_id": "blob"})


def test_empty_embedding_write_fails_loudly() -> None:
    store = _service()._get_database()
    with pytest.raises(ValueError, match="Embedding vector cannot be empty"):
        store.resource_repo.create_resource(
            url="empty",
            modality="conversation",
            local_path="empty",
            caption=None,
            embedding=[],
            user_data={"user_id": "blob"},
        )


@pytest.mark.parametrize("embedding", [[float("nan")], [1e39]])
def test_non_finite_float32_embedding_write_fails_loudly(embedding) -> None:
    store = _service()._get_database()
    with pytest.raises(ValueError, match="finite"):
        store.resource_repo.create_resource(
            url="non-finite",
            modality="conversation",
            local_path="non-finite",
            caption=None,
            embedding=embedding,
            user_data={"user_id": "blob"},
        )
