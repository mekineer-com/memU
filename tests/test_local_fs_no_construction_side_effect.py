"""Contract test for LocalFS: no disk side effect at construction.

History: `BlobConfig.resources_dir` defaults to `"./data/resources"` — a
CWD-relative path. `LocalFS.__init__` previously called `.mkdir(...)`
unconditionally, so every `MemoryService(...)` without an explicit
`blob_config` left an empty `./data/resources/` wherever Python happened
to start from. Running the test suite from `apps-codex/` created
`apps-codex/data/resources/`; running from `memu/` created
`memu/data/resources/`.

Fix: `LocalFS.__init__` stores `self.base` only. `mkdir` happens in
`fetch()` immediately before the first write.

Without this guard, the silent-side-effect pattern could creep back in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from memu.app.service import MemoryService
from memu.blob.local_fs import LocalFS


class Scope(BaseModel):
    user_id: str | None = None


def test_local_fs_ctor_does_not_mkdir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "blob_root"
    assert not target.exists()
    fs = LocalFS(str(target))
    assert fs.base == target
    assert not target.exists()


def test_memory_service_without_blob_config_leaves_no_data_resources(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": Scope},
    )
    assert not (tmp_path / "data" / "resources").exists(), (
        "MemoryService construction created a CWD-relative data/resources directory"
    )
    assert os.listdir(tmp_path) == [], f"unexpected filesystem churn: {os.listdir(tmp_path)}"


@pytest.mark.asyncio
async def test_relative_resource_is_jailed_but_absolute_source_still_works(tmp_path: Path) -> None:
    base = tmp_path / "resources"
    relative = base / "mentra_media" / "image.txt"
    relative.parent.mkdir(parents=True)
    relative.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    fs = LocalFS(str(base))

    path, text = await fs.fetch("mentra_media/image.txt", "text")
    assert (path, text) == (str(relative.resolve()), "inside")
    absolute_path, absolute_text = await fs.fetch(str(outside), "text")
    assert (absolute_path, absolute_text) == (str(outside.resolve()), "outside")
    with pytest.raises(ValueError, match="escapes LocalFS base"):
        await fs.fetch("../outside.txt", "text")
