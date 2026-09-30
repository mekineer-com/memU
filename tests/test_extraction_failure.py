from __future__ import annotations

from pathlib import Path

import pytest

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(
        database_config={
            "metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}
        }
    )


class _Stub:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def chat(self, _prompt: str) -> str:
        self.calls += 1
        return self.response


@pytest.mark.asyncio
async def test_extraction_failure_dumps_once_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *_a, **_k: "")
    monkeypatch.setattr(service.fs, "base", tmp_path)
    stub = _Stub("not xml")

    with pytest.raises(ValueError, match="Extraction reply invalid.*not xml"):
        await service._generate_entries_from_text(
            resource_text="[0] [user]: A fictional event.",
            store=None,  # type: ignore[arg-type]
            memory_types=["knowledge"],
            categories_prompt_str="Fictional life",
            llm_client=stub,
        )

    dumps = list((tmp_path / "extraction_dumps").iterdir())
    assert stub.calls == 1
    assert len(dumps) == 1
    assert "attempt1" in dumps[0].name
    assert dumps[0].read_text() == "not xml"


@pytest.mark.asyncio
async def test_router_failure_dumps_once_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service()
    monkeypatch.setattr(service.fs, "base", tmp_path)
    stub = _Stub("not json")

    with pytest.raises(ValueError, match="Router reply invalid"):
        await service._route_segment(
            "fictional episode",
            ["knowledge"],
            source_days=["2026-01-02"],
            llm_client=stub,
        )

    dumps = list((tmp_path / "extraction_dumps").iterdir())
    assert stub.calls == 1
    assert len(dumps) == 1
    assert "router_attempt1" in dumps[0].name
    assert dumps[0].read_text() == "not json"
