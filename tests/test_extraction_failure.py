from __future__ import annotations

from pathlib import Path

import pytest

from memu.app.service import MemoryService
from memu.app.memorize import SpeakerRosterEntry


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
@pytest.mark.parametrize("later_invalid", [False, True])
async def test_extraction_parses_each_reply_once_and_keeps_batch_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later_invalid: bool,
) -> None:
    service = _service()
    monkeypatch.setattr(service.fs, "base", tmp_path)
    valid = (
        "<item><memory><content>A fictional\n fact: a<b & x < y.</content>"
        "<speaker_ref>entity:guest</speaker_ref></memory></item>"
    )
    replies = [valid, "not xml" if later_invalid else valid]
    pending = iter(replies)
    parsed = []
    parse = service._parse_memory_type_response_xml

    def record_parse(response):
        parsed.append(response)
        return parse(response)

    async def chat(_prompt):
        return next(pending)

    stub = _Stub(valid)
    monkeypatch.setattr(stub, "chat", chat)
    monkeypatch.setattr(service, "_parse_memory_type_response_xml", record_parse)
    kwargs = dict(
        resource_text="", store=None, memory_types=["knowledge"], categories_prompt_str="",
        prepared_prompts=[("knowledge", "first"), ("knowledge", "second")],
        speaker_roster=[SpeakerRosterEntry("entity:guest", "Fictional Guest", "peer")],
        llm_client=stub,
    )
    if later_invalid:
        with pytest.raises(ValueError, match="Extraction reply invalid"):
            await service._generate_entries_from_text(**kwargs)
        assert len(list((tmp_path / "extraction_dumps").iterdir())) == 1
    else:
        entries = await service._generate_entries_from_text(**kwargs)
        assert [entry.content for entry in entries] == ["A fictional fact: a<b & x < y."]
    assert parsed == replies


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["not xml", "", " \n\t", "<item><memory/><memory><content> </content></memory></item>"])
async def test_extraction_failure_dumps_once_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *_a, **_k: "")
    monkeypatch.setattr(service.fs, "base", tmp_path)
    stub = _Stub(response)

    with pytest.raises(ValueError, match="Extraction reply invalid"):
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
    assert dumps[0].read_text() == response


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
