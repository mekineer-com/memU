"""Tests for retry-on-unparseable behavior in extraction and router paths."""
import pathlib

import pytest

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


_VALID_XML = """
<item>
  <memory>
    <content>Marcos values clear communication</content>
    <categories><category>communication</category></categories>
  </memory>
</item>
""".strip()

_GARBAGE = "not xml at all %%% garbage"

_VALID_ROUTER = (
    '{"excluded_types": [], "episodes": '
    '[{"title": "T", "episode_summary": "Full story.", "episode_item": "Compact story.", '
    '"categories": ["Daily life"], "day": "2026-01-02"}]}'
)


class _ExtractionStub:
    """Returns garbage on first call, valid XML on second (per prompt)."""

    def __init__(self, first: str, second: str) -> None:
        self._calls: dict[str, int] = {}
        self._first = first
        self._second = second

    async def chat(self, prompt: str) -> str:
        count = self._calls.get(prompt, 0) + 1
        self._calls[prompt] = count
        return self._first if count == 1 else self._second


class _RouterStub:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    async def chat(self, _prompt: str) -> str:
        return next(self._responses)


@pytest.mark.asyncio
async def test_extraction_retries_day_outside_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *a, **kw: "")
    invalid = _VALID_XML.replace("<content>", "<day>2026-01-01</day><content>")
    valid = _VALID_XML.replace("<content>", "<day>2026-01-02</day><content>")

    entries = await service._generate_entries_from_text(
        resource_text="2026-01-02: Something happened.",
        store=None,  # type: ignore[arg-type]
        memory_types=["knowledge"],
        categories_prompt_str="communication",
        source_days=["2026-01-02"],
        llm_client=_ExtractionStub(first=invalid, second=valid),
    )

    assert entries[0].memory_date == "2026-01-02"


@pytest.mark.asyncio
async def test_extraction_retry_succeeds_and_logs_error(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """garbage → valid: retry succeeds, error logged, items returned."""
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *a, **kw: "")
    stub = _ExtractionStub(first=_GARBAGE, second=_VALID_XML)

    import logging
    with caplog.at_level(logging.ERROR):
        entries = await service._generate_entries_from_text(
            resource_text="[0] [user]: Something happened.",
            store=None,  # type: ignore[arg-type]
            memory_types=["knowledge"],
            categories_prompt_str="communication",
            llm_client=stub,
        )

    assert len(entries) == 1
    assert entries[0].content == "Marcos values clear communication"
    assert any(
        "unparseable" in r.message.lower() or "retry" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    )


@pytest.mark.asyncio
async def test_extraction_retry_raises_on_double_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    """garbage → garbage: raises ValueError."""
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *a, **kw: "")
    stub = _ExtractionStub(first=_GARBAGE, second=_GARBAGE)

    with pytest.raises(ValueError, match="unparseable"):
        await service._generate_entries_from_text(
            resource_text="[0] [user]: Something happened.",
            store=None,  # type: ignore[arg-type]
            memory_types=["knowledge"],
            categories_prompt_str="communication",
            llm_client=stub,
        )


@pytest.mark.asyncio
async def test_extraction_double_garbage_dumps_files_and_snippet(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """garbage → garbage: dump files written, ValueError carries reply snippet."""
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *a, **kw: "")
    monkeypatch.setattr(service.fs, "base", tmp_path)
    stub = _ExtractionStub(first=_GARBAGE, second=_GARBAGE)

    with pytest.raises(ValueError) as exc_info:
        await service._generate_entries_from_text(
            resource_text="[0] [user]: Something happened.",
            store=None,  # type: ignore[arg-type]
            memory_types=["social"],
            categories_prompt_str="communication",
            llm_client=stub,
        )

    dump_dir = tmp_path / "extraction_dumps"
    dumps = sorted(dump_dir.iterdir())
    assert len(dumps) == 2, f"expected 2 dump files, got {[d.name for d in dumps]}"
    assert "attempt1" in dumps[0].name
    assert "attempt2" in dumps[1].name
    assert dumps[0].read_text() == _GARBAGE
    assert dumps[1].read_text() == _GARBAGE

    msg = str(exc_info.value)
    assert "unparseable" in msg
    # snippet: first 200 chars of the reply must appear (repr-escaped) in the message
    assert repr(_GARBAGE[:200]) in msg


@pytest.mark.asyncio
async def test_extraction_dump_failure_does_not_mask_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    monkeypatch.setattr(service, "_format_soul_context_for_prompt", lambda *a, **kw: "")
    monkeypatch.setattr(service.fs, "base", None)
    stub = _ExtractionStub(first=_GARBAGE, second=_GARBAGE)

    with pytest.raises(ValueError, match="unparseable"):
        await service._generate_entries_from_text(
            resource_text="[Marcos] Something happened.",
            store=None,  # type: ignore[arg-type]
            memory_types=["social"],
            categories_prompt_str="communication",
            llm_client=stub,
        )


@pytest.mark.asyncio
async def test_router_retry_succeeds_and_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    """Router: garbage → valid JSON: retry succeeds, error logged."""
    service = _service()
    stub = _RouterStub([_GARBAGE, _VALID_ROUTER])

    import logging
    with caplog.at_level(logging.ERROR):
        routed, episodes = await service._route_segment(
            "episode text",
            ["knowledge"],
            llm_client=stub,
            source_days=["2026-01-02"],
        )

    assert "knowledge" in routed
    assert episodes[0]["item"] == "Compact story."
    assert any(
        "unparseable" in r.message.lower() or "retry" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    )


@pytest.mark.asyncio
async def test_router_retry_raises_on_double_garbage(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Router: garbage → garbage: raises ValueError."""
    service = _service()
    monkeypatch.setattr(service.fs, "base", tmp_path)
    stub = _RouterStub([_GARBAGE, _GARBAGE])

    with pytest.raises(ValueError, match="invalid"):
        await service._route_segment(
            "episode text",
            ["knowledge"],
            llm_client=stub,
            source_days=["2026-01-02"],
        )

    dumps = list((tmp_path / "extraction_dumps").iterdir())
    assert len(dumps) == 1
    assert "router_attempt2" in dumps[0].name
    assert dumps[0].read_text() == _GARBAGE


@pytest.mark.asyncio
async def test_router_retries_invalid_episode_shape() -> None:
    service = _service()
    invalid = '{"excluded_types": [], "episodes": [{"title": "", "episode_summary": "Story."}]}'
    stub = _RouterStub([invalid, _VALID_ROUTER])

    routed, episodes = await service._route_segment(
        "episode text",
        ["knowledge"],
        llm_client=stub,
        source_days=["2026-01-02"],
    )

    assert routed == ["knowledge"]
    assert episodes == [{
        "title": "T",
        "summary": "Full story.",
        "item": "Compact story.",
        "categories": ["Daily life"],
        "day": "2026-01-02",
    }]


@pytest.mark.asyncio
async def test_router_retries_empty_episode_list() -> None:
    service = _service()
    stub = _RouterStub(['{"excluded_types": [], "episodes": []}', _VALID_ROUTER])

    routed, episodes = await service._route_segment(
        "ordinary logistics still form a story",
        ["knowledge"],
        llm_client=stub,
        source_days=["2026-01-02"],
    )

    assert routed == ["knowledge"]
    assert episodes[0]["title"] == "T"
