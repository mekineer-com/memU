"""Tests for retry-on-unparseable behavior in extraction and router paths."""
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

_VALID_ROUTER = '{"excluded_types": [], "episode_summary": "S", "episode_items": [{"title": "T", "summary": "S"}]}'


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
async def test_router_retry_succeeds_and_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    """Router: garbage → valid JSON: retry succeeds, error logged."""
    service = _service()
    stub = _RouterStub([_GARBAGE, _VALID_ROUTER])

    import logging
    with caplog.at_level(logging.ERROR):
        routed, summary, items = await service._route_episode(
            "episode text",
            ["knowledge"],
            llm_client=stub,
        )

    assert "knowledge" in routed
    assert any(
        "unparseable" in r.message.lower() or "retry" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    )


@pytest.mark.asyncio
async def test_router_retry_raises_on_double_garbage() -> None:
    """Router: garbage → garbage: raises ValueError."""
    service = _service()
    stub = _RouterStub([_GARBAGE, _GARBAGE])

    with pytest.raises(ValueError, match="unparseable"):
        await service._route_episode(
            "episode text",
            ["knowledge"],
            llm_client=stub,
        )
