import json
import time
from datetime import datetime

import pytest

from memu.app.service import MemoryService
from memu.app.memorize_segments import grouped_chat_happened_at, _render_grouped_chat_messages
from memu.utils.conversation import format_grouped_chat_history, parse_happened_at


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_extract_message_happened_at_map_uses_grouped_chat_timestamp_order() -> None:
    service = _service()
    raw_text = json.dumps([
        {"role": "user", "content": "one", "ts_ms": 1737849600000},
        {"role": "assistant", "content": "two", "timestamp": "2025-01-26T03:40:49.205Z"},
        {"role": "user", "content": "three", "received_at": "2025-01-27T01:02:03Z"},
        {"role": "user", "content": "four", "created_at": "2025-01-28T01:02:03Z"},
    ])

    happened_at_map = service._extract_message_happened_at_map(raw_text)

    assert sorted(happened_at_map) == [0, 1, 2, 3]
    assert happened_at_map[1] is not None
    assert happened_at_map[2] is not None
    assert happened_at_map[3] is not None
    assert happened_at_map[0] == datetime.fromtimestamp(1737849600)
    for idx, row in enumerate(json.loads(raw_text)[1:], start=1):
        raw = row.get("timestamp") or row.get("received_at") or row.get("created_at")
        assert happened_at_map[idx] == datetime.fromisoformat(raw).astimezone().replace(tzinfo=None)


def test_grouped_chat_happened_at_accepts_timestamp_fallback() -> None:
    happened_at = grouped_chat_happened_at({"timestamp": "2026-01-02T10:00:00-05:00"})

    assert happened_at is not None
    assert happened_at == datetime.fromisoformat("2026-01-02T10:00:00-05:00").astimezone().replace(tzinfo=None)


def test_grouped_chat_happened_at_preserves_calendar_only_day() -> None:
    assert grouped_chat_happened_at({"received_at": "2026-01-02"}) == datetime(2026, 1, 2)


def test_imported_source_day_and_app_label_survive_utc_normalization() -> None:
    row = {"conversation_id": "import:dm:fictional-chat", "app_label": "Nomi",
           "source_day": "2025-02-01", "received_at": "2025-01-31T22:30:00+00:00",
           "role": "user", "name": "TestSpeaker", "content": "hello"}
    assert grouped_chat_happened_at(row) == datetime(2025, 2, 1)
    rendered = format_grouped_chat_history([row], time_label_resolver=lambda value: value)
    assert "My Nomi Conversations:" in rendered and "--- 2025-02-01 ---" in rendered
    assert "[TestSpeaker] hello" in rendered and "My SillyTavern" not in rendered


def test_resolve_entry_happened_at_uses_memory_selected_day() -> None:
    service = _service()
    happened_at_map = {
        "2025-01-25": datetime.fromtimestamp(1737849600),
        "2025-01-26": datetime.fromtimestamp(1737936000),
    }

    assert service._resolve_entry_happened_at("2025-01-26", happened_at_map) == datetime.fromtimestamp(1737936000)
    assert service._resolve_entry_happened_at(None, happened_at_map) is None


@pytest.fixture(params=["UTC", "America/Lima", "America/New_York"])
def host_timezone(request, monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("Timezone switching requires tzset")
    with monkeypatch.context() as patch:
        patch.setenv("TZ", request.param)
        time.tzset()
        try:
            yield request.param
        finally:
            patch.undo()
            time.tzset()


@pytest.mark.parametrize("stamp", [
    "2026-10-09T02:30:00Z",
    "2026-10-08T23:30:00-05:00",
    "2026-03-08T07:30:00+00:00",
])
def test_source_instants_share_local_calendar_dates(host_timezone, stamp) -> None:
    instant = datetime.fromisoformat(stamp)
    epoch = instant.timestamp()
    expected = datetime.fromtimestamp(epoch)
    for raw in (stamp, instant, epoch, epoch * 1000, str(epoch * 1000)):
        assert parse_happened_at(raw) == expected

    row = {"conversation_id": "chat:fictional-local", "_message_index": 0,
           "role": "user", "content": "A fictional evening.",
           "received_at": stamp, "ts_ms": epoch * 1000}
    original = dict(row)
    happened = grouped_chat_happened_at(row)
    assert happened == expected
    assert _service()._extract_message_happened_at_map(json.dumps([row])) == {0: expected}
    assert f"--- {expected.date().isoformat()} (" in _render_grouped_chat_messages([row])
    assert row == original


def test_calendar_dates_and_source_day_are_not_shifted(host_timezone) -> None:
    for raw in ("2026-01-02", "2026-01-02T10:30:00", datetime(2026, 1, 2, 10, 30)):
        expected = datetime(2026, 1, 2) if raw == "2026-01-02" else datetime(2026, 1, 2, 10, 30)
        assert parse_happened_at(raw) == expected
    row = {"conversation_id": "import:dm:fictional-chat", "app_label": "OtherApp",
           "source_day": "2025-02-01", "received_at": "2025-01-31T22:30:00+00:00",
           "_message_index": 0, "role": "user", "content": "An imported message."}
    original = dict(row)
    assert grouped_chat_happened_at(row) == datetime(2025, 2, 1)
    assert "--- 2025-02-01 (" in _render_grouped_chat_messages([row])
    assert row == original
    assert parse_happened_at("2026-02-30") is None
