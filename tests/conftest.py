import pytest

from memu.app import category_summary_journal


@pytest.fixture(autouse=True)
def temporary_summary_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path / "journal")
