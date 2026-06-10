"""Tests for memory item reference utilities."""

from __future__ import annotations

from memu.utils.references import extract_references


class TestExtractReferences:
    """Tests for extract_references function."""

    def test_extract_single_reference(self):
        text = "User loves coffee [ref:abc123]."
        assert extract_references(text) == ["abc123"]

    def test_extract_multiple_references(self):
        text = "User loves coffee [ref:abc123]. Also tea [ref:def456]."
        assert extract_references(text) == ["abc123", "def456"]

    def test_extract_comma_separated_references(self):
        text = "User prefers hot drinks [ref:abc,def,ghi]."
        assert extract_references(text) == ["abc", "def", "ghi"]

    def test_extract_no_duplicates(self):
        text = "Coffee [ref:abc]. More coffee [ref:abc]. Tea [ref:def]."
        assert extract_references(text) == ["abc", "def"]

    def test_extract_empty_text(self):
        assert extract_references("") == []
        assert extract_references(None) == []

    def test_extract_no_references(self):
        assert extract_references("User loves coffee and tea.") == []

    def test_extract_with_hyphens_and_underscores(self):
        text = "Info [ref:item_abc-123]."
        assert extract_references(text) == ["item_abc-123"]
