"""Tests for Tool Memory feature - specialized memory type for tracking tool usage."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for direct import - MUST be before any memu imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest  # noqa: E402

# Import directly from the models file path to avoid circular import through database/__init__.py
# We use importlib to import the module directly without triggering the package __init__
spec = importlib.util.spec_from_file_location("models", src_path / "memu" / "database" / "models.py")
assert spec is not None
assert spec.loader is not None
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

# Rebuild models to resolve forward references with proper namespace
rebuild_ns = {
    "Any": Any,
    "datetime": datetime,
    "MemoryType": models.MemoryType,
}
models.MemoryItem.model_rebuild(_types_namespace=rebuild_ns)

MemoryItem = models.MemoryItem
MemoryType = models.MemoryType


class TestMemoryItemToolType:
    """Tests for MemoryItem with tool type."""

    def test_tool_memory_type_literal(self):
        """Test that 'tool' is a valid MemoryType."""
        from typing import get_args

        valid_types = get_args(MemoryType)
        assert "tool" in valid_types

    def test_create_tool_memory(self):
        """Test creating a tool type memory item with tool fields in extra."""
        item = MemoryItem(
            resource_id=None,
            memory_type="tool",
            summary="file_reader tool usage for config files",
            extra={
                "when_to_use": "When needing to read configuration files",
                "metadata": {"tool_name": "file_reader", "avg_success_rate": 0.95},
            },
        )

        assert item.memory_type == "tool"
        assert item.extra["when_to_use"] == "When needing to read configuration files"
        assert item.extra["metadata"]["tool_name"] == "file_reader"



class TestMemoryItemNewFields:
    """Tests for tool-related fields stored in extra."""

    def test_when_to_use_field(self):
        """Test when_to_use field stored in extra for retrieval hints."""
        item = MemoryItem(
            resource_id=None,
            memory_type="profile",
            summary="User prefers dark mode",
            extra={"when_to_use": "When configuring UI settings or themes"},
        )

        assert item.extra["when_to_use"] == "When configuring UI settings or themes"

    def test_metadata_field(self):
        """Test metadata field stored in extra for type-specific data."""
        item = MemoryItem(
            resource_id=None,
            memory_type="knowledge",
            summary="User attended conference",
            extra={
                "metadata": {
                    "event_date": "2026-01-15",
                    "location": "San Francisco",
                    "attendees": ["Alice", "Bob"],
                }
            },
        )

        assert item.extra.get("metadata") is not None
        assert item.extra["metadata"]["event_date"] == "2026-01-15"
        assert item.extra["metadata"]["location"] == "San Francisco"
        assert len(item.extra["metadata"]["attendees"]) == 2

    def test_default_values(self):
        """Test that extra defaults to empty dict."""
        item = MemoryItem(
            resource_id=None,
            memory_type="knowledge",
            summary="Python is a programming language",
        )

        assert item.extra.get("when_to_use") is None
        assert item.extra.get("metadata") is None
        assert item.extra.get("tool_calls") is None
