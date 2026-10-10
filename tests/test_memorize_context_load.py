from datetime import UTC, datetime
import json

import pytest

from memu.app.dossier import select_active_dossiers
from memu.app.memorize import estimate_memorize_context_load, render_memory_type_prompt, router_episode_guidance
from memu.app.service import MemoryService
from memu.app.settings import CustomPrompt, MemorizeConfig, PromptBlock
from memu.database.models import MemoryCategory


def _anchors():
    return [MemoryCategory(name=f"Test {role}", description="Fictional anchor", kind="lore", anchor_role=role)
            for role in ("soul", "user")]


def test_forecast_is_pure_includes_dominant_prose_and_tracks_threshold_guidance(monkeypatch):
    monkeypatch.setattr(MemoryService, "__init__", lambda *_a, **_k: pytest.fail("Forecast must not build a service"))
    settings = MemorizeConfig()
    categories = _anchors() + [MemoryCategory(
        name=f"Topic {index}", description="Fictional description", kind="topic",
        summary="memory " * (index + 1) * 100, last_evidence_at=datetime(2026, 1, 1, tzinfo=UTC),
    ) for index in range(20)]
    def forecast(rows=categories, threshold=8000):
        return estimate_memorize_context_load(categories=rows, narrative_self="Fictional narrative",
            memorize_config=settings.model_copy(update={"min_chunk_tokens": threshold}))
    original = [row.model_dump() for row in categories]
    assert forecast() > forecast(categories[:2]) + 1000
    assert forecast(threshold=10_001) > forecast(threshold=10_000)
    assert forecast(threshold=4000) < forecast()
    assert forecast(list(reversed(categories))) == forecast()
    legacy = MemoryCategory(name="Legacy", description="legacy " * 100_000, kind=None)
    assert forecast(categories + [legacy]) == forecast()
    assert [row.model_dump() for row in categories] == original
    assert forecast([]) is None
    assert [router_episode_guidance(n) for n in (0, 2500, 2501, 10000, 10001, 75000, 75001)] == [1, 1, 2, 4, 5, 9, 10]


def test_active_policy_spills_reserved_slots_and_forecast_excludes_inactive_prose():
    categories = _anchors() + [MemoryCategory(
        name=f"Topic {index}", description="Domain", kind="topic", summary="prose " * 100,
        last_evidence_at=datetime(2026, 1, index + 1, tzinfo=UTC),
    ) for index in range(5)]
    settings = MemorizeConfig(active_dossiers_per_kind=1)
    active = select_active_dossiers(categories, 1)
    assert {row.name for row in active if row.anchor_role is None} == {"Topic 2", "Topic 3", "Topic 4"}
    def forecast():
        return estimate_memorize_context_load(categories=categories, narrative_self=None, memorize_config=settings)
    before = forecast()
    categories[2].summary = "inactive prose " * 100_000
    assert forecast() == before
    categories[-1].summary = "active prose " * 1000
    assert forecast() > before


def test_structured_custom_prompt_rendering_preserves_order_escaping_and_user_name():
    settings = MemorizeConfig()
    settings.memory_type_prompts["knowledge"] = CustomPrompt(root={
        "later": PromptBlock(ordinal=2, prompt="Second {categories_str} / {user_id}"),
        "first": PromptBlock(ordinal=0, prompt="First {resource} / {soul_context} / {target_items}"),
        "off": PromptBlock(ordinal=-1, prompt="Hidden"),
    })
    kwargs = dict(memory_type="knowledge", resource_text="<raw>{value}", categories_str="Known",
                  soul_context_str="Context", target_items="up to 3", user_id="Configured User")
    expected = "First &lt;raw&gt;{{value}} / Context / up to 3\n\nSecond Known / Configured User"
    assert render_memory_type_prompt(memorize_config=settings, **kwargs) == expected
    service = MemoryService(database_config={"metadata_store": {"dsn": "sqlite:///:memory:"}},
                            memorize_config=settings, user_config={"user_name": "Configured User"})
    try:
        assert service._build_memory_type_prompt(**{**kwargs, "user_id": "ignored"}) == expected
    finally:
        service.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["batch", "image", "resume"])
@pytest.mark.parametrize("missing", ["extract", "category", "cli"])
async def test_missing_capacity_stops_both_profiles_before_any_embedding(monkeypatch, entry, missing):
    service = MemoryService(database_config={"metadata_store": {"dsn": "sqlite:///:memory:"}},
                            memorize_config={"enforce_input_budget": True}, claude_code=missing == "cli")
    default = service.llm_profiles.profiles["default"]
    default.context_window_tokens = None if missing == "extract" else 100_000
    service.llm_profiles.profiles["category"] = default.model_copy(update={
        "context_window_tokens": None if missing == "category" else 100_000, "chat_model": "fictional-category",
    })
    service.memorize_config.category_update_llm_profile = "category"
    monkeypatch.setattr(service, "_select_embedding_client", lambda *_a, **_k: pytest.fail("Paid embedding selected"))
    scope = {"user_id": "test-user", "soul_id": "test-soul"}
    try:
        with pytest.raises(ValueError, match="context_window_tokens is required"):
            if entry == "batch":
                await service.memorize_segments_batch(modality="conversation", user=scope,
                    segments=[{"resource_url": "memory://fictional", "segment": {}}])
            elif entry == "image":
                await service.memorize(resource_url="memory://fictional", modality="image", user=scope,
                                       local_path="not-read.png", caption="Fictional caption")
            else:
                await service.resume_memorize_segment(segment_id="saved", phase="dedupe", user=scope,
                                                      on_dedupe_complete=lambda _session: None)
    finally:
        service.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_context_only_batch_needs_no_unused_chat_capacity_or_paid_client(monkeypatch, missing):
    service = MemoryService(database_config={"metadata_store": {"dsn": "sqlite:///:memory:"}},
                            memorize_config={"enforce_input_budget": True, "background_extra_messages_tokens": 1})
    service.llm_profiles.profiles["default"].context_window_tokens = None if missing else 100_000
    service.llm_profiles.profiles["category"] = service.llm_profiles.profiles["default"].model_copy(
        update={"context_window_tokens": None},
    )
    service.memorize_config.category_update_llm_profile = "category"
    class Client:
        chat_model = "fictional-model"
        async def chat(self, *_a, **_k):
            pytest.fail("Context-only batch must not extract memories")
        async def embed(self, *_a, **_k):
            pytest.fail("Context-only batch must not embed memory items")
    service._llm_clients.update(default=Client(), embedding=Client())
    monkeypatch.setattr(service, "_select_chat_client", lambda *_a, **_k: pytest.fail("No chat call to budget"))
    kwargs = dict(modality="conversation", user={"user_id": "test-user", "soul_id": "test-soul"},
        segments=[{"resource_url": "memory://context-only", "segment": {"context_only": True},
                   "raw_text": json.dumps([{"role": "user", "content": "background " * 100,
                                            "memorize_chat": False, "received_at": "2026-01-01T00:00:00Z"}])}])
    try:
        result = await service.memorize_segments_batch(**kwargs)
        assert result and result[0]["items"] == []
    finally:
        service.database.close()
