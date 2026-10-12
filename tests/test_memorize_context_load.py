from datetime import UTC, datetime
from fractions import Fraction
import json
import math

import pytest

from memu.app.dossier import render_dossier_index, select_active_dossiers
from memu.app.dossier_revision import estimate_prompt_tokens
from memu.app import memorize as engine
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
            memorize_config=settings.model_copy(update={"min_chunk_tokens": threshold}),
            primary_messages=[{"role": "user", "content": "fictional chat " * 1000}])
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


@pytest.mark.parametrize("messages", [[], [{"content": " \n\t"}],
    [{"content": "caption " * 1000, "event_kind": "image"}],
    [{"content": "background " * 1000, "memorize_chat": False}]])
def test_without_primary_words_reports_known_fixed_support_not_threshold_density(messages):
    categories = _anchors()
    settings = MemorizeConfig(memory_types=["knowledge"],
        memory_type_prompts={"knowledge": "{soul_context}\n{resource}"})
    def forecast(threshold, rows=categories):
        return estimate_memorize_context_load(categories=rows, narrative_self=None,
            memorize_config=settings.model_copy(update={"min_chunk_tokens": threshold}),
            primary_messages=messages)
    fixed = forecast(1)
    assert isinstance(fixed, int) and fixed > 0
    assert forecast(100_000) == fixed
    categories[0].summary = "stored supporting memory " * 1000
    assert forecast(1) > fixed
    assert forecast(1, categories[:1]) is None


@pytest.mark.parametrize("threshold,custom,support,content", [
    pytest.param(1, True, "", "a b c", id="short-sample-rounding"),
    pytest.param(4000, True, "a " * 3000, "denseidentifier" * 20 + " b c",
                 id="word-support-byte-chat"),
    pytest.param(4000, True, "denseidentifier" * 3000, "a b c", id="byte-support-word-chat"),
    pytest.param(7, True, "", "\u6f22\u5b57<raw>{value}" * 100 + " b c", id="utf8-escaping"),
    pytest.param(4, False, "denseidentifier" * 3000, "a b c", id="default-renderer-at-threshold"),
])
def test_projection_combines_stage_words_and_bytes_before_maximum(threshold, custom, support, content):
    categories = _anchors()
    categories[0].summary = support
    settings = MemorizeConfig(min_chunk_tokens=threshold, memory_types=["knowledge"])
    if custom:
        settings.memory_type_prompts["knowledge"] = "{soul_context}\n{resource}\n{target_items}\n{user_id}"
    messages = [{"role": "assistant", "content": content, "conversation_id": "activity:dm:fictional"}]
    sample = engine.segment_helpers._render_grouped_chat_messages(messages, soul_name="Fictional Soul")
    scale = Fraction(3 * threshold, 4 * len(content.split()))
    guide = router_episode_guidance(int(Fraction(len(sample.split()) * 4, 3) * scale))
    context = engine.MemorizeMixin._format_soul_context_for_prompt({
        "narrative_self": "Fictional narrative", "anchor_dossiers": categories,
        "dossier_index": render_dossier_index(categories), "relevant_dossiers": [],
    })
    resource = f"\n\n{engine._EPISODE_REVIEW_EXTRACTION_GUIDANCE}\n\n---\n"
    router = engine.render_router_prompt("", settings.memory_types, guide, "", "Fictional narrative")
    kwargs = dict(memorize_config=settings, memory_type="knowledge", categories_str="",
        soul_context_str=context, target_items=engine.memory_type_target_items(threshold), user_id="Test User")
    extraction = render_memory_type_prompt(resource_text=resource, **kwargs)
    expected = max(max(
        int(Fraction(len(prompt.split()) + math.ceil(len(stage.split()) * scale), 1) / Fraction(3, 4)),
        math.ceil(Fraction(len(prompt.encode()) + 1 + math.ceil(len(stage.encode()) * scale), 1) / Fraction(5, 2)),
    ) for prompt, stage in [(router, sample), (extraction, engine.escape_prompt_value(sample))])
    original = settings.model_dump(), [row.model_dump() for row in categories], [dict(row) for row in messages]
    assert estimate_memorize_context_load(categories=categories, narrative_self="Fictional narrative",
        memorize_config=settings, user_id="Test User", primary_messages=messages,
        soul_name="Fictional Soul") == expected
    assert (settings.model_dump(), [row.model_dump() for row in categories], messages) == original
    assert engine.segment_helpers._estimate_text_tokens(content) == 4
    assert engine.memory_type_target_items(threshold) == f"up to {max(1, math.ceil(1.2 * threshold / 1000))}"
    if threshold == 4:
        assert expected == max(estimate_prompt_tokens("\n" + engine.render_router_prompt(
            sample, settings.memory_types, guide, "", "Fictional narrative")),
            estimate_prompt_tokens("\n" + render_memory_type_prompt(resource_text=resource + sample, **kwargs)))


def test_wrapper_newline_is_fixed_overhead_without_chat():
    settings = MemorizeConfig(memory_types=["knowledge"], memory_type_prompts={"knowledge": "x" * 10_000})
    assert estimate_memorize_context_load(categories=_anchors(), narrative_self=None,
        memorize_config=settings) == estimate_prompt_tokens("\n" + "x" * 10_000) == 4001


@pytest.mark.parametrize("resource_copies", [0, 2])
def test_custom_resource_placement_uses_actual_extraction_renderer(resource_copies):
    categories = _anchors()
    categories[0].summary = "a " * 3000
    settings = MemorizeConfig(min_chunk_tokens=4, memory_types=["knowledge"],
        memory_type_prompts={"knowledge": "{soul_context}\n" + "{resource}\n" * resource_copies})
    messages = [{"role": "user", "content": "<raw>{value} b c"}]
    sample = engine.segment_helpers._render_grouped_chat_messages(messages)
    context = engine.MemorizeMixin._format_soul_context_for_prompt({"anchor_dossiers": categories})
    extraction = render_memory_type_prompt(memorize_config=settings, memory_type="knowledge",
        resource_text=f"\n\n{engine._EPISODE_REVIEW_EXTRACTION_GUIDANCE}\n\n---\n{sample}",
        categories_str="", soul_context_str=context, target_items=engine.memory_type_target_items(4))
    router = engine.render_router_prompt(sample, settings.memory_types,
        router_episode_guidance(engine.segment_helpers._estimate_text_tokens(sample)), "", None)
    assert estimate_memorize_context_load(categories=categories, narrative_self=None,
        memorize_config=settings, primary_messages=messages) == max(
            estimate_prompt_tokens("\n" + router), estimate_prompt_tokens("\n" + extraction))


def test_rendered_word_pressure_sets_guidance_and_relevant_union(monkeypatch):
    categories = _anchors() + [MemoryCategory(name=f"Topic {index}", kind="topic", description="Topic",
        summary=f"support-{index} " * (index + 1) * 100,
        last_evidence_at=datetime(2026, 1, index + 1, tzinfo=UTC)) for index in range(8)]
    messages = [{"role": "user", "content": "a " * 1900}]
    settings = MemorizeConfig(min_chunk_tokens=2500, active_dossiers_per_kind=10, memory_types=["knowledge"])
    sample = engine.segment_helpers._render_grouped_chat_messages(messages)
    pressure = len(sample.split()) * 2500 // 1900
    assert router_episode_guidance(2500) == 1
    assert router_episode_guidance(pressure) == 2
    contexts = []
    render = engine.render_memory_type_prompt
    def capture(**kwargs):
        contexts.append(kwargs["soul_context_str"])
        return render(**kwargs)
    monkeypatch.setattr(engine, "render_memory_type_prompt", capture)
    estimate_memorize_context_load(categories=categories, narrative_self=None,
        memorize_config=settings, primary_messages=messages)
    assert all(f"support-{index}" in contexts[0] for index in range(2, 8))
    assert all(f"support-{index}" not in contexts[0] for index in range(2))


def test_activity_recaps_are_sampled_but_images_and_mem_off_rows_are_not():
    settings = MemorizeConfig()
    categories = _anchors()
    recap = {"role": "assistant", "conversation_id": "activity:dm:fictional",
             "content": "Completed fictional task " * 30}
    def forecast(messages):
        return estimate_memorize_context_load(categories=categories, narrative_self=None,
            memorize_config=settings, primary_messages=messages, soul_name="Fictional Soul")
    assert forecast([recap]) > forecast([])
    assert forecast([recap, {"event_kind": "image", "content": "caption " * 10000},
        {"memorize_chat": False, "content": "background " * 10000}]) == forecast([recap])


def test_proportional_sample_growth_does_not_scale_backlog_cost(monkeypatch):
    messages = [{"role": "user", "content": "a b c"}]
    rendered = engine.segment_helpers._render_grouped_chat_messages(messages)
    # Hold grouping/header density constant; real grouping can change sample density.
    monkeypatch.setattr(engine.segment_helpers, "_render_grouped_chat_messages",
        lambda rows, **_kwargs: (rendered + "\n") * len(rows))
    kwargs = dict(categories=_anchors(), narrative_self=None, memorize_config=MemorizeConfig())
    assert estimate_memorize_context_load(primary_messages=messages, **kwargs) == \
        estimate_memorize_context_load(primary_messages=messages * 100, **kwargs)


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
