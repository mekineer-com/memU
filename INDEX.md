# memu — Memory Engine Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Module Map (`src/memu/`)

| Package | Purpose |
|---------|---------|
| `app/service.py` | `MemoryService` — top-level facade, only public API |
| `app/memorize.py` | Memorize workflow: preprocess → route → extract → store; `all_categories_summary` + `soul_card` threaded into extraction soul-context; deterministic speaker attribution (`speaker_id`/`speaker_label`) from episode message metadata + message-index provenance; extraction roster now supports two trigger paths: ambiguous same-role episodes, or episodes that mention user-declared relationship entities (so 1:1 chats can still emit valid `<speaker_ref>` for quoted third parties); roster refs are fail-closed (unknown refs ignored); semantic dedupe now keys by `(source_role, speaker_id, summary)` so same content from different speakers does not collapse; reinforcement roll-up on dedupe merge (`reinforcement_count` + `last_reinforced_at` accumulated on survivor when `enable_item_reinforcement=true`); conversation path now exposes `split_segment_into_episodes()` + `memorize_episode()` + `memorize_episodes_batch()` (router exclusion + cross-episode extraction with strict `<episode_ref>` attribution); after each MemoryItem is created, `<entities>` XML from the extraction response is parsed → `entity_repo.get_or_create()` per entity → `mentions` triple written; supersession also writes `evolved_into` triple (old item → new) |
| `app/memorize_parsing.py` | Parsing/normalization seam for memorize extraction inputs/outputs: message-index extraction, source-message-id normalization fallback, conversation timestamp parsing, and XML/JSON memory-type response parsing |
| `app/memorize_speakers.py` | Speaker attribution seam for memorize: speaker-id slugging, roster construction/validation, prompt-label sanitization, parsed speaker_ref resolution, and source-message speaker attribution helpers |
| `app/memorize_dedupe.py` | Dedupe/supersession seam for memorize: semantic dedupe scope/filtering, similarity scoring and re-embed fallback, merged-category-update filtering, and `replaces_previous_fact` supersede target resolution |
| `app/memorize_categories.py` | Category seam for memorize: homeless-entry clustering, dynamic-category planning/creation, category init/scope mapping, and category-summary update rendering |
| `app/memorize_persistence.py` | Persistence seam for memorize: resource creation, item/link/triple writes, item-reference backfill, and happened-at resolution |
| `app/memorize_episodes.py` | Episode/preprocess seam for memorize: modality preprocessing dispatch, conversation episode splitting, background-context rendering, multimodal response parsing, and episode payload normalization |
| `app/retrieve.py` | Retrieve workflow: rewrite query → embed → rank → judge; server-provided context queries (identity, summaries, cache, intentions, recent history) are preserved across steps and rendered as plain text in soul context; optional `as_of` filters graph edges by `valid_from`/`valid_to`; serialized retrieved memory items explicitly carry `speaker_id` + `speaker_label` when present |
| `app/settings.py` | Pydantic config models (MemorizeConfig, RetrieveConfig, LLMProfile, etc.) |
| `database/models.py` | Backend-agnostic data models (MemoryItem, MemoryCategory, Resource, Entity, Triple); `EntityType` literal; `PREDICATES` literal (`caused_by`, `evokes`, `evolved_into`, `conflicts_with`, `parallels`, `shaped_by`, `mentions`) |
| `database/factory.py` | `build_database()` — sqlite backend selector (Postgres backend removed) |
| `database/interfaces.py` | `Database` Protocol — the repo surface engine code programs against |
| `database/state.py` | `DatabaseState` dataclass — in-memory cache of loaded categories/resources used by workflow ctx |
| `database/vector.py` | Cosine + RRF helpers: `cosine_topk`, `reciprocal_rank_fusion`, `salience_score`, `rerank_by_salience` |
| `database/sqlite/sqlite.py` | `SQLiteStore` — concrete backend; includes idempotent `_ensure_*_columns` migration helpers |
| `database/sqlite/schema.py` | Per-scope SQLAlchemy model factory (`get_sqlite_sqlalchemy_models`); deep-copies columns per derivation |
| `database/sqlite/models.py` | Per-table model classes + `build_sqlite_table_model` — wires scope fields into each table |
| `database/sqlite/session.py` | Session factory + async engine wrapper |
| `database/postgres/` | Removed. If Postgres returns, rebuild as a thin adapter over shared repo logic. |
| `database/repositories/` | Backend-agnostic Protocol contracts: `memory_item.py`, `memory_category.py`, `resource.py`, `entity.py`, `triple.py`, `category_item.py` |
| `llm/wrapper.py` | LLM client factory — dispatches to backends |
| `llm/backends/` | Provider impls: `openai.py` (httpx-based, covers OpenAI-compatible APIs) |
| `embedding/` | Embedding client factory + backends (same pattern as llm/) |
| `workflow/` | DAG runner: `step.py` (unit), `pipeline.py` (graph), `runner.py` (executor) |
| `blob/local_fs.py` | Local filesystem media storage |
| `utils/` | Format converters (conversation, references, video) |

## Prompts (`src/memu/prompts/`)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `memory_type/` | `profile.py`, `behavior.py`, `knowledge.py`, `social.py` | Per-type extraction prompts (PROMPT + CUSTOM_PROMPT). These four are active (DEFAULT_MEMORY_TYPES). `skill.py` and `tool.py` exist but are not active extraction types. `event.py` removed — episodes are episodic memory; archived to `_archive/event-memory-type/`. |
| `memory_type/__init__.py` | — | PROMPTS dict, DEFAULT_MEMORY_TYPES list |
| `preprocess/` | `conversation.py`, `document.py`, `image.py`, `audio.py`, `video.py` | Input normalization per modality |
| `router/router.py` | — | Classify input → memory type(s) and `notable` flag in one pass; writes `episode_summary` (resource caption) and optional `episode_item` (condensed 1-2 sentence memory when summary >2 sentences) |
| `retrieve/` | `llm_category_ranker.py`, `llm_item_ranker.py`, `llm_resource_ranker.py`, `judger.py`, `pre_retrieval_decision.py` | Retrieval ranking & judgment |
| `category_patch/` | `category.py` | Dynamic category update prompts |
| `category_summary/` | `category.py`, `category_with_refs.py` | Category synthesis; both prompts treat `[reinforced Nx]` markers as frequency signals — instruct LLM to use "often", "frequently", "tends to" rather than treating as a one-off fact |
| `consolidation/` | `consolidation.py` | Consolidation prompt: narrative_self, life_goals, intentions, edges, companion_memory. Weekly reflection cycle. |

## Task → Files

| Task | Read first | Then modify |
|------|-----------|-------------|
| Add memory type | `prompts/memory_type/__init__.py`, `database/models.py` | New `prompts/memory_type/{type}.py`, update `__init__.py` PROMPTS dict, add to MemoryType literal |
| Tune extraction | `prompts/memory_type/{type}.py` | Edit PROMPT / CUSTOM_PROMPT in that file |
| Tune routing | `prompts/router/router.py` | Edit routing prompt directly |
| Change categories | `app/settings.py` (CategoryConfig), `prompts/category_summary/` | Target prompt file + settings. **Seed defaults live in `mcp-memu-server/config.json` `categories.defaults[]`** — engine settings.py defaults are overridden by the server. |
| Modify retrieval | `app/retrieve.py`, `prompts/retrieve/` | Ranker prompts or retrieve.py logic |
| Add LLM provider | `llm/backends/base.py`, any existing backend | New `llm/backends/{provider}.py`, register in `llm/wrapper.py` |
| Add embedding provider | `embedding/backends/base.py` | New `embedding/backends/{provider}.py`, register in `embedding/http_client.py` |
| Change DB schema | `database/models.py`, `database/sqlite/schema.py` | Both files (sqlite only in current codebase) |
| Run the test suite | `tests/README.md` | — |

## Database Tables

| Table | Key Fields |
|-------|-----------|
| `MemoryItem` | id, memory_type, summary, embedding, happened_at, source_role, speaker_id, speaker_label, confidence, emotional_intensity, source_message_ids, reflection_salience, conversation_id, episode_id, merged_into, extra (JSON) |
| `MemoryCategory` | id, name, description, embedding, summary |
| `CategoryItem` | id, item_id, category_id |
| `Resource` | id, url, modality, local_path, caption, embedding |
| `Entity` | id, name, entity_type (person/topic/place/project), normalized (lookup key), properties (JSON) |
| `Triple` | id, subject_id, subject_kind (entity/memory), predicate, object_id, object_kind, valid_from, valid_to (NULL=current), confidence, source_memory_id, properties (JSON) |
