# memu — Memory Engine Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Module Map (`src/memu/`)

| Package | Purpose |
|---------|---------|
| `app/service.py` | `MemoryService` — top-level facade, only public API |
| `app/memorize.py` | Memorize workflow: preprocess → route → extract → store; `all_categories_summary` + `soul_card` threaded into extraction soul-context; reinforcement roll-up on dedupe merge (`reinforcement_count` + `last_reinforced_at` accumulated on survivor when `enable_item_reinforcement=true`); after each MemoryItem is created, `<entities>` XML from the extraction response is parsed → `entity_repo.get_or_create()` per entity → `mentions` triple written; supersession also writes `evolved_into` triple (old item → new) |
| `app/retrieve.py` | Retrieve workflow: rewrite query → embed → rank → judge; `_split_context_queries()` keeps both history windows for route (`history_from_second_chat_x` = previous-window slice, `history_from_chat_x` = current-window slice), while downstream sufficiency steps keep only `history_from_chat_x`; `identity_context` is preserved across all steps and rendered as plain text at top of soul context |
| `app/settings.py` | Pydantic config models (MemorizeConfig, RetrieveConfig, LLMProfile, etc.) |
| `app/crud.py` | Low-level memory CRUD |
| `app/patch.py` | Memory patching / update logic |
| `database/models.py` | Backend-agnostic data models (MemoryItem, MemoryCategory, Resource, Entity, Triple); `EntityType` literal; `PREDICATES` literal (`caused_by`, `evokes`, `evolved_into`, `conflicts_with`, `contextualizes`, `parallels`, `shaped_by`, `mentions`) |
| `database/sqlite/schema.py` | SQLAlchemy ORM schema (SQLite) |
| `database/postgres/schema.py` | SQLAlchemy ORM schema (Postgres) + alembic migrations in `postgres/migrations/` |
| `database/repositories/` | Data access layer: `memory_item.py`, `memory_category.py`, `resource.py`, `entity.py`, `triple.py` |
| `llm/wrapper.py` | LLM client factory — dispatches to backends |
| `llm/backends/` | Provider impls: `openai.py`, `openrouter.py`, `grok.py`, `doubao.py` |
| `embedding/` | Embedding client factory + backends (same pattern as llm/) |
| `workflow/` | DAG runner: `step.py` (unit), `pipeline.py` (graph), `runner.py` (executor) |
| `blob/local_fs.py` | Local filesystem media storage |
| `utils/` | Format converters (conversation, references, video) |

## Prompts (`src/memu/prompts/`)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `memory_type/` | `profile.py`, `event.py`, `behavior.py`, `knowledge.py`, `social.py` | Per-type extraction prompts (PROMPT + CUSTOM_PROMPT). These five are active (DEFAULT_MEMORY_TYPES). `skill.py`, `tool.py`, `diary.py` exist but are not active extraction types. |
| `memory_type/__init__.py` | — | PROMPTS dict, DEFAULT_MEMORY_TYPES list |
| `preprocess/` | `conversation.py`, `document.py`, `image.py`, `audio.py`, `video.py` | Input normalization per modality |
| `router/router.py` | — | Classify input → memory type(s) and `diary_worthy` flag in one pass |
| `retrieve/` | `query_rewriter.py`, `llm_category_ranker.py`, `llm_item_ranker.py`, `llm_resource_ranker.py`, `judger.py`, `pre_retrieval_decision.py` | Retrieval ranking & judgment |
| `category_patch/` | `category.py` | Dynamic category update prompts |
| `category_summary/` | `category.py`, `category_with_refs.py` | Category synthesis; both prompts treat `[reinforced Nx]` markers as frequency signals — instruct LLM to use "often", "frequently", "tends to" rather than treating as a one-off fact |
| `diary/` | `__init__.py` | Diary prompt package placeholder (consolidation now drives diary writes from server-side orchestration). |
| `consolidation/` | `consolidation.py` | Consolidation prompt contract (broad review + per-episode diary outputs). Contains explicit `SONNET WANTED` placeholders for voice-tuning blocks. |

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
| Change DB schema | `database/models.py`, `database/sqlite/schema.py` | Both files + postgres schema if needed |

## Database Tables

| Table | Key Fields |
|-------|-----------|
| `MemoryItem` | id, memory_type, summary, embedding, happened_at, source_role, confidence, reflection_salience, conversation_id, affective_tags, merged_into, superseded_by, extra (JSON) |
| `MemoryCategory` | id, name, description, embedding, summary |
| `CategoryItem` | id, item_id, category_id |
| `Resource` | id, url, modality, local_path, caption, embedding |
| `Entity` | id, name, entity_type (person/topic/place/project), normalized (lookup key), properties (JSON) |
| `Triple` | id, subject_id, subject_kind (entity/memory), predicate, object_id, object_kind, valid_from, valid_to (NULL=current), confidence, source_memory_id, properties (JSON) |
