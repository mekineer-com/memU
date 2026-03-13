# memu-1.4.0 — Memory Engine Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Module Map (`src/memu/`)

| Package | Purpose |
|---------|---------|
| `app/service.py` | `MemoryService` — top-level facade, only public API |
| `app/memorize.py` | Memorize workflow: preprocess → route → extract → store |
| `app/retrieve.py` | Retrieve workflow: rewrite query → embed → rank → judge |
| `app/settings.py` | Pydantic config models (MemorizeConfig, RetrieveConfig, LLMProfile, etc.) |
| `app/crud.py` | Low-level memory CRUD |
| `app/patch.py` | Memory patching / update logic |
| `database/models.py` | Backend-agnostic data models (MemoryItem, MemoryCategory, Resource) |
| `database/sqlite/schema.py` | SQLAlchemy ORM schema (SQLite) |
| `database/postgres/schema.py` | SQLAlchemy ORM schema (Postgres) + alembic migrations in `postgres/migrations/` |
| `database/repositories/` | Data access layer: `memory_item.py`, `memory_category.py`, `resource.py` |
| `llm/wrapper.py` | LLM client factory — dispatches to backends |
| `llm/backends/` | Provider impls: `openai.py`, `openrouter.py`, `grok.py`, `doubao.py` |
| `embedding/` | Embedding client factory + backends (same pattern as llm/) |
| `workflow/` | DAG runner: `step.py` (unit), `pipeline.py` (graph), `runner.py` (executor) |
| `blob/local_fs.py` | Local filesystem media storage |
| `utils/` | Format converters (conversation, references, video) |

## Prompts (`src/memu/prompts/`)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `memory_type/` | `profile.py`, `event.py`, `behavior.py`, `knowledge.py`, `skill.py`, `tool.py`, `diary.py` | Per-type extraction prompts (PROMPT + CUSTOM_PROMPT) |
| `memory_type/__init__.py` | — | PROMPTS dict, DEFAULT_MEMORY_TYPES list |
| `preprocess/` | `conversation.py`, `document.py`, `image.py`, `audio.py`, `video.py` | Input normalization per modality |
| `router/router.py` | — | Classify input → memory type(s) |
| `retrieve/` | `query_rewriter.py`, `llm_category_ranker.py`, `llm_item_ranker.py`, `llm_resource_ranker.py`, `judger.py`, `pre_retrieval_decision.py` | Retrieval ranking & judgment |
| `category_patch/` | `category.py` | Dynamic category update prompts |
| `category_summary/` | `category.py`, `category_with_refs.py` | Category synthesis |
| `diary/` | `diary_worthy.py`, `self_model_update.py` | Diary generation & self-model reflection |

## Task → Files

| Task | Read first | Then modify |
|------|-----------|-------------|
| Add memory type | `prompts/memory_type/__init__.py`, `database/models.py` | New `prompts/memory_type/{type}.py`, update `__init__.py` PROMPTS dict, add to MemoryType literal |
| Tune extraction | `prompts/memory_type/{type}.py` | Edit PROMPT / CUSTOM_PROMPT in that file |
| Tune routing | `prompts/router/router.py` | Edit routing prompt directly |
| Change categories | `app/settings.py` (CategoryConfig), `prompts/category_summary/` | Target prompt file + settings |
| Modify retrieval | `app/retrieve.py`, `prompts/retrieve/` | Ranker prompts or retrieve.py logic |
| Add LLM provider | `llm/backends/base.py`, any existing backend | New `llm/backends/{provider}.py`, register in `llm/wrapper.py` |
| Add embedding provider | `embedding/backends/base.py` | New `embedding/backends/{provider}.py`, register in `embedding/http_client.py` |
| Change DB schema | `database/models.py`, `database/sqlite/schema.py` | Both files + postgres schema if needed |

## Database Tables

| Table | Key Fields |
|-------|-----------|
| `MemoryItem` | id, memory_type, summary, embedding, happened_at, source_role, confidence, conversation_id, affective_tags, merged_into, extra (JSON) |
| `MemoryCategory` | id, name, description, embedding, summary |
| `CategoryItem` | id, item_id, category_id |
| `Resource` | id, url, modality, local_path, caption, embedding |
