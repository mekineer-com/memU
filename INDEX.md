# memu — Memory Engine Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Module Map (`src/memu/`)

| Package | Purpose |
|---------|---------|
| `app/service.py` | `MemoryService` — top-level facade, only public API |
| `app/dossier.py` | Dossier core policy: soul/user anchors, active/inactive sets, sparse memorize context, deterministic compact index, identity/content search views, strict `[M#]` handles, revision preparation/generation, reusable prompt rendering, and atomic revision apply |
| `app/dossier_revision.py` | Pure dossier-revision rendering, strict XML normalization, and deterministic section-patch assembly; one leading `##` section is structured prose |
| `app/memorize.py` | Memorize workflow: preprocess → route → extract → store. Roster supports same-role ambiguity + relationship-entity triggers; dedupe keys by `(source_role, speaker_id, summary)`; parse failures retry once before propagating. |
| `app/memorize_parsing.py` | Parsing/normalization seam: message-index extraction, source-message-id normalization, timestamp parsing, XML/JSON memory-type response parsing |
| `app/memorize_speakers.py` | Speaker attribution seam: speaker-id slugging, roster construction/validation, prompt-label sanitization, speaker_ref resolution |
| `app/memorize_dedupe.py` | Dedupe/supersession seam: semantic dedupe, similarity scoring, re-embed fallback, `replaces_previous_fact` supersede resolution |
| `app/memorize_categories.py` | Category seam: shared clustering, legacy dynamic-category planning, and dormant durable proposal/review prepare-render-generate-apply operations. Runtime cutover remains Slice H. Seed defaults live in `mcp-memu-server/config.json`. |
| `app/category_summary_journal.py` | Append-only category-summary journal under `memu/journal/`; writes journal then updates `MemoryCategory.summary` + `previous_summary` |
| `app/graph.py` | `GraphMixin` — graph reads, Atomic read surfaces (atoms/tags/canvas/neighborhood/similar/search), memory/category edits, approval review, hard-delete. Edge predicates: `caused_by`, `evokes`, `conflicts_with`, `parallels`, `shaped_by`. |
| `app/memorize_persistence.py` | Persistence seam: resource creation, item/link/triple writes, item-reference backfill, happened-at resolution |
| `app/memorize_segments.py` | Segment/preprocess seam: modality dispatch, segment text prep, background-tail summarization, rolling-summary merge, `on_extraction_progress` callback |
| `app/retrieve.py` | Retrieve workflow: derive `active_query` → embed → rank → judge. `force_retrieve` skips the retrieve/no-retrieve gate. |
| `app/settings.py` | Pydantic config models (MemorizeConfig, RetrieveConfig, LLMProfile, etc.) |
| `database/models.py` | Backend-agnostic records, including dossier metadata, scoped `memory_ref`, and `DossierCandidate` |
| `database/factory.py` | `build_database()` — sqlite backend selector (Postgres removed) |
| `database/interfaces.py` | `Database` Protocol — the repo surface engine code programs against |
| `database/state.py` | `DatabaseState` dataclass — in-memory cache of loaded categories/resources |
| `database/vector.py` | `cosine_topk`, `reciprocal_rank_fusion`, `salience_score`, `rerank_by_salience` |
| `database/sqlite/sqlite.py` | `SQLiteStore` — concrete backend; idempotent `_ensure_*_columns` migration helpers |
| `database/sqlite/schema.py` | Per-scope SQLAlchemy model factory (`get_sqlite_sqlalchemy_models`) |
| `database/sqlite/models.py` | Per-table model classes + `build_sqlite_table_model` |
| `database/sqlite/session.py` | Session factory; loads the required package-local `vec0.so` built by `scripts/build-sqlite-vec.sh` on every connection |
| `database/sqlite/repositories/base.py` | Base repository class for SQLite backend |
| `database/sqlite/repositories/memory_item_repo.py` | Scoped memory-item search plus atomic `[M#]` allocation and explicit migration-only ref backfill |
| `database/sqlite/repositories/memory_category_repo.py` | Category/dossier persistence, anchor reads, and deterministic activity ordering |
| `database/sqlite/repositories/category_item_repo.py` | Category–item link persistence |
| `database/sqlite/repositories/resource_repo.py` | Resource (segment file) persistence |
| `database/sqlite/repositories/entity_repo.py` | Entity persistence and lookup |
| `database/sqlite/repositories/triple_repo.py` | Triple (graph edge) persistence and temporal queries |
| `database/sqlite/repositories/dossier_candidate_repo.py` | Durable unresolved category proposals with idempotent create, review-consideration state, and atomic resolution |
| `scripts/migrate-embeddings-to-blob.py` | Offline dry-run/backup/migration tool for converting one explicitly named stopped soul DB from legacy JSON TEXT embeddings to canonical float32 BLOBs |
| `database/postgres/` | Removed |
| `database/repositories/` | Backend-agnostic Protocol contracts: memory_item, memory_category, resource, entity, triple, category_item |
| `llm/wrapper.py` | LLM client factory — dispatches to backends |
| `llm/http_client.py` | LLM HTTP client |
| `llm/backends/` | Provider impls: `openai.py` (httpx-based, covers OpenAI-compatible APIs) |
| `llm/claude_cli.py` | `ClaudeCLIClient` — Claude Code CLI adapter. Uses soul workspace for session/resume calls, neutral workspace for no-session calls (prevents persona bleed). |
| `embedding/` | Embedding client factory + backends (same pattern as llm/): `openai.py`, `doubao.py` |
| `workflow/` | DAG runner: `step.py` (unit), `pipeline.py` (graph), `runner.py` (executor), `interceptor.py` (hook mechanism) |
| `blob/local_fs.py` | Local filesystem media storage |
| `utils/conversation.py` | Canonical source for all AI-facing chat display: `format_grouped_chat_history()`, platform/chat headings, date dividers, `My Activities:` always first. Used by turn_contract, consolidation, and memorize rendering. |
| `utils/references.py` | Legacy `[ref:ITEM_ID]` inline citations in category summaries; canonical dossier citations use `[M#]` |
| `utils/taxonomy.py` | Shared dossier kinds, scope/embedding validation, category-name normalization, and title/description identity text |
| `utils/video.py` | Video processing utilities for frame extraction |

## Prompts (`src/memu/prompts/`)

`dossier_revision.py` and `dynamic_dossier_review.py` contain Marcos-reviewed
dossier prompts; both callables remain dormant until the taxonomy cutover.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `memory_type/` | `profile.py`, `behavior.py`, `knowledge.py`, `social.py` | Per-type extraction prompts (PROMPT + CUSTOM_PROMPT). These four are active (DEFAULT_MEMORY_TYPES). `event.py` removed — archived to `_archive/event-memory-type/`. |
| `memory_type/__init__.py` | — | PROMPTS dict, DEFAULT_MEMORY_TYPES list |
| `preprocess/` | `document.py`, `image.py`, `audio.py`, `video.py` | Input normalization for non-chat modalities |
| `router/router.py` | — | Route input by excluded memory types; produce 1–3 titled episodes with separate full summaries, compact items, category proposals, and a source-valid day |
| `retrieve/` | `pre_retrieval_decision.py` | Retrieve/no-retrieve and active-query prompt |
| `category_summary/` | `category.py`, `category_with_refs.py` | Category synthesis; treat `[reinforced Nx]` markers as frequency signals, not one-off facts |
| `consolidation/` | `consolidation.py` | Consolidation prompt: narrative_self, life_goals, intentions, edges, companion_memory |

## Task → Files

| Task | Read first | Then modify |
|------|-----------|-------------|
| Add memory type | `prompts/memory_type/__init__.py`, `database/models.py` | New `prompts/memory_type/{type}.py`, update PROMPTS dict + MemoryType literal |
| Tune extraction | `prompts/memory_type/{type}.py` | Edit PROMPT / CUSTOM_PROMPT |
| Tune routing | `prompts/router/router.py` | Edit routing prompt directly |
| Change categories | `app/settings.py`, `prompts/category_summary/` | Prompt file + settings; seed defaults in `mcp-memu-server/config.json` |
| Modify retrieval | `app/retrieve.py`, `app/settings.py`, `prompts/retrieve/pre_retrieval_decision.py` | — |
| Add LLM provider | `llm/backends/base.py` | New `llm/backends/{provider}.py`, register in `llm/wrapper.py` |
| Add embedding provider | `embedding/backends/base.py` | New `embedding/backends/{provider}.py`, register in `embedding/http_client.py` |
| Change DB schema | `database/models.py`, `database/sqlite/schema.py`, `database/sqlite/sqlite.py` | Domain model, fresh schema, then additive legacy DDL |
| Run the test suite | `tests/README.md` | — |

## Database Tables

| Table | Key Fields |
|-------|-----------|
| `MemoryItem` | id, scoped memory_ref, memory_type, summary, embedding, happened_at, provenance, merged_into, extra (JSON), approved_at |
| `MemoryCategory` | id, name, description, embedding, summary approval fields, kind/subtype/entity/anchor, evidence/revision timestamps |
| `DossierCandidate` | proposed/normalized name, item/segment/day provenance, consideration timestamp, optional resolved category/timestamp |
| `memory_ref_counters` | scoped next `[M#]` value; owned by `memory_item_repo` |
| `CategoryItem` | id, item_id, category_id |
| `Resource` | id, url, modality, local_path, caption, embedding |
| `Entity` | id, name, entity_type (person/topic/place/project), normalized, properties (JSON) |
| `Triple` | id, subject_id, subject_kind, predicate, object_id, object_kind, valid_from, valid_to (NULL=current), confidence, source_memory_id, properties (JSON) |
| `memory_item_edit_history` | item_id, old_summary, new_summary, edited_at — append-only audit log |
| `MemoryCategory.previous_summary` | stores summary before each overwrite |
