# memu — Memory Engine Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

`# Marcos' reminder: a category is a dossier.`

## Module Map (`src/memu/`)

| Package | Purpose |
|---------|---------|
| `app/service.py` | `MemoryService` — top-level facade, only public API |
| `app/dossier.py` | Dossier core policy: soul/user anchors, active/inactive sets, sparse memorize context, deterministic compact index, identity/content search views, strict `[M#]` handles, revision preparation/generation, reusable prompt rendering, and atomic revision apply |
| `app/dossier_revision.py` | Pure dossier-revision rendering, strict XML normalization, and deterministic section-patch assembly; one leading `##` section is structured prose |
| `app/memorize.py` | Memorize workflow: preprocess → route → extract → store. Roster supports same-role ambiguity + relationship-entity triggers; dedupe keys by `(source_role, speaker_id, summary)`; parse failures retry once before propagating. |
| `app/memorize_parsing.py` | Parsing/normalization seam: message-index extraction, source-message-id normalization, timestamp parsing, XML/JSON memory-type response parsing |
| `app/memorize_speakers.py` | Speaker attribution seam: stable entity-ID resolution, roster construction/validation, prompt-label sanitization, speaker_ref resolution |
| `app/memorize_dedupe.py` | Dedupe/supersession seam: semantic dedupe, similarity scoring, re-embed fallback, `replaces_previous_fact` supersede resolution |
| `app/memorize_categories.py` | Dossier filing seam: category proposals, reviewed candidate clustering, and dynamic dossier prepare-render-generate-apply operations. |
| `app/category_summary_journal.py` | Append-only category-summary journal under `memu/journal/`; writes journal then updates `MemoryCategory.summary` + `previous_summary` |
| `app/graph.py` | `GraphMixin` — graph reads, exact-`M#` and filtered Atomic curation search, detail-only dossier membership with transactional attach/detach, canonical dossier metadata/citation projection, scoped entity list/detail/create/update/merge/ignore/restore/safe-delete and transactional memory-entity assignment, memory/category edits, approval review, hard-delete. Edge predicates: `caused_by`, `evokes`, `conflicts_with`, `parallels`, `shaped_by`. |
| `app/memorize_persistence.py` | Persistence seam: resource creation, item/triple writes, and happened-at resolution |
| `app/memorize_segments.py` | Segment/preprocess seam: modality dispatch, segment text prep, background-tail summarization, rolling-summary merge, `on_extraction_progress` callback |
| `app/retrieve.py` | Retrieve workflow: derive `active_query` plus optional temporal bounds → dossier sufficiency (optional exact `[M#]` follow-up) → hybrid/graph recall. Temporal matches are softly promoted; exact requested memories supplement item `top_k`; entity-linked graph additions must clear the configured query-similarity floor. `force_retrieve` skips the retrieve/no-retrieve gate. |
| `app/settings.py` | Pydantic config models (MemorizeConfig, RetrieveConfig, LLMProfile, etc.) |
| `database/models.py` | Backend-agnostic records, including dossier metadata, scoped `memory_ref`, and `DossierCandidate` |
| `database/factory.py` | `build_database()` — sqlite backend selector (Postgres removed) |
| `database/interfaces.py` | `Database` Protocol — the repo surface engine code programs against |
| `database/state.py` | `DatabaseState` dataclass — in-memory cache of loaded categories/resources |
| `database/vector.py` | `cosine_topk`, `reciprocal_rank_fusion`, `relative_score_fusion`, `autocut_first_cluster`, `salience_score`, `rerank_by_salience` |
| `database/sqlite/sqlite.py` | `SQLiteStore` — concrete backend; idempotent `_ensure_*_columns` migration helpers |
| `database/sqlite/schema.py` | Per-scope SQLAlchemy model factory (`get_sqlite_sqlalchemy_models`) |
| `database/sqlite/models.py` | Per-table model classes + `build_sqlite_table_model` |
| `database/sqlite/session.py` | Session factory; loads the required package-local `vec0.so` built by `scripts/build-sqlite-vec.sh` on every connection |
| `database/sqlite/repositories/base.py` | Base repository class for SQLite backend |
| `database/sqlite/repositories/memory_item_repo.py` | Scoped memory-item search plus atomic `[M#]` allocation and explicit migration-only ref backfill |
| `database/sqlite/repositories/memory_category_repo.py` | Category/dossier persistence, anchor reads, and deterministic activity ordering |
| `database/sqlite/repositories/category_item_repo.py` | Category–item link persistence |
| `database/sqlite/repositories/resource_repo.py` | Resource (segment file) persistence |
| `database/sqlite/repositories/entity_repo.py` | Scoped entity persistence, stable-ID edits, aliases, and source-reference lookup |
| `database/sqlite/repositories/triple_repo.py` | Triple (graph edge) persistence and temporal queries |
| `database/sqlite/repositories/dossier_candidate_repo.py` | Durable unresolved category proposals with idempotent create, review-consideration state, and atomic resolution |
| `scripts/build-sqlite-vec.sh` | Builds the package-local `vec0.so` sqlite-vec extension |
| `scripts/migrate-embeddings-to-blob.py` | Offline dry-run/backup/migration tool for converting one explicitly named stopped soul DB from legacy JSON TEXT embeddings to canonical float32 BLOBs |
| `scripts/migrate-entity-speaker-ids.py` | One-time dry-run/backup migration from legacy name-derived entity speaker refs to stable entity IDs; archive after the release cutover |
| `database/postgres/` | Removed |
| `database/repositories/` | Backend-agnostic Protocol contracts: memory_item, memory_category, resource, entity, triple, category_item, dossier_candidate |
| `llm/wrapper.py` | LLM client factory — dispatches to backends |
| `llm/http_client.py` | LLM HTTP client |
| `llm/backends/` | Provider impls: `openai.py` (httpx-based, covers OpenAI-compatible APIs) |
| `llm/claude_cli.py` | `ClaudeCLIClient` — Claude Code CLI adapter. Uses soul workspace for session/resume calls, neutral workspace for no-session calls (prevents persona bleed). |
| `embedding/` | Embedding client factory + backends (same pattern as llm/): `openai.py`, `doubao.py` |
| `workflow/` | DAG runner: `step.py` (unit), `pipeline.py` (graph), `runner.py` (executor), `interceptor.py` (hook mechanism) |
| `blob/local_fs.py` | Local filesystem media storage |
| `utils/conversation.py` | Canonical source for all AI-facing chat display: `format_grouped_chat_history()`, ST/Atomic/WhatsApp/Smartglasses headings, date dividers, `My Activities:` always first. Used by turn_contract, consolidation, and memorize rendering. |
| `utils/taxonomy.py` | Shared dossier kinds, scope/embedding validation, category-name normalization, and title/description identity text |
| `utils/video.py` | Video processing utilities for frame extraction |

## Prompts (`src/memu/prompts/`)

`dossier_revision.py`, `dynamic_dossier_review.py`, and `consolidation/` contain
Marcos-reviewed dossier prompts.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `memory_type/` | `profile.py`, `behavior.py`, `knowledge.py`, `social.py` | Per-type extraction prompts (PROMPT + CUSTOM_PROMPT). Each generalized item chooses a source-valid day; whole-segment provenance remains code-owned. These four are active (DEFAULT_MEMORY_TYPES). |
| `memory_type/__init__.py` | — | PROMPTS dict, DEFAULT_MEMORY_TYPES list |
| `preprocess/` | `document.py`, `image.py`, `audio.py`, `video.py` | Input normalization for non-chat modalities |
| `router/router.py` | — | Route input into the configured maximum number of titled episodes, with separate full summaries, compact items, category proposals, and a source-valid day |
| `retrieve/` | `pre_retrieval_decision.py` | Retrieve/no-retrieve, active-query, and optional temporal-bound prompt |
| `consolidation/` | `dossiers.py`, `anchors.py` | Two-call reflection prompts: due life-domain dossiers, then narrative_self + soul/user anchors + goals + intentions + edges + companion memory |

## Task → Files

| Task | Read first | Then modify |
|------|-----------|-------------|
| Add memory type | `prompts/memory_type/__init__.py`, `database/models.py` | New `prompts/memory_type/{type}.py`, update PROMPTS dict + MemoryType literal |
| Tune extraction | `prompts/memory_type/{type}.py` | Edit PROMPT / CUSTOM_PROMPT |
| Tune routing | `prompts/router/router.py` | Edit routing prompt directly |
| Change dossier runtime | `app/dossier.py`, `app/memorize_categories.py`, `prompts/consolidation/` | Keep filing, revision, and reflection contracts aligned |
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
| `Entity` | id, name, free-text entity_type, normalized, properties (JSON) |
| `Triple` | id, subject_id, subject_kind, predicate, object_id, object_kind, valid_from, valid_to (NULL=current), confidence, source_memory_id, properties (JSON) |
| `memory_item_edit_history` | item_id, old_summary, new_summary, edited_at — append-only audit log |
| `MemoryCategory.previous_summary` | stores summary before each overwrite |
