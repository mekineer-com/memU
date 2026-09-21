# memU — Memory Engine

Local-first memory engine for AI companions. Extracts, stores, and retrieves structured memories from conversation history.

Part of [OpenAlma](https://github.com/mekineer-com/OpenAlma). For project-wide questions and ideas, use [OpenAlma Discussions](https://github.com/mekineer-com/OpenAlma/discussions).

Based on [memU v1.4.0](https://github.com/NevaMind-AI/memU/blob/v1.4.0/README.md), significantly diverged. Unofficial fork, not affiliated with NevaMind-AI. Does not connect to any cloud service.

---

## What it does

`memu` is the core extraction and retrieval library. It is not a server — it is imported by `mcp-memu-server`, which provides the API layer.

**Four extraction types:**

| Type | What it captures |
|------|-----------------|
| `profile` | What someone said or declared — about themselves or someone else |
| `knowledge` | What was learned — facts, mechanisms, things worth carrying |
| `behavior` | How someone acted — patterns, interaction styles, ways of being |
| `social` | Dynamics between 2+ beings — how they are together, what they mean to each other |

Pipeline-owned types (not extracted from conversation): `subconscious` (APImw background thoughts), `reflection` (consolidation), `episode` (summaries).

**Batch extraction:** Episodes are routed individually (exclusion model — all types included by default), then combined into one LLM call per type with `<episode_ref>` provenance. Conversations marked `memorize_chat=false` become background context (inline summaries for routing/extraction) rather than primary extraction targets.

**Three-layer storage:**

```
Resource  →  MemoryItem  →  MemoryCategory
(raw input)   (extracted fact)   (auto-organized topic)
```

---

## Procedural memory (sidecar)

Alongside memories extracted from conversation, the soul can opt into a **curated, shared knowledge base** of professional frameworks she draws on when the moment calls for them. The soul emits a second query rewrite specifically for the sidecar; retrieve merges the top hit into the response-side memory list, tagged `[mental-health-procedural-memory]` so it's distinguishable from experiential memory.

First domain: **mental health**, 15 anchor entries at `memu/procedural/mental_health.yaml`. Scenarios:

- **Core emotion patterns:** anxious rumination, panic, low-mood / motivation, anger, perfectionism, self-criticism
- **Grief & distress:** loss acceptance, acute distress tolerance
- **Sleep:** insomnia / stimulus control
- **Relational:** conflict / validation, boundaries / people-pleasing, attachment anxiety
- **Identity & withdrawal:** loneliness, life transitions, avoidance / procrastination

Each entry reads as internalized professional knowledge — no framework names or citations appear in the rendered text; `framework` and `source` live in metadata for audit. Frameworks drawn from: CBT, DBT, CBT-I, self-compassion, attachment theory, developmental/transition models, relational/assertiveness, social-cognitive.

**Status:** live since 2026-04-22. Storage in `memu/sqlite/procedural.db` (cosine-vector lookup). Wired into `/retrieve` — the `route_intention` LLM emits a `mental_health_query` rewrite when relevant; extension's Mental Health Addon checkbox gates the lookup.

---

## Requirements

- Python 3.12
- SQLite
- An LLM provider API key (OpenAI-compatible)

---

## Installation

```bash
pip install -e .
python scripts/install-sqlite-vec.py
```

The Python build must support SQLite loadable extensions. Some macOS Python builds disable them; if the installer reports this, use a Python 3.12 build with loadable-extension support.

Or with the monorepo venv:
```bash
cd memu
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/install-sqlite-vec.py
```

SQLite storage requires the pinned `sqlite-vec` extension installed by that script. It verifies and installs the official loadable artifact for glibc Linux x86-64/ARM64, macOS Intel/Apple Silicon, or Windows x86-64. Because sqlite-vec's Linux artifact requires glibc, Alpine uses the retained source-build fallback. The generated package-local extension is ignored by Git and loaded automatically by memU.

Before the sqlite-vec storage cutover, inspect and migrate each stopped soul database explicitly:

```bash
scripts/migrate-embeddings-to-blob.py path/to/soul.db
scripts/migrate-embeddings-to-blob.py path/to/soul.db --apply
```

Dry-run is the default. `--apply` requires all processes using that database to be stopped and creates a timestamped backup before conversion.

---

## Usage

`memu` is used via `mcp-memu-server`. Direct library use:

```python
from memu.app import MemoryService

svc = MemoryService(
    llm_profiles={
        "default": {
            "provider": "openai",
            "api_key": "sk-...",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "gpt-4o-mini",
            "embed_model": "text-embedding-3-large",
        }
    },
    database_config={
        "metadata_store": {"provider": "sqlite", "dsn": "memu.db"}
    },
)

# Memorize a conversation
await svc.memorize(resource_url="path/to/conversation.json", modality="conversation")

# Retrieve memories
result = await svc.retrieve(queries=[{"role": "user", "content": {"text": "what does Marcos think about work?"}}])
```

---

## Codebase orientation

See `INDEX.md` for the full module map, prompt inventory, and task→file guide.

```
src/memu/
├── app/           # Public API: service.py, memorize.py, retrieve.py, settings.py
├── database/      # Models, ORM schemas (SQLite), repositories
├── llm/           # LLM client factory + backends (openai, openrouter, grok, doubao)
├── embedding/     # Embedding client factory + backends
├── prompts/       # All extraction, routing, retrieval, and consolidation prompts
├── workflow/      # DAG runner: step, pipeline, runner
└── utils/         # Format converters
```

**Key prompt locations:**
- Extraction: `prompts/memory_type/{profile,knowledge,behavior,social}.py`
- Router: `prompts/router/router.py`
- Non-chat preprocessors: `prompts/preprocess/{document,image,audio,video}.py`
- Retrieval query decision: `prompts/retrieve/pre_retrieval_decision.py` (ranking lives in `app/retrieve.py` + `RetrieveItemConfig`)
- Consolidation: `prompts/consolidation/`

---

## License

GPLv3. This fork and new contributions are GPLv3. Upstream-derived portions retain Apache 2.0 attribution; see `NOTICE` and `LICENSE-APACHE.txt`.
