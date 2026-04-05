# memU — Memory Engine

Local-first memory engine for AI companions. Extracts, stores, and retrieves structured memories from conversation history.

This is a private fork of the upstream NevaMind memU project, significantly diverged. It is not affiliated with NevaMind-AI and does not connect to any cloud service.

---

## What it does

`memu` is the core extraction and retrieval library. It is not a server — it is imported by `mcp-memu-server`, which provides the API layer.

**Five active memory types:**

| Type | What it captures |
|------|-----------------|
| `profile` | Who someone is — identity, traits, personality, inner life |
| `event` | What happened — episodic, time-anchored experiences |
| `knowledge` | What was learned — facts, mechanisms, things worth carrying |
| `behavior` | How someone acts — patterns, interaction styles, ways of being |
| `social` | Third parties in the user's life — family, friends, coworkers, pets |

Plus **diary** on a separate generation path — auto-triggered after memorize when diary-worthy episodes are queued. Diary-worthiness is output from the episode router (same JSON call as memory-type routing; no separate classifier).

**Three-layer storage:**

```
Resource  →  MemoryItem  →  MemoryCategory
(raw input)   (extracted fact)   (auto-organized topic)
```

---

## Requirements

- Python 3.12+
- SQLite (default) or PostgreSQL
- An LLM provider API key (OpenAI-compatible)

---

## Installation

```bash
pip install -e .
```

Or with the monorepo venv:
```bash
cd memu
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
```

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
├── database/      # Models, ORM schemas (SQLite + Postgres), repositories
├── llm/           # LLM client factory + backends (openai, openrouter, grok, doubao)
├── embedding/     # Embedding client factory + backends
├── prompts/       # All extraction, routing, retrieval, and diary prompts
├── workflow/      # DAG runner: step, pipeline, runner
└── utils/         # Format converters
```

**Key prompt locations:**
- Extraction: `prompts/memory_type/{profile,event,knowledge,behavior,social}.py`
- Router: `prompts/router/router.py`
- Retrieval ranking: `prompts/retrieve/`
- Diary: `prompts/diary/`

---

## License

GPLv3. This fork and new contributions are GPLv3. Upstream-derived portions retain Apache 2.0 attribution; see `NOTICE` and `LICENSE-APACHE.txt`.
