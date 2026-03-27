# Changelog

This fork diverged from NevaMind-AI/memU at **v1.2.0** (early 2026).
For upstream history before that point, see the [NevaMind-AI/memU](https://github.com/NevaMind-AI/memU) repo.

---

## v0.0.5-buildfix — 2026-03-28

- Soul turn loop: `/conversation/{id}/turn` endpoint; per-turn JSON contract (response, cache\_entry, intention\_action, annulments, inner\_thought)
- Conversation state: `prior_context` + `memory_cache` replace `working_note`
- Category seed defaults settled: Identity, Preferences, Relationships, Experiences
- Event extraction anchor rule: ground memories in time, place, and reason
- Soul card passthrough from ST character description; category block formatting
- README and documentation sprint: replaced all upstream NevaMind content

## v0.0.4-buildfix — 2026-03-21

- Phase 1.5 extraction quality: soul context injection, name passthrough (Marcos/Siri instead of user/assistant), narration verb ban, per-segment top-3 budget
- Sleep-gap-only memorize timing; minimum chunk token gate (default 2000)
- Router: fail-closed on parse errors, decision logging, memorable gating
- Dynamic category formation: centroid gate + homeless clustering + planner
- `happened_at` timestamp propagation from conversation messages
- Salience rerank: additive formula (similarity + salience\_score); retrieve gates disabled
- Semantic dedupe: `merged_into` (cosine ≥ 0.89) + `superseded_by` (explicit corrections, threshold 0.75)
- BM25 hybrid search: FTS5 virtual table + RRF fusion alongside vector search
- Async `/memorize`: returns 202; batch processing in BackgroundTask
- Diary auto-trigger after memorize if `pending_diary_memory_ids` non-empty
- Salience heuristic removed; LLM scores directly, NULL stored for unscored items
- Segment → episode rename throughout engine (segment = server-level sleep-gap block; episode = LLM-identified unit within a segment)

## v0.0.3-buildfix — 2026-03-10

- Rebased onto NevaMind v1.4.0 upstream
- Build and CI alignment for fork

## v0.0.2-buildfix — 2026-03-09

- Initial fork from NevaMind v1.2.0
- Alpine Linux / Python 3.12 build fixes
- Phase 2 diary flow: generation, self-model rows (tendencies + tension pairs), intention rows
- Segment routing before extraction
- Scope-aware SQLite paths; `soul_id` in active DB path
