# memU Memory Engine

Keep README.md for people choosing, installing and using this project. Put durable
working guidance here, code navigation in an existing index, and task history in
the relevant plan or audit; do not turn this file into a session log.

Read [INDEX.md](INDEX.md) before exploring the engine. It owns the module map,
prompt inventory and task-to-file guide; do not duplicate them here. For work
that crosses the server boundary, read MCP's `MEMORIZE_PIPELINE.md`.

memU is a library, not a client or HTTP server. Keep orchestration in MCP.
A category is a dossier; retain `category` identifiers rather than renaming them.
Storage requires the pinned sqlite-vec extension installed by
`scripts/install-sqlite-vec.py`; do not substitute a different search path.
