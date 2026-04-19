# Running the test suite

```sh
cd /home/marcos/apps-codex/memu
PYTHONPATH=src ./.venv/bin/python -m pytest tests/ -q \
  --ignore=tests/llm \
  --ignore=tests/rust_entry_test.py \
  --ignore=tests/test_lazyllm.py
```

Notes:
- Use the local `./.venv/bin/python`, not system python3.
- `PYTHONPATH=src` exposes the `memu` package.
- Ignored sets:
  - `tests/llm/` — need env-dependent API keys.
  - `tests/rust_entry_test.py` — needs a built rust extension.
  - `tests/test_lazyllm.py` + other async tests — need pytest-asyncio (not installed); they error with `async def functions are not natively supported`.
- Current baseline: 76 pass, 6 pre-existing failures in memory-type parsing / semantic dedupe / openrouter / retrieve cache. Investigate before assuming your change caused them.
