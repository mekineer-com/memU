# Running the test suite

```sh
cd /home/marcos/apps-codex/memu
PYTHONPATH=src ./.venv/bin/python -m pytest tests/ -q
```

Notes:
- Use the local `./.venv/bin/python`, not system python3.
- `PYTHONPATH=src` exposes the `memu` package.
- Async tests run through pytest-asyncio (`asyncio_mode = "auto"` in `pyproject.toml`).
- Baseline on 2026-10-28: 1029 passed, 0 failed. Treat any failure as new.
