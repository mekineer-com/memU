# Contributing

This is a private fork of NevaMind-AI/memU. It is not actively maintained as a public open-source project, but issues and PRs are welcome.

## Reporting bugs

Open a GitHub issue with:
- What you were doing
- What you expected
- What happened instead (error message / stack trace)
- Python version and OS

## Submitting changes

1. Fork, create a feature branch
2. Run `make check` before pushing (pre-commit hygiene checks + mypy)
3. Python syntax check: `python3 -m py_compile src/memu/app/memorize.py`
4. Open a pull request with a short description of what changed and why

## Code style

- Python 3.12+
- Type hints on public functions

## License

New contributions are GPLv3. Upstream-derived portions retain Apache 2.0 attribution — see `NOTICE` and `LICENSE-APACHE.txt`.
