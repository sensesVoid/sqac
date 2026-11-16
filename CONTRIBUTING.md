# Contributing to SQAC

## Development Setup

```bash
git clone https://github.com/sensesVoid/sqac.git
cd sqac
pip install -e ".[all]"
pip install pytest ruff
```

## Running Tests

```bash
python -m pytest -q                    # full suite
python -m pytest tests/test_simd.py -q  # SIMD parity
python -m ruff check sqac tests sqac_mcp.py  # lint
```

## Code Style

- Python: ruff with target Python 3.10, line length 100.
- Rust: cargo clippy with `-D warnings`.
- Follow existing patterns: dataclasses for models, type hints, docstrings.

## Pull Requests

1. Create a feature branch from `main`.
2. Write tests first (TDD).
3. Ensure all tests pass and ruff/clippy are clean.
4. Open a PR with a clear description.
5. CI must be green before merge.

## Reporting Bugs

Use `sqac report "title" -d "description"` to file a GitHub issue with diagnostics.

## Security

See [SECURITY.md](SECURITY.md) for the security model and vulnerability reporting.
Do not open public issues for security vulnerabilities — email the maintainer directly.
