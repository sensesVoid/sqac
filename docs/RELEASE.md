# Release Process

This document describes how SQAC versions are cut and published.

## 1. Prepare the release

1. Confirm the target branch is merged or ready to merge:
   - All tests pass locally (`python -m pytest -q`).
   - Lint clean (`python -m ruff check sqac tests sqac_mcp.py`).
   - PR CI is green (see `CONTRIBUTING.md` for the workflow).
2. Update `CHANGELOG.md` with a section for the new version
   (`Added / Changed / Fixed / Security` as appropriate).
3. Bump the version in **`sqac/_version.py`** (single source of truth;
   `pyproject.toml` reads it dynamically).
4. Commit and push.

## 2. Tag and publish

Releases are driven by tags and PyPI **Trusted Publishing** (OIDC; no API token
is stored in GitHub):

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The `release.yml` workflow then:

1. Builds the sdist and wheels (see `.github/workflows/wheels.yml`).
2. Publishes to PyPI (or TestPyPI when triggered manually with `testpypi: true`).
3. Creates a GitHub Release with artifacts and generated notes.

## 3. Verify the release

```bash
# fresh venv smoke test (PyPI installs need --no-cache-dir to avoid stale index)
python -m venv .venv-smoke
.venv-smoke/Scripts/pip install --no-cache-dir sqac==X.Y.Z
# run the smoke test OUTSIDE the repo so the local ./sqac/ does not shadow the package
cd ..
.venv-smoke/Scripts/python -c "import sqac; print(sqac.__version__)"
```

Checklist:

- [ ] PyPI page shows the new version.
- [ ] Fresh-venv import works and version matches.
- [ ] GitHub Release exists with wheels + sdist attached.
- [ ] `CHANGELOG.md` section matches the release.
- [ ] `pip-audit` clean for runtime dependencies (fresh venv:
      `.venv-smoke/Scripts/python -m pip_audit`; a `sqac` skip line is
      expected for brand-new versions, since PyPI may not have indexed it yet —
      every other package must report clean).

## 4. After the release

- Announce in the repository (GitHub Release notes are the canonical record).
- If a hotfix is needed: branch from the release tag, fix, bump the patch
  version, and repeat from step 1.
- Post-1.0: re-add deferred wheels (aarch64 Linux, macOS) when CI time allows —
  see `improvement_plans.md`.
