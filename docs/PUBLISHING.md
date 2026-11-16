# Publishing SQAC

Releases use PyPI Trusted Publishing. There is no API token in GitHub.

## One-time setup

1. PyPI → project `sqac` → Manage → Publishing → add a GitHub publisher:
   owner `sensesVoid`, repository `sqac`, workflow `release.yml`,
   environment `pypi`.
2. TestPyPI: the same publisher with workflow `release.yml` and environment
   `testpypi`.
3. GitHub → Settings → Environments: create `pypi` and `testpypi`;
   optionally require a reviewer on `pypi`.

## Release

1. Bump `sqac/_version.py`, update `CHANGELOG.md`, commit.
2. `git tag v0.2.0 && git push origin v0.2.0`
3. The Release workflow builds the sdist and wheels (Linux x86_64 + Windows
   AMD64), publishes, and creates a GitHub Release.
   Note: aarch64 (Linux) and macOS wheels are deferred to v1.0.0. Source builds
   with Rust installed still produce native wheels for those platforms.

Dry run: Actions → Release → Run workflow → `testpypi: true`.
