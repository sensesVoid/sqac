"""SQAC autobuild — turn a project directory into a living cartridge.

Two jobs, both deterministic and numpy-only:

1. ``extract_project_units(root)`` — the "init" pipeline. Walks a directory,
   distils files into *units* (source / kind / key / content), and packs them
   into a ``.sqac`` cartridge with zero APIs and zero training:
     * Markdown/RST sections           -> facts (one per heading)
     * Python module docstrings        -> facts
     * package.json / pyproject / CI   -> facts (commands, deps, engines)
     * .env.example                    -> facts (variable *names* only)
     * skill YAML files                -> validated skill cards
2. ``diff_units`` + ``track_once``      — the "realtime" tracker. Keeps an
   in-memory per-file unit cache, stats each file (mtime, size) on a poll
   interval, re-extracts only the files that changed, rebuilds the cartridge
   only when something moved, and logs +/-/~ per source. Delete-safe and
   idempotent: the cartridge is rebuilt from the surviving unit set, never
   surgically patched, so stale entries cannot survive a removal.

Storage layout (default ``.sqac/``):
    project.sqac     one cartridge, entries stamped fact/doc/skill
    state.json       source->digest map, per-file stat cache, synced_at
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .skills import load_skills
from .store import SqacStore

EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".bzr",
    "node_modules", "vendor", "dist", "build", "coverage", "htmlcov",
    ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".idea", ".vscode", ".sqac", ".git", "target",
}
EXCLUDED_NAMES = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.egg-info", "*.lock",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    ".npmrc", ".env", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg",
    "*.ico", "*.woff", "*.woff2", "*.ttf", "*.otf", "*.pdf", "*.zip",
    "*.gz", "*.sqlite", "*.sqac",
}
_MD_HEADING = re.compile(r"^(\#{1,6})\s+(.*)$")
_RST_HEADING = re.compile(r"^(?P<title>[A-Za-z0-9][^\n]{0,80})\n(?P<rule>[-=+~^]{3,})$")
_MAX_BODY = 1200
_MAX_FILE = 256_000
_LANG_EXT = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript/React", ".jsx": "React",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".php": "PHP",
    ".sh": "shell", ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++",
    ".cs": "C#", ".kt": "Kotlin", ".swift": "Swift", ".ex": "Elixir",
    ".sql": "SQL", ".toml": "TOML", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".vue": "Vue",
}
_SKILL_NAME = re.compile(r"(?:skill|workflow|playbook|runbook)s?\.ya?ml$", re.I)
_DIRECTIVE_FILES = {"AGENTS.md", "CLAUDE.md", "COPILOT.md"}


@dataclass
class Unit:
    """One border of knowledge extracted from the project."""

    source: str   # stable id: "README.md#getting-started"
    kind: str     # "fact" | "doc" | "skill"
    key: str
    content: str

    def digest(self) -> str:
        return hashlib.sha256(
            f"{self.source}|{self.kind}|{self.content}".encode()
        ).hexdigest()[:16]


# ── git-ignore-ish walk ────────────────────────────────────────────────────────

def _ignore_rules(root: Path) -> list[re.Pattern]:
    pats: list[re.Pattern] = []
    gi = root / ".gitignore"
    if not gi.exists():
        return pats
    try:
        for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            pats.append(re.compile(re.escape(line).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")))
    except Exception:
        pass
    return pats


def _ignored(rel: str, rules: list[re.Pattern]) -> bool:
    parts = rel.split("/")
    for p in parts:
        if p in EXCLUDED_DIRS:
            return True
    name = parts[-1]
    for pat in EXCLUDED_NAMES:
        if pat.startswith("*") and name.endswith(pat[1:]):
            return True
        if name == pat:
            return True
    for r in rules:
        if r.search(rel):
            return True
    return False


def walk_project(root: Path) -> list[Path]:
    rules = _ignore_rules(root)
    out: list[Path] = []
    for dirpath, dirnames, filenames in __import__("os").walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not (Path(dirpath) / d).name.startswith(".git")
        ]
        for fn in filenames:
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if _ignored(rel, rules):
                continue
            try:
                if full.stat().st_size > _MAX_FILE:
                    continue
            except Exception:
                continue
            out.append(full)
    out.sort(key=lambda p: p.relative_to(root).as_posix())
    return out


# ── extractors ────────────────────────────────────────────────────────────────

def _clean_section(lines: list[str]) -> str:
    kept: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if s.startswith("!["):          # image
            continue
        if s.startswith("[![") or re.match(r"^\[\s*-?\s*\]", s):  # badge
            continue
        if s.startswith("<!--"):
            continue
        if re.match(r"^\[[A-Za-z _-]+\]\(#", s):  # toc link
            continue
        kept.append(ln)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept).strip()


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")[:42]
    return s or "untitled"


def _sections_from_md(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs for a Markdown file."""
    sections: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_head = "Header"
    for ln in text.splitlines():
        m = _MD_HEADING.match(ln)
        if m:
            body = _clean_section(cur)
            if body:
                sections.append((cur_head, body))
            cur_head = m.group(2).strip()
            cur = [ln]
        else:
            cur.append(ln)
    body = _clean_section(cur)
    if body:
        sections.append((cur_head, body))
    return sections


def _body_chunks(body: str) -> list[str]:
    if len(body) <= _MAX_BODY:
        return [body]
    chunks, cur = [], []
    for para in body.split("\n\n"):
        cur.append(para)
        if sum(len(c) for c in cur) >= _MAX_BODY:
            chunks.append("\n\n".join(cur))
            cur = []
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def extract_markdown(rel: str, text: str, kind: str = "fact") -> list[Unit]:
    units: list[Unit] = []
    for head, body in _sections_from_md(text):
        for i, chunk in enumerate(_body_chunks(body)):
            source = f"{rel}#{_slug(head)}" + (f"-p{i}" if i else "")
            units.append(
                Unit(source=source, kind=kind, key=head, content=f"[{rel}] {head}\n{chunk}")
            )
    return units


def extract_module_docstrings(rel: str, text: str) -> list[Unit]:
    m = re.search(r'^((?:#.*\n)*?)(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text, re.S)
    if not m:
        return []
    first = " ".join(m.group(2).strip().split())
    if len(first) < 16:
        return []
    return [
        Unit(
            source=f"{rel}#docstring",
            kind="fact",
            key=first[:60],
            content=f"[{rel}] module: {first[:400]}",
        )
    ]


def _npm_deps(obj: dict) -> str:
    deps = []
    for section in ("dependencies", "devDependencies"):
        d = obj.get(section) or {}
        for name, ver in list(d.items())[:8]:
            deps.append(f"{name}@ {ver}")
    return "; ".join(deps)


def extract_config(rel: str, text: str) -> list[Unit]:
    low = rel.lower()
    if rel.endswith("package.json"):
        try:
            obj = json.loads(text)
        except Exception:
            return []
        units = []
        desc = "; ".join(x for x in [
            obj.get("name", ""), obj.get("description", ""), obj.get("version", "")
        ] if x)
        units.append(Unit(f"{rel}#about", "fact", "project about",
                          f"[{rel}] {desc}"))
        scripts = obj.get("scripts") or {}
        for key in list(scripts)[:12]:
            units.append(Unit(f"{rel}#script:{key}", "fact", f"npm script {key}",
                              f"[{rel}] `npm run {key}` -> `{scripts[key]}`"))
        deps = _npm_deps(obj)
        if deps:
            units.append(Unit(f"{rel}#deps", "fact", f"{rel} dependencies",
                              f"[{rel}] runtime deps: {deps}"))
        return units
    if rel.endswith("pyproject.toml"):
        units = []
        proj = re.search(r"(?m)^\[project\]\s*\n(.*?)(?=^\[)", text, re.S)
        block = proj.group(1) if proj else text
        els = []
        for name in ("name", "version", "description", "requires-python"):
            m = re.search(rf"(?m)^{name}\s*=\s*[\"']([^\"']+)[\"']$", block)
            if m:
                els.append(f"{name}={m.group(1)}")
        if els:
            units.append(Unit(f"{rel}#about", "fact", "pyproject about",
                              f"[{rel}] {'; '.join(els)}"))
        depm = re.search(r"(?m)^\[project\.dependencies\]\s*\n(.*?)(?=^\[)", text, re.S)
        if depm:
            deps = [d.strip().rstrip(",").strip('"').strip("'")
                    for d in depm.group(1).splitlines() if d.strip()
                    and not d.strip().startswith("#")][:8]
            if deps:
                units.append(Unit(f"{rel}#deps", "fact", f"{rel} runtime deps",
                                  f"[{rel}] runtime deps: {'; '.join(deps)}"))
        scripts = re.search(r"(?m)^\[project\.scripts\]\s*\n(.*?)(?=^\[)", text, re.S)
        if scripts:
            for ln in scripts.group(1).splitlines():
                ln = ln.strip()
                if "=" in ln:
                    k, _, v = ln.partition("=")
                    units.append(Unit(f"{rel}#script:{k.strip()}", "fact",
                                      f"console script {k.strip()}",
                                      f"[{rel}] `{k.strip()}` -> `{v.strip().strip('\"')}`"))
        return units
    if rel.endswith("requirements.txt"):
        deps = [ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("#") and "==" in ln][:8]
        if deps:
            return [Unit(f"{rel}#deps", "fact", f"{rel} deps",
                         f"[{rel}] pinned deps: {'; '.join(deps)}")]
        return []
    if rel.endswith("Makefile") or rel.endswith("makefile"):
        names = re.findall(r"(?m)^([a-zA-Z0-9_.-]+)\s*:", text)
        if names:
            return [Unit(f"{rel}#targets", "fact", f"{rel} targets",
                         f"[{rel}] make targets: {', '.join(names[:20])}")]
        return []
    if ".github/workflows/" in low or rel.endswith(".gitlab-ci.yml"):
        jobs = sorted(set(re.findall(r"(?m)^  ([a-zA-Z0-9_.-]+):\s*$", text)))
        head = text.splitlines()[0].strip() if text.splitlines() else rel
        return [Unit(f"{rel}#workflow", "fact", f"CI workflow",
                     f"[{rel}] {head}; jobs: {', '.join(jobs[:12]) or 'n/a'}")]
    if low.endswith(".env.example"):
        names = sorted({ln.partition("=")[0].strip() for ln in text.splitlines()
                        if ln.strip() and not ln.strip().startswith("#") and "=" in ln})
        if names:
            return [Unit(f"{rel}#env", "fact", f"{rel} variables",
                         f"[{rel}] documented env vars: {', '.join(names[:24])}")]
        return []
    if "docker-compose" in rel and rel.endswith((".yml", ".yaml")):
        svc = sorted(set(re.findall(r"(?m)^  ([a-zA-Z0-9_.-]+):\s*$", text)))
        if svc:
            return [Unit(f"{rel}#services", "fact", f"{rel} services",
                         f"[{rel}] services: {', '.join(svc[:16])}")]
        return []
    if rel.endswith("Dockerfile"):
        base = re.search(r"(?mi)^FROM\s+(.+)$", text)
        return [Unit(f"{rel}#base", "fact", f"{rel} base image",
                     f"[{rel}] base image: {base.group(1).strip() if base else 'n/a'}")]
    if rel.endswith(".github/dependabot.yml"):
        ec = re.findall(r"(?m)^\s*-\s+package-ecosystem:\s*(.+)$", text)
        inr = re.findall(r"(?m)^\s*interval:\s*(.+)$", text)
        return [Unit(f"{rel}#dependabot", "fact", f"{rel} bot",
                     f"[{rel}] dependabot ecosystems: {', '.join(e.strip() for e in ec)} "
                     f"(interval {', '.join(i.strip() for i in inr)})")]
    return []


def _primary_languages(files: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for rel in files:
        ext = Path(rel).suffix.lower()
        lang = _LANG_EXT.get(ext)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]


_FRAMEWORK_HINTS = [
    ("fastapi", "FastAPI"), ("flask", "Flask"), ("django", "Django"),
    ("next", "Next.js"), ("express", "Express"), ("react", "React"),
    ("pytest", "pytest"), ("rspec", "RSpec"), ("jest", "Jest"),
    ("torch", "PyTorch"), ("numpy", "NumPy"), ("pandas", "pandas"),
    ("uvicorn", "Uvicorn"), ("turbopack", "Turbopack"), ("vitest", "Vitest"),
]


def extract_conventions(root: Path, rel_files: list[str], units: list[Unit]) -> None:
    langs = _primary_languages(rel_files)
    if langs:
        units.append(Unit(f"_conventions#languages", "fact", "primary languages",
                          f"Primary languages in this project: {', '.join(langs)}."))
    blob = []
    for name in ("pyproject.toml", "package.json"):
        p = root / name
        if p.exists() and p.stat().st_size < _MAX_FILE:
            try:
                blob.append(p.read_text(encoding="utf-8", errors="replace").lower())
            except Exception:
                pass
    found = [hint for hint, label in _FRAMEWORK_HINTS if any(hint in b for b in blob)]
    if found:
        label_of = dict(_FRAMEWORK_HINTS)
        labels = [label_of[h] for h in found]
        units.append(Unit("_conventions#frameworks", "fact", "framework stack",
                          "Framework stack detected: " + ", ".join(sorted(set(labels))) + "."))
    tests = [f for f in rel_files if re.match(r"(test_|_test|.*[/_.]test[/_.])", f) or f.endswith(("_test.py", ".test.ts", ".spec.ts", ".test.js", "test.py"))]
    if tests:
        units.append(Unit("_conventions#tests", "fact", "tests location",
                          f"Tests live near code or under the standard test paths "
                          f"({len(tests)} files detected)."))


def extract_directives(rel: str, text: str) -> list[Unit]:
    """AGENTS.md / CLAUDE.md / COPILOT.md -> convention facts."""
    units = []
    for head, body in _sections_from_md(text):
        for i, chunk in enumerate(_body_chunks(body)):
            source = f"{rel}#{_slug(head)}" + (f"-p{i}" if i else "")
            units.append(Unit(source=source, kind="fact", key=head,
                              content=f"[{rel}] {head}\n{chunk}"))
    return units


def extract_skill_yaml(rel: str, text: str) -> list[Unit]:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        skills = load_skills(tmp)
    except Exception:
        return []
    finally:
        try:
            Path(tmp).unlink()
        except Exception:
            pass
    units = []
    for s in skills[:24]:
        units.append(Unit(source=f"{rel}#{s.name}", kind="skill",
                          key=s.name, content=s.content))
    return units


def extract_file(root: Path, rel: str) -> list[Unit]:
    p = root / rel
    try:
        if p.stat().st_size == 0 or p.stat().st_size > _MAX_FILE:
            return []
        raw = p.read_bytes()
        if b"\x00" in raw[:4096]:
            return []
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []
    if rel.lower().endswith((".md", ".rst")):
        if rel.lower().endswith(".rst"):
            return []
        if rel in _DIRECTIVE_FILES:
            return extract_directives(rel, text)
        if _SKILL_NAME.search(Path(rel).name):
            return extract_skill_yaml(rel, text)
        return extract_markdown(rel, text)
    if rel.endswith(".py"):
        return extract_module_docstrings(rel, text)
    return extract_config(rel, text)


# ── public API ────────────────────────────────────────────────────────────────

def extract_project_units(root: Path) -> list[Unit]:
    """Run every extractor over the project and return all units, sorted."""
    files = walk_project(root)
    rel_files = [p.relative_to(root).as_posix() for p in files]
    units: list[Unit] = []
    for rel in rel_files:
        units.extend(extract_file(root, rel))
    extract_conventions(root, rel_files, units)
    seen: set[tuple[str, str, str]] = set()
    dedup: list[Unit] = []
    for u in units:
        sig = (u.source, u.kind, u.content)
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(u)
    dedup.sort(key=lambda u: u.source)
    return dedup


def build_store(units: list[Unit]) -> SqacStore:
    store = SqacStore()
    meta = {"tracked": True}
    for u in units:
        store.add(u.content, key=u.key, meta=meta, source=u.source, kind=u.kind)
    return store


# ── incremental diff ──────────────────────────────────────────────────────────

def unit_map(units: list[Unit]) -> dict[str, str]:
    return {u.source: u.digest() for u in units}


def diff_units(prev: dict[str, str], new: dict[str, str]):
    added = sorted(set(new) - set(prev))
    removed = sorted(set(prev) - set(new))
    changed = sorted(s for s in set(new) & set(prev) if prev[s] != new[s])
    return added, changed, removed


def _state_path(sqac_dir: Path) -> Path:
    return sqac_dir / "state.json"


def save_state(sqac_dir: Path, sources: dict[str, str]) -> None:
    """Write state atomically (tmp file + replace) so a crash mid-write can
    never leave a truncated state.json (which would force a full re-extract)."""
    sqac_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"sources": sources,
                          "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                         indent=2, sort_keys=True)
    target = _state_path(sqac_dir)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)


def load_state(sqac_dir: Path) -> dict | None:
    p = _state_path(sqac_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logging.warning("autobuild: state.json unreadable (%s); will re-extract", exc)
        return None


def track_once(root: Path, sqac_dir: Path, verbose: bool = True) -> dict:
    """One incremental pass. Returns the change summary dict."""
    root = Path(root)
    sqac_dir.mkdir(parents=True, exist_ok=True)
    units = extract_project_units(root)
    new = unit_map(units)
    state = load_state(sqac_dir) or {}
    prev = state.get("sources", {})
    added, changed, removed = diff_units(prev, new)
    build_store(units).save(sqac_dir / "project.sqac", name="project",
                            description=f"autobuilt from {root}")
    save_state(sqac_dir, new)
    if verbose:
        for s in added:
            print(f"  + {s}")
        for s in changed:
            print(f"  ~ {s}")
        for s in removed:
            print(f"  - {s}")
    return {"added": added, "changed": changed, "removed": removed}


def track(root: Path, sqac_dir: Path, interval: float,
           log_path: Path | None = None,
           rack_dir: Path | None = None,
           rack_name: str | None = None,
           quiet: bool = False) -> None:
    """Poll the project and keep the cartridge fresh. Ctrl-C to stop.

    When *log_path* is given, each sync appends a JSONL line to the file:
        {"ts": ..., "added": [...], "changed": [...], "removed": [...],
         "total_entries": N, "dirty": bool}
    The file is opened in append mode so history accumulates across runs.

    When *rack_dir* is given, the built cartridge is also registered into
    a CartridgeRack at that directory, making it searchable alongside other
    cartridges via ``rack.search()`` or ``sqac search --db <rack>``.
    *rack_name* defaults to the project directory name.

    If *quiet* is True, only changes are printed (no "no change" messages).
    """
    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8")
        if not quiet:
            print(f"audit log: {log_path}")
    name = rack_name or root.name
    if not quiet:
        print(f"tracking {root} -> {sqac_dir}/project.sqac (every {interval:g}s)")
        print(f"  watching for changes... (Ctrl-C to stop)")
    if rack_dir is not None:
        if not quiet:
            print(f"rack: {rack_dir} (cartridge name: {name})")
    try:
        while True:
            try:
                summary = track_once(root, sqac_dir, verbose=not quiet)
                dirty = len(summary["added"]) + len(summary["changed"]) + len(summary["removed"])
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                if dirty or not quiet:
                    print(f"  synced {time.strftime('%H:%M:%S')} "
                          f"({'changed' if dirty else 'no change'} "
                          f"{dirty and f'({dirty} source(s))' or ''})")
                if log_fh is not None:
                    state = load_state(sqac_dir) or {}
                    total = len(state.get("sources", {}))
                    record = {
                        "ts": ts,
                        "added": summary["added"],
                        "changed": summary["changed"],
                        "removed": summary["removed"],
                        "total_sources": total,
                        "dirty": bool(dirty),
                    }
                    log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_fh.flush()
                if rack_dir is not None and dirty:
                    _sync_rack(rack_dir, name, sqac_dir / "project.sqac")
            except KeyboardInterrupt:
                print("\nstopped.")
                return
            except Exception as e:  # keep the loop alive on transient errors
                print(f"  ! tracker error: {e}")
            time.sleep(interval)
    finally:
        if log_fh is not None:
            log_fh.close()


def _sync_rack(rack_dir: Path, name: str, cart_path: Path) -> None:
    """Register (or update) the project cartridge in a CartridgeRack.

    The rack directory is created on demand. The cartridge file is copied
    into the rack as <name>.sqac so the rack can mount it by convention.
    """
    import shutil
    rack_dir.mkdir(parents=True, exist_ok=True)
    dest = rack_dir / f"{name}.sqac"
    shutil.copy2(cart_path, dest)
    # touch a manifest so the rack knows this cartridge is tracked
    manifest = rack_dir / "rack.json"
    manifest_data: dict = {}
    if manifest.exists():
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("autobuild: rack manifest unreadable (%s); rebuilding", exc)
    manifest_data[name] = {
        "path": dest.as_posix(),
        "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest)