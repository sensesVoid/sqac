"""Universal skill formatter — YAML skill cards for SQAC cartridges.

Turns the measured skill-card lessons (multi-key triggers, zero answers in
content, concrete situation phrases) into an enforced, human-readable format.

Format (YAML):

    name: weighted-index
    domain: logic
    difficulty: hard
    pattern: combinatorial
    content: |
      SKILL weighted-index: label items 1..N. Take i coins from item i.
      The total excess weight tells you which item has the defect.
    keys:
      - when exactly one of many items has a hidden property and you can weigh once
      - one defective batch among many, one weighing to find it
      - 12 bags of coins, one heavier, one weighing on a scale
      - find the odd bag with coins weighing more, one digital scale

Usage:
    from sqac.skills import load_skills, validate, to_entries

    skills = load_skills("logic_skills.yaml")
    report = validate(skills)
    entries = to_entries(skills)  # -> list of (key, content, meta)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from .store import SqacStore

# ── Skill data model ────────────────────────────────────────────────────────

@dataclass
class Skill:
    """One reasoning procedure with routing metadata."""

    name: str
    content: str
    keys: list[str] = field(default_factory=list)
    domain: str = "general"
    difficulty: str = "medium"
    pattern: str = ""
    source: str = ""

    @property
    def badge(self) -> str:
        return f"SKILL {self.name}"

    def content_with_badge(self) -> str:
        if self.content.startswith("SKILL "):
            return self.content
        return f"{self.badge}: {self.content}"


# ── Parsing ─────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> list[Skill]:
    """Parse a YAML file containing one or more skill cards.

    Accepts a single skill (dict) or a list of skills.
    """
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    skills = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        keys = item.get("keys", [])
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("\n") if k.strip()]
        # YAML colons in trigger phrases produce dicts; flatten them
        flat_keys = []
        for k in keys:
            if isinstance(k, dict):
                flat_keys.append(" ".join(f"{vk} {vv}" for vk, vv in k.items()))
            elif isinstance(k, str):
                flat_keys.append(k)
        keys = flat_keys
        skills.append(
            Skill(
                name=item.get("name", ""),
                content=item.get("content", ""),
                keys=keys,
                domain=item.get("domain", "general"),
                difficulty=item.get("difficulty", "medium"),
                pattern=item.get("pattern", ""),
                source=item.get("source", ""),
            )
        )
    return skills


def load_skills(path: str | Path) -> list[Skill]:
    """Load skills from YAML (.yaml/.yml) or JSON (.json) file."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return load_yaml(p)
    elif p.suffix == ".json":
        import json
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]
        return [
            Skill(
                name=item.get("name", ""),
                content=item.get("content", ""),
                keys=item.get("keys", []),
                domain=item.get("domain", "general"),
                difficulty=item.get("difficulty", "medium"),
                pattern=item.get("pattern", ""),
                source=item.get("source", ""),
            )
            for item in raw
            if isinstance(item, dict)
        ]
    else:
        raise ValueError(f"unsupported format: {p.suffix}")


# ── Validation ──────────────────────────────────────────────────────────────

# Heuristic patterns for detecting common anti-patterns
_ABSTRACT_TRIGGERS = re.compile(
    r"\bwhen (a|the|an) (problem|question|situation|case|scenario) "
    r"(asks|involves|requires|contains|has|is about)\b",
    re.IGNORECASE,
)
_NUMERIC_ANSWERS = re.compile(r"\b\d+\s*[+\-*/=]\s*\d+\b")
_SINGLE_KEY = re.compile(r"^[a-z]+$")


@dataclass
class Issue:
    skill_name: str
    severity: str  # "error" | "warning" | "info"
    message: str


def validate(skills: list[Skill]) -> list[Issue]:
    """Validate skill cards against measured best practices.

    Returns a list of issues. Errors = must fix. Warnings = should fix.
    """
    issues: list[Issue] = []
    seen_names: set[str] = set()

    for s in skills:
        # --- errors ---
        if not s.name:
            issues.append(Issue("", "error", "missing 'name'"))
            continue
        if s.name in seen_names:
            issues.append(Issue(s.name, "error", f"duplicate name '{s.name}'"))
        seen_names.add(s.name)

        if not s.content or not s.content.strip():
            issues.append(Issue(s.name, "error", "empty content — must contain the procedure"))
            continue

        if not s.keys or len(s.keys) == 0:
            issues.append(Issue(s.name, "error", "no trigger keys — at least 2 required"))
            continue

        if len(s.keys) < 2:
            issues.append(
                Issue(s.name, "warning", f"only {len(s.keys)} trigger key — measured minimum is 2")
            )

        # --- warnings (measured anti-patterns) ---

        # 1. Abstract-only triggers (measured: 0/5 retrieval without concrete phrases)
        abstract_count = sum(1 for k in s.keys if _ABSTRACT_TRIGGERS.search(k))
        if abstract_count == len(s.keys):
            issues.append(
                Issue(
                    s.name,
                    "warning",
                    "all triggers are abstract — add concrete situation phrases "
                    "(measured: abstract-only keys get 0% retrieval on concrete problems)",
                )
            )

        # 2. Answer leakage (anti-transliteration check)
        lines = [l.strip() for l in s.content.split("\n") if l.strip()]
        for line in lines:
            stripped = line.lstrip("#").strip()
            if _NUMERIC_ANSWERS.search(stripped):
                issues.append(
                    Issue(
                        s.name,
                        "warning",
                        f"content may contain numeric answers ({stripped[:60]}…) — "
                        "skill content should be pure procedure (zero answers)",
                    )
                )
                break

        # 3. Very short content (likely incomplete)
        word_count = len(s.content.split())
        if word_count < 10:
            issues.append(
                Issue(s.name, "warning", f"content is only {word_count} words — may be incomplete")
            )

        # 4. Single-word triggers (too narrow)
        short_keys = [k for k in s.keys if len(k.split()) < 4]
        if short_keys:
            issues.append(
                Issue(
                    s.name,
                    "info",
                    f"short trigger(s): {short_keys[:3]} — consider expanding with situation details",
                )
            )

    return issues


def print_validation(skills: list[Skill], issues: list[Issue]) -> bool:
    """Pretty-print validation report. Returns True if no errors."""
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos = [i for i in issues if i.severity == "info"]

    print(f"\nvalidating {len(skills)} skills...")
    for i in errors:
        print(f"  ERROR   [{i.skill_name}] {i.message}")
    for i in warnings:
        print(f"  WARNING [{i.skill_name}] {i.message}")
    for i in infos:
        print(f"  INFO    [{i.skill_name}] {i.message}")

    if not errors and not warnings:
        print("  all clean")
    print()
    return len(errors) == 0


# ── Conversion to SQAC entries ──────────────────────────────────────────────

def to_entries(skills: list[Skill]) -> list[dict[str, Any]]:
    """Convert skill cards to SQAC-ready entries (key, content, meta).

    Each skill produces len(keys) entries — one per trigger phrase —
    all pointing at the same procedure content. This is the multi-key
    routing pattern that measured 5/5 retrieval.
    """
    entries = []
    for s in skills:
        content = s.content_with_badge()
        meta = {
            "skill": s.name,
            "domain": s.domain,
            "difficulty": s.difficulty,
            "pattern": s.pattern,
        }
        if s.source:
            meta["source"] = s.source
        for key in s.keys:
            entries.append({"key": key, "content": content, "meta": meta, "source": f"skill:{s.name}"})
    return entries


def pack_cartridge(skills: list[Skill], path: str | Path, name: str = "skill-library", semantic: bool = False) -> None:
    """Pack validated skills into a .sqac cartridge (multi-key routing)."""
    store = SqacStore(semantic=semantic)
    for entry in to_entries(skills):
        store.add(entry["content"], key=entry["key"], meta=entry["meta"], source=entry["source"])
    store.save(path, name=name, description=f"{len(skills)} skills, {len(store)} entries")
    size = Path(path).stat().st_size
    print(f"packed {len(skills)} skills -> {path} ({len(store)} entries, {size / 1024:.1f} KB, semantic={'on' if semantic else 'off'})")


# ── Skill templates ─────────────────────────────────────────────────────────

DOMAINS = {
    "logic": [
        "deductive reasoning", "inductive reasoning", "abductive reasoning",
        "proof by contradiction", "proof by induction", "pigeonhole principle",
        "invariant finding", "backward chaining", "forward chaining",
        "divide and conquer", "reduce to known problem", "weighted indexing",
        "state search", "overlap counting", "direct proportion",
    ],
    "debugging": [
        "binary search for bugs", "reproduce then isolate",
        "check assumptions first", "simplify test case",
        "add logging at boundaries", "rubber duck debugging",
    ],
    "coding": [
        "edge cases first", "decompose function", "test-driven development",
        "refactor to patterns", "cache repeated computation",
        "guard clause early return", "single responsibility",
    ],
    "writing": [
        "reverse outline", "write the summary first",
        "one idea per paragraph", "kill your darlings",
        "show don't tell", "vary sentence length",
    ],
}


def generate_triggers(skill: Skill, max_triggers: int = 5) -> list[str]:
    """Generate concrete situation triggers from skill content + domain.

    Strategy:
      1. Extract action verbs and domain nouns from the content.
      2. Combine with domain-specific situation templates.
      3. Return concrete phrases the user can accept or edit.
    """
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "or", "but", "not", "if", "then", "else", "when", "each",
        "every", "all", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "so", "up", "out", "about", "that",
        "this", "which", "what", "who", "whom", "it", "its", "each",
    }
    words = [w.strip(".,;:!?()[]{}'") for w in skill.content.lower().split()]
    verbs = [w for w in words if len(w) > 4 and w.isalpha() and w not in stop]
    nouns = verbs[:6]  # top meaningful words

    suggestions = []

    # template 1: "when you need to <verb>..."
    if nouns:
        suggestions.append(f"when you need to {nouns[0]}")

    # template 2: "<noun> problem, <noun> issue, <noun> question"
    if nouns[:2]:
        suggestions.append(f"{nouns[0]} problem, {nouns[0]} issue")

    # template 3: domain-specific triggers from DOMAINS
    if skill.domain in DOMAINS:
        for pattern in DOMAINS[skill.domain]:
            pattern_words = set(pattern.split())
            if len(pattern_words & set(nouns)) >= 1:
                suggestions.append(f"when the situation calls for {pattern}")

    # template 4: content-derived situation phrases
    if len(nouns) >= 2:
        suggestions.append(f"dealing with {nouns[0]} and {nouns[1]}")

    # deduplicate, remove existing keys, cap
    seen = set(skill.keys)
    out = []
    for s in suggestions:
        if s not in seen and s not in out:
            seen.add(s)
            out.append(s)
    return out[:max_triggers]


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="sqac.skills", description="Universal skill formatter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="validate skill YAML file(s)")
    p_val.add_argument("input", nargs="+", help=".yaml or .json skill file(s) — validated together")
    p_val.add_argument("--strict", action="store_true", help="treat warnings as errors")

    p_pack = sub.add_parser("pack", help="pack skills into .sqac cartridge")
    p_pack.add_argument("input", nargs="+", help=".yaml or .json skill file(s), merged in order")
    p_pack.add_argument("-o", "--out", required=True)
    p_pack.add_argument("--name", default="skill-library")
    p_pack.add_argument("--semantic", action="store_true", help="enable semantic tier (static int8 by default)")

    p_suggest = sub.add_parser("suggest", help="suggest additional trigger keys")
    p_suggest.add_argument("input", nargs="+", help=".yaml or .json skill file(s)")

    p_template = sub.add_parser("template", help="show skill template or domain list")
    p_template.add_argument("--domain", default=None, help="show triggers for this domain")

    args = ap.parse_args(argv)

    if args.cmd == "template":
        if args.domain:
            triggers = DOMAINS.get(args.domain, [])
            print(f"domain: {args.domain}")
            for t in triggers:
                print(f"  - when you need {t}")
        else:
            print("available domains:", ", ".join(DOMAINS.keys()))
        return 0

    skills = []
    for path in args.input:
        skills += load_skills(path)

    if args.cmd == "validate":
        issues = validate(skills)
        ok = print_validation(skills, issues)
        if args.strict and any(i.severity == "warning" for i in issues):
            return 1
        return 0 if ok else 1

    if args.cmd == "suggest":
        for s in skills:
            suggestions = generate_triggers(s)
            if suggestions:
                print(f"\n{s.name} — suggested triggers:")
                for k in suggestions:
                    print(f"  - {k}")
        return 0

    if args.cmd == "pack":
        issues = validate(skills)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            print("cannot pack: validation errors exist")
            for i in errors:
                print(f"  ERROR [{i.skill_name}] {i.message}")
            return 1
        pack_cartridge(skills, args.out, name=args.name, semantic=getattr(args, 'semantic', False))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
