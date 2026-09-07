"""Thought-as-a-skill engine: store a curated reasoning procedure as an SQAC
skill card, route a question to the right card, and inject the procedure into
the model's prompt / reasoning to steer its thinking.

Styles:
  direct           - no scaffold (model answers in one shot)
  cot              - free chain-of-thought seed
  injected         - verbose: "follow every step in order" instruction
  injected_concise - terse: "apply the steps silently, output ANSWER = only"
  expression       - Program-of-Thoughts: emit one arithmetic expression;
                     arithmetic is evaluated externally (see eval_expression)
  cod              - Chain-of-Draft: one short draft (<5 words) per step,
                     then ANSWER = <number>

Backend: any OpenAI-compatible chat-completions API (Groq by default).
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from sqac.skills import load_skills, validate
from sqac.store import KIND_SKILL, SqacStore

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_ENV_FILE = ""
# Env candidates, in order: os.environ, then these files if present.
_ENV_CANDIDATES = [
    os.path.expanduser("~/.config/kilo/.env"),
    os.path.expanduser("~/content/.config/kilo/.env"),
    os.path.expanduser("~/.env"),
]


@dataclass
class RoutedSkill:
    content: str
    skill: str | None
    confidence: float
    mode: str


@dataclass
class Plan:
    action: str  # "inject" | "direct"
    style: str   # style passed to prompt()
    routed: RoutedSkill | None
    reason: str


# Deterministic direction resolver for single-pair operator cards. The fuzzy
# tier cannot see word order ("X meters into feet" vs "X feet to meters"
# are ~0.64 vs ~0.64), so we pin direction with a model-free rule:
#   source = the unit sitting next to the number; target = the other unit.
_UNIT_ALIASES = {
    "miles": "mile", "mile": "mile",
    "kilometers": "km", "kilometer": "km", "km": "km",
    "feet": "foot", "foot": "foot",
    "meters": "m", "meter": "m", "m": "m",
    "inches": "inch", "inch": "inch",
    "centimeters": "cm", "centimeter": "cm", "cm": "cm",
    "pounds": "lb", "pound": "lb", "lbs": "lb", "lb": "lb",
    "kilograms": "kg", "kilogram": "kg", "kg": "kg",
    "ounces": "oz", "ounce": "oz", "oz": "oz",
    "grams": "g", "gram": "g", "g": "g",
    "celsius": "c", "fahrenheit": "f", "c": "c", "f": "f",
}
_PAIR_NAMES = {
    ("mile", "km"): "miles-to-km", ("km", "mile"): "km-to-miles",
    ("foot", "m"): "feet-to-m", ("m", "foot"): "m-to-feet",
    ("inch", "cm"): "inches-to-cm", ("cm", "inch"): "cm-to-inches",
    ("lb", "kg"): "lbs-to-kg", ("kg", "lb"): "kg-to-lbs",
    ("oz", "g"): "oz-to-g", ("g", "oz"): "g-to-oz",
    ("c", "f"): "c-to-f", ("f", "c"): "f-to-c",
}
_CARD_UNITS = {v: set(k) for k, v in _PAIR_NAMES.items()}


def extract_direction_card(question: str) -> str | None:
    """Return the operator-card name matching the question's source->target
    direction, or None if it cannot be pinned (no two units / no number)."""
    low = question.lower()
    present = set()
    for word, unit in _UNIT_ALIASES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", low):
            present.add(unit)
    if len(present) != 2:
        return None
    num = re.search(r"-?\d+(?:\.\d+)?", low)
    if not num:
        return None
    after = re.search(r"\s*(?:degrees\s+)?([a-z]+)", low[num.end():])
    if not after or after.group(1) not in _UNIT_ALIASES:
        return None
    src = _UNIT_ALIASES[after.group(1)]
    tgt = next(u for u in present if u != src)
    return _PAIR_NAMES.get((src, tgt))


class ThoughtSkillEngine:
    """Loads skill cards into an SQAC store, routes questions, and builds
    injection prompts.

    Injection policy: inject a stored procedure only when the router is
    confident (>= min_confidence) AND the card is a clean single-step
    operator (pattern in injectable_patterns). Everything else answers
    directly. This is the Round 3 finding encoded as a decision rule:
    rate\times -type operators gain up to +60pp when injected; fragile
    multi-step/verbatim procedures and low-confidence routes default to
    direct answering.
    """

    def __init__(
        self,
        skills_path: str,
        semantic: bool = False,
        min_confidence: float = 0.6,
        injectable_patterns: set[str] | None = None,
    ):
        self.skills = load_skills(skills_path)
        issues = validate(self.skills)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise ValueError(
                "skill validation errors: "
                + "; ".join(f"{e.skill_name}: {e.message}" for e in errors)
            )
        self._n_warnings = len([i for i in issues if i.severity == "warning"])
        self.min_confidence = min_confidence
        self.injectable_patterns = injectable_patterns or {"operator"}
        self.store = SqacStore(semantic=semantic)
        for s in self.skills:
            content = s.content_with_badge()
            meta = {"skill": s.name, "domain": s.domain, "pattern": s.pattern}
            for key in s.keys:
                self.store.add(
                    content, key=key, meta=meta, source=f"skill:{s.name}", kind=KIND_SKILL
                )

    def route(self, question: str) -> RoutedSkill | None:
        hits = self.store.search_grouped(question, top_k=1, kind=KIND_SKILL)
        if not hits:
            return None
        h = hits[0]
        return RoutedSkill(
            content=h.content,
            skill=h.meta.get("skill") if isinstance(h.meta, dict) else None,
            confidence=h.confidence,
            mode=h.mode,
        )

    def skill_pattern(self, name: str | None) -> str:
        if name is None:
            return ""
        for s in self.skills:
            if s.name == name:
                return s.pattern
        return ""

    def plan(self, question: str, prefer: str = "expression") -> Plan:
        r = self.route(question)
        if (
            r is None
            or r.skill is None
            or r.confidence < self.min_confidence
            or self.skill_pattern(r.skill) not in self.injectable_patterns
        ):
            reason = (
                "gate: hit=None"
                if r is None
                else f"gate: {r.skill}/{r.confidence:.2f}/pattern="
                f"{self.skill_pattern(r.skill)}"
            )
            return Plan(action="direct", style="direct", routed=r, reason=reason)
        # Deterministic direction pin: the fuzzy tier can't see word order
        # (direction twins tie at ~0.70). For single-pair operators we read
        # which unit sits next to the number and force the exact card, but
        # only when the routed skill is the direction twin of that card.
        dir_name = extract_direction_card(question)
        if (
            dir_name is not None
            and _CARD_UNITS.get(r.skill) == _CARD_UNITS.get(dir_name)
            and r.skill != dir_name
        ):
            target = next((s for s in self.skills if s.name == dir_name), None)
            if target is not None:
                r = RoutedSkill(
                    content=target.content_with_badge(),
                    skill=dir_name,
                    confidence=r.confidence,
                    mode=f"{r.mode}+dir",
                )
        return Plan(
            action="inject",
            style=prefer,
            routed=r,
            reason=f"operator {r.skill} at {r.confidence:.2f}",
        )

    def solve(
        self,
        api_key: str,
        model: str,
        question: str,
        prefer: str = "expression",
        max_tokens: int = 120,
        api_url: str = GROQ_URL,
    ) -> dict:
        """Route -> decide (plan) -> prompt -> complete. Returns the plan,
        the completion fields, and the parsed prediction per style."""
        plan = self.plan(question, prefer)
        msgs = self.prompt(question, plan.style, plan.routed)
        resp = complete(api_key, model, msgs, max_tokens=max_tokens, api_url=api_url)
        pred = eval_expression(resp["text"]) if plan.style == "expression" else None
        return {"plan": plan, "text": resp["text"], "pred": pred, **resp}

    # ── prompt styles ──────────────────────────────────────────────────────

    def prompt(self, question: str, style: str, routed: RoutedSkill | None = None):
        if style == "direct":
            return [
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer with a single number, nothing else.",
                }
            ]
        if style == "cot":
            return [
                {"role": "user", "content": f"Question: {question}\nPlease solve the problem."},
                {"role": "assistant", "content": "Let's think step by step."},
            ]
        if style in ("injected", "injected_concise"):
            if routed is None:
                raise ValueError(f"style={style} requires a routed skill")
            if style == "injected":
                guide = (
                    "Use these reasoning steps to solve the problem. Follow every step "
                    "in order, then give the final answer as ANSWER = <number>.\n\n"
                    f"{routed.content}\n\n"
                    f"Question: {question}"
                )
            else:
                guide = (
                    "Apply the following reasoning procedure silently. Do not write "
                    "the steps out. Finish with exactly one line: ANSWER = <number>.\n\n"
                    f"{routed.content}\n\n"
                    f"Question: {question}"
                )
            return [{"role": "user", "content": guide}]
        if style == "expression":
            if routed is None:
                raise ValueError("style=expression requires a routed skill")
            guide = (
                "Use the reasoning procedure below. Write ONE valid arithmetic "
                "expression (numbers and the operators + - * / ( ) only) that "
                "computes the answer. Do not write any words. Output only the "
                "expression.\n\n"
                f"{routed.content}\n\n"
                f"Question: {question}"
            )
            return [{"role": "user", "content": guide}]
        if style == "cod":
            if routed is None:
                raise ValueError("style=cod requires a routed skill")
            guide = (
                "Use the reasoning procedure below. For each step write at most a "
                "few words as a short draft, then finish with exactly one line: "
                "ANSWER = <number>.\n\n"
                f"{routed.content}\n\n"
                f"Question: {question}"
            )
            return [{"role": "user", "content": guide}]
        raise ValueError(f"unknown style: {style}")


# ── provider call ────────────────────────────────────────────────────────────

def load_api_key(env_var: str = "GROQ_API_KEY", env_file: str | None = None) -> str:
    if os.environ.get(env_var):
        return os.environ[env_var]
    names = {env_var, env_var.replace("_API_KEY", ""), "GROQ"}
    files = [_ENV_CANDIDATES[0]] if env_file else _ENV_CANDIDATES
    for path in files:
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() in names:
                    return v.strip().strip('"').strip("'")
    raise RuntimeError(f"no API key found (set {env_var} or pass a .env file)")


def complete(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    api_url: str = GROQ_URL,
    temperature: float = 0.0,
    retries: int = 6,
    timeout: float = 120.0,
) -> dict:
    """OpenAI-compatible chat completion with 429/5xx backoff. Returns
    text, reasoning, token usage, and latency."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    delay = 1.0
    t0 = time.time()
    for attempt in range(retries):
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sqac-thought-injection/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                jitter = random.uniform(0.0, delay * 0.25)
                time.sleep(delay + jitter)
                delay *= 2
                continue
            raise
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    usage = r.get("usage", {})
    return {
        "text": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning") or msg.get("reasoning_content") or "").strip(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ),
        "latency_s": round(dt, 2),
    }


# ── safe expression evaluation (Program-of-Thoughts arithmetic) ─────────────

_ALLOWED_NODES = frozenset(
    {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd,
    }
)


def _is_safe(tree) -> bool:
    return all(type(n) in _ALLOWED_NODES for n in ast.walk(tree))


def eval_expression(text: str) -> float | None:
    """Interpret the model's output as a bare arithmetic expression
    (tolerates an `ANSWER =` prefix and stray non-math characters).
    Returns the numeric result or None if it is not evaluable."""
    lines = text.splitlines()
    candidate = lines[-1].strip() if lines else text.strip()
    candidate = re.sub(r"(?i)^\s*answer\s*[:=]\s*", "", candidate).strip()
    candidate = re.sub(r"\s*x\s*", "*", candidate)
    candidate = candidate.replace("×", "*").replace("÷", "/")
    cleaned = re.sub(r"[^0-9+\-*/().]", " ", candidate)
    cleaned = " ".join(cleaned.split()).strip().strip("=").strip()
    if not cleaned:
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None
    if not _is_safe(tree):
        return None
    try:
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None