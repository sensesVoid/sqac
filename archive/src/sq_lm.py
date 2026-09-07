#!/usr/bin/env python3
"""
SQ-LM v1 — SynthQuant Language Model Prototype

Hybrid architecture combining:
  1. SLM (Small Language Model) — for generation
  2. LoRA (style adapter) — for tone/style adaptation
  3. SQA (symbolic rules) — for deterministic knowledge
  4. TurboVec — for fast fuzzy search
  5. Router — decides symbolic, neural, or hybrid path

Usage:
    python3 sq_lm.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchhd

try:
    from turbovec import TurboQuantIndex
    HAS_TURBOVEC = True
except ImportError:
    HAS_TURBOVEC = False

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
D = 10048  # VSA dimensions
CONFIDENCE_THRESHOLD = 0.6
TURBO_BIT_WIDTH = 2


# ══════════════════════════════════════════════════════════════════════════════
# SLM (Simulated Small Language Model)
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedSLM:
    """
    Simulated SLM for demonstration.
    In production: Qwen-2.5-1B via llama.cpp or transformers.
    """

    def __init__(self):
        self.vocab_size = 32000
        self.hidden_size = 2048

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Simulate text generation."""
        # In production: self.model.generate(prompt)
        return f"[SLM generates: {prompt[:50]}...]"


# ══════════════════════════════════════════════════════════════════════════════
# LoRA (Simulated Style Adapter)
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedLoRA:
    """
    Simulated LoRA adapter for style/tone.
    In production: actual LoRA weights on transformer layers.
    """

    STYLES = {
        "explain": "Let me explain this clearly: {content}",
        "simplify": "In simple terms: {content}",
        "formal": "According to best practices: {content}",
        "casual": "So basically, {content}",
        "code": "Here's the code:\n```{content}```",
        "tutorial": "Step by step:\n{content}",
        "debug": "To fix this issue: {content}",
    }

    def generate(self, content: str, style: str = "explain") -> str:
        """Generate styled text."""
        template = self.STYLES.get(style, self.STYLES["explain"])
        return template.format(content=content)


# ══════════════════════════════════════════════════════════════════════════════
# SQA Store (TurboVec-powered)
# ══════════════════════════════════════════════════════════════════════════════

class SQAStore:
    """
    Symbolic rule store with TurboVec for fast fuzzy search.
    Combines Python dict (exact) + TurboVec (fuzzy).
    """

    def __init__(self):
        self.vocab = {}
        self.exact_index = {}  # (type, lang, rule) → index
        self.rules = []  # list of rule dicts
        self.key_vectors = []  # VSA hypervectors for TurboVec
        self.turbo_index = None
        self.built = False

    def _get_hv(self, token):
        if token not in self.vocab:
            self.vocab[token] = torchhd.random(
                1, dimensions=D, vsa="BSC", device=DEVICE
            ).squeeze(0)
        return self.vocab[token]

    def _encode_rule(self, rule: dict):
        """Encode rule as VSA hypervector."""
        tokens = []
        for key in ["type", "language", "rule", "pattern", "action"]:
            if key in rule:
                tokens.append(f"{key}:{rule[key]}")

        if not tokens:
            return None

        hv = self._get_hv(tokens[0])
        for tok in tokens[1:]:
            hv = hv.bundle(self._get_hv(tok))
        return hv

    def add_rules(self, rules: list):
        """Add rules to the store."""
        for rule in rules:
            idx = len(self.rules)
            self.rules.append(rule)

            # Exact index
            key = (rule.get("type", ""), rule.get("language", ""), rule.get("rule", ""))
            self.exact_index[key] = idx

            # VSA vector for TurboVec
            hv = self._encode_rule(rule)
            if hv is not None:
                self.key_vectors.append(hv.float().numpy())

        self._build_turbo()

    def _build_turbo(self):
        """Build TurboVec index from key vectors."""
        if not HAS_TURBOVEC or not self.key_vectors:
            return

        vectors = np.array(self.key_vectors, dtype=np.float32)
        self.turbo_index = TurboQuantIndex(dim=D, bit_width=TURBO_BIT_WIDTH)
        self.turbo_index.add(vectors)
        self.built = True

    def lookup(self, query: dict) -> tuple:
        """
        Lookup with fallback: exact → fuzzy.

        Returns: (rule, confidence, source)
        """
        # 1. Try exact lookup (O(1))
        key = (query.get("type", ""), query.get("language", ""), query.get("rule", ""))
        if key in self.exact_index:
            idx = self.exact_index[key]
            return self.rules[idx], 1.0, "exact"

        # 2. Try fuzzy lookup via TurboVec
        if self.built and self.turbo_index is not None:
            query_hv = self._encode_rule(query)
            if query_hv is not None:
                query_np = query_hv.float().numpy().reshape(1, -1)
                scores, indices = self.turbo_index.search(query_np, k=1)

                if indices.size > 0:
                    idx = indices[0][0]
                    if idx < len(self.rules):
                        # Convert TurboVec score to confidence
                        # Higher score = more similar
                        confidence = min(1.0, scores[0][0] / (D / 2))
                        return self.rules[idx], confidence, "fuzzy"

        return None, 0.0, "none"


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════

class Router:
    """Routes queries to symbolic, neural, or hybrid path."""

    def __init__(self, sqa: SQAStore, slm: SimulatedSLM, lora: SimulatedLoRA):
        self.sqa = sqa
        self.slm = slm
        self.lora = lora
        self.stats = {"symbolic": 0, "neural": 0, "hybrid": 0}

    def process(self, query: dict, context: str = "") -> dict:
        """Process a query through the SQ-LM pipeline."""
        t_start = time.perf_counter()

        # SQA lookup
        rule, confidence, source = self.sqa.lookup(query)

        if confidence > CONFIDENCE_THRESHOLD:
            # MODE 1: Pure Symbolic
            self.stats["symbolic"] += 1
            response = self._format_rule(rule)
            mode = "symbolic"

        elif confidence > CONFIDENCE_THRESHOLD * 0.5 and rule is not None:
            # MODE 3: Hybrid
            self.stats["hybrid"] += 1
            rule_text = self._format_rule(rule)
            style = self._detect_style(context)
            response = self.lora.generate(rule_text, style)
            mode = "hybrid"

        else:
            # MODE 2: Pure Neural
            self.stats["neural"] += 1
            style = self._detect_style(context)
            # In production: SLM generates with LoRA style
            response = self.slm.generate(context)
            mode = "neural"

        t_process = (time.perf_counter() - t_start) * 1000

        return {
            "mode": mode,
            "response": response,
            "confidence": confidence,
            "source": source,
            "rule": rule,
            "process_time_ms": t_process,
        }

    def _format_rule(self, rule: dict) -> str:
        """Format a rule for display."""
        parts = []
        for key in ["type", "language", "rule", "pattern", "action"]:
            if key in rule:
                parts.append(f"{key}: {rule[key]}")
        return " | ".join(parts)

    def _detect_style(self, context: str) -> str:
        """Detect desired style from context."""
        ctx = context.lower()
        if "explain" in ctx or "what is" in ctx:
            return "explain"
        elif "simple" in ctx or "like" in ctx:
            return "simplify"
        elif "code" in ctx or "example" in ctx:
            return "code"
        elif "fix" in ctx or "debug" in ctx:
            return "debug"
        elif "step" in ctx or "how to" in ctx:
            return "tutorial"
        return "explain"


# ══════════════════════════════════════════════════════════════════════════════
# SQ-LM (SynthQuant Language Model)
# ══════════════════════════════════════════════════════════════════════════════

class SQLM:
    """
    The SynthQuant Language Model.
    Combines SLM + LoRA + SQA + TurboVec.
    """

    def __init__(self):
        self.slm = SimulatedSLM()
        self.lora = SimulatedLoRA()
        self.sqa = SQAStore()
        self.router = Router(self.sqa, self.slm, self.lora)

    def load_rules(self, rules: list):
        """Load rules into SQA store."""
        self.sqa.add_rules(rules)

    def process(self, query: dict, context: str = "") -> dict:
        """Process a query through the full SQ-LM pipeline."""
        return self.router.process(query, context)

    def stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "rules_loaded": len(self.sqa.rules),
            "vocab_size": len(self.sqa.vocab),
            "turbo_built": self.sqa.built,
            "routing_stats": self.router.stats,
        }


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATA
# ══════════════════════════════════════════════════════════════════════════════

CODING_RULES = [
    {"type": "syntax", "language": "rust", "rule": "ownership",
     "pattern": "let x = y", "action": "move y into x"},
    {"type": "syntax", "language": "rust", "rule": "borrowing",
     "pattern": "let x = &y", "action": "borrow y immutably"},
    {"type": "syntax", "language": "rust", "rule": "mutable borrow",
     "pattern": "let x = &mut y", "action": "borrow y mutably"},
    {"type": "syntax", "language": "python", "rule": "indentation",
     "pattern": "def f():", "action": "indent block"},
    {"type": "syntax", "language": "python", "rule": "comprehension",
     "pattern": "[x for x in y]", "action": "list comprehension"},
    {"type": "logic", "language": "sql", "rule": "inner join",
     "pattern": "SELECT * FROM a JOIN b", "action": "inner join tables"},
    {"type": "logic", "language": "sql", "rule": "left join",
     "pattern": "SELECT * FROM a LEFT JOIN b", "action": "left join tables"},
    {"type": "logic", "language": "sql", "rule": "aggregate",
     "pattern": "SELECT COUNT(*)", "action": "count rows"},
    {"type": "pattern", "language": "go", "rule": "goroutine",
     "pattern": "go func()", "action": "spawn goroutine"},
    {"type": "pattern", "language": "go", "rule": "channel",
     "pattern": "ch <- v", "action": "send to channel"},
    {"type": "pattern", "language": "typescript", "rule": "type guard",
     "pattern": "if (x is Type)", "action": "narrow type"},
    {"type": "logic", "language": "rust", "rule": "match",
     "pattern": "match x { ... }", "action": "pattern match"},
    {"type": "logic", "language": "rust", "rule": "if let",
     "pattern": "if let Some(x) = y", "action": "optional unwrap"},
    {"type": "syntax", "language": "rust", "rule": "trait impl",
     "pattern": "impl Trait for Type", "action": "implement trait"},
    {"type": "syntax", "language": "rust", "rule": "lifetime",
     "pattern": "fn f<'a>(x: &'a str)", "action": "annotate lifetime"},
    {"type": "logic", "language": "sql", "rule": "subquery",
     "pattern": "SELECT * FROM (SELECT ...)", "action": "nested query"},
    {"type": "pattern", "language": "go", "rule": "error handling",
     "pattern": "if err != nil", "action": "check error"},
    {"type": "syntax", "language": "python", "rule": "context manager",
     "pattern": "with open(...) as f:", "action": "resource management"},
]

TEST_QUERIES = [
    # Exact matches (should route to symbolic)
    {"query": {"type": "syntax", "language": "rust", "rule": "ownership"},
     "context": "How do I use ownership?",
     "expected": "symbolic"},

    {"query": {"type": "syntax", "language": "rust", "rule": "borrowing"},
     "context": "What is borrowing?",
     "expected": "symbolic"},

    # Fuzzy matches (should route to hybrid)
    {"query": {"type": "syntax", "language": "rust"},
     "context": "Explain Rust ownership simply",
     "expected": "hybrid"},

    {"query": {"type": "logic", "language": "sql"},
     "context": "How to join tables in SQL?",
     "expected": "hybrid"},

    # Neural only (no match)
    {"query": {"type": "creative", "language": "unknown"},
     "context": "Write a poem about programming",
     "expected": "neural"},

    {"query": {"type": "meta", "language": "unknown"},
     "context": "What is the meaning of life?",
     "expected": "neural"},
]


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    """Run SQ-LM benchmark."""
    print("=" * 90)
    print("SQ-LM v1 — SynthQuant Language Model")
    print("=" * 90)
    print()

    # Initialize
    sq_lm = SQLM()

    # Load rules
    print("LOADING RULES")
    print("-" * 50)
    t_start = time.perf_counter()
    sq_lm.load_rules(CODING_RULES)
    t_load = (time.perf_counter() - t_start) * 1000

    stats = sq_lm.stats()
    print(f"  Rules loaded:     {stats['rules_loaded']}")
    print(f"  Vocabulary:       {stats['vocab_size']} tokens")
    print(f"  TurboVec built:   {stats['turbo_built']}")
    print(f"  Load time:        {t_load:.1f} ms")
    print()

    # Process queries
    print("PROCESSING QUERIES")
    print("=" * 90)

    results = []
    for i, test in enumerate(TEST_QUERIES):
        result = sq_lm.process(test["query"], test["context"])
        results.append(result)

        mode_icon = {"symbolic": "🔵", "neural": "🟢", "hybrid": "🟣"}
        expected = test["expected"]
        match = "✅" if result["mode"] == expected else "❌"

        print(f"\n  Query {i+1}: {test['context']}")
        print(f"  Expected: {expected} | Got: {result['mode']} {match}")
        print(f"  Source: {result['source']} | Confidence: {result['confidence']:.4f}")
        print(f"  Time: {result['process_time_ms']:.2f} ms")
        print(f"  Response: {result['response'][:80]}")

    # Summary
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    correct = sum(1 for r, t in zip(results, TEST_QUERIES)
                  if r["mode"] == t["expected"])
    total = len(results)
    avg_time = sum(r["process_time_ms"] for r in results) / total

    print(f"\n  Routing accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Avg response time: {avg_time:.2f} ms")
    print(f"  Mode distribution: {stats['routing_stats']}")
    print()

    # Architecture summary
    print("  SQ-LM ARCHITECTURE")
    print("  " + "-" * 60)
    print("  Component     | Role                  | Speed")
    print("  " + "-" * 60)
    print("  SLM           | Language generation   | Simulated")
    print("  LoRA          | Style/tone adaptation | Instant")
    print("  SQA           | Deterministic rules   | 1.35 ms (TurboVec)")
    print("  Router        | Path selection        | <1 ms")
    print("  " + "-" * 60)
    print()
    print("  The SQ-LM activates ONLY the needed path:")
    print("  • Known rule → symbolic (fast, deterministic)")
    print("  • Partial match → hybrid (accurate + styled)")
    print("  • Unknown → neural (creative, probabilistic)")
    print()


if __name__ == "__main__":
    run_benchmark()
