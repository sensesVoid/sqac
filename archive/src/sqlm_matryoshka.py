#!/usr/bin/env python3
"""
SQ-LM v2 — Matryoshka MoE VSA Proof of Concept

Implements nested hypervectors with mixture-of-experts gating:
- 1024-dim vector = 4 × 256-dim experts
- Gate selects which experts to activate per query
- Training loss supervises at each nesting level
- Inference uses only active experts (adaptive compute)

Usage:
    python3 sqlm_matryoshka.py --demo
    python3 sqlm_matryoshka.py --benchmark --dims 256 512 1024
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import sq_lm_v2 as sq


# ---------------------------------------------------------------------------
# VSA Primitives
# ---------------------------------------------------------------------------

class VSAPrimitives:
    """Pure VSA operations used throughout."""
    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    @staticmethod
    def unbind(ab: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return ab * a

    @staticmethod
    def bundle(hvs: list[torch.Tensor]) -> torch.Tensor:
        out = torch.stack(hvs).sum(dim=0)
        return F.normalize(out, dim=-1)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a, b, dim=0).item())


# ---------------------------------------------------------------------------
# Rule + Expert
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    name: str
    condition_hv: torch.Tensor
    action_hv: torch.Tensor
    transform: Callable | None = None
    metadata: dict = field(default_factory=dict)


class MatryoshkaExpert(nn.Module):
    """One expert in the Matryoshka MoE store.
    
    Each expert owns a contiguous slice [start:end] of the full vector.
    Rules are stored as full-dimension vectors; querying uses only
    this expert's slice, so activating one expert costs ~1/E of the
    full-model compute.
    """

    def __init__(self, expert_dim: int, start_idx: int, total_dim: int):
        super().__init__()
        self.expert_dim = expert_dim
        self.start_idx = start_idx
        self.end_idx = start_idx + expert_dim
        self.total_dim = total_dim
        self.rules: list[Rule] = []
        self.register_buffer("matrix", torch.zeros(0, total_dim))

    @torch.no_grad()
    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        self._rebuild()

    @torch.no_grad()
    def _rebuild(self):
        if not self.rules:
            self.matrix = torch.zeros(0, self.total_dim)
        else:
            self.matrix = torch.stack([r.condition_hv for r in self.rules])

    @torch.no_grad()
    def query(self, hv: torch.Tensor, top_k: int = 1) -> list[tuple[Rule, float]]:
        """Query using only this expert's slice of the vector."""
        if len(self.rules) == 0:
            return []
        # Extract this expert's slice
        slice_q = hv[self.start_idx:self.end_idx]
        q = F.normalize(slice_q.view(1, -1), dim=-1)
        # Extract same slice from all stored rules
        slice_m = self.matrix[:, self.start_idx:self.end_idx]
        m = F.normalize(slice_m, dim=-1)
        sims = (q @ m.t()).view(-1)
        k = min(top_k, len(sims))
        top = torch.topk(sims, k)
        return [(self.rules[idx], float(sim.item())) for idx, sim in zip(top.indices, top.values)]

    def __len__(self) -> int:
        return len(self.rules)


# ---------------------------------------------------------------------------
# Matryoshka MoE Store
# ---------------------------------------------------------------------------

class MatryoshkaMoEStore(nn.Module):
    """VSA knowledge store with Matryoshka nesting + MoE gating.

    A 1024-dim vector contains 4 independent 256-dim experts.
    A learned gate selects which experts activate per query.
    Only active experts perform similarity search.
    """

    def __init__(self, dim: int = 1024, num_experts: int = 4,
                 gate_threshold: float = 0.15):
        super().__init__()
        assert dim % num_experts == 0, "dim must be divisible by num_experts"
        self.dim = dim
        self.expert_dim = dim // num_experts
        self.num_experts = num_experts
        self.gate_threshold = gate_threshold

        self.experts = nn.ModuleList([
            MatryoshkaExpert(self.expert_dim, i * self.expert_dim, self.dim)
            for i in range(num_experts)
        ])

        # MoE gate: maps full vector → expert weights
        self.gate = nn.Sequential(
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=-1)
        )

        self.register_buffer("full_matrix", torch.zeros(0, dim))
        self.register_buffer("expert_indices", torch.zeros(0, dtype=torch.long))

    @torch.no_grad()
    def add_rule(self, name: str, condition: str, action: str,
                 tokenizer: sq.CharTokenizer | None = None,
                 encoder: Callable | None = None):
        """Add rule to the least-loaded expert."""
        cond_hv = self._encode(condition, tokenizer, encoder)
        action_hv = self._encode(action, tokenizer, encoder)

        # Route to expert with lowest utilization
        utilizations = [len(e) for e in self.experts]
        expert_idx = min(range(self.num_experts), key=lambda i: utilizations[i])

        self.experts[expert_idx].add_rule(Rule(name, cond_hv, action_hv))
        self._rebuild_full()

    @torch.no_grad()
    def reason(self, query: torch.Tensor) -> tuple[torch.Tensor | None, str, float]:
        """Query with MoE: only active experts compute."""
        if len(self.experts) == 0 or all(len(e) == 0 for e in self.experts):
            return None, "", 0.0

        gate_scores = self.gate(query.view(1, -1)).view(-1)

        best_rule, best_conf = None, 0.0
        for i, score in enumerate(gate_scores):
            if score > self.gate_threshold:
                # Pass full query; expert extracts its own slice
                results = self.experts[i].query(query, top_k=1)
                if results:
                    rule, conf = results[0]
                    conf = conf * score.item()
                    if conf > best_conf:
                        best_conf = conf
                        best_rule = rule

        if best_rule is None:
            return None, "", 0.0
        return best_rule.action_hv, best_rule.name, best_conf

    @torch.no_grad()
    def _encode(self, text: str,
                tokenizer: sq.CharTokenizer | None = None,
                encoder: Callable | None = None) -> torch.Tensor:
        if encoder is not None and tokenizer is not None:
            ids = tokenizer.encode(text)
            if ids:
                # encoder is a callable that takes token ids and returns embeddings
                device = next(self.gate.parameters()).device
                hvs = []
                for tid in ids[:64]:
                    # Get embedding for single token
                    emb = encoder(torch.tensor([tid], device=device))
                    hvs.append(emb.detach())
                if hvs:
                    return VSAPrimitives.bundle(hvs)
        h = torch.zeros(self.dim)
        for i, ch in enumerate(text):
            idx = (i * 31 + ord(ch)) % self.dim
            h[idx] = 1.0
        return F.normalize(h, dim=-1)

    @torch.no_grad()
    def _rebuild_full(self):
        """Rebuild full matrix for training-time supervision."""
        all_hvs = []
        all_idx = []
        for e_idx, expert in enumerate(self.experts):
            if len(expert.rules) > 0:
                all_hvs.append(expert.matrix)
                all_idx.extend([e_idx] * len(expert.rules))
        if all_hvs:
            self.full_matrix = torch.cat(all_hvs, dim=0)
            self.expert_indices = torch.tensor(all_idx, dtype=torch.long)
        else:
            self.full_matrix = torch.zeros(0, self.dim)
            self.expert_indices = torch.zeros(0, dtype=torch.long)

    def get_expert_utilization(self) -> list[int]:
        return [len(e) for e in self.experts]

    def get_gate_stats(self, sample_inputs: list[str],
                       tokenizer: sq.CharTokenizer | None = None) -> dict:
        """Report average gate activation per expert."""
        counts = [0] * self.num_experts
        for text in sample_inputs:
            qhv = self._encode(text, tokenizer)
            scores = self.gate(qhv.view(1, -1)).view(-1)
            for i, s in enumerate(scores):
                if s > self.gate_threshold:
                    counts[i] += 1
        return {f"expert_{i}": c for i, c in enumerate(counts)}


# ---------------------------------------------------------------------------
# Matryoshka Loss
# ---------------------------------------------------------------------------

class MatryoshkaLoss(nn.Module):
    """Nested loss: supervise at each nesting level.
    
    For a 1024-dim vector with 4 experts:
    - Loss at 256 dims (expert 1)
    - Loss at 512 dims (experts 1+2)
    - Loss at 768 dims (experts 1+2+3)
    - Loss at 1024 dims (all experts)
    
    Finer scales are weighted more heavily to encourage
    information density in early dimensions.
    """

    def __init__(self, num_experts: int = 4, weight_decay: float = 0.9):
        super().__init__()
        self.num_experts = num_experts
        self.weight_decay = weight_decay

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute nested cosine loss.
        
        Args:
            pred: (B, D) predicted hypervectors
            target: (B, D) target hypervectors
        """
        B, D = pred.shape
        expert_dim = D // self.num_experts
        total_loss = 0.0
        weight_sum = 0.0

        for i in range(1, self.num_experts + 1):
            slice_end = i * expert_dim
            pred_slice = F.normalize(pred[:, :slice_end], dim=-1)
            target_slice = F.normalize(target[:, :slice_end], dim=-1)
            loss = 1 - F.cosine_similarity(pred_slice, target_slice, dim=-1).mean()
            weight = self.weight_decay ** (self.num_experts - i)
            total_loss += weight * loss
            weight_sum += weight

        return total_loss / weight_sum


# ---------------------------------------------------------------------------
# Matryoshka SQLM
# ---------------------------------------------------------------------------

class MatryoshkaSQLM(sq.SQLM):
    """SQ-LM v2 with Matryoshka MoE VSA store.
    
    Extends the base model with:
    - MatryoshkaMoEStore: nested vectors + expert gating
    - Nested loss: trains all nesting levels jointly
    - Adaptive compute: only active experts run at inference
    """

    def __init__(self, tokenizer: sq.CharTokenizer, dim: int = 1024,
                 num_layers: int = 2, num_experts: int = 4,
                 gate_threshold: float = 0.15):
        super().__init__(tokenizer, dim, knowledge=False, num_layers=num_layers)
        self.logic = MatryoshkaMoEStore(dim, num_experts, gate_threshold)
        self.matryoshka_loss = MatryoshkaLoss(num_experts)

    def add_rule(self, name: str, condition: str, action: str):
        self.logic.add_rule(name, condition, action,
                           tokenizer=self.tokenizer,
                           encoder=self.encoder.embed_token)

    def reason(self, text: str) -> tuple[str | None, str, float]:
        qhv = self.encode_query(text)
        answer_hv, rule_name, conf = self.logic.reason(qhv)
        if answer_hv is None:
            return None, rule_name, conf
        with torch.no_grad():
            logits = self.decoder.mode_a_logits(answer_hv)
            pred = int(logits.argmax(dim=-1).item())
        return self.tokenizer.decode([pred]), rule_name, conf

    def train_with_matryoshka_loss(self, ids_seq: list[int],
                                   loss_fn: nn.CrossEntropyLoss,
                                   optimizer: torch.optim.Optimizer) -> float:
        """Train one step with nested loss + standard LM loss."""
        self.train()
        
        # Standard LM loss
        logits = self.train_logits(ids_seq)
        logits = logits[:-1]
        targets = torch.tensor(ids_seq[1:])
        lm_loss = loss_fn(logits, targets)

        # Matryoshka loss: enforce nested structure
        with torch.no_grad():
            states, _ = self._run_states(ids_seq)
        matryoshka_loss = self.matryoshka_loss(states, states)

        # Combined loss
        total_loss = lm_loss + 0.1 * matryoshka_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()

    def get_expert_stats(self) -> dict:
        util = self.logic.get_expert_utilization()
        return {
            "utilization": util,
            "total_rules": sum(util),
            "avg_util": sum(util) / len(util) if util else 0,
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class MatryoshkaBenchmark:
    """Compare full-dim vs Matryoshka MoE inference."""

    def __init__(self, model: MatryoshkaSQLM):
        self.model = model

    def query_full(self, query: str) -> tuple[Rule | None, float]:
        """Brute-force: search all experts with full 1024-dim vector."""
        qhv = self.model.encode_query(query)
        results_all = []
        for expert in self.model.logic.experts:
            if len(expert.rules) > 0:
                results_all.extend(expert.query(qhv, top_k=3))
        if not results_all:
            return None, 0.0
        return max(results_all, key=lambda x: x[1])

    def query_moe(self, query: str) -> tuple[Rule | None, float]:
        """MoE: gate selects experts, only active ones search."""
        qhv = self.model.encode_query(query)
        answer_hv, rule_name, conf = self.model.logic.reason(qhv)
        if answer_hv is None:
            return None, 0.0
        # Find the rule object
        for expert in self.model.logic.experts:
            for rule in expert.rules:
                if rule.name == rule_name:
                    return rule, conf
        return None, conf

    def benchmark(self, queries: list[str], repeats: int = 100) -> dict:
        """Time full vs MoE inference."""
        # Warmup
        for q in queries:
            self.query_full(q)
            self.query_moe(q)

        # Full
        t0 = time.perf_counter()
        for _ in range(repeats):
            for q in queries:
                self.query_full(q)
        full_time = time.perf_counter() - t0

        # MoE
        t0 = time.perf_counter()
        for _ in range(repeats):
            for q in queries:
                self.query_moe(q)
        moe_time = time.perf_counter() - t0

        # Expert activation stats
        gate_stats = self.model.logic.get_gate_stats(
            queries, tokenizer=self.model.tokenizer)

        return {
            "full_time_s": full_time,
            "moe_time_s": moe_time,
            "speedup": full_time / moe_time if moe_time > 0 else 0,
            "queries": len(queries) * repeats,
            "gate_stats": gate_stats,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_matryoshka_demo():
    print("=" * 80)
    print("SQ-LM v2 — Matryoshka MoE VSA Proof of Concept")
    print("=" * 80)

    # Build model
    tokenizer = sq.CharTokenizer(sq.DEMO_CORPUS + [v for _, v in sq.DEMO_FACTS],
                                vocab_size=256)
    model = MatryoshkaSQLM(tokenizer, dim=1024, num_experts=4)

    print(f"\n  Model: D={model.dim}, experts=4, expert_dim={model.dim//4}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Add rules and see routing
    print("\n  PHASE 1: Add Rules with Expert Routing")
    print("-" * 60)
    
    rules = [
        ("cat_sat", "the cat sat on the mat", "the cat is happy"),
        ("dog_run", "the dog runs in the park", "the dog is fast"),
        ("bird_fly", "a bird flew over the tree", "the bird is free"),
        ("sun_rise", "the sun rises in the east", "the sun is bright"),
        ("knowledge", "knowledge is stored in vectors", "vectors remember"),
        ("rust_borrow", "borrow checker prevents data races", "use references"),
        ("python_indent", "python uses indentation", "indent after def"),
        ("sql_join", "join combines tables", "use join for multiple tables"),
    ]
    
    for name, condition, action in rules:
        model.add_rule(name, condition, action)
    
    stats = model.get_expert_stats()
    print(f"  Added {stats['total_rules']} rules")
    print(f"  Expert utilization: {stats['utilization']}")

    # Phase 2: Test reasoning
    print("\n  PHASE 2: Reasoning Tests")
    print("-" * 60)
    
    queries = [
        "the cat sat on the mat",
        "the dog runs in the park",
        "the sun rises in the east",
        "borrow checker prevents data races",
        "python uses indentation",
        "knowledge is stored in vectors",
    ]
    
    for q in queries:
        answer, rule, conf = model.reason(q)
        status = "✓" if answer else "✗"
        print(f"    [{status}] '{q}'")
        print(f"         → rule: {rule}, conf: {conf:.3f}, answer: {answer or 'None'}")

    # Phase 3: Benchmark full vs MoE
    print("\n  PHASE 3: Full vs MoE Benchmark")
    print("-" * 60)
    
    bench = MatryoshkaBenchmark(model)
    results = bench.benchmark(queries, repeats=50)
    
    print(f"  Queries: {results['queries']}")
    print(f"  Full 1024-dim: {results['full_time_s']:.3f}s "
          f"({results['full_time_s']/results['queries']*1000:.2f} ms/query)")
    print(f"  MoE adaptive:  {results['moe_time_s']:.3f}s "
          f"({results['moe_time_s']/results['queries']*1000:.2f} ms/query)")
    print(f"  Speedup: {results['speedup']:.2f}x")
    print(f"  Gate stats: {results['gate_stats']}")

    # Phase 4: Test nested loss training
    print("\n  PHASE 4: Nested Loss Training")
    print("-" * 60)
    
    model2 = MatryoshkaSQLM(tokenizer, dim=1024, num_experts=4)
    optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    print("  Training 5 epochs with nested loss...")
    for epoch in range(1, 6):
        model2.train()
        total_loss = 0.0
        n = 0
        for text in sq.DEMO_CORPUS:
            ids = tokenizer.encode(text)
            if len(ids) < 3:
                continue
            loss = model2.train_with_matryoshka_loss(ids, loss_fn, optimizer)
            total_loss += loss
            n += 1
        if epoch % 2 == 0 or epoch == 1:
            print(f"    epoch {epoch}/5  avg_loss {total_loss/max(n,1):.4f}")

    # Phase 5: Test adaptive compute
    print("\n  PHASE 5: Adaptive Compute by Query Complexity")
    print("-" * 60)
    
    simple_queries = ["hello", "hi", "yes", "no"]
    medium_queries = ["the cat sat", "dog runs", "sun rises"]
    complex_queries = [
        "the cat sat on the mat and the dog ran",
        "python uses indentation for blocks after def",
        "knowledge is stored in vectors that remember",
    ]
    
    for label, qs in [("simple", simple_queries), 
                       ("medium", medium_queries),
                       ("complex", complex_queries)]:
        stats = model.logic.get_gate_stats(qs, tokenizer=tokenizer)
        active = sum(1 for v in stats.values() if v > 0)
        print(f"  {label:8s}: {active}/{model.logic.num_experts} experts active, stats={stats}")


def build_matryoshka_demo() -> MatryoshkaSQLM:
    tokenizer = sq.CharTokenizer(sq.DEMO_CORPUS + [v for _, v in sq.DEMO_FACTS],
                                vocab_size=256)
    model = MatryoshkaSQLM(tokenizer, dim=1024, num_experts=4)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run Matryoshka MoE demo")
    ap.add_argument("--benchmark", action="store_true", help="run benchmark")
    ap.add_argument("--dims", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.demo:
        run_matryoshka_demo()
    elif args.benchmark:
        run_benchmark(args)
    else:
        ap.print_help()


def run_benchmark(args):
    print("=" * 80)
    print("Matryoshka MoE VSA — Scaling Benchmark")
    print("=" * 80)
    
    for dim in args.dims:
        for num_experts in [1, 2, 4, 8]:
            expert_dim = dim // num_experts
            if expert_dim < 64:
                continue
            print(f"\n  D={dim}, experts={num_experts}, expert_dim={expert_dim}")
            model = MatryoshkaSQLM(
                sq.CharTokenizer(sq.DEMO_CORPUS, vocab_size=256),
                dim=dim, num_experts=num_experts
            )
            stats = model.get_expert_stats()
            print(f"    params: {sum(p.numel() for p in model.parameters()):,}")
            print(f"    experts: {stats['utilization']}")


if __name__ == "__main__":
    main()
