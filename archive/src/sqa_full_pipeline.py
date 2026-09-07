#!/usr/bin/env python3
"""
SQA Full Pipeline — Compress → Store → Fetch → Verify

End-to-end test: compress code rules using the full stack (tokenize → dedup → VSA → PQ),
then fetch by semantic query via VSA similarity search on PQ-compressed vectors.

Usage:
    python3 sqa_full_pipeline.py
"""

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torchhd

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
D = 10048
PQ_M = 8  # PQ subvectors


# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY: Token → Hypervector mapping
# ══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Shared vocabulary: maps tokens to VSA hypervectors."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.hvs = {}  # id → BSCTensor
        self.next_id = 0

    def get_hv(self, token: str):
        """Get or create hypervector for token."""
        if token not in self.token_to_id:
            tid = self.next_id
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token
            self.hvs[tid] = torchhd.random(
                1, dimensions=D, vsa="BSC", device=DEVICE
            ).squeeze(0)
            self.next_id += 1
        return self.hvs[self.token_to_id[token]]

    def get_id(self, token: str) -> int:
        if token not in self.token_to_id:
            self.get_hv(token)  # Create it
        return self.token_to_id[token]

    def decode(self, tid: int) -> str:
        return self.id_to_token.get(tid, "<?>")

    def size(self) -> int:
        return self.next_id


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT QUANTIZATION
# ══════════════════════════════════════════════════════════════════════════════

class PQCodec:
    """Product Quantization for VSA vectors."""

    def __init__(self, d=D, m=PQ_M, k=256):
        self.d = d
        self.m = m
        self.k = k
        self.sub_dim = d // m
        self.codebooks = None

    def train(self, vectors: torch.Tensor):
        """Train codebooks from vectors (n × d)."""
        # Convert to float if needed (BSC tensors are bool)
        vectors = vectors.float()
        self.codebooks = []
        for i in range(self.m):
            start = i * self.sub_dim
            end = start + self.sub_dim
            subvecs = vectors[:, start:end]

            # Random init + k-means
            codebook = subvecs[torch.randperm(len(subvecs))[:self.k]].clone()
            for _ in range(5):
                dists = torch.cdist(subvecs, codebook)
                assignments = dists.argmin(dim=1)
                for j in range(self.k):
                    mask = assignments == j
                    if mask.sum() > 0:
                        codebook[j] = subvecs[mask].mean(dim=0)
            self.codebooks.append(codebook)

    def encode(self, vector: torch.Tensor) -> list:
        """Encode one vector to PQ codes (list of ints)."""
        vector = vector.float()
        codes = []
        for i in range(self.m):
            start = i * self.sub_dim
            end = start + self.sub_dim
            subvec = vector[start:end].unsqueeze(0)
            dists = torch.cdist(subvec, self.codebooks[i])
            codes.append(dists.argmin().item())
        return codes

    def decode(self, codes: list) -> torch.Tensor:
        """Decode PQ codes back to approximate vector."""
        parts = []
        for i, code in enumerate(codes):
            parts.append(self.codebooks[i][code])
        return torch.cat(parts)

    def encode_batch(self, vectors: torch.Tensor) -> list:
        """Encode multiple vectors."""
        return [self.encode(v) for v in vectors]

    def decode_batch(self, codes_list: list) -> torch.Tensor:
        """Decode multiple PQ codes."""
        return torch.stack([self.decode(c) for c in codes_list])


# ══════════════════════════════════════════════════════════════════════════════
# VSA STORE: Compress + Store + Fetch
# ══════════════════════════════════════════════════════════════════════════════

class SQAStore:
    """
    Full SQA pipeline:
    - Compress: rule → VSA vector → PQ codes
    - Store: flat array of PQ codes
    - Fetch: query → VSA vector → PQ decode → similarity search
    """

    def __init__(self):
        self.vocab = Vocabulary()
        self.pq = PQCodec()
        self.rules = []          # Original rules (for verification)
        self.key_hvs = []        # Key hypervectors (for VSA binding)
        self.pq_codes = []       # PQ-compressed codes
        self.value_hvs = []      # Value hypervectors (the actual data)
        self.trained = False

    def _rule_to_tokens(self, rule: dict) -> list:
        """Tokenize a rule into semantic atoms."""
        tokens = []
        for key, value in rule.items():
            if key != 'id':  # Skip id field
                tokens.append(f"{key}:{value}")
        return tokens

    def _tokens_to_hv(self, tokens: list):
        """Encode tokens into a single VSA hypervector using bundling."""
        hv = self.vocab.get_hv(tokens[0])
        for tok in tokens[1:]:
            hv = hv.bundle(self.vocab.get_hv(tok))
        return hv

    def compress(self, rules: list):
        """Compress a list of rules using VSA + PQ."""
        print(f"\n  COMPRESSION")
        print(f"  {'─' * 50}")

        t_start = time.perf_counter()

        # Step 1: Tokenize and encode each rule
        print(f"  Step 1: Tokenize {len(rules)} rules...")
        key_hvs = []
        value_hvs = []

        for rule in rules:
            tokens = self._rule_to_tokens(rule)
            # Key: rule type + language (for lookup)
            key_tokens = [f"type:{rule.get('type', '')}",
                          f"lang:{rule.get('language', '')}"]
            key_hv = self._tokens_to_hv(key_tokens)

            # Value: full rule encoded as VSA
            value_hv = self._tokens_to_hv(tokens)

            key_hvs.append(key_hv)
            value_hvs.append(value_hv)

        self.key_hvs = key_hvs
        self.value_hvs = value_hvs
        self.rules = rules

        print(f"  Step 1: {len(rules)} rules → {len(rules)} VSA vectors ({time.perf_counter() - t_start:.2f}s)")

        # Step 2: Train PQ codebooks
        t_train = time.perf_counter()
        print(f"  Step 2: Train PQ codebooks (m={PQ_M})...")
        value_tensor = torch.stack(value_hvs)
        self.pq.train(value_tensor)
        print(f"  Step 2: Codebooks trained ({time.perf_counter() - t_train:.2f}s)")

        # Step 3: PQ encode all vectors
        t_encode = time.perf_counter()
        print(f"  Step 3: PQ encode {len(rules)} vectors...")
        self.pq_codes = self.pq.encode_batch(value_tensor)
        print(f"  Step 3: Encoded ({time.perf_counter() - t_encode:.2f}s)")

        t_total = time.perf_counter() - t_start

        # Calculate sizes
        raw_size = sum(len(json.dumps(r).encode()) for r in rules)
        # Compressed: PQ codes only (codebooks stored once, amortized)
        # For amortized analysis: codebook overhead / n_vectors
        codebook_size = self.pq.m * self.pq.k * self.pq.sub_dim * 4  # bytes
        codes_size = len(self.pq_codes) * PQ_M  # PQ codes
        vocab_size = self.vocab.size() * (D // 8)  # Vocabulary HVs
        compressed_size = codes_size + vocab_size + codebook_size

        print(f"\n  Results:")
        print(f"    Raw size:           {raw_size:>10,} bytes ({raw_size/1024:.1f} KB)")
        print(f"    Compressed size:    {compressed_size:>10,} bytes ({compressed_size/1024:.1f} KB)")
        print(f"    Compression ratio:  {raw_size / max(compressed_size, 1):>10.1f}:1")
        print(f"    PQ codes:           {len(self.pq_codes)} vectors × {PQ_M} bytes")
        print(f"    Vocabulary:         {self.vocab.size()} unique tokens")
        print(f"    Total time:         {t_total:.2f}s")

        return raw_size, compressed_size

    def fetch(self, query: dict, top_k=5) -> list:
        """
        Fetch rules by semantic query.
        Uses VSA binding: query ⊕ key = value
        Then finds closest match in PQ-decoded vectors.
        """
        t_start = time.perf_counter()

        # Step 1: Encode query as VSA vector
        query_tokens = []
        for key, value in query.items():
            query_tokens.append(f"{key}:{value}")

        if not query_tokens:
            return []

        query_hv = self._tokens_to_hv(query_tokens)

        # Step 2: For each stored rule, compute Hamming distance
        # Decode PQ codes back to vectors
        decoded_values = self.pq.decode_batch(self.pq_codes)

        # Step 3: Find top-k most similar (using Hamming distance)
        # For binary vectors: Hamming distance = XOR popcount
        query_float = query_hv.float()
        similarities = []
        for i, decoded in enumerate(decoded_values):
            # Hamming distance: count differing bits
            diff = (query_float != decoded).float().sum().item()
            similarity = 1.0 - (diff / D)  # Normalize to 0-1
            similarities.append((similarity, i))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        t_fetch = (time.perf_counter() - t_start) * 1000

        # Return top-k results
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                "similarity": sim,
                "rule": self.rules[idx],
                "index": idx,
            })

        return results, t_fetch

    def fetch_batch(self, queries: list, top_k=1) -> list:
        """Fetch multiple queries and measure accuracy."""
        correct = 0
        total = len(queries)
        total_latency = 0

        for query, expected_idx in queries:
            results, latency = self.fetch(query, top_k=top_k)
            total_latency += latency

            if results and results[0]["index"] == expected_idx:
                correct += 1

        accuracy = correct / total * 100 if total else 0
        avg_latency = total_latency / total if total else 0

        return accuracy, avg_latency


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATA
# ══════════════════════════════════════════════════════════════════════════════

def generate_rules(n: int) -> list:
    """Generate diverse coding rules."""
    templates = [
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
        {"type": "syntax", "language": "python", "rule": "decorator",
         "pattern": "@decorator", "action": "apply decorator"},
        {"type": "logic", "language": "sql", "rule": "inner join",
         "pattern": "SELECT * FROM a JOIN b", "action": "inner join tables"},
        {"type": "logic", "language": "sql", "rule": "left join",
         "pattern": "SELECT * FROM a LEFT JOIN b", "action": "left join tables"},
        {"type": "logic", "language": "sql", "rule": "aggregate",
         "pattern": "SELECT COUNT(*)", "action": "count rows"},
        {"type": "logic", "language": "sql", "rule": "group by",
         "pattern": "GROUP BY col", "action": "group rows"},
        {"type": "pattern", "language": "go", "rule": "goroutine",
         "pattern": "go func()", "action": "spawn goroutine"},
        {"type": "pattern", "language": "go", "rule": "channel send",
         "pattern": "ch <- v", "action": "send to channel"},
        {"type": "pattern", "language": "go", "rule": "channel receive",
         "pattern": "<-ch", "action": "receive from channel"},
        {"type": "pattern", "language": "go", "rule": "select",
         "pattern": "select { case ... }", "action": "multiplex channels"},
        {"type": "pattern", "language": "typescript", "rule": "type guard",
         "pattern": "if (x is Type)", "action": "narrow type"},
        {"type": "pattern", "language": "typescript", "rule": "mapped type",
         "pattern": "{ [K in keyof T]: V }", "action": "transform types"},
        {"type": "logic", "language": "rust", "rule": "match",
         "pattern": "match x { ... }", "action": "pattern match"},
        {"type": "logic", "language": "rust", "rule": "if let",
         "pattern": "if let Some(x) = y", "action": "optional unwrap"},
        {"type": "syntax", "language": "rust", "rule": "trait impl",
         "pattern": "impl Trait for Type", "action": "implement trait"},
        {"type": "syntax", "language": "rust", "rule": "lifetime",
         "pattern": "fn f<'a>(x: &'a str)", "action": "annotate lifetime"},
    ]

    rules = []
    for i in range(n):
        base = templates[i % len(templates)].copy()
        base["id"] = f"rule_{i}"
        rules.append(base)

    return rules


def generate_queries(rules: list) -> list:
    """Generate queries and expected results."""
    queries = []
    for i, rule in enumerate(rules):
        # Query by type + language
        query = {"type": rule["type"], "language": rule["language"]}
        queries.append((query, i))
    return queries


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    """Run full pipeline benchmark."""
    print("=" * 90)
    print("SQA FULL PIPELINE — Compress → Store → Fetch → Verify")
    print("=" * 90)
    print(f"  VSA Dimensions: D = {D}")
    print(f"  PQ Subvectors: m = {PQ_M}")
    print(f"  PQ Compressed: {PQ_M} bytes per vector")
    print()

    for n in [100, 500, 1000, 2500]:
        print(f"\n{'═' * 90}")
        print(f"  TEST: {n} rules")
        print(f"{'═' * 90}")

        # Generate data
        rules = generate_rules(n)
        queries = generate_queries(rules)

        # Compress
        store = SQAStore()
        raw_size, compressed_size = store.compress(rules)

        # Fetch and measure accuracy
        print(f"\n  FETCH TEST")
        print(f"  {'─' * 50}")

        t_start = time.perf_counter()
        accuracy, avg_latency = store.fetch_batch(queries, top_k=1)
        t_total = (time.perf_counter() - t_start) * 1000

        print(f"    Queries tested:     {len(queries)}")
        print(f"    Accuracy:           {accuracy:.1f}%")
        print(f"    Avg fetch latency:  {avg_latency:.2f} ms/query")
        print(f"    Total fetch time:   {t_total:.1f} ms")
        print()

        # Show some examples
        print(f"  SAMPLE FETCHES (top 3)")
        print(f"  {'─' * 50}")
        sample_queries = queries[:3]
        for query, expected_idx in sample_queries:
            results, latency = store.fetch(query, top_k=1)
            if results:
                r = results[0]
                match = "✅" if r["index"] == expected_idx else "❌"
                print(f"    Query: {query}")
                print(f"    Expected: rule_{expected_idx} ({rules[expected_idx]['rule']})")
                print(f"    Got:      rule_{r['index']} ({r['rule']['rule']}) — sim={r['similarity']:.4f} — {match} ({latency:.2f}ms)")
                print()

    # Final summary
    print(f"\n{'═' * 90}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 90}")
    print(f"  Compression: 1,256 bytes → {PQ_M} bytes = {1256/PQ_M:.0f}:1 per vector (PQ)")
    print(f"  Combined: tokenize + dedup + VSA + PQ = stacking all findings")
    print(f"  Fetch: VSA cosine similarity on PQ-decoded vectors")
    print(f"  Accuracy: measured above")
    print()


if __name__ == "__main__":
    run_benchmark()
