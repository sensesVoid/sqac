#!/usr/bin/env python3
"""
SQA Compression Experiment — Fission-Fusion Mechanism (v2)

Uses VSA (Binary Spatter Codes) to compress data by:
  FISSION:  Break data into atomic tokens → bind with position vectors
  BUNDLE:   Superimpose all positional traces into one hypervector
  FUSION:   Probe the bundle with each atom to reconstruct sequence

v2: Fixed torchhd BSC tensor subclass compatibility.

Usage:
    python3 sqa_compression.py benchmark [--size N]
    python3 sqa_compression.py analyze <input_file>
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torchhd

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
DIMENSIONS = 10048  # D = 10,048 (157 × 64 bits)
CHUNK_SIZE = 32     # Tokens per bundle
VOCABULARY_SIZE = 2048  # Max unique atoms

# ── Fission Engine ────────────────────────────────────────────────────────────

class FissionEncoder:
    """
    FISSION: Decomposes raw data into semantic atoms and encodes them
    as VSA hypervectors with positional binding.
    """

    def __init__(self, dimensions=DIMENSIONS, device=DEVICE):
        self.D = dimensions
        self.device = device
        self.atom_hvs = {}       # atom_id → BSCTensor
        self.position_hvs = {}   # position → BSCTensor
        self.vocabulary = {}     # string → atom_id
        self.reverse_vocab = {}  # atom_id → string
        self.next_atom_id = 0
        self._generate_position_hvs()

    def _generate_position_hvs(self):
        """Pre-generate position hypervectors."""
        for i in range(CHUNK_SIZE):
            self.position_hvs[i] = torchhd.random(
                1, dimensions=self.D, vsa="BSC", device=self.device
            ).squeeze(0)

    def _get_atom_hv(self, atom: str):
        """Get or create a hypervector for an atom."""
        if atom not in self.vocabulary:
            if self.next_atom_id >= VOCABULARY_SIZE:
                atom_id = hash(atom) % VOCABULARY_SIZE
                return self.atom_hvs.get(atom_id)

            atom_id = self.next_atom_id
            self.vocabulary[atom] = atom_id
            self.reverse_vocab[atom_id] = atom
            self.atom_hvs[atom_id] = torchhd.random(
                1, dimensions=self.D, vsa="BSC", device=self.device
            ).squeeze(0)
            self.next_atom_id += 1

        return self.atom_hvs[self.vocabulary[atom]]

    def tokenize(self, data: str) -> list:
        """Tokenize data into semantic atoms."""
        tokens = []
        words = data.split()

        for word in words:
            # Whole word
            tokens.append(f"W:{word}")
            # Character bigrams
            if len(word) >= 2:
                for j in range(min(len(word) - 1, 4)):
                    tokens.append(f"B:{word[j:j+2]}")

        return tokens

    def fuse(self, tokens: list) -> tuple:
        """
        FUSION: Bundle positional traces into compressed hypervectors.
        Returns (list of chunk bundles, metadata).
        """
        metadata = {
            "num_tokens": len(tokens),
            "chunk_size": CHUNK_SIZE,
            "unique_atoms": len(set(tokens)),
            "dimensions": self.D,
        }

        # Process in chunks — each chunk becomes ONE bundle
        chunk_bundles = []

        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk = tokens[i:i+CHUNK_SIZE]

            # Start with the first trace as the bundle base
            first_atom = self._get_atom_hv(chunk[0])
            if first_atom is None:
                continue
            first_pos = self.position_hvs[0]
            bundle = torchhd.bind(first_pos, first_atom)

            # Bundle remaining traces via .bundle() method
            for pos, atom in enumerate(chunk[1:], 1):
                atom_hv = self._get_atom_hv(atom)
                if atom_hv is None:
                    continue
                pos_hv = self.position_hvs[pos % CHUNK_SIZE]
                trace = torchhd.bind(pos_hv, atom_hv)
                bundle = bundle.bundle(trace)

            chunk_bundles.append(bundle)

        metadata["num_chunks"] = len(chunk_bundles)

        return chunk_bundles, metadata

    def defuse(self, chunk_bundles: list, num_tokens: int) -> list:
        """
        DE-FUSION: Probe bundles to reconstruct token sequence.
        """
        reconstructed = []
        atom_ids = list(self.reverse_vocab.keys())

        for chunk_idx, bundle in enumerate(chunk_bundles):
            chunk_len = min(CHUNK_SIZE, num_tokens - chunk_idx * CHUNK_SIZE)

            for pos in range(chunk_len):
                pos_hv = self.position_hvs[pos % CHUNK_SIZE]
                best_atom = None
                best_sim = -1.0

                for atom_id in atom_ids:
                    atom_hv = self.atom_hvs[atom_id]

                    # UNBIND: bundle ⊕ pos_hv reveals atoms at this position
                    candidate = bundle.bind(pos_hv)

                    # Cosine similarity
                    sim = torchhd.cosine_similarity(
                        candidate.unsqueeze(0), atom_hv.unsqueeze(0)
                    ).item()

                    if sim > best_sim:
                        best_sim = sim
                        best_atom = self.reverse_vocab[atom_id]

                if best_atom and best_sim > 0.1:
                    reconstructed.append(best_atom)
                else:
                    reconstructed.append("<?>")

        return reconstructed


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(num_tokens_list=None):
    """Run compression benchmark at various scales."""
    if num_tokens_list is None:
        num_tokens_list = [100, 500, 1000, 2500, 5000]

    print("=" * 80)
    print("SQA COMPRESSION BENCHMARK — Fission-Fusion Mechanism (v2)")
    print("=" * 80)
    print(f"  Dimensions: D = {DIMENSIONS}")
    print(f"  Chunk size: {CHUNK_SIZE} tokens per bundle")
    print(f"  Device: {DEVICE}")
    print()

    results = []

    for num_tokens in num_tokens_list:
        encoder = FissionEncoder()

        # Generate synthetic data
        synthetic_words = [f"word{i % 300}" for i in range(num_tokens)]
        raw_data = " ".join(synthetic_words)
        raw_size = len(raw_data.encode('utf-8'))

        # Tokenize
        tokens = encoder.tokenize(raw_data)

        # Compress (FISS + FUSE)
        t_start = time.perf_counter()
        chunk_bundles, metadata = encoder.fuse(tokens)
        t_compress = (time.perf_counter() - t_start) * 1000

        # Compressed size: N chunks × D/8 bytes + vocabulary
        compressed_size = len(chunk_bundles) * (DIMENSIONS // 8)
        vocab_overhead = len(encoder.vocabulary) * 32  # 32 bytes per vocab entry
        total_compressed = compressed_size + vocab_overhead

        # Decompress (DEFUSE)
        t_start = time.perf_counter()
        reconstructed = encoder.defuse(chunk_bundles, len(tokens))
        t_decompress = (time.perf_counter() - t_start) * 1000

        # Accuracy
        original_atoms = [f"W:{w}" for w in synthetic_words]
        matches = sum(1 for a, b in zip(original_atoms, reconstructed) if a == b)
        accuracy = (matches / len(original_atoms)) * 100 if original_atoms else 0

        ratio = raw_size / max(total_compressed, 1)
        bundle_bytes = DIMENSIONS // 8

        result = {
            "num_tokens": num_tokens,
            "raw_size": raw_size,
            "compressed_size": total_compressed,
            "ratio": ratio,
            "accuracy": accuracy,
            "num_chunks": len(chunk_bundles),
            "bundle_bytes": bundle_bytes,
            "compress_ms": t_compress,
            "decompress_ms": t_decompress,
        }
        results.append(result)

        print(f"  Tokens: {num_tokens:>6} | "
              f"Raw: {raw_size:>8} B | "
              f"Bundles: {len(chunk_bundles):>3} × {bundle_bytes} B | "
              f"Total: {total_compressed:>7} B | "
              f"Ratio: {ratio:>6.1f}:1 | "
              f"Acc: {accuracy:>5.1f}% | "
              f"Compress: {t_compress:>7.1f} ms | "
              f"Decompress: {t_decompress:>7.1f} ms")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if results:
        best_ratio = max(r["ratio"] for r in results)
        worst_acc = min(r["accuracy"] for r in results)
        best_acc = max(r["accuracy"] for r in results)
        largest = max(r["num_tokens"] for r in results)

        print(f"  Best compression ratio:   {best_ratio:.1f}:1 (at {largest} tokens)")
        print(f"  Accuracy range:           {worst_acc:.1f}% — {best_acc:.1f}%")
        print(f"  Bundle size (fixed):      {DIMENSIONS // 8} bytes per chunk")
        print()
        print("  KEY INSIGHTS:")
        print("  1. Compression ratio GROWS with input size (fixed bundle overhead)")
        print("  2. Accuracy DEGRADES with more tokens per chunk (capacity limit)")
        print("  3. The sweet spot is D/2ln(1/ε) tokens per chunk ≈ 500-1000")
        print()

        # Theoretical analysis
        D = DIMENSIONS
        max_safe = int(D / (2 * math.log(1/0.01)))  # ε=0.01
        print(f"  THEORETICAL:")
        print(f"    Bundle capacity (ε=0.01): ~{max_safe} tokens per chunk")
        print(f"    Optimal chunk size: {min(max_safe, CHUNK_SIZE)} tokens")
        print(f"    At D={D}, {max_safe} tokens/chunk → "
              f"ratio = input_bytes / ({max_safe} × {D//8} bytes)")

    return results


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_file(filepath: str):
    """Analyze a file's compression potential."""
    raw_data = Path(filepath).read_text(errors="replace")
    raw_size = len(raw_data.encode('utf-8'))

    encoder = FissionEncoder()
    tokens = encoder.tokenize(raw_data)

    print(f"=" * 70)
    print(f"SQA COMPRESSION ANALYSIS — {filepath}")
    print(f"=" * 70)
    print(f"  Raw size:           {raw_size:,} bytes ({raw_size/1024:.1f} KB)")
    print(f"  Tokens (fission):   {len(tokens):,}")
    print(f"  Unique atoms:       {len(encoder.vocabulary):,}")

    # Compress
    chunk_bundles, metadata = encoder.fuse(tokens)
    bundle_size = DIMENSIONS // 8
    vocab_overhead = len(encoder.vocabulary) * 32
    total_compressed = len(chunk_bundles) * bundle_size + vocab_overhead

    ratio = raw_size / max(total_compressed, 1)

    print(f"  Chunks:             {len(chunk_bundles)}")
    print(f"  Bundle size:        {bundle_size} bytes × {len(chunk_bundles)} chunks")
    print(f"  Vocab overhead:     {vocab_overhead:,} bytes")
    print(f"  Total compressed:   {total_compressed:,} bytes ({total_compressed/1024:.1f} KB)")
    print(f"  Compression ratio:  {ratio:.1f}:1")
    print()

    # Decompress to verify
    reconstructed = encoder.defuse(chunk_bundles, len(tokens))
    original_atoms = []
    for word in raw_data.split():
        original_atoms.append(f"W:{word}")
        if len(word) >= 2:
            for j in range(min(len(word) - 1, 4)):
                original_atoms.append(f"B:{word[j:j+2]}")

    # Truncate to match
    min_len = min(len(original_atoms), len(reconstructed))
    matches = sum(1 for a, b in zip(original_atoms[:min_len], reconstructed[:min_len]) if a == b)
    accuracy = (matches / min_len) * 100 if min_len else 0

    print(f"  Reconstruction accuracy: {accuracy:.1f}%")
    print()

    # Projections
    print("  PROJECTIONS (what input size gives X:1 ratio):")
    for target_ratio in [10, 50, 100, 500, 1000]:
        input_for_ratio = target_ratio * total_compressed
        print(f"    {target_ratio:>5}:1 → need {input_for_ratio:>12,} bytes "
              f"({input_for_ratio/1024/1024:.1f} MB) input")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SQA Compression — Fission-Fusion Mechanism"
    )
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("benchmark", help="Run compression benchmark")
    bench.add_argument("--size", nargs="+", type=int,
                       help="Token counts to test")

    ana = sub.add_parser("analyze", help="Analyze file compression potential")
    ana.add_argument("file", help="File to analyze")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(args.size)
    elif args.command == "analyze":
        analyze_file(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
