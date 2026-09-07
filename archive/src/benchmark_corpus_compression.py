#!/usr/bin/env python3
"""
SQ-LM v2 — Corpus Compression Benchmark (synthetic duplicates)

Tests whether VSA-based semantic deduplication improves training speed/quality
on a corpus with intentional near-duplicates.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

import sq_lm_v2 as sq


def make_synthetic_corpus() -> list[str]:
    base = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a bird flew over the tree",
        "the sun rises in the east",
        "water flows down the hill",
        "knowledge is stored in vectors",
        "vectors encode meaning in high dimensions",
        "the model remembers without forgetting",
        "binding associates two concepts",
        "bundling combines many ideas into one",
    ]
    corpus = []
    for _ in range(5):
        for s in base:
            corpus.append(s)
    noise = [
        "the cat sat on the mat today",
        "the dog ran quickly in the park",
        "a small bird flew over the tall tree",
        "the sun rises in the eastern sky",
        "water flows down the steep hill",
        "knowledge is stored in high dimensions",
        "vectors encode deep meaning",
        "the model remembers facts without forgetting",
        "binding joins two concepts together",
        "bundling merges many ideas",
    ]
    corpus.extend(noise)
    return corpus


def train_on_corpus(model: sq.SQLM, texts: list[str], epochs: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for text in texts:
            ids = model.tokenizer.encode(text)
            if len(ids) < 3:
                continue
            logits = model.train_logits(ids)
            logits = logits[:-1]
            targets = torch.tensor(ids[1:])
            loss = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            dt = time.perf_counter() - t0
            print(f"    epoch {epoch+1:2d}/{epochs}  loss {total_loss/max(n,1):.4f}  ({dt:.1f}s)")


def evaluate_loss(model: sq.SQLM, texts: list[str]) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for text in texts:
            ids = model.tokenizer.encode(text)
            if len(ids) < 3:
                continue
            logits = model.train_logits(ids)
            logits = logits[:-1]
            targets = torch.tensor(ids[1:])
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--vocab", type=int, default=256)
    args = ap.parse_args()

    print("=" * 70)
    print("SQ-LM v2 — Corpus Compression Benchmark")
    print("=" * 70)
    print(f"  dim={args.dim}  epochs={args.epochs}  threshold={args.threshold}")

    texts = make_synthetic_corpus()
    print(f"\n  Full corpus: {len(texts)} sentences")

    tokenizer = sq.CharTokenizer(texts, vocab_size=args.vocab)
    model_full = sq.SQLM(tokenizer, dim=args.dim)
    model_comp = sq.SQLM(tokenizer, dim=args.dim)

    compressed = model_full.compress_corpus(texts, threshold=args.threshold)
    print(f"  Compressed corpus: {len(compressed)} sentences "
          f"({len(compressed)/len(texts):.1%} of original)")

    print("\n  Training on FULL corpus ...")
    t_full = time.perf_counter()
    train_on_corpus(model_full, texts, args.epochs, args.lr)
    full_time = time.perf_counter() - t_full

    print("\n  Training on COMPRESSED corpus ...")
    t_comp = time.perf_counter()
    train_on_corpus(model_comp, compressed, args.epochs, args.lr)
    comp_time = time.perf_counter() - t_comp

    full_loss = evaluate_loss(model_full, texts)
    comp_loss = evaluate_loss(model_comp, texts)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Full:     {len(texts):>3} sentences  loss={full_loss:.4f}  time={full_time:.1f}s")
    print(f"  Compressed: {len(compressed):>3} sentences  loss={comp_loss:.4f}  time={comp_time:.1f}s")
    print(f"  Speedup: {full_time/max(comp_time,1e-9):.2f}x")
    print(f"  Loss delta: {comp_loss - full_loss:+.4f}")


if __name__ == "__main__":
    main()
