#!/usr/bin/env python3
"""
SQ-LM v2 scaled benchmark: sweep (dim, layers) and report perplexity,
top-1 accuracy, and generation quality.

Usage:
    python3 benchmark_sqlm_v2.py --dims 512 1024 --layers 2 3 --epochs 30
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn as nn

import sq_lm_v2 as sq
from eval_sqlm import make_corpus, pad_batch, evaluate


def train_and_eval(dim, layers, vocab, epochs, lr, batch, seed, device):
    train_texts, val_texts = make_corpus(seed=seed, n_train=600, n_val=200)
    tok = sq.CharTokenizer(train_texts, vocab_size=vocab)
    model = sq.SQLM(tok, dim=dim, num_layers=layers)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        rng = random.Random(seed + epoch)
        order = rng.sample(train_texts, len(train_texts))
        ep_loss, ep_tok = 0.0, 0
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            seq_len = max(len(tok.encode(t)) for t in chunk)
            X = pad_batch(tok, chunk, len(chunk), device=model.device)
            logits = model.train_logits_batch(X)
            targets = torch.tensor(X[:, 1:].tolist(), device=model.device)
            lg = logits[:, :-1, :]
            loss = loss_fn(lg.reshape(-1, model.vocab_size), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * targets.numel()
            ep_tok += targets.numel()

        if epoch % 5 == 0 or epoch == epochs:
            v_loss, v_acc = evaluate(model, val_texts, batch=batch)
            train_ppl = math.exp(ep_loss / max(ep_tok, 1))
            val_ppl = math.exp(v_loss)
            print(f"  D={dim} L={layers} ep={epoch:2d}  "
                  f"train_ppl={train_ppl:.2f}  val_ppl={val_ppl:.2f}  "
                  f"val_acc={v_acc:.1%}")

    v_loss, v_acc = evaluate(model, val_texts, batch=batch)
    train_loss, _ = evaluate(model, train_texts, batch=batch)
    dt = time.perf_counter() - t0

    model.eval()
    prompts = ["the cat", "the sun", "water"]
    gens = []
    with torch.no_grad():
        for p in prompts:
            gens.append(model.generate(p, max_tokens=24, temperature=0.4))

    return {
        "dim": dim,
        "layers": layers,
        "train_ppl": math.exp(train_loss),
        "val_ppl": math.exp(v_loss),
        "val_acc": v_acc,
        "time_s": dt,
        "gens": gens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 70)
    print("SQ-LM v2 Scaled Benchmark")
    print("=" * 70)

    results = []
    for dim in args.dims:
        for layers in args.layers:
            print(f"\n--- Config: D={dim}  layers={layers} ---")
            r = train_and_eval(
                dim=dim,
                layers=layers,
                vocab=args.vocab,
                epochs=args.epochs,
                lr=args.lr,
                batch=args.batch,
                seed=args.seed,
                device=args.device,
            )
            results.append(r)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'D':>6} {'L':>4} {'Train PPL':>10} {'Val PPL':>10} {'Val Acc':>10} {'Time':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['dim']:>6} {r['layers']:>4} {r['train_ppl']:>10.2f} "
              f"{r['val_ppl']:>10.2f} {r['val_acc']:>10.1%} "
              f"{r['time_s']:>7.1f}s")

    print("\nGeneration samples:")
    for r in results:
        print(f"\nD={r['dim']} L={r['layers']}:")
        for p, g in zip(["the cat", "the sun", "water"], r["gens"]):
            print(f"  '{p}' -> {g[:80]}")


if __name__ == "__main__":
    main()
