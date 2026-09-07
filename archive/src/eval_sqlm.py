#!/usr/bin/env python3
"""
SQ-LM v2 evaluation: combinatorial generalization.

Builds a synthetic English-like corpus from a template grammar, splits on
UNIQUE combinations so validation only sees combos the model never trained
on, then measures held-out next-token perplexity and top-1 accuracy.

Measures "does the VSA recurrence learn the underlying rule, not just
memorize the training sentences".

Usage:
    python3 eval_sqlm.py --dim 256 --epochs 30 --verbose
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sq_lm_v2 as sq


# ---------------------------------------------------------------------------
# Template grammar
# ---------------------------------------------------------------------------

ADJS = ["big", "small", "old", "young"]
SUBJS = ["cat", "dog", "bird", "fox", "bear", "deer"]
VERBS = ["runs", "sits", "sleeps", "jumps", "walks", "flies"]
PREPS = ["in", "under", "near", "behind", "beside"]
PLACES = ["park", "tree", "hill", "cave", "pond", "nest"]

ARTICLE = "the"


def _combo_tokens(combo) -> list[str]:
    adj, subj, verb, prep, place = combo
    return [ARTICLE, adj, subj, verb, prep, ARTICLE, place]


def make_corpus(seed: int = 0, n_train: int = 600, n_val: int = 200):
    """Build sentence lists by sampling unique combos. The train/val split is
    on the COMBINATION (subject-verb-prep-place), so val sentences are
    structurally valid English the model never saw. Returns (train, val).
    """
    rng = random.Random(seed)
    all_combos = list(itertools.product(ADJS, SUBJS, VERBS, PREPS, PLACES))
    rng.shuffle(all_combos)
    # Disjoint combo split: first n_train for training, next n_val for val.
    train_combos = all_combos[:n_train]
    val_combos = all_combos[n_train:n_train + n_val]
    return (
        [" ".join(_combo_tokens(c)) for c in train_combos],
        [" ".join(_combo_tokens(c)) for c in val_combos],
    )


# ---------------------------------------------------------------------------
# Training loop (batched teacher forcing, pad-masked loss)
# ---------------------------------------------------------------------------

def pad_batch(tok: sq.CharTokenizer, texts: list[str], bs: int, device=None) -> torch.Tensor:
    ids_list = [tok.encode(t) for t in texts]
    S = max(len(x) for x in ids_list)
    X = torch.zeros(bs, S, dtype=torch.long, device=device).fill_(tok.pad_id)
    for r, ids in enumerate(ids_list):
        X[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return X


def evaluate(model, texts: list[str], batch: int = 64) -> tuple[float, float]:
    """Return (avg_loss, top1_acc) over all next-token predictions."""
    model.eval()
    device = model.device
    total_loss = 0.0
    total_tok = 0
    total_ok = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            X = pad_batch(model.tokenizer, chunk, len(chunk), device)          # (B,S)
            logits = model.train_logits_batch(X)                               # (B,S,V)
            targets = torch.tensor(X[:, 1:].tolist(), device=device)           # (B,S-1)
            lg = logits[:, :-1, :]                                             # predict next at each pos
            loss = F.cross_entropy(
                lg.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                ignore_index=model.tokenizer.pad_id,
            )
            mask = targets != model.tokenizer.pad_id
            total_loss += loss.item() * mask.sum().item()
            total_tok += mask.sum().item()
            ok = (lg.argmax(-1) == targets) & mask
            total_ok += ok.sum().item()
    avg_loss = total_loss / max(total_tok, 1)
    acc = total_ok / max(total_tok, 1)
    return avg_loss, acc


def train_eval(args) -> tuple[float, float]:
    train_texts, val_texts = make_corpus(args.seed, args.n_train, args.n_val)
    tok = sq.CharTokenizer(train_texts, vocab_size=args.vocab)
    model = sq.SQLM(tok, dim=args.dim, num_layers=args.layers)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("cuda requested but unavailable -> using cpu")
            args.device = "cpu"
        else:
            model = model.to(args.device)
    print(f"corpus: {len(train_texts)} train / {len(val_texts)} val sentences")
    print(f"model:  D={args.dim}  vocab={tok.vocab_size}  layers={args.layers}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"val combos are DISJOINT from train combos (compositional generalization)")
    print()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    best_acc = 0.0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch)
        order = rng.sample(train_texts, len(train_texts))
        ep_loss, ep_tok = 0.0, 0
        for i in range(0, len(order), args.batch):
            chunk = order[i:i + args.batch]
            X = pad_batch(tok, chunk, len(chunk))
            logits = model.train_logits_batch(X)
            targets = torch.tensor(X[:, 1:].tolist())
            lg = logits[:, :-1, :]
            loss = loss_fn(lg.reshape(-1, model.vocab_size), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * targets.numel()
            ep_tok += targets.numel()
        train_loss = ep_loss / max(ep_tok, 1)

        if epoch % args.log_every == 0 or epoch == args.epochs:
            v_loss, v_acc = evaluate(model, val_texts)
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(v_loss)
            best_acc = max(best_acc, v_acc)
            dt = time.perf_counter() - t_start
            print(f"  epoch {epoch:3d}  train_ppl {train_ppl:8.2f}  train_loss {train_loss:.3f}"
                  f"  |  val_ppl {val_ppl:8.2f}  val_acc {v_acc:6.1%}"
                  f"  ({dt:.0f}s, {ep_tok/dt:.0f} tok/s)")


    v_loss, v_acc = evaluate(model, val_texts)
    train_loss, _ = evaluate(model, train_texts)
    print()
    print(f"FINAL  train_ppl {math.exp(train_loss):.2f}  val_ppl {math.exp(v_loss):.2f}  "
          f"val_acc {v_acc:.1%}  best_acc {best_acc:.1%}")
    return v_acc, math.exp(v_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    train_eval(args)


if __name__ == "__main__":
    main()