#!/usr/bin/env python3
"""
SQ-LM v2 — formal training on real data streamed from Hugging Face.

Streams a real corpus (default: wikitext-2-raw), trains a fresh byte-level BPE
tokenizer on a sample of it, packs text into fixed-length windows, and trains
the VSA recurrence with batched teacher forcing. Reports train/valid
perplexity and generates continuation samples.

Usage:
    python3 train_real.py --dim 256 --vocab 1024 --budget-tokens 300000 \
        --epochs 1 --batch 64 --seq 64
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

import sq_lm_v2 as sq

PAD, UNK = "<pad>", "<unk>"


class BPETokenizer:
    """Minimal byte-level BPE with the same interface the model expects:
    encode/decode/vocab_size/pad_id/unk_id. pad=0, unk=1 (matches CharTokenizer)."""

    def __init__(self, tok: Tokenizer):
        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.pad_id = tok.token_to_id(PAD) or 0
        self.unk_id = tok.token_to_id(UNK) or 1

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def train_bpe(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[PAD, UNK],
                         min_frequency=2, show_progress=False)
    tok.train_from_iterator(texts, trainer)
    tok.enable_truncation(max_length=1024)
    return tok


def clean_texts(rows):
    """Yield non-empty, non-whitespace Wikipedia lines, stripped."""
    for row in rows:
        t = row.get("text")
        if isinstance(t, str):
            t = t.strip()
            if t:
                yield t


def pack_windows(ids_iter, seq_len: int, budget: int):
    """Concatenate token streams and slice into fixed-length windows."""
    buf: list[int] = []
    n = 0
    for ids in ids_iter:
        buf.extend(ids)
        while len(buf) >= seq_len:
            yield list(buf[:seq_len])
            buf = buf[seq_len:]
            n += seq_len
            if n >= budget:
                return


def build_splits(args):
    """Stream train (sampled for tokenizer + budget) and validation splits."""
    print(f"[1/3] streaming '{args.dataset}' ({args.subset}) ...")
    ds_train = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    ds_val = load_dataset(args.dataset, args.subset, split=args.val_split, streaming=True)

    print(f"[2/3] training BPE tokenizer (vocab={args.vocab}) on sample ...")
    sample = []
    for t in clean_texts(ds_train):
        sample.append(t)
        if len(sample) >= args.tok_samples:
            break
    tok_raw = train_bpe(sample, args.vocab)
    tok = BPETokenizer(tok_raw)

    train_ids = [tok.encode(t) for t in clean_texts(ds_train)]
    val_ids = [tok.encode(t) for t in clean_texts(ds_val)]
    return tok, train_ids, val_ids


def pad_batch(tok, ids_list, seq_len: int, bs: int, device=None) -> torch.Tensor:
    X = torch.zeros(bs, seq_len, dtype=torch.long, device=device).fill_(tok.pad_id)
    for r, ids in enumerate(ids_list[:bs]):
        X[r, :min(len(ids), seq_len)] = torch.tensor(ids[:seq_len], dtype=torch.long)
    return X


def evaluate(model, ids_list: list[list[int]], batch: int):
    model.eval()
    device = model.device
    total_loss, total_tok, total_ok = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(ids_list), batch):
            chunk = ids_list[i:i + batch]
            X = pad_batch(model.tokenizer, chunk, max(len(x) for x in chunk), len(chunk), device)
            logits = model.train_logits_batch(X)
            targets = torch.tensor(X[:, 1:].tolist(), device=device)
            lg = logits[:, :-1, :]
            loss = nn.functional.cross_entropy(
                lg.reshape(-1, model.vocab_size), targets.reshape(-1),
                ignore_index=model.tokenizer.pad_id)
            mask = targets != model.tokenizer.pad_id
            total_loss += loss.item() * mask.sum().item()
            total_tok += mask.sum().item()
            total_ok += ((lg.argmax(-1) == targets) & mask).sum().item()
    return total_loss / max(total_tok, 1), total_ok / max(total_tok, 1)


def train(args):
    tok, train_ids, val_ids = build_splits(args)
    model = sq.SQLM(tok, dim=args.dim, num_layers=args.layers)
    if args.load is not None:
        model = load_model(args.load, tok)
        print(f"loaded model from {args.load}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable -> falling back to cpu")
        args.device = "cpu"
    model = model.to(args.device)
    print(f"model: D={args.dim} vocab={tok.vocab_size} layers={args.layers} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"data: {sum(len(x) for x in train_ids):,} tokens train, "
          f"{sum(len(x) for x in val_ids):,} tokens valid\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    t_start = time.perf_counter()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Stream training windows on the fly (true streaming, bounded budget).
        windows = pack_windows(iter(train_ids), args.seq, args.budget_tokens)
        ep_loss, ep_tok, epoch_t0 = 0.0, 0, time.perf_counter()
        batch_windows: list[list[int]] = []
        for w in windows:
            batch_windows.append(w)
            if len(batch_windows) == args.batch:
                X = torch.tensor(batch_windows, dtype=torch.long, device=model.device)
                logits = model.train_logits_batch(X)
                targets = torch.tensor(X[:, 1:].tolist(), device=model.device)
                lg = logits[:, :-1, :]
                loss = loss_fn(lg.reshape(-1, model.vocab_size), targets.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item() * targets.numel()
                ep_tok += targets.numel()
                step += 1
                batch_windows = []
                if step % args.log_every == 0:
                    dt = time.perf_counter() - epoch_t0
                    print(f"  step {step:6d}  loss {ep_loss/ep_tok:.3f}  "
                          f"ppl {math.exp(ep_loss/ep_tok):8.2f}  ({ep_tok/max(dt,1e-9):.0f} tok/s)")
        if batch_windows:
            X = torch.tensor(batch_windows, dtype=torch.long, device=model.device)
            logits = model.train_logits_batch(X)
            targets = torch.tensor(X[:, 1:].tolist())
            loss = loss_fn(logits[:, :-1, :].reshape(-1, model.vocab_size),
                           targets.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * targets.numel()
            ep_tok += targets.numel()
        v_loss, v_acc = evaluate(model, val_ids, args.batch)
        print(f"[epoch {epoch}] train_ppl {math.exp(ep_loss/ep_tok):.2f} | "
              f"valid_ppl {math.exp(v_loss):.2f}  valid_acc {v_acc:.1%}  "
              f"({time.perf_counter()-epoch_t0:.0f}s/epoch)")

    print(f"\ntotal {step} steps in {time.perf_counter()-t_start:.0f}s")
    if args.save is not None:
        tok_meta = {"stoi": {k: v for k, v in tok._tok.get_vocab().items()},
                    "vocab_size": tok.vocab_size}
        save_model(model, args.save, tok_meta)
        print(f"saved model -> {args.save}")
    print("\ngeneration samples:")
    for p in args.prompts:
        print(f"  {p!r} ->\n      {model.generate(p, max_tokens=48, temperature=0.5)!r}\n")


def save_model(model, path: str, tok_meta: dict):
    torch.save({
        "dim": model.dim,
        "num_layers": model.num_layers,
        "state": model.state_dict(),
        "tok": tok_meta,
    }, path)


def load_model(path: str, tok: BPETokenizer):
    ckpt = torch.load(path, map_location="cpu")
    m = sq.SQLM(tok, dim=ckpt["dim"], num_layers=ckpt.get("num_layers", 2))
    m.load_state_dict(ckpt["state"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--subset", default="wikitext-2-raw-v1")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--tok-samples", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--budget-tokens", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--prompts", nargs="+", default=[
        "The VSA model is", "Hyperdimensional computing was first proposed by"])
    ap.add_argument("--save", default=None, help="path to save trained model .pt")
    ap.add_argument("--load", default=None, help="path to load a saved model .pt")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()