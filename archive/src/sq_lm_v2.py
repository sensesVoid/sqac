#!/usr/bin/env python3
"""
SQ-LM v2 — SynthQuant Vector-Symbolic Language Model

A pure VSA (Vector Symbolic Architecture) language model built on
hyperdimensional computing primitives (bind, bundle, permute) instead of
transformer self-attention.

Design goals:
  - Fast to build (CPU-native, small dimensions)
  - No catastrophic forgetting (additive VSA knowledge store)
  - Unlimited context (constant-size recurrence state)
  - Learnable encoder (THDC-style learned token embeddings)

Architecture:
  Encoder (learnable token -> hypervector + position permute)
    -> Sequence Memory (VSA-gated recurrence, O(1) state)
    -> Associative Knowledge Store (non-forgetting VSA lookup)
    -> Decoder (cleanup network: state -> vocab -> next token)

Usage:
    python3 sq_lm_v2.py                 # run demo
    python3 sq_lm_v2.py --train         # train on small corpus
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd


DEFAULT_D = 1024          # VSA dimensionality (THDC showed 64-2048 works)
DEFAULT_VOCAB = 4096      # vocab size for prototype
CONTEXT_LIMIT = 64        # tokens considered in a bundle (v1)
BIND_DIM = 32             # rolling position-window for long sequences


# ---------------------------------------------------------------------------
# Tokenizer (subword-lite)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Simple character-level tokenizer for the prototype. Yields full control
    over vocab, no external data, works on CPU for any corpus.
    """

    def __init__(self, texts: list[str] | None = None, vocab_size: int = DEFAULT_VOCAB):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.stoi: dict[str, int] = {chr(i): i + 2 for i in range(128)}  # ASCII reserve
        self.itos: dict[int, str] = {v: k for k, v in self.stoi.items()}
        if texts:
            self.build(texts)

    def build(self, texts: list[str]):
        counter: dict[str, int] = {}
        for t in texts:
            for ch in t:
                counter[ch] = counter.get(ch, 0) + 1
        # Extend vocab from most common chars beyond ASCII
        for ch, _ in sorted(counter.items(), key=lambda kv: -kv[1]):
            if ch not in self.stoi and len(self.stoi) < self.vocab_size:
                self.stoi[ch] = len(self.stoi)
                self.itos[len(self.stoi) - 1] = ch
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)


# ---------------------------------------------------------------------------
# SQ-LM v2 model
# ---------------------------------------------------------------------------

@dataclass
class SQLMResult:
    mode: str                 # 'vsa' | 'knowledge' | 'hybrid'
    token_logits: torch.Tensor | None = None
    next_token: int | None = None
    state_hv: torch.Tensor | None = None
    knowledge_hit: bool = False
    knowledge_conf: float = 0.0
    latency_ms: float = 0.0


class Encoder(nn.Module):
    """Learned token -> hypervector mapping (THDC-style), plus VSA position
    permutation so the state is order-sensitive.
    """

    def __init__(self, vocab_size: int, dim: int, position_window: int = BIND_DIM):
        super().__init__()
        self.dim = dim
        self.position_window = position_window
        # Learnable embeddings. In BSC the forward maps to {0,1}; we keep a
        # real-valued embedding, binarize for VSA storage, but use soft features
        # for training gradients via a straight-through estimator.
        self.embed = nn.Embedding(vocab_size, dim)
        # Position hypervectors (fixed random BSC) for order sensitivity.
        # Registered as a buffer so .to(device) moves them with the module.
        self.register_buffer("pos", torchhd.random(position_window, dimensions=dim, vsa="BSC"))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map a (batch, seq) tensor of token ids to (batch, seq, dim) real
        hypervector features, unit-normalized for stable accumulation.
        """
        emb = self.embed(token_ids)                # (B, S, D)
        return F.normalize(emb, dim=-1)

    def embed_token(self, token_id: int) -> torch.Tensor:
        return self.forward(torch.tensor([[token_id]]))[0, 0]

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    def position_hv(self, idx: int) -> torch.Tensor:
        return self.pos[idx % self.position_window]


class SequenceMemory(nn.Module):
    """VSA-gated recurrence over hypervectors. Collapses a whole sequence into
    one bounded state hypervector (O(1) memory = unlimited context).

    The recurrence uses binding to write (associate state with token) and
    bundling + gates to update, approximating Gated-DeltaNet's erase-then-write
    in VSA terms. Real-valued gates let this train via backprop while the
    forward path stays binary.
    """

    def __init__(self, dim: int, input_dim: int | None = None):
        super().__init__()
        self.dim = dim
        f_dim = input_dim or dim
        # Gate networks map [state; token] -> gate scalar (keep/write)
        self.gate_g = nn.Linear(dim + f_dim, 1)
        self.gate_i = nn.Linear(dim + f_dim, 1)

    def step(self, state: torch.Tensor, token_hv: torch.Tensor) -> torch.Tensor:
        """One recurrence step. state: (B, D), token_hv: (B, D).

        Gated accumulation (VSA-flavored, like simplified gated linear
        attention): gates decide how much old state to keep and how much new
        token-feature to write. No per-step normalization keeps the backward
        graph cheap (norm/div over unrolled steps is the dominant CPU cost).
        """
        if token_hv.dim() == 1:
            token_hv = token_hv.unsqueeze(0)
        s = state if state.dim() == 2 else state.unsqueeze(0)
        x = torch.cat([s, token_hv], dim=-1)
        g = torch.sigmoid(self.gate_g(x))   # keep-proportion of old state
        i = torch.sigmoid(self.gate_i(x))   # write-proportion of token features

        # Erase-then-write (delta-rule-like): forget g, write i, keep the rest.
        new_state = (1 - g) * token_hv + (1 - i) * s
        return new_state

    def reset(self):
        pass


class KnowledgeStore(nn.Module):
    """Associative VSA knowledge store. Non-forgetting: adding a fact is just
    appending a hypervector. Lookup is Hamming-based (SIMD-able).

    keys:     (K, D) binary hypervectors (question/key)
    values:   (K, D) binary hypervectors (answer/assoc content)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("keys", torch.zeros(0, dim))
        self.register_buffer("values", torch.zeros(0, dim))
        self.metadata: list[dict] = []

    @property
    def device(self) -> torch.device:
        return self.keys.device

    @torch.no_grad()
    def add(self, key_hv: torch.Tensor, value_hv: torch.Tensor, meta: dict | None = None):
        k = key_hv.detach().view(1, -1).float().to(self.device)
        v = value_hv.detach().view(1, -1).float().to(self.device)
        self.keys = torch.cat([self.keys.detach(), k]).to(self.device)
        self.values = torch.cat([self.values.detach(), v]).to(self.device)
        self.metadata.append(meta or {})

    @torch.no_grad()
    def lookup(self, query_hv: torch.Tensor, top_k: int = 1):
        """Return (indices, cosine_scores). Higher = more similar."""
        if self.keys.shape[0] == 0:
            return torch.empty(0, dtype=torch.long), None
        q = query_hv.detach().float().view(1, -1)
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(self.keys, dim=-1)
        sim = (qn @ kn.t()).view(-1)                    # (K,)
        top = torch.topk(sim, min(top_k, len(sim))).indices
        return top, sim

    def knowledge(self, idx: torch.Tensor) -> torch.Tensor:
        return self.values[idx].view(1, -1)


class Decoder(nn.Module):
    """Cleanup network: state hypervector -> next-token logits.

    Two modes:
      A (pure VSA): cosine of state against all embedding rows -> logits
      B (learned):  small linear head
    We use A, with the learned embeddings as the cleanup memory (nearest vocab).
    A batched MLP transforms states before the cosine so the model gets the
    nonlinearity it needs for next-token prediction without unrolling cost.
    """

    def __init__(self, dim: int, vocab_size: int, encoder: Encoder,
                 hidden: int | None = None):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.encoder = encoder
        hidden = hidden or dim * 4
        self.project = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def mode_a_logits(self, states: torch.Tensor) -> torch.Tensor:
        """Cosine similarity against the learned embedding rows (nearest-vocab
        cleanup memory). states: (S, D) or (1, D) -> (S, V). The projection
        MLP is applied in one batched matmul over the sequence (no per-step
        unroll), then compared against the embedding rows.
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        feat = self.project(states)                              # (S, D)
        emb = F.normalize(self.encoder.embed.weight, dim=-1)     # (V, D)
        st = F.normalize(feat, dim=-1)                           # (S, D)
        logits = (st @ emb.t()) * 8.0                            # (S, V)
        return logits

    def forward(self, state_hv: torch.Tensor,
                knowledge_hv: torch.Tensor | None = None,
                alpha: float = 0.0) -> torch.Tensor:
        if knowledge_hv is not None:
            state_hv = (1 - alpha) * state_hv + alpha * knowledge_hv
        logits = self.mode_a_logits(state_hv)
        return logits


class SQLM(nn.Module):
    """Full VSA language model."""

    def __init__(self, tokenizer: CharTokenizer, dim: int = DEFAULT_D,
                 knowledge: bool = True, num_layers: int = 2):
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        self.vocab_size = tokenizer.vocab_size
        self.num_layers = num_layers

        self.encoder = Encoder(self.vocab_size, dim)
        self.layers = nn.ModuleList([
            SequenceMemory(dim, input_dim=dim) for _ in range(num_layers)
        ])
        self.decoder = Decoder(dim, self.vocab_size, self.encoder)
        self.knowledge = KnowledgeStore(dim) if knowledge else None

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    # -- helpers -------------------------------------------------------------

    def _embed_ids(self, ids: list[int]) -> torch.Tensor:
        """Encode a sequence with position permutation, return (S, D). Real."""
        hvs = [self.encoder.embed_token(tid) for tid in ids]
        out = []
        for t, h in enumerate(hvs):
            out.append(self._bind_position(h, t))
        return torch.stack(out)

    def encode_query(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        ids = ids[:CONTEXT_LIMIT]
        hvs = self._embed_ids(ids)
        # bundle all token hvs -> single query hv (superposition)
        hv = torch.zeros(self.dim, device=self.device)
        for h in hvs:
            hv = hv + h
        return hv

    def _bind_position(self, thv: torch.Tensor, pos: int) -> torch.Tensor:
        """Bind a token hv with positional phase (bipolar multiply)."""
        p = self.encoder.position_hv(pos).to(thv.device)
        p_bipolar = p * 2 - 1.0                     # {0,1} -> {-1,+1}
        return thv * p_bipolar

    # -- forward -------------------------------------------------------------

    def _run_states(self, ids_seq: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """N-layer VSA recurrence over the sequence.

        Returns (states, final_state) where states is (S, D) (the state after
        each position, last layer) and final_state is (1, D) at the last position.
        """
        states = [torch.zeros(1, self.dim, device=self.device) for _ in range(self.num_layers)]
        states_out = []
        for i, tid in enumerate(ids_seq):
            thv = self.encoder.embed_token(tid).view(1, -1)
            thv = self._bind_position(thv, i)
            layer_input = thv
            for j, layer in enumerate(self.layers):
                states[j] = layer.step(states[j], layer_input)
                layer_input = states[j]
            states_out.append(states[-1].squeeze(0))
        if not states_out:
            return torch.zeros(0, self.dim, device=self.device), torch.zeros(1, self.dim, device=self.device)
        return torch.stack(states_out), states[-1]

    def train_logits(self, ids_seq: list[int]) -> torch.Tensor:
        """One pass over a sequence, predicting next token at every position.

        Returns (S, V) logits where logits[t] predicts ids[t+1]. Runs the
        recurrence once over the sequence, then the decoder once as a batched
        matmul over all positions, so teacher forcing is O(S*D*V) total with
        minimal graph overhead.
        """
        states, _ = self._run_states(ids_seq)
        return self.decoder.mode_a_logits(states)

    def train_logits_batch(self, ids_batch: torch.Tensor) -> torch.Tensor:
        """(B, S) token ids -> (B, S, V) next-token logits, all rows of the
        batch run through the recurrence in parallel (same positions). Same
        objective as train_logits, ~Bx fewer Python loop iterations.
        """
        B, S = ids_batch.shape
        states = [torch.zeros(B, self.dim, device=self.device) for _ in range(self.num_layers)]
        emb = self.encoder.forward(ids_batch)          # (B, S, D), normalized
        states_out = []
        for i in range(S):
            thv = emb[:, i, :]                          # (B, D)
            p = self.encoder.position_hv(i)
            thv = thv * (p * 2 - 1.0)                   # position bind
            layer_input = thv
            for j, layer in enumerate(self.layers):
                states[j] = layer.step(states[j], layer_input)
                layer_input = states[j]
            states_out.append(states[-1])               # (B, D)
        states = torch.stack(states_out, dim=1)         # (B, S, D)
        flat = states.reshape(B * S, self.dim)
        logits = self.decoder.mode_a_logits(flat)       # (B*S, V)
        return logits.reshape(B, S, -1)

    def forward(self, ids_seq: list[int]) -> SQLMResult:
        """Process the full token sequence, predict the next token after it."""
        start = time.perf_counter()
        states, state2 = self._run_states(ids_seq)

        # knowledge lookup on the query (the user-facing text or last context)
        knowledge_hv = None
        kconf = 0.0
        if self.knowledge is not None:
            query_text = self.tokenizer.decode(ids_seq) if ids_seq else ""
            qhv = self.encode_query(query_text)
            idx, sims = self.knowledge.lookup(qhv)
            if idx.shape[0] > 0 and sims is not None:
                kconf = float(sims[idx[0]].item())       # cosine similarity, [-1, 1]
                if kconf > 0.6:
                    knowledge_hv = self.knowledge.values[idx[0]].view(1, -1).float()
                    self.knowledge._last_idx = idx[0]

        alpha = 0.6 if knowledge_hv is not None else 0.0
        logits = self.decoder(state2, knowledge_hv, alpha)

        result = SQLMResult(
            mode="knowledge" if knowledge_hv is not None else "vsa",
            token_logits=logits,
            state_hv=state2,
            knowledge_hit=knowledge_hv is not None,
            knowledge_conf=kconf,
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def knowledge_retrieve(self, ids_seq: list[int],
                           threshold: float = 0.6) -> tuple[str | None, float]:
        """Retrieve a stored fact value if the context matches a key.

        Uses the same query encoding (position-bound bundle) that keys were
        stored with, so an exact key match gives confidence ~1.0. Returns
        (value_text, confidence) or (None, conf).
        """
        if self.knowledge is None or not ids_seq:
            return None, 0.0
        query_text = self.tokenizer.decode(ids_seq)
        qhv = self.encode_query(query_text)
        idx, sims = self.knowledge.lookup(qhv)
        if idx.shape[0] == 0 or sims is None:
            return None, 0.0
        conf = float(sims[idx[0]].item())
        if conf > threshold:
            meta = self.knowledge.metadata[idx[0]]
            return meta.get("value"), conf
        return None, conf

    def next_token(self, ids_seq: list[int], temperature: float = 1.0) -> int:
        r = self.forward(ids_seq)
        logits = r.token_logits[0] / max(temperature, 1e-6)
        return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())

    def binarized(self):
        """Return a copy with hypervectors compressed to {+1,-1} (BSC).

        Binarizes the token embeddings and knowledge-store key/value vectors,
        which are the dominant storage cost. Gates/recurrence weights stay
        real (they're cheap and need precision). This mirrors the real edge
        path: HV tables are bit-packed, so D floats -> D bits (32x smaller).
        The model still normalizes in the decoder, so sign-embeddings work.
        """
        import copy
        m = copy.deepcopy(self).cpu().eval()
        with torch.no_grad():
            w = m.encoder.embed.weight
            m.encoder.embed.weight.copy_(torch.where(w > 0, 1.0, -1.0))
            if m.knowledge is not None:
                for name in ("keys", "values"):
                    buf = getattr(m.knowledge, name)
                    setattr(m.knowledge, name,
                            torch.where(buf > 0, 1.0, -1.0).float())
        return m

    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 1.0) -> str:
        ids = self.tokenizer.encode(prompt)
        for _ in range(max_tokens):
            value, conf = self.knowledge_retrieve(ids)
            if value is not None:
                ids += self.tokenizer.encode(value)
                break
            nxt = self.next_token(ids, temperature)
            ids.append(nxt)
        return self.tokenizer.decode(ids)

    def compress_corpus(self, texts: list[str], threshold: float = 0.85) -> list[str]:
        """Remove near-duplicate texts using VSA semantic similarity.

        Encodes each text as a position-bound bundle hypervector, then
        greedily removes texts whose cosine similarity to any kept text
        exceeds ``threshold``. Returns the deduplicated list preserving
        original order.

        This applies the SQA compression research (semantic dedup) to
        training data: fewer examples, less overfitting, faster training.
        """
        if len(texts) <= 1:
            return list(texts)
        hvs = [self.encode_query(t) for t in texts]
        kept = []
        kept_hvs = []
        for i, hv in enumerate(hvs):
            if not kept_hvs:
                kept.append(texts[i])
                kept_hvs.append(hv)
                continue
            sims = torch.stack([F.cosine_similarity(hv, kh, dim=0) for kh in kept_hvs])
            if sims.max().item() < threshold:
                kept.append(texts[i])
                kept_hvs.append(hv)
        return kept


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

DEMO_CORPUS = [
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
    "the sun sets in the west",
    "the moon shines at night",
    "stars twinkle in the dark sky",
    "rain falls from the clouds",
    "the river runs to the sea",
    "memory grows with each new fact",
    "adding knowledge never erases old knowledge",
    "the hypervector holds the whole context",
    "permutation tracks the order of tokens",
    "the decoder finds the nearest word",
]

DEMO_FACTS = [
    ("the capital of france is", "paris"),
    ("the largest planet is", "jupiter"),
    ("water freezes at", "zero degrees"),
    ("the speed of light is", "fast"),
    ("rust uses the borrow checker to", "prevent data races"),
]


def build_demo() -> tuple[SQLM, CharTokenizer]:
    tokenizer = CharTokenizer(DEMO_CORPUS + [v for _, v in DEMO_FACTS], vocab_size=256)
    model = SQLM(tokenizer, dim=512)
    return model, tokenizer


def train_demo(model: SQLM, epochs: int = 60, lr: float = 1e-2):
    """Full teacher-forcing next-token training on the demo corpus.

    Every position in every sequence contributes a next-token prediction
    (standard LM objective), computed in O(S) per sequence via train_logits.
    """
    obs = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    texts = DEMO_CORPUS      # facts live ONLY in the knowledge store, not the corpus
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for text in texts:
            ids = model.tokenizer.encode(text)
            if len(ids) < 3:
                continue
            logits = model.train_logits(ids)             # (S, V): logits[i] predicts ids[i+1]
            logits = logits[:-1]                          # drop prediction past sequence end
            targets = torch.tensor(ids[1:])               # next tokens
            loss = loss_fn(logits, targets)
            obs.zero_grad()
            loss.backward()
            obs.step()
            total_loss += loss.item()
            n += 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  loss {total_loss/max(n,1):.4f}")
    # Add knowledge facts non-destructively after training
    for k, v in DEMO_FACTS:
        khv = model.encode_query(k)
        vhv = model.encode_query(v)
        model.knowledge.add(khv, vhv, {"key": k, "value": v})


def run_demo(train: bool = True):
    print("=" * 80)
    print("SQ-LM v2 — SynthQuant Vector-Symbolic Language Model")
    print("=" * 80)
    print("\n  Reading order: bind -> bundle -> permute -> cleanup\n")

    model, tokenizer = build_demo()

    if train:
        print("  TRAINING on demo corpus (CPU, ~seconds)\n")
        train_demo(model)

    # Generation probe
    model.eval()
    prompts = [
        "the cat",
        "the sun",
        "water",
        "memory",
        "the capital of france",
        "rust uses the borrow checker",
    ]
    print("\n  GENERATION (greedy, cold)\n" + "-" * 60)
    for p in prompts:
        out = model.generate(p, max_tokens=24, temperature=0.4)
        print(f"  '{p}' ->")
        print(f"      {out}")
        print()

    # Knowledge recall test (non-forgetting)
    if model.knowledge is not None:
        print("\n  KNOWLEDGE RECALL (non-forgetting)\n" + "-" * 60)
        for k, v in DEMO_FACTS:
            r = model.forward(model.tokenizer.encode(k))
            hit = "HIT " if r.knowledge_hit else "miss"
            print(f"  [{hit}] '{k}' -> conf {r.knowledge_conf:.3f}  (stored: '{v}')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train on demo corpus")
    ap.add_argument("--no-train", action="store_true", help="skip training")
    args = ap.parse_args()
    run_demo(train=not args.no_train)


if __name__ == "__main__":
    main()
