#!/usr/bin/env python3
"""
Unified Model: SQA + Matryoshka MoE + FissFus + TurboVec

Integrates:
- FissFus Stage 1: VQ-VAE compressor with PQ subspaces
- FissFus Stage 2: VSA structural memory with permutation addressing
- Matryoshka MoE: Nested hypervectors with expert gating
- TurboVec: Fast approximate nearest neighbor search (Google TurboQuant)
- SQA: Learned token embeddings + cleanup decoder

Usage:
    python unified_model.py --demo
    python unified_model.py --benchmark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, '/teamspace/studios/this_studio/Project_ SynthQuant/src')
from fissfus_stage1_improved import VQVAE, text_to_chunks, chunks_to_text, reconstruction_error, codebook_utilization
from fissfus_stage2 import FissFusStage2
from fissfus_matryoshka import MatryoshkaMoEStage2, VSAPrimitives

CHARS = string.printable
VOCAB_SIZE = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

class TurboVecRetriever:
    """TurboVec-accelerated codebook lookup.
    
    Uses Google's TurboQuant for 2-4 bit vector compression
    and SIMD-optimized nearest neighbor search.
    """
    
    def __init__(self, num_codes: int, dim: int, bit_width: int = 2):
        self.num_codes = num_codes
        self.dim = dim
        self.bit_width = bit_width
        self.index = None
        self.codebook_vectors = None
        
    def build_index(self, codebook_vectors: np.ndarray):
        """Build TurboVec index from codebook vectors."""
        self.codebook_vectors = codebook_vectors.copy()
        
        try:
            from turbovec import TurboQuantIndex
            self.index = TurboQuantIndex(self.dim, bit_width=self.bit_width)
            self.index.add(codebook_vectors)
            self.use_turbovec = True
        except Exception as e:
            print(f"TurboVec not available, falling back to numpy: {e}")
            self.use_turbovec = False
    
    def search(self, query: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest codes using TurboVec or fallback."""
        if self.use_turbovec and self.index is not None:
            scores, indices = self.index.search(query.reshape(1, -1), k=top_k)
            return indices[0], scores[0]
        else:
            # Fallback: numpy cosine similarity
            q = query.reshape(1, -1)
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            k_norm = self.codebook_vectors / (np.linalg.norm(self.codebook_vectors, axis=1, keepdims=True) + 1e-8)
            sims = (q_norm @ k_norm.T)[0]
            top_indices = np.argsort(-sims)[:top_k]
            return top_indices, sims[top_indices]

class UnifiedModel(nn.Module):
    """Unified SQA + Matryoshka MoE + FissFus + TurboVec model."""
    
    def __init__(self, chunk_size: int = 8, latent_dim: int = 128, 
                 codebook_size: int = 256, num_subspaces: int = 4,
                 vsa_dim: int = 1024, num_experts: int = 4,
                 gate_threshold: float = 0.15, turbovec_bits: int = 2):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_subspaces = num_subspaces
        self.vsa_dim = vsa_dim
        self.num_experts = num_experts
        self.expert_dim = vsa_dim // num_experts
        self.turbovec_bits = turbovec_bits
        
        # Stage 1: FissFus VQ-VAE compressor
        self.stage1 = VQVAE(chunk_size, latent_dim, codebook_size, num_subspaces)
        
        # Stage 2: FissFus VSA structural memory
        self.stage2 = FissFusStage2(codebook_size, vsa_dim, use_krop=True)
        
        # Matryoshka MoE: nested retrieval
        self.moe_store = MatryoshkaMoEStage2(num_codes=codebook_size, dim=vsa_dim, num_experts=num_experts, gate_threshold=gate_threshold)
        
        # TurboVec: fast codebook search
        self.turbovec = TurboVecRetriever(codebook_size, vsa_dim, bit_width=turbovec_bits)
        
        # SQA-style learned token embeddings for decoding
        self.token_embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, VOCAB_SIZE)
        )
    
    def encode(self, text: str) -> Tuple[np.ndarray, List[int]]:
        """Encode text into codes and bucket IDs."""
        chunks = text_to_chunks(text, self.chunk_size)
        if len(chunks) == 0:
            raise ValueError("Empty text after chunking")
        
        dataset = torch.tensor(np.stack(chunks), dtype=torch.float32)
        
        codes = []
        bucket_ids = []
        for i, chunk in enumerate(dataset):
            code_vec = self.stage1.encode(chunk.unsqueeze(0))
            code = int(code_vec[0, 0]) if code_vec.ndim > 1 else int(code_vec[0])
            bucket_id = self.stage2.add(code, pos=i)
            codes.append(code)
            bucket_ids.append(bucket_id)
        
        return np.array(codes), bucket_ids
    
    def retrieve(self, query_text: str, bucket_ids: List[int], 
                 top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve codes using TurboVec + Matryoshka MoE."""
        chunks = text_to_chunks(query_text, self.chunk_size)
        if len(chunks) == 0:
            raise ValueError("Empty query after chunking")
        
        dataset = torch.tensor(np.stack(chunks), dtype=torch.float32)
        
        retrieved_codes = []
        similarities = []
        for i, chunk in enumerate(dataset):
            code_vec = self.stage1.encode(chunk.unsqueeze(0))
            code = int(code_vec[0, 0]) if code_vec.ndim > 1 else int(code_vec[0])
            
            # Get candidate code vector
            V = self.stage2._get_code_vector(code)
            
            # TurboVec fast search
            turbo_codes, turbo_sims = self.turbovec.search(V, top_k=top_k)
            turbo_code = int(turbo_codes[0])
            
            # Matryoshka MoE gating
            gate_scores = self.moe_store.gate(V)
            expert_idx = i % self.num_experts
            if gate_scores[expert_idx] > self.moe_store.gate_threshold:
                retrieved_codes.append(turbo_code)
                similarities.append(float(turbo_sims[0]))
            else:
                retrieved_codes.append(code)
                similarities.append(0.0)
        
        return np.array(retrieved_codes), np.array(similarities)
    
    def decode(self, codes: np.ndarray) -> str:
        """Decode codes back to text using Stage 1 decoder."""
        recon_chunks = self.stage1.decode_codes(codes)
        recon_chunks = recon_chunks.reshape(-1, self.chunk_size)
        return chunks_to_text(recon_chunks)
    
    def forward(self, text: str) -> Tuple[str, dict]:
        """Full encode → retrieve → decode pipeline."""
        codes, bucket_ids = self.encode(text)
        retrieved, sims = self.retrieve(text, bucket_ids)
        reconstructed = self.decode(retrieved)
        
        stats = {
            "num_chunks": len(codes),
            "unique_codes": len(np.unique(codes)),
            "avg_similarity": float(np.mean(sims)),
            "compression_ratio": len(text.encode('utf-8')) / max(len(codes) * 4, 1),
            "turbovec_enabled": self.turbovec.use_turbovec
        }
        
        return reconstructed, stats
    
    def build_turbovec_index(self):
        """Build TurboVec index from current codebook."""
        if self.stage2.use_krop:
            vectors = self.stage2.codebook.codebook
            if hasattr(vectors, 'cpu'):
                vectors = vectors.cpu().numpy()
        else:
            vectors = self.stage2.codebook_vectors
        
        self.turbovec.build_index(vectors)
        print(f"TurboVec index built: {vectors.shape[0]} codes, {vectors.shape[1]} dims, {self.turbovec.bit_width}-bit")

def demo():
    """Run unified model demo."""
    print("=" * 80)
    print("Unified Model: SQA + Matryoshka MoE + FissFus + TurboVec")
    print("=" * 80)
    
    model = UnifiedModel(chunk_size=8, latent_dim=128, codebook_size=256,
                        num_subspaces=4, vsa_dim=1024, num_experts=4,
                        turbovec_bits=2)
    model.eval()
    
    text = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    Sphinx of black quartz, judge my vow.
    """ * 10
    
    print(f"\nInput: {len(text)} characters")
    
    # Build TurboVec index
    model.build_turbovec_index()
    
    reconstructed, stats = model(text)
    
    print(f"\nReconstruction: {stats['num_chunks']} chunks")
    print(f"Unique codes: {stats['unique_codes']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Avg similarity: {stats['avg_similarity']:.3f}")
    print(f"TurboVec enabled: {stats['turbovec_enabled']}")
    print(f"\nSample output: {reconstructed[:200]}")

def benchmark():
    """Benchmark unified model."""
    print("=" * 80)
    print("Unified Model Benchmark")
    print("=" * 80)
    
    configs = [
        {"chunk": 8, "latent": 128, "codebook": 256, "vsa": 1024, "experts": 4, "name": "Small"},
        {"chunk": 8, "latent": 256, "codebook": 512, "vsa": 2048, "experts": 8, "name": "Medium"},
        {"chunk": 16, "latent": 512, "codebook": 1024, "vsa": 4096, "experts": 16, "name": "Large"},
    ]
    
    text = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    Sphinx of black quartz, judge my vow.
    """ * 20
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        model = UnifiedModel(
            chunk_size=config['chunk'],
            latent_dim=config['latent'],
            codebook_size=config['codebook'],
            num_subspaces=4,
            vsa_dim=config['vsa'],
            num_experts=config['experts'],
            turbovec_bits=2
        )
        model.eval()
        model.build_turbovec_index()
        
        reconstructed, stats = model(text)
        print(f"  Chunks: {stats['num_chunks']}")
        print(f"  Unique codes: {stats['unique_codes']}")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")
        print(f"  Similarity: {stats['avg_similarity']:.3f}")
        print(f"  TurboVec: {stats['turbovec_enabled']}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    
    if args.demo:
        demo()
    elif args.benchmark:
        benchmark()
    else:
        demo()
