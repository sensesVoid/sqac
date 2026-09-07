import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Bucket:
    bound: np.ndarray
    code: int
    stored_pos: int
    
class VSAPrimitives:
    """VSA operations using circular convolution (HRR-style)."""
    
    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    
    @staticmethod
    def unbind(ab: np.ndarray, a: np.ndarray) -> np.ndarray:
        a_inv = np.roll(a[::-1].copy(), 1)
        return np.real(np.fft.ifft(np.fft.fft(ab) * np.fft.fft(a_inv)))
    
    @staticmethod
    def bundle(vecs: List[np.ndarray]) -> np.ndarray:
        result = np.sum(vecs, axis=0)
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result = result / norm
        return result
    
    @staticmethod
    def permute(v: np.ndarray, k: int) -> np.ndarray:
        return np.roll(v, k)
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

class KROPCleanup:
    def __init__(self, dim: int):
        assert dim & (dim - 1) == 0
        self.dim = dim
        self.K = int(math.log2(dim))
        self.H, self.thetas = self._build_matrix()
    
    def _build_matrix(self):
        H = np.array([[1.0]], dtype=np.float32)
        thetas = np.linspace(0, 2 * np.pi, self.K + 2)[1:-1]
        for theta in thetas:
            c, s = math.cos(theta), math.sin(theta)
            H = np.block([[c * H, s * H], [s * H, -c * H]])
        return H.astype(np.float32), thetas
    
    def cleanup(self, u: np.ndarray) -> int:
        U = u.copy()
        for k, theta in reversed(list(enumerate(self.thetas))):
            c, s = math.cos(theta), math.sin(theta)
            U = U.reshape(-1, 2 ** k)
            U_top, U_bot = U[::2], U[1::2]
            U[::2] = c * U_top + s * U_bot
            U[1::2] = s * U_top - c * U_bot
        return int(np.argmax(U))
    
    def reconstruct(self, idx: int) -> np.ndarray:
        v = np.array([1.0], dtype=np.float32)
        for k, t in enumerate(self.thetas):
            kth_bit = (idx >> k) & 1
            if kth_bit == 0:
                v = np.concatenate([math.cos(t) * v, math.sin(t) * v])
            else:
                v = np.concatenate([math.sin(t) * v, -math.cos(t) * v])
        return v.astype(np.float32)

class KROPCodebook:
    def __init__(self, num_codes: int, dim: int):
        assert dim & (dim - 1) == 0
        self.num_codes = num_codes
        self.dim = dim
        self.cleanup = KROPCleanup(dim)
        rng = np.random.RandomState(42)
        self.thetas = rng.uniform(0, 2 * np.pi, size=self.cleanup.K).astype(np.float32)
        self.cleanup.thetas = self.thetas
        H = self._build_H()
        self.codebook = H[:num_codes].copy()
        norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
        self.codebook = self.codebook / (norms + 1e-8)
    
    def _build_H(self):
        H = np.array([[1.0]], dtype=np.float32)
        for theta in self.thetas:
            c, s = math.cos(theta), math.sin(theta)
            H = np.block([[c * H, s * H], [s * H, -c * H]])
        return H.astype(np.float32)
    
    def lookup(self, query: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        sims = VSAPrimitives.cosine_matrix(query.reshape(1, -1), self.codebook)[0]
        top_indices = np.argsort(-sims)[:top_k]
        top_sims = sims[top_indices]
        return top_indices, top_sims

class MatryoshkaMoEStage2:
    """Stage 2 VSA with Matryoshka MoE: nested vectors + expert gating.
    
    - D-dimensional vector = E experts of D/E dims each
    - Gate selects which experts activate per query
    - Only active experts perform similarity search
    - Supports hierarchical retrieval: coarse → medium → fine
    """
    
    def __init__(self, num_codes: int, dim: int, num_experts: int = 4, 
                 gate_threshold: float = 0.15, use_krop: bool = True):
        assert dim % num_experts == 0
        self.num_codes = num_codes
        self.dim = dim
        self.num_experts = num_experts
        self.expert_dim = dim // num_experts
        self.gate_threshold = gate_threshold
        
        if use_krop and dim & (dim - 1) == 0:
            self.codebook = KROPCodebook(num_codes, dim)
            self.use_krop = True
        else:
            rng = np.random.RandomState(42)
            self.codebook_vectors = rng.normal(0, 1.0 / math.sqrt(dim), (num_codes, dim)).astype(np.float32)
            norms = np.linalg.norm(self.codebook_vectors, axis=1, keepdims=True)
            self.codebook_vectors = self.codebook_vectors / (norms + 1e-8)
            self.use_krop = False
        
        self.buckets: Dict[int, Bucket] = {}
        self.next_bucket_id = 0
        
        # MoE gate: learns to select experts based on query
        self.gate_logits = nn.Parameter(torch.randn(num_experts) * 0.01)
        self.gate_bias = nn.Parameter(torch.zeros(num_experts))
    
    def _get_code_vector(self, code: int) -> np.ndarray:
        if self.use_krop:
            return self.codebook.codebook[code]
        return self.codebook_vectors[code]
    
    def gate(self, query: np.ndarray) -> np.ndarray:
        """Compute expert activation scores."""
        print(f"[DEBUG] gate called, query type={type(query)}, shape={getattr(query, 'shape', 'N/A')}")
        q = torch.from_numpy(np.array(query, copy=True)).float().view(1, -1)
        with torch.no_grad():
            norm = q.norm()
            logits = self.gate_logits * norm + self.gate_bias
            scores = torch.sigmoid(logits).cpu().numpy()
        print(f"[DEBUG] gate scores={scores}")
        return scores
    
    def add(self, code: int, pos: int, level: int = 0, context: int = 0) -> int:
        """Store code at position using permutation + expert routing."""
        bucket_id = self.next_bucket_id
        self.next_bucket_id += 1
        
        V = self._get_code_vector(code)
        bound = VSAPrimitives.permute(V, pos)
        
        self.buckets[bucket_id] = Bucket(bound=bound, code=code, stored_pos=pos)
        return bucket_id
    
    def query(self, pos: int, level: int = 0, context: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve code with MoE: only active experts search."""
        best_code = -1
        best_sim = -1.0
        
        # Get gate scores for this query position
        X = np.random.RandomState(42).normal(0, 1, self.dim).astype(np.float32)
        X = X / (np.linalg.norm(X) + 1e-8)
        gate_scores = self.gate(X)
        
        for bucket_id, bucket in self.buckets.items():
            # Only search if the expert for this position is active
            expert_idx = pos % self.num_experts
            if gate_scores[expert_idx] < self.gate_threshold:
                continue
            
            unbound = VSAPrimitives.permute(bucket.bound, pos - bucket.stored_pos)
            V = self._get_code_vector(bucket.code)
            sim = VSAPrimitives.similarity(unbound, V)
            if sim > best_sim:
                best_sim = sim
                best_code = bucket.code
        
        return np.array([best_code]), np.array([best_sim])
    
    def exact_retrieval_rate(self, codes: np.ndarray, positions: np.ndarray) -> float:
        correct = 0
        for code, pos in zip(codes, positions):
            retrieved, _ = self.query(pos)
            if retrieved[0] == code:
                correct += 1
        return correct / len(codes)
    
    def get_expert_stats(self) -> dict:
        return {
            "num_buckets": len(self.buckets),
            "dim": self.dim,
            "num_experts": self.num_experts,
            "expert_dim": self.expert_dim,
            "use_krop": self.use_krop
        }

if __name__ == "__main__":
    dim = 256
    num_codes = 512
    stage2 = MatryoshkaMoEStage2(num_codes, dim, num_experts=4, use_krop=True)
    
    for i in range(10):
        stage2.add(code=i % num_codes, pos=i, level=0, context=0)
    
    for i in range(10):
        retrieved, sim = stage2.query(pos=i, level=0, context=0)
        print(f"pos={i:2d}  expected={i % num_codes:3d}  retrieved={retrieved[0]:3d}  sim={sim[0]:.3f}")
    
    print(stage2.get_expert_stats())
