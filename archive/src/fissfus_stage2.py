import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
        """Cyclic shift permutation."""
        return np.roll(v, k)
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def cosine_matrix(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        k_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
        return q_norm @ k_norm.T

class KROPCleanup:
    """Linearithmic cleanup using Kronecker rotation products."""
    
    def __init__(self, dim: int):
        assert dim & (dim - 1) == 0, "dim must be power of 2"
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
        thetas = self.thetas
        for k, theta in reversed(list(enumerate(thetas))):
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
    """KROP-structured codebook for fast cleanup."""
    
    def __init__(self, num_codes: int, dim: int):
        assert dim & (dim - 1) == 0, "dim must be power of 2"
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

class FissFusStage2:
    """Stage 2 VSA structural layer with permutation-based addressing.
    
    Uses standard VSA sequence encoding:
    - Store: bucket_vec = bundle([π^0(V_0), π^1(V_1), ...])
    - Query: query_vec = π^{-pos}(bucket_vec) → cleanup → code
    """
    
    def __init__(self, num_codes: int, dim: int, max_bucket_size: int = 64, use_krop: bool = True):
        self.num_codes = num_codes
        self.dim = dim
        self.max_bucket_size = max_bucket_size
        
        if use_krop and dim & (dim - 1) == 0:
            self.codebook = KROPCodebook(num_codes, dim)
            self.use_krop = True
        else:
            rng = np.random.RandomState(42)
            self.codebook_vectors = rng.normal(0, 1.0 / math.sqrt(dim), (num_codes, dim)).astype(np.float32)
            norms = np.linalg.norm(self.codebook_vectors, axis=1, keepdims=True)
            self.codebook_vectors = self.codebook_vectors / (norms + 1e-8)
            self.use_krop = False
        
        self.buckets: Dict[int, List[np.ndarray]] = {}
        self.next_bucket_id = 0
    
    def _get_code_vector(self, code: int) -> np.ndarray:
        if self.use_krop:
            return self.codebook.codebook[code]
        return self.codebook_vectors[code]
    
    def create_bucket(self) -> int:
        bucket_id = self.next_bucket_id
        self.next_bucket_id += 1
        self.buckets[bucket_id] = []
        return bucket_id
    
    def add(self, code: int, pos: int, level: int = 0, context: int = 0) -> int:
        """Store code at position. Creates a new bucket automatically."""
        bucket_id = self.create_bucket()
        self.add_to_bucket(bucket_id, code, pos, level, context)
        return bucket_id
    
    def add_to_bucket(self, bucket_id: int, code: int, pos: int, level: int = 0, context: int = 0):
        """Add code to existing bucket at position."""
        V = self._get_code_vector(code)
        permuted = VSAPrimitives.permute(V, pos)
        self.buckets[bucket_id].append(permuted)
    
    def query_bucket(self, bucket_id: int, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query bucket for item at position."""
        if bucket_id not in self.buckets or not self.buckets[bucket_id]:
            return np.array([-1]), np.array([-1.0])
        
        bucket_vec = VSAPrimitives.bundle(self.buckets[bucket_id])
        query_vec = VSAPrimitives.permute(bucket_vec, -pos)
        
        retrieved_code = self.codebook.lookup(query_vec)[0][0]
        V = self._get_code_vector(retrieved_code)
        sim = VSAPrimitives.similarity(query_vec, V)
        
        return np.array([retrieved_code]), np.array([sim])
    
    def exact_retrieval_rate(self, codes: np.ndarray, positions: np.ndarray, bucket_ids: np.ndarray) -> float:
        correct = 0
        for code, pos, bid in zip(codes, positions, bucket_ids):
            retrieved, _ = self.query_bucket(bid, pos)
            if retrieved[0] == code:
                correct += 1
        return correct / len(codes)
    
    def capacity_check(self) -> dict:
        return {
            "num_buckets": len(self.buckets),
            "max_bucket_size": self.max_bucket_size,
            "dim": self.dim,
            "num_codes": self.num_codes,
            "use_krop": self.use_krop
        }

if __name__ == "__main__":
    dim = 256
    num_codes = 512
    stage2 = FissFusStage2(num_codes, dim, use_krop=True)
    
    bid = stage2.create_bucket()
    for i in range(5):
        stage2.add_to_bucket(bid, code=i, pos=i)
    
    for i in range(5):
        retrieved, sim = stage2.query_bucket(bid, pos=i)
        print(f"pos={i} expected={i} retrieved={retrieved[0]} sim={sim[0]:.3f}")
    
    print(stage2.capacity_check())
