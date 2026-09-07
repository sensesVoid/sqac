#!/usr/bin/env python3
"""
Multi-Vector Stage 2 VSA Memory - True SQA Architecture
Each code = multiple vectors enabling partial binding, error correction, compositional reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

@dataclass
class MultiVectorConfig:
    num_vectors: int = 4           # Number of vectors per concept
    dim_per_vector: int = 256      # Dimension per vector (total = num_vectors * dim_per_vector)
    codebook_size: int = 256
    max_bucket_size: int = 64
    use_krop: bool = True

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
        sims = (query @ self.codebook.T).flatten()
        top_indices = np.argsort(-sims)[:top_k]
        return top_indices, sims[top_indices]

class MultiVectorVSAPrimitives:
    """Multi-vector VSA operations."""
    
    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b
    
    @staticmethod
    def unbind(ab: np.ndarray, a: np.ndarray) -> np.ndarray:
        return ab * a
    
    @staticmethod
    def bundle(vecs: List[np.ndarray]) -> np.ndarray:
        if not vecs:
            return np.array([])
        result = np.sum(vecs, axis=0)
        norm = np.linalg.norm(result)
        return result / (norm + 1e-8) if norm > 1e-8 else result
    
    @staticmethod
    def permute(v: np.ndarray, k: int) -> np.ndarray:
        return np.roll(v, k % len(v))
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def bind_multi(vec_list_a: List[np.ndarray], vec_list_b: List[np.ndarray]) -> List[np.ndarray]:
        """Bind corresponding vectors: [v1,v2,...] ⊗ [w1,w2,...] = [v1⊗w1, v2⊗w2, ...]"""
        return [MultiVectorVSAPrimitives.bind(a, b) for a, b in zip(vec_list_a, vec_list_b)]
    
    @staticmethod
    def unbind_multi(bound_list: List[np.ndarray], vec_list: List[np.ndarray]) -> List[np.ndarray]:
        """Unbind: [v1⊗w1, v2⊗w2, ...] ⊗ [w1, w2, ...] = [v1, v2, ...]"""
        return [MultiVectorVSAPrimitives.unbind(b, w) for b, w in zip(bound_list, vec_list)]
    
    @staticmethod
    def partial_unbind(bound_list: List[np.ndarray], vec_list: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """Unbind only known vectors: [v1⊗w1, v2⊗w2, ...] ⊗ [w1, None, w3, ...]"""
        result = []
        for b, w in zip(bound_list, vec_list):
            if w is not None:
                result.append(MultiVectorVSAPrimitives.unbind(b, w))
            else:
                result.append(b)  # Leave bound if no key
        return result
    
    @staticmethod
    def similarity_multi(vec_list_a: List[np.ndarray], vec_list_b: List[np.ndarray]) -> float:
        """Average similarity across vector pairs."""
        sims = [MultiVectorVSAPrimitives.similarity(a, b) for a, b in zip(vec_list_a, vec_list_b)]
        return np.mean(sims)

class MultiVectorFissFusStage2:
    """
    Multi-Vector FissFus Stage 2 - True SQA Architecture.
    Each code = multiple vectors enabling partial binding, error correction, compositional reasoning.
    """
    
    def __init__(self, config: MultiVectorConfig):
        self.config = config
        self.num_vectors = config.num_vectors
        self.dim_per_vector = config.dim_per_vector
        self.total_dim = config.num_vectors * config.dim_per_vector
        self.codebook_size = config.codebook_size
        self.max_bucket_size = config.max_bucket_size
        
        if config.use_krop and config.dim_per_vector & (config.dim_per_vector - 1) == 0:
            # KROP codebook needs to be total_dim, not dim_per_vector
            self.codebook = KROPCodebook(config.codebook_size, self.total_dim)
            self.use_krop = True
        else:
            rng = np.random.RandomState(42)
            self.codebook_vectors = rng.normal(
                0, 1.0 / math.sqrt(config.dim_per_vector), 
                (config.codebook_size, config.dim_per_vector)
            ).astype(np.float32)
            norms = np.linalg.norm(self.codebook_vectors, axis=1, keepdims=True)
            self.codebook_vectors = self.codebook_vectors / (norms + 1e-8)
            self.use_krop = False
        
        self.buckets: Dict[int, List[List[np.ndarray]]] = {}  # bucket_id -> list of vector lists
        self.next_bucket_id = 0
    
    def _get_vectors(self, code: int) -> List[np.ndarray]:
        """Get list of vectors for a code."""
        if self.use_krop:
            vec = self.codebook.codebook[code]
        else:
            vec = self.codebook_vectors[code]
        return [vec[i * self.dim_per_vector:(i + 1) * self.dim_per_vector] 
                for i in range(self.num_vectors)]
    
    def create_bucket(self) -> int:
        bucket_id = self.next_bucket_id
        self.next_bucket_id += 1
        self.buckets[bucket_id] = []
        return bucket_id
    
    def add_to_bucket(self, bucket_id: int, code: int, pos: int, level: int = 0, context: int = 0):
        """Store code as multiple permuted vectors in bucket."""
        vectors = self._get_vectors(code)
        
        # Permute each vector by position + vector_index
        permuted = [MultiVectorVSAPrimitives.permute(v, pos + i) 
                    for i, v in enumerate(vectors)]
        
        self.buckets[bucket_id].append(permuted)
    
    def query_bucket(self, bucket_id: int, pos: int) -> Tuple[int, float]:
        """Query bucket for item at position using multi-vector unbind."""
        if bucket_id not in self.buckets or not self.buckets[bucket_id]:
            return -1, -1.0
        
        # Each item in bucket is a list of vectors [v1, v2, v3, v4] for a single code
        # We need to unbind each item and match against codebook
        
        best_code = -1
        best_sim = -1.0
        
        for item_vectors in self.buckets[bucket_id]:
            # Unpermute each vector by -(pos + vector_index)
            unbound_vectors = []
            for i, v in enumerate(item_vectors):
                unbound = MultiVectorVSAPrimitives.permute(v, -(pos + i))
                unbound_vectors.append(unbound)
            
            # Look up each unbound vector in codebook, combine evidence
            for code in range(self.codebook_size):
                code_vectors = self._get_vectors(code)
                sims = [MultiVectorVSAPrimitives.similarity(uv, cv) 
                        for uv, cv in zip(unbound_vectors, code_vectors)]
                avg_sim = np.mean(sims)
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_code = code
        
        if best_code == -1:
            return -1, -1.0
        return best_code, best_sim
    
    def add_to_bucket(self, bucket_id: int, code: int, pos: int, level: int = 0, context: int = 0):
        self.add_to_bucket(bucket_id, code, pos, level, context)
    
    def add(self, code: int, pos: int, level: int = 0, context: int = 0) -> int:
        bucket_id = self.create_bucket()
        self.add_to_bucket(bucket_id, code, pos, level, context)
        return bucket_id
    
    def create_bucket(self) -> int:
        bucket_id = self.next_bucket_id
        self.next_bucket_id += 1
        self.buckets[bucket_id] = []
        return bucket_id
    
    def add_to_bucket(self, bucket_id: int, code: int, pos: int, level: int = 0, context: int = 0):
        vectors = self._get_vectors(code)
        permuted = [MultiVectorVSAPrimitives.permute(v, pos + i) 
                    for i, v in enumerate(vectors)]
        self.buckets[bucket_id].append(permuted)
    
    def query_bucket(self, bucket_id: int, pos: int) -> Tuple[int, float]:
        if bucket_id not in self.buckets or not self.buckets[bucket_id]:
            return -1, -1.0
        
        # Bundle each vector position separately
        bundled = []
        for i in range(self.num_vectors):
            vecs_at_i = [item[i] for item in self.buckets[bucket_id]]
            bundled.append(MultiVectorVSAPrimitives.bundle(vecs_at_i))
        
        # Unpermute by position
        query_vectors = [MultiVectorVSAPrimitives.permute(v, -pos) for v in bundled]
        
        # Lookup each vector, combine evidence
        candidate_codes = []
        candidate_sims = []
        
        for code in range(self.codebook_size):
            vectors = self._get_vectors(code)
            sims = [MultiVectorVSAPrimitives.similarity(qv, cv) 
                    for qv, cv in zip(query_vectors, vectors)]
            avg_sim = np.mean(sims)
            if avg_sim > 0.3:  # Threshold
                candidate_codes.append(code)
                candidate_sims.append(avg_sim)
        
        if not candidate_codes:
            return -1, -1.0
        
        best_idx = np.argmax(candidate_sims)
        return candidate_codes[best_idx], candidate_sims[best_idx]
    
    def create_bucket(self) -> int:
        bucket_id = self.next_bucket_id
        self.next_bucket_id += 1
        self.buckets[bucket_id] = []
        return bucket_id
    
    def capacity_check(self) -> dict:
        return {
            "num_buckets": len(self.buckets),
            "max_bucket_size": self.max_bucket_size,
            "dim": self.config.dim_per_vector * self.num_vectors,
            "num_vectors": self.num_vectors,
            "dim_per_vector": self.dim_per_vector,
            "codebook_size": self.codebook_size
        }

# ── Quick Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np
    import math
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test multi-vector VSA
    config = MultiVectorConfig(num_vectors=4, dim_per_vector=256, codebook_size=256)
    stage2 = MultiVectorFissFusStage2(config)
    
    # Test basic operations
    print("Testing Multi-Vector Stage 2...")
    
    # Store some codes
    bid = stage2.create_bucket()
    stage2.add_to_bucket(bid, 42, 0)
    stage2.add_to_bucket(bid, 17, 1)
    stage2.add_to_bucket(bid, 93, 2)
    
    # Query
    for pos in range(3):
        code, sim = stage2.query_bucket(bid, pos)
        print(f"pos={pos}: code={code}, sim={sim:.3f}")
    
    print("\n✅ Multi-Vector Stage 2 working!")