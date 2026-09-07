import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

from fissfus_stage1 import VQVAE, text_to_chunks, chunks_to_text, reconstruction_error, codebook_utilization
from fissfus_stage2 import FissFusStage2

CHARS = string.printable
VOCAB_SIZE = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

@dataclass
class FissFusResult:
    original_text: str
    reconstructed_text: str
    codes: np.ndarray
    retrieval_codes: np.ndarray
    stage1_recon_error: float
    stage2_retrieval_accuracy: float
    compression_ratio: float
    encode_time: float
    query_time: float

class FissFus(nn.Module):
    """End-to-end FissFus: VQ-VAE + VSA structural layer."""
    
    def __init__(self, stage1: VQVAE, stage2: FissFusStage2, chunk_size: int = 8):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.chunk_size = chunk_size
    
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
            code = int(code_vec[0, 0])
            bucket_id = self.stage2.add(code, pos=i)
            codes.append(code)
            bucket_ids.append(bucket_id)
        
        return np.array(codes), bucket_ids
    
    def query(self, text: str, bucket_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Query text against stored buckets."""
        chunks = text_to_chunks(text, self.chunk_size)
        if len(chunks) == 0:
            raise ValueError("Empty text after chunking")
        
        dataset = torch.tensor(np.stack(chunks), dtype=torch.float32)
        
        retrieved_codes = []
        similarities = []
        for i, chunk in enumerate(dataset):
            code_vec = self.stage1.encode(chunk.unsqueeze(0))
            code = int(code_vec[0, 0])
            retrieved, sim = self.stage2.query_bucket(bucket_ids[i], pos=i)
            retrieved_codes.append(retrieved[0])
            similarities.append(sim[0])
        
        return np.array(retrieved_codes), np.array(similarities)
    
    def end_to_end(self, text: str) -> FissFusResult:
        """Full encode→store→query pipeline."""
        start_encode = time.time()
        codes, bucket_ids = self.encode(text)
        encode_time = time.time() - start_encode
        
        recon_chunks = self.stage1.decode_codes(codes).reshape(-1, self.chunk_size)
        reconstructed = chunks_to_text(recon_chunks)
        
        start_query = time.time()
        retrieved_codes, _ = self.query(text, bucket_ids)
        query_time = time.time() - start_query
        
        original_chunks = text_to_chunks(text, self.chunk_size)
        original_array = np.stack(original_chunks)
        recon_array = recon_chunks
        stage1_error = reconstruction_error(original_array, recon_array)
        
        stage2_accuracy = np.mean(codes == retrieved_codes)
        
        original_bytes = len(text.encode('utf-8'))
        compressed_bytes = len(codes) * 4 + len(bucket_ids) * 8
        compression_ratio = original_bytes / max(compressed_bytes, 1)
        
        return FissFusResult(
            original_text=text,
            reconstructed_text=reconstructed,
            codes=codes,
            retrieval_codes=retrieved_codes,
            stage1_recon_error=stage1_error,
            stage2_retrieval_accuracy=float(stage2_accuracy),
            compression_ratio=compression_ratio,
            encode_time=encode_time,
            query_time=query_time
        )

def load_trained_stage1(model_path: str, chunk_size: int = 8, latent_dim: int = 128,
                        codebook_size: int = 512) -> VQVAE:
    """Load trained Stage 1 model."""
    model = VQVAE(chunk_size, latent_dim, codebook_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def create_end_to_end_pipeline(chunk_size: int = 8, latent_dim: int = 128,
                               codebook_size: int = 512, vsa_dim: int = 1024) -> FissFus:
    """Create full FissFus pipeline."""
    stage1 = VQVAE(chunk_size, latent_dim, codebook_size, num_subspaces=1)
    stage1.eval()
    
    stage2 = FissFusStage2(codebook_size, vsa_dim, use_krop=True)
    
    return FissFus(stage1, stage2, chunk_size)

def benchmark_compression_ratios():
    """Benchmark different compression configurations."""
    configs = [
        {"chunk": 8, "latent": 128, "codebook": 512, "vsa": 1024, "name": "2x"},
        {"chunk": 16, "latent": 256, "codebook": 1024, "vsa": 2048, "name": "4x"},
        {"chunk": 32, "latent": 512, "codebook": 2048, "vsa": 4096, "name": "8x"},
    ]
    
    text = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    Sphinx of black quartz, judge my vow.
    """ * 20
    
    results = []
    for config in configs:
        print(f"\n--- {config['name']} compression ---")
        model = create_end_to_end_pipeline(
            chunk_size=config['chunk'],
            latent_dim=config['latent'],
            codebook_size=config['codebook'],
            vsa_dim=config['vsa']
        )
        
        result = model.end_to_end(text)
        
        print(f"  Stage 1 recon error: {result.stage1_recon_error:.4f}")
        print(f"  Stage 2 retrieval:   {result.stage2_retrieval_accuracy:.4f}")
        print(f"  Compression ratio:   {result.compression_ratio:.2f}x")
        print(f"  Encode time:         {result.encode_time:.4f}s")
        print(f"  Query time:          {result.query_time:.4f}s")
        
        results.append({
            "config": config,
            "stage1_recon": result.stage1_recon_error,
            "stage2_retrieval": result.stage2_retrieval_accuracy,
            "compression": result.compression_ratio,
            "encode_time": result.encode_time,
            "query_time": result.query_time
        })
    
    return results

if __name__ == "__main__":
    import glob
    doc_texts = []
    for path in glob.glob("/teamspace/studios/this_studio/Project_ SynthQuant/docs/*.md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            doc_texts.append(f.read())
    text = "\n".join(doc_texts)
    print(f"Loaded {len(text)} characters from docs")
    
    model = create_end_to_end_pipeline()
    result = model.end_to_end(text)
    
    print(f"\nStage 1 reconstruction error: {result.stage1_recon_error:.4f}")
    print(f"Stage 2 retrieval accuracy:   {result.stage2_retrieval_accuracy:.4f}")
    print(f"Compression ratio:            {result.compression_ratio:.2f}x")
    print(f"Encode time:                  {result.encode_time:.4f}s")
    print(f"Query time:                   {result.query_time:.4f}s")
