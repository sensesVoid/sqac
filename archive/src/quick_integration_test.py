import torch
import numpy as np
import string
from fissfus_stage1 import VQVAE, text_to_chunks, chunks_to_text, reconstruction_error
from fissfus_stage2 import FissFusStage2

CHARS = string.printable
VOCAB_SIZE = len(CHARS)

def quick_integration_test():
    """Quick smoke test of Stage 1 + Stage 2 pipeline."""
    print("=== Quick Integration Test ===")
    
    text = "Hello world! This is a test. " * 50
    print(f"Text length: {len(text)} chars")
    
    chunk_size = 8
    chunks = text_to_chunks(text, chunk_size)
    print(f"Chunks: {len(chunks)}")
    
    stage1 = VQVAE(chunk_size=chunk_size, latent_dim=64, codebook_size=128, num_subspaces=1)
    stage1.eval()
    
    with torch.no_grad():
        dataset = torch.tensor(np.stack(chunks), dtype=torch.float32)
        codes = stage1.encode(dataset)
    
    print(f"Stage 1 codes shape: {codes.shape}")
    print(f"Unique codes: {len(np.unique(codes))}")
    
    stage2 = FissFusStage2(num_codes=128, dim=256, use_krop=True)
    
    bucket_ids = []
    for i, code in enumerate(codes):
        bid = stage2.add(int(code), pos=i)
        bucket_ids.append(bid)
    
    retrieved_codes = []
    for i in range(len(codes)):
        retrieved, sim = stage2.query_bucket(bucket_ids[i], pos=i)
        retrieved_codes.append(int(retrieved[0]))
    
    accuracy = np.mean(np.array(codes) == np.array(retrieved_codes))
    print(f"Stage 2 retrieval accuracy: {accuracy:.4f}")
    
    recon_chunks = stage1.decode_codes(codes)
    recon_text = chunks_to_text(recon_chunks.reshape(-1, chunk_size))
    
    original_chunks = np.stack(chunks)
    recon_error = reconstruction_error(original_chunks, recon_chunks.reshape(-1, chunk_size))
    print(f"Stage 1 reconstruction error: {recon_error:.4f}")
    
    print(f"Sample original: {text[:100]}")
    print(f"Sample recon:    {recon_text[:100]}")
    
    return {
        "stage1_recon_error": float(recon_error),
        "stage2_retrieval_accuracy": float(accuracy),
        "num_chunks": len(chunks),
        "unique_codes": int(len(np.unique(codes)))
    }

if __name__ == "__main__":
    result = quick_integration_test()
    print("\nResult:", result)
