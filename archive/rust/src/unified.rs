//! Unified model integrating all components
//!
//! Combines:
//! - Stage 1: VQ-VAE compressor
//! - Stage 2: VSA structural memory
//! - Matryoshka MoE: efficient retrieval
//! - TurboVec: fast codebook search

use crate::stage1::VQVAE;
use crate::stage2::FissFusStage2;
use crate::matryoshka::MatryoshkaMoEStore;
use crate::turbovec_retriever::TurboVecRetriever;

/// Unified model combining all stages
pub struct UnifiedModel {
    stage1: VQVAE,
    stage2: FissFusStage2,
    moe: MatryoshkaMoEStore,
    turbovec: TurboVecRetriever,
}

impl UnifiedModel {
    /// Create new unified model
    pub fn new(
        chunk_size: usize,
        latent_dim: usize,
        codebook_size: usize,
        num_subspaces: usize,
        vsa_dim: usize,
        num_experts: usize,
        gate_threshold: f32,
        turbovec_bits: usize,
    ) -> Self {
        let stage1 = VQVAE::new(chunk_size, latent_dim, codebook_size, num_subspaces, vsa_dim);
        let stage2 = FissFusStage2::new(codebook_size, vsa_dim, 64);
        let moe = MatryoshkaMoEStore::new(vsa_dim, num_experts, gate_threshold);
        let turbovec = TurboVecRetriever::new(codebook_size, vsa_dim, turbovec_bits);
        
        Self {
            stage1,
            stage2,
            moe,
            turbovec,
        }
    }
    
    /// Encode text to codes
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.stage1.encode(text)
    }
    
    /// Decode codes to text
    pub fn decode(&self, codes: &[usize]) -> String {
        self.stage1.decode(codes)
    }
    
    /// Store codes in Stage 2 memory
    pub fn store(&mut self, codes: &[usize]) -> Vec<usize> {
        let mut bucket_ids = Vec::with_capacity(codes.len());
        
        for (i, &code) in codes.iter().enumerate() {
            let bid = self.stage2.create_bucket();
            self.stage2.add_to_bucket(bid, code, i);
            bucket_ids.push(bid);
        }
        
        bucket_ids
    }
    
    /// Retrieve codes using Matryoshka MoE
    pub fn retrieve(&self, codes: &[usize], bucket_ids: &[usize]) -> Vec<usize> {
        let mut retrieved = Vec::with_capacity(codes.len());
        
        for (i, &code) in codes.iter().enumerate() {
            let V = self.stage2.get_code_vector(code);
            let active_experts = self.moe.active_experts(&V);
            
            if active_experts.contains(&(i % self.moe.num_experts)) {
                let (retrieved_code, _) = self.stage2.query_bucket(bucket_ids[i], i);
                retrieved.push(retrieved_code);
            } else {
                retrieved.push(code);
            }
        }
        
        retrieved
    }
    
    /// Full encode-store-retrieve-decode pipeline
    pub fn forward(&mut self, text: &str) -> (String, usize, f32) {
        let codes = self.encode(text);
        let bucket_ids = self.store(&codes);
        let retrieved = self.retrieve(&codes, &bucket_ids);
        let decoded = self.decode(&retrieved);
        
        let unique_codes = codes.iter().collect::<std::collections::HashSet<_>>().len();
        let compression = text.len() as f32 / (codes.len() * 4).max(1) as f32;
        
        (decoded, unique_codes, compression)
    }
    
    /// Get model statistics
    pub fn stats(&self) -> ModelStats {
        ModelStats {
            stage1_params: 0, // Would compute actual params
            stage2_buckets: self.stage2.num_buckets(),
            moe_experts: self.moe.num_experts,
            turbovec_enabled: self.turbovec.use_turbovec,
        }
    }
}

/// Model statistics
#[derive(Debug, Clone)]
pub struct ModelStats {
    pub stage1_params: usize,
    pub stage2_buckets: usize,
    pub moe_experts: usize,
    pub turbovec_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_model() {
        let mut model = UnifiedModel::new(8, 128, 256, 4, 1024, 4, 0.15, 2);
        let text = "Hello world!";
        
        let (decoded, unique, compression) = model.forward(text);
        assert!(!decoded.is_empty());
        assert!(unique > 0);
        assert!(compression > 0.0);
    }
}
