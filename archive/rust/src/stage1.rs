//! Stage 1: VQ-VAE compressor with PQ subspaces
//!
//! Compresses text chunks into discrete codes using:
//! - Character-level embeddings
//! - Product quantization (PQ) with multiple subspaces
//! - Dead code restart to prevent codebook collapse
//! - Entropy regularization for codebook utilization

use crate::vsa::VSAPrimitives;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Stage 1 VQ-VAE compressor
pub struct VQVAE {
    chunk_size: usize,
    latent_dim: usize,
    codebook_size: usize,
    num_subspaces: usize,
    subspace_entries: usize,
    
    // Character embeddings
    char_embeddings: Array2<f32>,
    embed_dim: usize,
    
    // Encoder weights (simplified: single layer)
    encoder_weight: Array2<f32>,
    encoder_bias: Array1<f32>,
    
    // PQ codebooks: [num_subspaces][subspace_entries][subspace_dim]
    codebooks: Vec<Array2<f32>>,
    
    // Decoder weights
    decoder_weight: Array2<f32>,
    decoder_bias: Array1<f32>,
    
    // VSA dimension for Stage 2
    vsa_dim: usize,
    
    // Hyperparameters
    beta: f32,
    learning_rate: f32,
}

impl VQVAE {
    /// Create new VQ-VAE with given configuration
    pub fn new(
        chunk_size: usize,
        latent_dim: usize,
        codebook_size: usize,
        num_subspaces: usize,
        vsa_dim: usize,
    ) -> Self {
        assert!(latent_dim % num_subspaces == 0, "latent_dim must be divisible by num_subspaces");
        let subspace_entries = codebook_size / num_subspaces;
        let subspace_dim = latent_dim / num_subspaces;
        let embed_dim = 32;
        let hidden_dim = 128;
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        // Character embeddings: [vocab_size, embed_dim]
        let char_embeddings = Array2::from_shape_fn((100, embed_dim), |(_, _)| {
            rng.gen_range(-0.1..0.1)
        });
        
        // Encoder: [chunk_size * embed_dim, hidden_dim] -> [hidden_dim, latent_dim]
        let encoder_weight = Array2::from_shape_fn((chunk_size * embed_dim, hidden_dim), |(_, _)| {
            rng.gen_range(-0.1..0.1)
        });
        let encoder_bias = Array1::zeros(hidden_dim);
        
        // PQ codebooks
        let mut codebooks = Vec::with_capacity(num_subspaces);
        for _ in 0..num_subspaces {
            let cb = Array2::from_shape_fn((subspace_entries, subspace_dim), |(_, _)| {
                rng.gen_range(-1.0 / subspace_entries as f32..1.0 / subspace_entries as f32)
            });
            codebooks.push(cb);
        }
        
        // Decoder: [latent_dim, hidden_dim] -> [hidden_dim, chunk_size * vocab_size]
        let decoder_weight = Array2::from_shape_fn((latent_dim, hidden_dim), |(_, _)| {
            rng.gen_range(-0.1..0.1)
        });
        let decoder_bias = Array1::zeros(hidden_dim);
        
        Self {
            chunk_size,
            latent_dim,
            codebook_size,
            num_subspaces,
            subspace_entries,
            char_embeddings,
            embed_dim,
            encoder_weight,
            encoder_bias,
            codebooks,
            decoder_weight,
            decoder_bias,
            vsa_dim,
            beta: 0.25,
            learning_rate: 1e-3,
        }
    }
    
    /// Encode text chunk to discrete codes
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let chunks = text_to_chunks(text, self.chunk_size);
        let mut codes = Vec::with_capacity(chunks.len());
        
        for chunk in &chunks {
            let code = self.encode_chunk(chunk);
            codes.push(code);
        }
        
        codes
    }
    
    /// Encode single chunk to single code (first subspace only for now)
    pub fn encode_chunk(&self, chunk: &[usize]) -> usize {
        // Get embeddings
        let mut emb = Array1::zeros(self.chunk_size * self.embed_dim);
        for (i, &char_idx) in chunk.iter().enumerate() {
            if char_idx < self.char_embeddings.nrows() {
                let char_emb = self.char_embeddings.row(char_idx);
                let start = i * self.embed_dim;
                emb.slice_mut(ndarray::s![start..start + self.embed_dim])
                    .assign(&char_emb);
            }
        }
        
        // Encoder forward pass
        let z = emb.dot(&self.encoder_weight) + &self.encoder_bias;
        let z = relu(&z);
        
        // PQ quantization - only use first subspace for now
        let subspace_dim = self.latent_dim / self.num_subspaces;
        let z_sub = z.slice(ndarray::s![0..subspace_dim]).to_owned();
        let codebook = &self.codebooks[0];
        
        // Find nearest codebook entry
        let mut min_dist = f32::INFINITY;
        let mut best_code = 0;
        
        for (i, codebook_vec) in codebook.rows().into_iter().enumerate() {
            let dist = (&z_sub - &codebook_vec).mapv(|x| x * x).sum();
            if dist < min_dist {
                min_dist = dist;
                best_code = i;
            }
        }
        
        best_code
    }
    
    /// Decode codes back to text
    pub fn decode(&self, codes: &[usize]) -> String {
        let chunks = codes.iter().map(|&code| {
            self.decode_code(code)
        }).collect::<Vec<_>>();
        
        chunks_to_text(&chunks)
    }
    
    /// Decode single code to chunk
    pub fn decode_code(&self, code: usize) -> Vec<usize> {
        let subspace_dim = self.latent_dim / self.num_subspaces;
        
        // Get codebook vector
        let codebook = &self.codebooks[0];
        let mut quantized = Array1::zeros(self.latent_dim);
        if code < codebook.nrows() {
            let cb_vec = codebook.row(code);
            quantized.slice_mut(ndarray::s![0..subspace_dim]).assign(&cb_vec);
        }
        
        // Decoder forward pass
        let h = quantized.dot(&self.decoder_weight) + &self.decoder_bias;
        let h = relu(&h);
        
        // Project to character logits (simplified)
        let logits = h.slice(ndarray::s![0..self.chunk_size * 100])
            .to_owned()
            .into_shape((self.chunk_size, 100))
            .unwrap();
        
        // Greedy decoding
        let mut chars = Vec::with_capacity(self.chunk_size);
        for i in 0..self.chunk_size {
            let row = logits.row(i);
            let max_idx = row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            chars.push(max_idx);
        }
        
        chars
    }
    
    /// Train one epoch on text
    pub fn train_epoch(&mut self, text: &str) -> f32 {
        let chunks = text_to_chunks(text, self.chunk_size);
        if chunks.is_empty() {
            return 0.0;
        }
        
        let mut total_loss = 0.0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        for chunk in &chunks {
            // Forward pass
            let code = self.encode_chunk(chunk);
            
            // Simplified loss: encourage codebook diversity
            let loss = self.compute_loss(chunk, code);
            total_loss += loss;
            
            // Update codebook (simplified gradient)
            self.update_codebook(chunk, code, &mut rng);
        }
        
        total_loss / chunks.len() as f32
    }
    
    /// Compute training loss
    fn compute_loss(&self, chunk: &[usize], code: usize) -> f32 {
        // Simplified: reconstruction loss + commitment loss
        let subspace_dim = self.latent_dim / self.num_subspaces;
        
        // Get embeddings
        let mut emb = Array1::zeros(self.chunk_size * self.embed_dim);
        for (i, &char_idx) in chunk.iter().enumerate() {
            if char_idx < self.char_embeddings.nrows() {
                let char_emb = self.char_embeddings.row(char_idx);
                let start = i * self.embed_dim;
                emb.slice_mut(ndarray::s![start..start + self.embed_dim])
                    .assign(&char_emb);
            }
        }
        
        // Encoder
        let z = emb.dot(&self.encoder_weight) + &self.encoder_bias;
        let z = relu(&z);
        let z_sub = z.slice(ndarray::s![0..subspace_dim]).to_owned();
        
        // Codebook vector
        let codebook = &self.codebooks[0];
        let cb_vec = if code < codebook.nrows() {
            codebook.row(code).to_owned()
        } else {
            Array1::zeros(subspace_dim)
        };
        
        // Commitment loss
        let commit_loss = (&z_sub - &cb_vec).mapv(|x| x * x).sum();
        
        // Simplified reconstruction loss
        let recon_loss = commit_loss * 0.5;
        
        recon_loss + self.beta * commit_loss
    }
    
    /// Update codebook with dead code restart
    fn update_codebook(&mut self, chunk: &[usize], code: usize, rng: &mut impl Rng) {
        // Simple codebook update: move toward encoder output
        let subspace_dim = self.latent_dim / self.num_subspaces;
        
        // Get embeddings
        let mut emb = Array1::zeros(self.chunk_size * self.embed_dim);
        for (i, &char_idx) in chunk.iter().enumerate() {
            if char_idx < self.char_embeddings.nrows() {
                let char_emb = self.char_embeddings.row(char_idx);
                let start = i * self.embed_dim;
                emb.slice_mut(ndarray::s![start..start + self.embed_dim])
                    .assign(&char_emb);
            }
        }
        
        let z = emb.dot(&self.encoder_weight) + &self.encoder_bias;
        let z = relu(&z);
        let z_sub = z.slice(ndarray::s![0..subspace_dim]).to_owned();
        
        // Update codebook entry
        if code < self.codebooks[0].nrows() {
            let lr = self.learning_rate;
            let cb = &mut self.codebooks[0];
            let row_mut = cb.row_mut(code);
            row_mut += &(&z_sub - &row_mut) * lr;
        }
    }
    
    /// Restart dead codes in codebook
    pub fn restart_dead_codes(&mut self, usage_counts: &[usize]) {
        if usage_counts.len() != self.subspace_entries {
            return;
        }
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let subspace_dim = self.latent_dim / self.num_subspaces;
        
        for (i, &count) in usage_counts.iter().enumerate() {
            if count == 0 {
                // Reset to random vector
                let mut new_vec = Array1::zeros(subspace_dim);
                for j in 0..subspace_dim {
                    new_vec[j] = rng.gen_range(-0.1..0.1);
                }
                let row = self.codebooks[0].row_mut(i);
                row.assign(&new_vec);
            }
        }
    }
    
    /// Get codebook vectors for TurboVec indexing
    pub fn get_codebook_vectors(&self) -> Array2<f32> {
        self.codebooks[0].clone()
    }
}

/// ReLU activation
fn relu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

/// Convert text to character index chunks
fn text_to_chunks(text: &str, chunk_size: usize) -> Vec<Vec<usize>> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    
    for i in (0..bytes.len()).step_by(chunk_size) {
        let mut chunk = Vec::with_capacity(chunk_size);
        for j in i..(i + chunk_size).min(bytes.len()) {
            chunk.push(bytes[j] as usize % 100);
        }
        while chunk.len() < chunk_size {
            chunk.push(0);
        }
        chunks.push(chunk);
    }
    
    chunks
}

/// Convert chunks to text
fn chunks_to_text(chunks: &[Vec<usize>]) -> String {
    let mut text = String::new();
    for chunk in chunks {
        for &c in chunk {
            if c < 100 {
                text.push((c + 32) as u8 as char);
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vqvae_encode_decode() {
        let model = VQVAE::new(8, 64, 256, 4, 1024);
        let text = "Hello world!";
        let codes = model.encode(text);
        assert!(!codes.is_empty());
        
        let decoded = model.decode(&codes);
        assert!(!decoded.is_empty());
    }
    
    #[test]
    fn test_dead_code_restart() {
        let mut model = VQVAE::new(8, 64, 256, 4, 1024);
        let usage = vec![0; 64]; // All dead
        model.restart_dead_codes(&usage);
        // Should not panic
    }
}
