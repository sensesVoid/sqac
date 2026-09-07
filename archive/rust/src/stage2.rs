//! Stage 2: VSA structural memory with permutation addressing
//!
//! Stores compressed codes in addressable VSA memory buckets:
//! - Permutation-based addressing (cyclic shift) for position encoding
//! - KROP cleanup for fast codebook lookup
//! - Bucket bundling for sequence-level storage

use crate::vsa::VSAPrimitives;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Memory bucket storing bound code vectors
#[derive(Debug, Clone)]
pub struct Bucket {
    pub bound: Array1<f32>,
    pub code: usize,
    pub stored_pos: usize,
}

/// Stage 2 VSA structural memory
pub struct FissFusStage2 {
    num_codes: usize,
    dim: usize,
    max_bucket_size: usize,
    
    // KROP-structured codebook for fast cleanup
    codebook: Array2<f32>,
    use_krop: bool,
    
    // Memory buckets: bucket_id -> items
    buckets: HashMap<usize, Vec<Bucket>>,
    
    // Next bucket ID
    next_bucket_id: usize,
}

impl FissFusStage2 {
    /// Create new Stage 2 VSA memory
    pub fn new(num_codes: usize, dim: usize, max_bucket_size: usize) -> Self {
        assert!(dim.is_power_of_two(), "VSA dim must be power of 2 for KROP");
        
        // Build KROP-structured codebook
        let codebook = build_krop_codebook(num_codes, dim);
        
        Self {
            num_codes,
            dim,
            max_bucket_size,
            codebook,
            use_krop: true,
            buckets: HashMap::new(),
            next_bucket_id: 0,
        }
    }
    
    /// Create a new empty bucket
    pub fn create_bucket(&mut self) -> usize {
        let id = self.next_bucket_id;
        self.next_bucket_id += 1;
        self.buckets.insert(id, Vec::new());
        id
    }
    
    /// Add code to bucket at position using permutation addressing
    pub fn add_to_bucket(&mut self, bucket_id: usize, code: usize, pos: usize) {
        let V = self.get_code_vector(code);
        let bound = VSAPrimitives::permute(&V, pos);
        
        let bucket = Bucket {
            bound,
            code,
            stored_pos: pos,
        };
        
        self.buckets.entry(bucket_id).or_default().push(bucket);
    }
    
    /// Query bucket for item at position
    pub fn query_bucket(&self, bucket_id: usize, pos: usize) -> (usize, f32) {
        if let Some(bucket_items) = self.buckets.get(&bucket_id) {
            let mut best_code = 0usize;
            let mut best_sim = -1.0f32;
            
            for bucket in bucket_items {
                let unbound = VSAPrimitives::permute(&bucket.bound, pos - bucket.stored_pos);
                let V = self.get_code_vector(bucket.code);
                let sim = VSAPrimitives::similarity(&unbound, &V);
                
                if sim > best_sim {
                    best_sim = sim;
                    best_code = bucket.code;
                }
            }
            
            return (best_code, best_sim);
        }
        
        (0, 0.0)
    }
    
    /// Get code vector from codebook
    pub fn get_code_vector(&self, code: usize) -> Array1<f32> {
        if code < self.num_codes && self.use_krop {
            self.codebook.row(code).to_owned()
        } else {
            let mut v = Array1::zeros(self.dim);
            if self.dim > 0 {
                v[0] = 1.0;
            }
            v
        }
    }
    
    /// Get codebook for TurboVec indexing
    pub fn get_codebook(&self) -> Array2<f32> {
        self.codebook.clone()
    }
    
    /// Check if KROP is enabled
    pub fn uses_krop(&self) -> bool {
        self.use_krop
    }
    
    /// Get number of buckets
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }
}

/// Build KROP-structured codebook for fast cleanup
fn build_krop_codebook(num_codes: usize, dim: usize) -> Array2<f32> {
    assert!(dim.is_power_of_two(), "KROP requires power-of-2 dimension");
    
    let mut codebook = Array2::zeros((num_codes, dim));
    
    // Build KROP matrix H
    let k = dim.trailing_zeros() as usize;
    let mut H = Array2::<f32>::zeros((dim, dim));
    
    // Initialize H[0,0] = 1
    H[(0, 0)] = 1.0;
    
    // Build KROP structure
    let thetas: Vec<f32> = (0..k).map(|i| {
        2.0 * std::f32::consts::PI * (i + 1) as f32 / (k + 2) as f32
    }).collect();
    
    let mut current_size = 1;
    for theta in &thetas {
        let c = theta.cos();
        let s = theta.sin();
        let new_size = current_size * 2;
        
        for i in 0..current_size {
            for j in 0..current_size {
                let val = H[(i, j)];
                H[(i, j)] = c * val;
                H[(i, j + current_size)] = s * val;
                H[(i + current_size, j)] = s * val;
                H[(i + current_size, j + current_size)] = -c * val;
            }
        }
        current_size = new_size;
    }
    
    // Use first num_codes rows as codebook
    for i in 0..num_codes.min(dim) {
        for j in 0..dim {
            codebook[(i, j)] = H[(i, j)];
        }
    }
    
    // Normalize each row
    for i in 0..num_codes {
        let row = codebook.row(i);
        let norm = (row.dot(&row)).sqrt();
        if norm > 1e-8 {
            let mut normalized = row.to_owned();
            normalized /= norm;
            codebook.row_mut(i).assign(&normalized);
        }
    }
    
    codebook
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stage2_basic() {
        let mut stage2 = FissFusStage2::new(256, 1024, 64);
        let bid = stage2.create_bucket();
        stage2.add_to_bucket(bid, 42, 7);
        
        let (code, sim) = stage2.query_bucket(bid, 7);
        assert_eq!(code, 42);
        assert!(sim > 0.9);
    }
    
    #[test]
    fn test_krop_codebook_shape() {
        let cb = build_krop_codebook(256, 1024);
        assert_eq!(cb.nrows(), 256);
        assert_eq!(cb.ncols(), 1024);
    }
}
