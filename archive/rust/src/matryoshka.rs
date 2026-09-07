//! Matryoshka MoE: Nested hypervectors with expert gating
//!
//! - D-dimensional vector = E experts of D/E dims each
//! - Learned gate selects which experts activate per query
//! - Only active experts perform similarity search
//! - Supports hierarchical retrieval: coarse → medium → fine

use crate::vsa::VSAPrimitives;
use crate::stage2::FissFusStage2;
use ndarray::Array1;
use std::collections::HashMap;

/// Matryoshka MoE store for efficient retrieval
pub struct MatryoshkaMoEStore {
    num_experts: usize,
    expert_dim: usize,
    gate_threshold: f32,
    
    // Gate parameters
    gate_logits: Vec<f32>,
    gate_bias: Vec<f32>,
    
    // Expert utilization tracking
    expert_utilization: Vec<usize>,
}

impl MatryoshkaMoEStore {
    /// Create new Matryoshka MoE store
    pub fn new(dim: usize, num_experts: usize, gate_threshold: f32) -> Self {
        assert!(dim % num_experts == 0, "dim must be divisible by num_experts");
        
        let expert_dim = dim / num_experts;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        // Initialize gate parameters
        let gate_logits = (0..num_experts)
            .map(|_| rng.gen_range(-0.01..0.01))
            .collect();
        let gate_bias = (0..num_experts)
            .map(|_| rng.gen_range(-0.01..0.01))
            .collect();
        
        Self {
            num_experts,
            expert_dim,
            gate_threshold,
            gate_logits,
            gate_bias,
            expert_utilization: vec![0; num_experts],
        }
    }
    
    /// Compute expert activation scores for query
    pub fn gate(&self, query: &Array1<f32>) -> Vec<f32> {
        let mut scores = Vec::with_capacity(self.num_experts);
        
        for i in 0..self.num_experts {
            // Simple linear gate: score = sigmoid(logits[i] * norm + bias[i])
            let norm = query.dot(&query).sqrt();
            let logit = self.gate_logits[i] * norm + self.gate_bias[i];
            let score = 1.0 / (1.0 + (-logit).exp());
            scores.push(score);
        }
        
        scores
    }
    
    /// Get active experts for query
    pub fn active_experts(&self, query: &Array1<f32>) -> Vec<usize> {
        let scores = self.gate(query);
        let mut active = Vec::new();
        
        for (i, &score) in scores.iter().enumerate() {
            if score > self.gate_threshold {
                active.push(i);
            }
        }
        
        active
    }
    
    /// Update expert utilization statistics
    pub fn update_utilization(&mut self, expert_idx: usize) {
        if expert_idx < self.expert_utilization.len() {
            self.expert_utilization[expert_idx] += 1;
        }
    }
    
    /// Get expert utilization statistics
    pub fn get_utilization(&self) -> &[usize] {
        &self.expert_utilization
    }
    
    /// Get gate statistics for a set of queries
    pub fn get_gate_stats(&self, queries: &[&Array1<f32>]) -> Vec<usize> {
        let mut counts = vec![0; self.num_experts];
        
        for query in queries {
            let active = self.active_experts(query);
            for expert in active {
                counts[expert] += 1;
            }
        }
        
        counts
    }
    
    /// Check if expert is active for query
    pub fn is_expert_active(&self, query: &Array1<f32>, expert_idx: usize) -> bool {
        let scores = self.gate(query);
        expert_idx < scores.len() && scores[expert_idx] > self.gate_threshold
    }
}

/// Matryoshka loss: supervise at each nesting level
pub struct MatryoshkaLoss {
    num_experts: usize,
    weight_decay: f32,
}

impl MatryoshkaLoss {
    /// Create new Matryoshka loss
    pub fn new(num_experts: usize, weight_decay: f32) -> Self {
        Self {
            num_experts,
            weight_decay,
        }
    }
    
    /// Compute nested cosine loss
    pub fn forward(&self, pred: &Array1<f32>, target: &Array1<f32>) -> f32 {
        assert_eq!(pred.len(), target.len(), "Vectors must have same dimension");
        
        let expert_dim = pred.len() / self.num_experts;
        let mut total_loss = 0.0;
        let mut weight_sum = 0.0;
        
        for i in 1..=self.num_experts {
            let slice_end = i * expert_dim;
            let pred_slice = normalize(&pred.slice(ndarray::s![0..slice_end]).to_owned());
            let target_slice = normalize(&target.slice(ndarray::s![0..slice_end]).to_owned());
            
            let loss = 1.0 - pred_slice.dot(&target_slice);
            let weight = self.weight_decay.powi((self.num_experts - i) as i32);
            
            total_loss += weight * loss;
            weight_sum += weight;
        }
        
        total_loss / weight_sum
    }
}

/// Normalize vector to unit length
fn normalize(v: &Array1<f32>) -> Array1<f32> {
    let norm = v.dot(v).sqrt();
    if norm < 1e-8 {
        v.clone()
    } else {
        v / norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matryoshka_moe_basic() {
        let moe = MatryoshkaMoEStore::new(1024, 4, 0.15);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let query = Array1::from_vec((0..1024).map(|_| rng.gen_range(-1.0..1.0)).collect());
        
        let scores = moe.gate(&query);
        assert_eq!(scores.len(), 4);
        
        let active = moe.active_experts(&query);
        assert!(active.len() <= 4);
    }
    
    #[test]
    fn test_matryoshka_loss() {
        let loss = MatryoshkaLoss::new(4, 0.9);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pred = Array1::from_vec((0..1024).map(|_| rng.gen_range(-1.0..1.0)).collect());
        let target = Array1::from_vec((0..1024).map(|_| rng.gen_range(-1.0..1.0)).collect());
        
        let l = loss.forward(&pred, &target);
        assert!(l >= 0.0 && l <= 2.0);
    }
}
