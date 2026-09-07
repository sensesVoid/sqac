//! Python bindings for the unified model
//!
//! Provides PyO3 bindings for experimentation from Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::{UnifiedModel, ModelStats};

/// Python wrapper for the unified model
#[pyclass]
pub struct PyUnifiedModel {
    inner: UnifiedModel,
}

#[pymethods]
impl PyUnifiedModel {
    /// Create new unified model
    #[new]
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
        Self {
            inner: UnifiedModel::new(
                chunk_size,
                latent_dim,
                codebook_size,
                num_subspaces,
                vsa_dim,
                num_experts,
                gate_threshold,
                turbovec_bits,
            ),
        }
    }
    
    /// Encode text to codes
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }
    
    /// Decode codes to text
    pub fn decode(&self, codes: Vec<usize>) -> String {
        self.inner.decode(&codes)
    }
    
    /// Store codes in memory
    pub fn store(&mut self, codes: Vec<usize>) -> Vec<usize> {
        self.inner.store(&codes)
    }
    
    /// Retrieve codes
    pub fn retrieve(&self, codes: Vec<usize>, bucket_ids: Vec<usize>) -> Vec<usize> {
        self.inner.retrieve(&codes, &bucket_ids)
    }
    
    /// Full forward pass
    pub fn forward(&mut self, text: &str) -> (String, usize, f32) {
        self.inner.forward(text)
    }
    
    /// Get model statistics
    pub fn stats(&self) -> PyObject {
        let stats = self.inner.stats();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("stage1_params", stats.stage1_params).unwrap();
            dict.set_item("stage2_buckets", stats.stage2_buckets).unwrap();
            dict.set_item("moe_experts", stats.moe_experts).unwrap();
            dict.set_item("turbovec_enabled", stats.turbovec_enabled).unwrap();
            dict.into()
        })
    }
}

/// Module definition
#[pymodule]
fn synthquant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyUnifiedModel>()?;
    Ok(())
}
