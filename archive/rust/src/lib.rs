//! SynthQuant Unified Model
//!
//! Integrates:
//! - FissFus Stage 1: VQ-VAE compressor with PQ subspaces
//! - FissFus Stage 2: VSA structural memory with permutation addressing
//! - Matryoshka MoE: Nested hypervectors with expert gating
//! - TurboVec: Fast approximate nearest neighbor search
//!
//! Python bindings via PyO3 for experimentation.

pub mod vsa;
pub mod stage1;
pub mod stage2;
pub mod matryoshka;
pub mod turbovec_retriever;
pub mod unified;
pub mod python;

pub use vsa::VSAPrimitives;
pub use stage1::{VQVAE, PQVectorQuantizer};
pub use stage2::{FissFusStage2, Bucket};
pub use matryoshka::{MatryoshkaMoEStore, MatryoshkaLoss};
pub use turbovec_retriever::TurboVecRetriever;
pub use unified::UnifiedModel;
