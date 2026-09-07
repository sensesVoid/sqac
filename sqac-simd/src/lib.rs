//! SIMD-accelerated XOR+popcount for SQAC fuzzy retrieval.
//!
//! Provides `bulk_similarity` — the hot path for lexical/semantic tier scanning.
//! Falls back to portable popcount when SIMD is unavailable.

use pyo3::prelude::*;

/// Compute Hamming similarity between a query vector and a matrix of key vectors.
///
/// `query`   — packed bits, D/8 bytes (e.g. 128 bytes for D=1024)
/// `keys`    — concatenated packed bits, n * D/8 bytes
/// `n`       — number of key vectors
/// `dims`    — bit dimension (must be multiple of 8)
///
/// Returns a Vec<f64> of length `n` with similarity = 1.0 - hamming/dims.
#[pyfunction]
fn bulk_similarity(query: &[u8], keys: &[u8], n: usize, dims: usize) -> PyResult<Vec<f64>> {
    let key_len = dims / 8;
    let expected = n * key_len;
    if keys.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "keys length {} != n * key_len = {} * {} = {}",
            keys.len(), n, key_len, expected
        )));
    }
    if query.len() != key_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "query length {} != key_len {}",
            query.len(), key_len
        )));
    }

    let mut results = Vec::with_capacity(n);
    let dims_f = dims as f64;

    for i in 0..n {
        let offset = i * key_len;
        let key = &keys[offset..offset + key_len];
        let hamming = xor_popcount(query, key);
        results.push(1.0 - hamming as f64 / dims_f);
    }

    Ok(results)
}

/// XOR + popcount of two equal-length byte slices.
/// Uses SIMD when available (AVX2 on x86_64, NEON on ARM).
#[inline]
fn xor_popcount(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt") {
            return unsafe { xor_popcount_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { xor_popcount_neon(a, b) };
    }

    // Portable fallback
    xor_popcount_portable(a, b)
}

/// Portable XOR+popcount — works everywhere.
#[inline]
fn xor_popcount_portable(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// AVX2 + POPCNT implementation for x86_64.
/// Processes 256 bits (32 bytes) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn xor_popcount_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 32;
    let mut total: u32 = 0;

    let mut i = 0;
    while i < chunks {
        let va = _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i);
        let xored = _mm256_xor_si256(va, vb);

        // POPCNT via lookup: _mm256_popcnt_epi8 requires VPOPCNTDQ (AVX512-VPOPCNTDQ)
        // Fallback: extract 64-bit lanes and use _mm_popcnt_epi64 equivalent
        let lo = _mm256_extracti128_si256(xored, 0);
        let hi = _mm256_extracti128_si256(xored, 1);

        // Use scalar popcount on each 64-bit chunk (still fast with POPCNT)
        let lo_ptr = &lo as *const _ as *const i64;
        let hi_ptr = &hi as *const _ as *const i64;
        for j in 0..2 {
            total += _popcnt64(*lo_ptr.add(j)) as u32;
            total += _popcnt64(*hi_ptr.add(j)) as u32;
        }

        i += 1;
    }

    // Handle remaining bytes
    let remainder = chunks * 32;
    for j in remainder..len {
        total += (a[j] ^ b[j]).count_ones();
    }

    total
}

/// NEON implementation for AArch64.
/// Processes 128 bits (16 bytes) per iteration.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_popcount_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 16;
    let mut total: u32 = 0;

    let lookup = vld1q_u8([
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    ].as_ptr());

    let mut i = 0;
    while i < chunks {
        let va = vld1q_u8(a.as_ptr().add(i * 16));
        let vb = vld1q_u8(b.as_ptr().add(i * 16));
        let xored = veorq_u8(va, vb);

        // Bit-parallel popcount using lookup table
        let lo = vandq_u8(xored, vdupq_n_u8(0x0F));
        let hi = vshrq_n_u8(xored, 4);
        let pop_lo = vqtbl1q_u8(lookup, lo);
        let pop_hi = vqtbl1q_u8(lookup, hi);
        let counts = vaddq_u8(pop_lo, pop_hi);

        // Horizontal sum of bytes
        let sum = vaddlvq_u8(counts);
        total += sum as u32;

        i += 1;
    }

    // Handle remaining bytes
    let remainder = chunks * 16;
    for j in remainder..len {
        total += (a[j] ^ b[j]).count_ones();
    }

    total
}

/// Module definition for Python
#[pymodule]
fn sqac_simd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bulk_similarity, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v = vec![0xAAu8; 128];
        let sim = xor_popcount(&v, &v);
        assert_eq!(sim, 0);
    }

    #[test]
    fn test_opposite_vectors() {
        let a = vec![0xFFu8; 128];
        let b = vec![0x00u8; 128];
        let sim = xor_popcount(&a, &b);
        assert_eq!(sim, 128 * 8); // all bits differ
    }

    #[test]
    fn test_one_bit_diff() {
        let mut a = vec![0x00u8; 128];
        let mut b = vec![0x00u8; 128];
        a[0] = 0x01;
        b[0] = 0x02;
        let sim = xor_popcount(&a, &b);
        assert_eq!(sim, 2); // 2 bits differ
    }

    #[test]
    fn test_bulk_similarity() {
        let query = vec![0xAAu8; 128];
        let mut keys = Vec::new();
        keys.extend_from_slice(&vec![0xAAu8; 128]); // identical
        keys.extend_from_slice(&vec![0x55u8; 128]); // opposite
        let sims = bulk_similarity(&query, &keys, 2, 1024).unwrap();
        assert!((sims[0] - 1.0).abs() < 1e-10); // identical = 1.0
        assert!((sims[1] - 0.0).abs() < 1e-10); // opposite = 0.0
    }

    #[test]
    fn test_matches_python_popcount() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Generate deterministic pseudo-random data
        let mut a = vec![0u8; 128];
        let mut b = vec![0u8; 128];
        for i in 0..128 {
            a[i] = (i.wrapping_mul(37).wrapping_add(13)) as u8;
            b[i] = (i.wrapping_mul(71).wrapping_add(29)) as u8;
        }

        let rust_result = xor_popcount(&a, &b);

        // Python reference: sum((x ^ y).bit_count() for x, y in zip(a, b))
        let python_result: u32 = a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum();

        assert_eq!(rust_result, python_result);
    }
}
