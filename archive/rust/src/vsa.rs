//! VSA primitives for real-valued hypervectors
//!
//! Uses circular convolution (HRR-style) for exact unbinding
//! and cyclic shift permutation for position encoding.

use ndarray::{Array1, Array2};
use rand::Rng;
use std::f32::consts::PI;

pub struct VSAPrimitives;

impl VSAPrimitives {
    /// Circular convolution binding: a ⊗ b
    /// Uses FFT for efficient computation
    pub fn bind(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        let n = a.len();
        assert_eq!(n, b.len(), "Vectors must have same dimension");
        
        // Use FFT-based circular convolution
        let a_fft = fft(&a);
        let b_fft = fft(&b);
        let ab_fft: Vec<Complex32> = a_fft.iter().zip(b_fft.iter())
            .map(|(x, y)| x * y)
            .collect();
        ifft(&ab_fft, n)
    }
    
    /// Circular correlation unbinding: a ⊖ b
    /// For real vectors, this is equivalent to convolution with reversed a
    pub fn unbind(ab: &Array1<f32>, a: &Array1<f32>) -> Array1<f32> {
        let n = a.len();
        assert_eq!(n, ab.len(), "Vectors must have same dimension");
        
        // Reverse a for correlation
        let a_rev = reverse(a);
        let ab_fft = fft(&ab);
        let a_rev_fft = fft(&a_rev);
        let result_fft: Vec<Complex32> = ab_fft.iter().zip(a_rev_fft.iter())
            .map(|(x, y)| x * y)
            .collect();
        ifft(&result_fft, n)
    }
    
    /// Bundle multiple vectors by summing and normalizing
    pub fn bundle(vecs: &[&Array1<f32>]) -> Array1<f32> {
        if vecs.is_empty() {
            return Array1::zeros(0);
        }
        if vecs.len() == 1 {
            return normalize(vecs[0]);
        }
        
        let mut result = Array1::zeros(vecs[0].len());
        for v in vecs {
            result += v;
        }
        normalize(&result)
    }
    
    /// Cyclic shift permutation: π^k(v)
    pub fn permute(v: &Array1<f32>, k: usize) -> Array1<f32> {
        let n = v.len();
        let k = k % n;
        if k == 0 {
            return v.clone();
        }
        
        let mut result = Array1::zeros(n);
        for i in 0..n {
            result[i] = v[(i + k) % n];
        }
        result
    }
    
    /// Cosine similarity between two vectors
    pub fn similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Generate random normalized hypervector
    pub fn random(dim: usize, rng: &mut impl Rng) -> Array1<f32> {
        let mut v = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)));
        normalize(&v)
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

/// Reverse vector
fn reverse(v: &Array1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(v.len());
    for i in 0..v.len() {
        result[i] = v[v.len() - 1 - i];
    }
    result
}

/// Complex number for FFT
#[derive(Clone, Copy, Debug)]
struct Complex32 {
    re: f32,
    im: f32,
}

impl Complex32 {
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Add for Complex32 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Sub for Complex32 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// Cooley-Tukey FFT for power-of-2 sizes
fn fft(x: &Array1<f32>) -> Vec<Complex32> {
    let n = x.len();
    assert!(n.is_power_of_two(), "FFT requires power-of-2 size");
    
    // Convert to complex
    let mut a: Vec<Complex32> = x.iter()
        .map(|&v| Complex32::new(v, 0.0))
        .collect();
    
    // Bit-reversal permutation
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits(i as u32, bits) as usize;
        if i < j {
            a.swap(i, j);
        }
    }
    
    // FFT butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f32;
        let w = Complex32::new(angle.cos(), angle.sin());
        
        for i in (0..n).step_by(len) {
            let mut wk = Complex32::new(1.0, 0.0);
            for j in 0..half {
                let u = a[i + j];
                let v = a[i + j + half].mul(wk);
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                wk = wk.mul(w);
            }
        }
        len *= 2;
    }
    
    a
}

/// Inverse FFT
fn ifft(x: &[Complex32], n: usize) -> Array1<f32> {
    assert!(n.is_power_of_two(), "IFFT requires power-of-2 size");
    
    // Conjugate
    let mut a: Vec<Complex32> = x.iter()
        .map(|&c| Complex32::new(c.re, -c.im))
        .collect();
    
    // Forward FFT
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits(i as u32, bits) as usize;
        if i < j {
            a.swap(i, j);
        }
    }
    
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f32;
        let w = Complex32::new(angle.cos(), angle.sin());
        
        for i in (0..n).step_by(len) {
            let mut wk = Complex32::new(1.0, 0.0);
            for j in 0..half {
                let u = a[i + j];
                let v = a[i + j + half].mul(wk);
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                wk = wk.mul(w);
            }
        }
        len *= 2;
    }
    
    // Scale and convert to real
    let scale = 1.0 / n as f32;
    Array1::from_iter(a.iter().map(|&c| c.re * scale))
}

/// Reverse bits of integer
fn reverse_bits(x: u32, bits: u32) -> u32 {
    let mut result = 0;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    
    #[test]
    fn test_bind_unbind_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let v = VSAPrimitives::random(256, &mut rng);
        let k = VSAPrimitives::random(256, &mut rng);
        
        let bound = VSAPrimitives::bind(&v, &k);
        let recovered = VSAPrimitives::unbind(&bound, &k);
        
        let sim = VSAPrimitives::similarity(&v, &recovered);
        assert!(sim > 0.95, "Bind-unbind identity failed: similarity = {}", sim);
    }
    
    #[test]
    fn test_permute_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let v = VSAPrimitives::random(256, &mut rng);
        
        let p = VSAPrimitives::permute(&v, 0);
        assert_eq!(v, p);
        
        let p = VSAPrimitives::permute(&v, 256);
        assert_eq!(v, p);
    }
}
