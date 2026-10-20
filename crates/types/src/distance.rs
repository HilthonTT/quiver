use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    #[default]
    Cosine,
    Euclidean,
    #[serde(alias = "dot")]
    DotProduct,
    #[serde(alias = "l1")]
    Manhattan,
    Hamming,
}

impl Metric {
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => 1.0 - dot(a, b),
            Metric::Euclidean => squared_euclidean(a, b).sqrt(),
            Metric::DotProduct => -dot(a, b),
            Metric::Manhattan => manhattan(a, b),
            Metric::Hamming => hamming(a, b),
        }
    }

    pub const ALL: [Metric; 5] = [
        Metric::Cosine,
        Metric::Euclidean,
        Metric::DotProduct,
        Metric::Manhattan,
        Metric::Hamming,
    ];

    pub fn prepare(self, v: &mut [f32]) -> bool {
        if self != Metric::Cosine {
            return true;
        }
        // In f64 so that tiny or huge (but finite, non-zero) vectors neither
        // underflow to a zero norm nor overflow to an infinite one.
        let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return false;
        }

        v.iter_mut().for_each(|x| *x = (f64::from(*x) / norm) as f32);
        v.iter().all(|x| x.is_finite())
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::DotProduct => "dot_product",
            Metric::Manhattan => "manhattan",
            Metric::Hamming => "hamming",
        }
    }
}

const LANES: usize = 8;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = [0.0f32; LANES];
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let tail: f32 = ra
        .iter()
        .zip(rb)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    acc.iter().sum::<f32>() + tail
}

#[inline]
pub fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            acc[i] += (x[i] - y[i]).abs();
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).sum();
    acc.iter().sum::<f32>() + tail
}

#[inline]
pub fn hamming(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_prepare_accepts_tiny_and_huge_vectors() {
        for scale in [1e-30f32, 1e-40, 2e19, 3e38] {
            let mut v = vec![scale, 0.0, scale];
            assert!(Metric::Cosine.prepare(&mut v), "{scale}");
            assert!((dot(&v, &v) - 1.0).abs() < 1e-6, "{scale}: {v:?}");
        }
        let mut zero = vec![0.0f32; 3];
        assert!(!Metric::Cosine.prepare(&mut zero));
    }
}
