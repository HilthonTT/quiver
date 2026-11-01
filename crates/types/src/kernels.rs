//! Distance kernels: an `f32` query against stored vectors of `f32`, `f16`
//! (as raw bits) or `u8` components.
//!
//! Each kernel has a portable version, written in eight independent lanes
//! so that compilers vectorize it for the baseline instruction set (SSE2 on
//! x86-64, NEON on AArch64), and on x86-64 an AVX2 + FMA version (with F16C
//! for `f16`) chosen at run time when the CPU has those features. The two
//! versions add in different orders, so their results may differ in the
//! last bits; one process always uses the same one.

/// Stored components that a query is compared with.
pub(crate) trait Element: Copy {
    fn to_f32(self) -> f32;
}

impl Element for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// `f16` bits.
impl Element for u16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        half::f16::from_bits(self).to_f32()
    }
}

impl Element for u8 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}

/// Below this many components the SIMD versions are not worth the call.
#[cfg(target_arch = "x86_64")]
const SIMD_MIN_LEN: usize = 16;

/// Dispatches to the AVX2 kernel `$simd` when the CPU (and length) allows,
/// else to the portable `$portable`.
macro_rules! dispatch {
    ($a:ident, $b:ident, $simd:path, $portable:expr, $($feature:tt),+) => {{
        debug_assert_eq!($a.len(), $b.len());
        #[cfg(target_arch = "x86_64")]
        if $a.len() >= SIMD_MIN_LEN && $(std::arch::is_x86_feature_detected!($feature))&&+ {
            // SAFETY: the CPU has the features the kernel is compiled for.
            return unsafe { $simd($a, $b) };
        }
        $portable
    }};
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    dispatch!(a, b, x86::dot_f32, portable::dot(a, b), "avx2", "fma")
}

pub(crate) fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    dispatch!(a, b, x86::l2_f32, portable::l2(a, b), "avx2", "fma")
}

pub(crate) fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    dispatch!(a, b, x86::l1_f32, portable::l1(a, b), "avx2", "fma")
}

pub(crate) fn hamming(a: &[f32], b: &[f32]) -> f32 {
    dispatch!(a, b, x86::hamming_f32, portable::hamming(a, b), "avx2", "fma")
}

pub(crate) fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
    dispatch!(a, b, x86::dot_f16, portable::dot(a, b), "avx2", "fma", "f16c")
}

pub(crate) fn squared_euclidean_f16(a: &[f32], b: &[u16]) -> f32 {
    dispatch!(a, b, x86::l2_f16, portable::l2(a, b), "avx2", "fma", "f16c")
}

pub(crate) fn manhattan_f16(a: &[f32], b: &[u16]) -> f32 {
    dispatch!(a, b, x86::l1_f16, portable::l1(a, b), "avx2", "fma", "f16c")
}

pub(crate) fn hamming_f16(a: &[f32], b: &[u16]) -> f32 {
    dispatch!(
        a,
        b,
        x86::hamming_f16,
        portable::hamming(a, b),
        "avx2",
        "fma",
        "f16c"
    )
}

pub(crate) fn dot_u8(a: &[f32], b: &[u8]) -> f32 {
    dispatch!(a, b, x86::dot_u8, portable::dot(a, b), "avx2", "fma")
}

pub(crate) fn squared_euclidean_u8(a: &[f32], b: &[u8]) -> f32 {
    dispatch!(a, b, x86::l2_u8, portable::l2(a, b), "avx2", "fma")
}

pub(crate) fn manhattan_u8(a: &[f32], b: &[u8]) -> f32 {
    dispatch!(a, b, x86::l1_u8, portable::l1(a, b), "avx2", "fma")
}

pub(crate) fn hamming_u8(a: &[f32], b: &[u8]) -> f32 {
    dispatch!(a, b, x86::hamming_u8, portable::hamming(a, b), "avx2", "fma")
}

pub(crate) mod portable {
    use super::Element;

    const LANES: usize = 8;

    /// `Σ f(a[i], b[i])` in eight lanes.
    #[inline(always)]
    fn sum<T: Element>(a: &[f32], b: &[T], f: impl Fn(f32, f32) -> f32) -> f32 {
        let mut acc = [0.0f32; LANES];
        let (ca, ra) = a.as_chunks::<LANES>();
        let (cb, rb) = b.as_chunks::<LANES>();
        for (x, y) in ca.iter().zip(cb) {
            for i in 0..LANES {
                acc[i] += f(x[i], y[i].to_f32());
            }
        }
        let tail: f32 = ra.iter().zip(rb).map(|(&x, &y)| f(x, y.to_f32())).sum();
        acc.iter().sum::<f32>() + tail
    }

    #[inline]
    pub(crate) fn dot<T: Element>(a: &[f32], b: &[T]) -> f32 {
        sum(a, b, |x, y| x * y)
    }

    #[inline]
    pub(crate) fn l2<T: Element>(a: &[f32], b: &[T]) -> f32 {
        sum(a, b, |x, y| {
            let d = x - y;
            d * d
        })
    }

    #[inline]
    pub(crate) fn l1<T: Element>(a: &[f32], b: &[T]) -> f32 {
        sum(a, b, |x, y| (x - y).abs())
    }

    #[inline]
    pub(crate) fn hamming<T: Element>(a: &[f32], b: &[T]) -> f32 {
        a.iter().zip(b).filter(|&(&x, &y)| x != y.to_f32()).count() as f32
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Element;

    /// Sum of the eight lanes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
        _mm_cvtss_f32(s)
    }

    /// Eight `f32` from `p`.
    ///
    /// # Safety
    /// `p` must be valid for reading eight `f32`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_f32(p: *const f32) -> __m256 {
        // SAFETY: the caller guarantees eight readable floats.
        unsafe { _mm256_loadu_ps(p) }
    }

    /// Eight `f16` from `p`, widened.
    ///
    /// # Safety
    /// `p` must be valid for reading eight `u16`.
    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn load_f16(p: *const u16) -> __m256 {
        // SAFETY: the caller guarantees eight readable halves (16 bytes).
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p.cast())) }
    }

    /// Eight `u8` from `p`, widened.
    ///
    /// # Safety
    /// `p` must be valid for reading eight `u8`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_u8(p: *const u8) -> __m256 {
        // SAFETY: the caller guarantees eight readable bytes.
        unsafe { _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p.cast()))) }
    }

    /// Generates the four kernels for one element type. Each runs four
    /// accumulators of eight lanes over 32 components at a time, then eight
    /// at a time, then finishes the tail in scalar code.
    macro_rules! kernels {
        ($elem:ty, $load:ident, $features:literal, $dot:ident, $l2:ident, $l1:ident, $ham:ident) => {
            /// # Safety
            /// The CPU must have the enabled features.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn $dot(a: &[f32], b: &[$elem]) -> f32 {
                // SAFETY: indices stay below `n`, the shorter length.
                unsafe {
                    accumulate(
                        a,
                        b,
                        |x, y, acc| _mm256_fmadd_ps(x, y, acc),
                        |x, y| x * y.to_f32(),
                    )
                }
            }

            /// # Safety
            /// The CPU must have the enabled features.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn $l2(a: &[f32], b: &[$elem]) -> f32 {
                // SAFETY: as above.
                unsafe {
                    accumulate(
                        a,
                        b,
                        |x, y, acc| {
                            let d = _mm256_sub_ps(x, y);
                            _mm256_fmadd_ps(d, d, acc)
                        },
                        |x, y| {
                            let d = x - y.to_f32();
                            d * d
                        },
                    )
                }
            }

            /// # Safety
            /// The CPU must have the enabled features.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn $l1(a: &[f32], b: &[$elem]) -> f32 {
                let sign = _mm256_set1_ps(-0.0);
                // SAFETY: as above.
                unsafe {
                    accumulate(
                        a,
                        b,
                        |x, y, acc| _mm256_add_ps(acc, _mm256_andnot_ps(sign, _mm256_sub_ps(x, y))),
                        |x, y| (x - y.to_f32()).abs(),
                    )
                }
            }

            /// # Safety
            /// The CPU must have the enabled features.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn $ham(a: &[f32], b: &[$elem]) -> f32 {
                let n = a.len().min(b.len());
                let mut count = 0u32;
                let mut i = 0;
                while i + 8 <= n {
                    // SAFETY: `i + 8 <= n`.
                    let (x, y) = unsafe { (load_f32(a.as_ptr().add(i)), $load(b.as_ptr().add(i))) };
                    // Unordered-or-not-equal, so NaN differs from everything, as `!=`.
                    let ne = _mm256_cmp_ps::<_CMP_NEQ_UQ>(x, y);
                    count += (_mm256_movemask_ps(ne) as u32).count_ones();
                    i += 8;
                }
                let tail = a[i..n]
                    .iter()
                    .zip(&b[i..n])
                    .filter(|&(&x, &y)| x != y.to_f32())
                    .count();
                (count as usize + tail) as f32
            }

            /// `Σ step` over the vectors, with `scalar` for the tail.
            ///
            /// # Safety
            /// The CPU must have the enabled features.
            #[target_feature(enable = $features)]
            #[inline]
            unsafe fn accumulate(
                a: &[f32],
                b: &[$elem],
                step: impl Fn(__m256, __m256, __m256) -> __m256,
                scalar: impl Fn(f32, $elem) -> f32,
            ) -> f32 {
                let n = a.len().min(b.len());
                let (pa, pb) = (a.as_ptr(), b.as_ptr());
                let mut acc = [_mm256_setzero_ps(); 4];
                let mut i = 0;
                // SAFETY (all loads): each reads eight components at an
                // index `j` with `j + 8 <= n`.
                while i + 32 <= n {
                    for (k, acc) in acc.iter_mut().enumerate() {
                        let j = i + 8 * k;
                        let (x, y) = unsafe { (load_f32(pa.add(j)), $load(pb.add(j))) };
                        *acc = step(x, y, *acc);
                    }
                    i += 32;
                }
                while i + 8 <= n {
                    let (x, y) = unsafe { (load_f32(pa.add(i)), $load(pb.add(i))) };
                    acc[0] = step(x, y, acc[0]);
                    i += 8;
                }
                let total = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
                let tail: f32 = a[i..n]
                    .iter()
                    .zip(&b[i..n])
                    .map(|(&x, &y)| scalar(x, y))
                    .sum();
                hsum(total) + tail
            }
        };
    }

    mod f32_kernels {
        use super::*;
        kernels!(f32, load_f32, "avx2,fma", dot, l2, l1, hamming);
    }
    mod f16_kernels {
        use super::*;
        kernels!(u16, load_f16, "avx2,fma,f16c", dot, l2, l1, hamming);
    }
    mod u8_kernels {
        use super::*;
        kernels!(u8, load_u8, "avx2,fma", dot, l2, l1, hamming);
    }

    pub(super) use f16_kernels::{dot as dot_f16, hamming as hamming_f16, l1 as l1_f16, l2 as l2_f16};
    pub(super) use f32_kernels::{dot as dot_f32, hamming as hamming_f32, l1 as l1_f32, l2 as l2_f32};
    pub(super) use u8_kernels::{dot as dot_u8, hamming as hamming_u8, l1 as l1_u8, l2 as l2_u8};
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs()))
    }

    /// Every length around the 8- and 32-component blocks and the SIMD
    /// threshold, so each loop and the tails are exercised.
    fn lengths() -> impl Iterator<Item = usize> {
        (0..=70).chain([127, 128, 129, 768, 1536])
    }

    #[test]
    fn f32_kernels_match_the_portable_ones() {
        let mut rng = Rng(7);
        for n in lengths() {
            let a: Vec<f32> = (0..n).map(|_| rng.next()).collect();
            let mut b: Vec<f32> = (0..n).map(|_| rng.next()).collect();
            // Some equal components for Hamming.
            for i in (0..n).step_by(3) {
                b[i] = a[i];
            }
            assert!(close(dot(&a, &b), portable::dot(&a, &b)), "dot {n}");
            assert!(close(squared_euclidean(&a, &b), portable::l2(&a, &b)), "l2 {n}");
            assert!(close(manhattan(&a, &b), portable::l1(&a, &b)), "l1 {n}");
            assert_eq!(hamming(&a, &b), portable::hamming(&a, &b), "hamming {n}");
            assert_eq!(squared_euclidean(&a, &a), 0.0, "self distance {n}");
        }
    }

    #[test]
    fn f16_and_u8_kernels_match_decoding_first() {
        let mut rng = Rng(11);
        for n in lengths() {
            let q: Vec<f32> = (0..n).map(|_| rng.next() * 100.0).collect();
            let halves: Vec<u16> = (0..n)
                .map(|_| half::f16::from_f32(rng.next()).to_bits())
                .collect();
            let bytes: Vec<u8> = (0..n).map(|_| ((rng.next() + 1.0) * 127.0) as u8).collect();
            let h: Vec<f32> = halves.iter().map(|&x| x.to_f32()).collect();
            let u: Vec<f32> = bytes.iter().map(|&x| f32::from(x)).collect();

            assert!(close(dot_f16(&q, &halves), portable::dot(&q, &h)), "dot f16 {n}");
            assert!(
                close(squared_euclidean_f16(&q, &halves), portable::l2(&q, &h)),
                "l2 f16 {n}"
            );
            assert!(
                close(manhattan_f16(&q, &halves), portable::l1(&q, &h)),
                "l1 f16 {n}"
            );
            assert_eq!(hamming_f16(&h, &halves), 0.0, "hamming f16 {n}");
            assert_eq!(
                hamming_f16(&q, &halves),
                portable::hamming(&q, &h),
                "hamming f16 {n}"
            );

            assert!(close(dot_u8(&q, &bytes), portable::dot(&q, &u)), "dot u8 {n}");
            assert!(
                close(squared_euclidean_u8(&q, &bytes), portable::l2(&q, &u)),
                "l2 u8 {n}"
            );
            assert!(close(manhattan_u8(&q, &bytes), portable::l1(&q, &u)), "l1 u8 {n}");
            assert_eq!(hamming_u8(&u, &bytes), 0.0, "hamming u8 {n}");
            assert_eq!(squared_euclidean_u8(&u, &bytes), 0.0, "self distance u8 {n}");
        }
    }

    #[test]
    fn hamming_treats_nan_as_different() {
        let a = vec![f32::NAN; 24];
        assert_eq!(hamming(&a, &a), 24.0);
        assert_eq!(portable::hamming(&a, &a), 24.0);
    }
}
