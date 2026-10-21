use std::collections::HashMap;

use quiver_types::{SparseModifier, SparseVector};

use crate::{Accept, Neighbor, TopK};

/// Rebuild once removed vectors leave at least this many stale postings...
const REBUILD_MIN_STALE: usize = 4_096;
/// ...making up at least this fraction of all postings.
const REBUILD_RATIO: usize = 2;

/// An inverted index over one sparse column: for each component index,
/// the slots whose vector has a non-zero value there. Searches are exact.
///
/// It is rebuilt from the store on open rather than persisted.
#[derive(Debug, Clone, Default)]
pub struct SparseIndex {
    postings: HashMap<u32, Vec<(u32, f32)>>,
    /// Live vectors with a non-zero value at each index.
    df: HashMap<u32, u32>,
    /// Live non-empty vectors.
    docs: usize,
    postings_len: usize,
    /// Postings of removed vectors, still in the lists until a rebuild.
    stale: usize,
}

impl SparseIndex {
    /// Indexes the vectors of the slots `alive` accepts.
    pub fn build(vectors: &[SparseVector], alive: impl Fn(u32) -> bool) -> Self {
        let mut index = Self::default();
        for (slot, v) in vectors.iter().enumerate() {
            let slot = slot as u32;
            if alive(slot) {
                index.add(slot, v);
            }
        }
        index
    }

    /// Indexes `v`, the vector of `slot`.
    pub fn add(&mut self, slot: u32, v: &SparseVector) {
        if v.is_empty() {
            return;
        }
        for (&i, &x) in v.indices.iter().zip(&v.values) {
            self.postings.entry(i).or_default().push((slot, x));
            *self.df.entry(i).or_default() += 1;
        }
        self.docs += 1;
        self.postings_len += v.len();
    }

    /// Forgets `v`, the vector of a slot that was just deleted. Its postings
    /// stay until a rebuild; searches skip them as dead slots.
    pub fn remove(&mut self, v: &SparseVector) {
        if v.is_empty() {
            return;
        }
        for i in &v.indices {
            if let Some(n) = self.df.get_mut(i) {
                *n -= 1;
                if *n == 0 {
                    self.df.remove(i);
                }
            }
        }
        self.docs -= 1;
        self.stale += v.len();
    }

    /// Whether enough postings are stale that a rebuild pays off.
    pub fn needs_rebuild(&self) -> bool {
        self.stale >= REBUILD_MIN_STALE && self.stale * REBUILD_RATIO >= self.postings_len
    }

    /// Inverse document frequency of component `i`.
    fn idf(&self, i: u32) -> f32 {
        let n = self.docs as f64;
        let df = f64::from(self.df.get(&i).copied().unwrap_or(0));
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
    }

    /// The `k` accepted slots scoring highest against `query` (a sorted
    /// sparse vector), best first. Scores are dot products, with `modifier`
    /// applied to the query weights; as distances they are negated.
    pub fn search(
        &self,
        query: &SparseVector,
        k: usize,
        modifier: SparseModifier,
        accept: Accept<'_>,
    ) -> Vec<Neighbor> {
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for (&i, &q) in query.indices.iter().zip(&query.values) {
            let Some(list) = self.postings.get(&i) else {
                continue;
            };
            let weight = match modifier {
                SparseModifier::None => q,
                SparseModifier::Idf => q * self.idf(i),
            };
            for &(slot, x) in list {
                *scores.entry(slot).or_default() += weight * x;
            }
        }
        let k = k.min(scores.len());
        if k == 0 {
            return Vec::new();
        }
        let mut top = TopK::new(k);
        for (slot, score) in scores {
            if accept(slot) {
                top.push(Neighbor { slot, dist: -score });
            }
        }
        top.into_sorted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(pairs: &[(u32, f32)]) -> SparseVector {
        let mut v = SparseVector::new(
            pairs.iter().map(|p| p.0).collect(),
            pairs.iter().map(|p| p.1).collect(),
        );
        v.normalize().unwrap();
        v
    }

    #[test]
    fn scores_are_exact_dot_products() {
        let vectors = vec![
            v(&[(1, 1.0), (2, 1.0)]),
            v(&[(2, 3.0)]),
            v(&[]),
            v(&[(1, 0.5), (7, 2.0)]),
        ];
        let index = SparseIndex::build(&vectors, |_| true);
        let q = v(&[(1, 2.0), (2, 1.0)]);
        let got = index.search(&q, 10, SparseModifier::None, &|_| true);
        let ranked: Vec<(u32, f32)> = got.iter().map(|n| (n.slot, -n.dist)).collect();
        assert_eq!(ranked, [(0, 3.0), (1, 3.0), (3, 1.0)]);
        for n in &got {
            assert_eq!(-n.dist, vectors[n.slot as usize].dot(&q));
        }
        let filtered = index.search(&q, 10, SparseModifier::None, &|s| s == 3);
        assert_eq!(filtered.len(), 1);
        assert!(
            index
                .search(&v(&[(99, 1.0)]), 10, SparseModifier::None, &|_| true)
                .is_empty()
        );
    }

    #[test]
    fn idf_favours_rare_components() {
        // Component 1 is everywhere, component 2 in one vector.
        let mut vectors: Vec<SparseVector> = (0..10).map(|_| v(&[(1, 1.0)])).collect();
        vectors.push(v(&[(1, 1.0), (2, 1.0)]));
        vectors.push(v(&[(1, 2.0)]));
        let index = SparseIndex::build(&vectors, |_| true);
        let q = v(&[(1, 1.0), (2, 1.0)]);
        // Plain: the double weight on the common component wins.
        let plain = index.search(&q, 1, SparseModifier::None, &|_| true);
        assert_eq!(plain[0].slot, 10, "ties broken by slot");
        let idf = index.search(&v(&[(1, 2.0), (2, 1.0)]), 2, SparseModifier::Idf, &|_| true);
        assert_eq!(idf[0].slot, 10, "the rare match outranks the common one");
    }

    #[test]
    fn removals_update_frequencies_and_ask_for_rebuilds() {
        let vectors: Vec<SparseVector> = (0..5_000).map(|i| v(&[(i % 3, 1.0)])).collect();
        let mut index = SparseIndex::build(&vectors, |_| true);
        let before = index.idf(0);
        for v in &vectors[..4_500] {
            index.remove(v);
        }
        assert_eq!(index.docs, 500);
        assert_ne!(index.idf(0), before);
        assert!(index.needs_rebuild());
        let rebuilt = SparseIndex::build(&vectors, |s| s >= 4_500);
        assert!(!rebuilt.needs_rebuild());
        assert_eq!(rebuilt.docs, index.docs);
        assert_eq!(rebuilt.df, index.df);
    }
}
