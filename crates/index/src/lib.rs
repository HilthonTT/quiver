mod flat;
mod hnsw;
mod kdtree;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use vectordb_storage::VectorStore;
use vectordb_storage::codec::{Reader, WriteExt};
use vectordb_types::{Error, IndexConfig, Metric, Result};

pub use flat::FlatIndex;
pub use hnsw::HnswIndex;
pub use kdtree::KdTreeIndex;

/// A candidate result: a slot and its distance to the query.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// Slot in the [`VectorStore`].
    pub slot: u32,
    /// Distance to the query (lower is closer).
    pub dist: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.slot.cmp(&other.slot))
    }
}

/// Bounded max-heap that keeps the `k` closest neighbours seen so far.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Neighbor>,
}

impl TopK {
    /// An empty set that will hold at most `k` neighbours.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Offers a candidate; keeps it only if it is among the `k` closest.
    #[inline]
    pub fn push(&mut self, n: Neighbor) {
        if self.heap.len() < self.k {
            self.heap.push(n);
        } else if let Some(worst) = self.heap.peek() {
            if n < *worst {
                self.heap.pop();
                self.heap.push(n);
            }
        }
    }

    /// Whether `k` neighbours have been collected.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Distance of the farthest kept neighbour (∞ while not full).
    #[inline]
    pub fn worst(&self) -> f32 {
        if self.is_full() {
            self.heap.peek().map_or(f32::INFINITY, |n| n.dist)
        } else {
            f32::INFINITY
        }
    }

    /// Combines two partial results.
    pub fn merge(mut self, other: TopK) -> TopK {
        for n in other.heap {
            self.push(n);
        }
        self
    }

    /// Results sorted from closest to farthest.
    pub fn into_sorted(self) -> Vec<Neighbor> {
        self.heap.into_sorted_vec()
    }
}

/// Predicate deciding whether a slot may appear in results.
pub type Accept<'a> = &'a (dyn Fn(u32) -> bool + Sync);

/// A collection's index. An enum rather than a trait object so it can be
/// persisted and restored without dynamic-dispatch tricks.
#[derive(Debug, Clone)]
pub enum Index {
    /// Exact brute-force search.
    Flat(FlatIndex),
    /// Approximate HNSW graph search.
    Hnsw(HnswIndex),
    /// Exact k-d tree search.
    KdTree(KdTreeIndex),
}

const TAG_FLAT: u8 = 0;
const TAG_HNSW: u8 = 1;
const TAG_KDTREE: u8 = 2;

impl Index {
    /// An empty index.
    pub fn new(config: &IndexConfig, metric: Metric) -> Self {
        match config {
            IndexConfig::Flat => Index::Flat(FlatIndex::new(metric)),
            IndexConfig::Hnsw(p) => Index::Hnsw(HnswIndex::new(*p, metric)),
            IndexConfig::KdTree(p) => Index::KdTree(KdTreeIndex::new(*p, metric)),
        }
    }

    /// Builds an index over every slot of `store`.
    pub fn build(config: &IndexConfig, metric: Metric, store: &VectorStore) -> Self {
        let mut index = Self::new(config, metric);
        index.add_batch(0..store.slots() as u32, store);
        index
    }

    /// Makes the record in `slot` searchable. Slots must be added in order.
    pub fn add(&mut self, slot: u32, store: &VectorStore) {
        self.add_batch(slot..slot + 1, store);
    }

    /// Adds a run of consecutive slots. Lets indexes amortise rebalancing.
    pub fn add_batch(&mut self, slots: std::ops::Range<u32>, store: &VectorStore) {
        match self {
            Index::Flat(_) => {}
            Index::Hnsw(h) => slots.for_each(|s| h.add(s, store)),
            Index::KdTree(t) => t.add_batch(slots, store),
        }
    }

    /// The `k` accepted slots closest to `query`, closest first.
    ///
    /// `ef` is the HNSW beam width (ignored by exact indexes).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        store: &VectorStore,
        accept: Accept<'_>,
    ) -> Vec<Neighbor> {
        // `k` comes straight from API callers; never size buffers beyond the data.
        let k = k.min(store.slots());
        if k == 0 {
            return Vec::new();
        }
        match self {
            Index::Flat(f) => f.search(query, k, store, accept),
            Index::Hnsw(h) => h.search(query, k, ef, store, accept),
            Index::KdTree(t) => t.search(query, k, store, accept),
        }
    }

    /// Serializes the index (but not the vectors) to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let written = match self {
            Index::Flat(_) => out.put_u8(TAG_FLAT),
            Index::Hnsw(h) => out.put_u8(TAG_HNSW).and_then(|()| h.write_to(&mut out)),
            Index::KdTree(t) => out.put_u8(TAG_KDTREE).and_then(|()| t.write_to(&mut out)),
        };
        written.expect("writing to a Vec cannot fail");
        out
    }

    /// Restores an index serialized by [`Index::to_bytes`] for `store`.
    pub fn from_bytes(
        bytes: &[u8],
        config: &IndexConfig,
        metric: Metric,
        store: &VectorStore,
    ) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let slots = store.slots();
        let index = match (r.u8()?, config) {
            (TAG_FLAT, IndexConfig::Flat) => Index::Flat(FlatIndex::new(metric)),
            (TAG_HNSW, IndexConfig::Hnsw(p)) => {
                Index::Hnsw(HnswIndex::read_from(&mut r, *p, metric, slots)?)
            }
            (TAG_KDTREE, IndexConfig::KdTree(p)) => Index::KdTree(KdTreeIndex::read_from(
                &mut r,
                *p,
                metric,
                slots,
                store.dim(),
            )?),
            (tag, _) => {
                return Err(Error::Corrupt(format!(
                    "stored index (tag {tag}) does not match the configured {} index",
                    config.kind()
                )));
            }
        };
        if !r.is_empty() {
            return Err(Error::Corrupt("trailing bytes after index".into()));
        }
        Ok(index)
    }
}

/// Deterministic pseudo-random vectors for tests.
#[cfg(test)]
pub(crate) fn random_store(n: usize, dim: usize, seed: u64) -> VectorStore {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut store = VectorStore::new(dim);
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| next() * 2.0 - 1.0).collect();
        store.push(i.to_string(), &v, None);
    }
    store
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectordb_types::{HnswParams, KdTreeParams};

    #[test]
    fn topk_keeps_closest() {
        let mut t = TopK::new(2);
        for (slot, dist) in [(0, 5.0), (1, 1.0), (2, 3.0), (3, 0.5)] {
            t.push(Neighbor { slot, dist });
        }
        assert!(t.is_full());
        assert_eq!(t.worst(), 1.0);
        let got: Vec<u32> = t.into_sorted().iter().map(|n| n.slot).collect();
        assert_eq!(got, vec![3, 1]);
    }

    #[test]
    fn bytes_roundtrip_for_every_index() {
        let store = random_store(300, 4, 1);
        for config in [
            IndexConfig::Flat,
            IndexConfig::Hnsw(HnswParams::default()),
            IndexConfig::KdTree(KdTreeParams::default()),
        ] {
            let index = Index::build(&config, Metric::Euclidean, &store);
            let restored =
                Index::from_bytes(&index.to_bytes(), &config, Metric::Euclidean, &store).unwrap();
            let q = store.vector(7);
            let a = index.search(q, 5, None, &store, &|_| true);
            let b = restored.search(q, 5, None, &store, &|_| true);
            assert_eq!(a, b, "{config:?}");
        }
    }

    #[test]
    fn huge_k_and_ef_do_not_allocate_unbounded() {
        let store = random_store(50, 3, 2);
        for config in [
            IndexConfig::Flat,
            IndexConfig::Hnsw(HnswParams::default()),
            IndexConfig::KdTree(KdTreeParams::default()),
        ] {
            let index = Index::build(&config, Metric::Euclidean, &store);
            let got = index.search(
                store.vector(0),
                usize::MAX,
                Some(usize::MAX),
                &store,
                &|_| true,
            );
            assert_eq!(got.len(), 50, "{config:?}");
        }
    }

    #[test]
    fn from_bytes_rejects_mismatched_config() {
        let store = random_store(10, 2, 1);
        let bytes = Index::build(&IndexConfig::Flat, Metric::Cosine, &store).to_bytes();
        let hnsw = IndexConfig::Hnsw(HnswParams::default());
        assert!(Index::from_bytes(&bytes, &hnsw, Metric::Cosine, &store).is_err());
    }
}
