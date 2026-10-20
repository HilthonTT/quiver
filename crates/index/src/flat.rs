use rayon::prelude::*;

use quiver_storage::VectorStore;
use quiver_types::Metric;

use crate::{Accept, Neighbor, TopK};

/// Below this many slots a sequential scan beats rayon's scheduling overhead.
const PARALLEL_THRESHOLD: usize = 16_384;
const CHUNK: usize = 4_096;

#[derive(Debug, Clone)]
pub struct FlatIndex {
    metric: Metric,
}

impl FlatIndex {
    pub fn new(metric: Metric) -> Self {
        Self { metric }
    }

    pub fn search(&self, query: &[f32], k: usize, store: &VectorStore, accept: Accept<'_>) -> Vec<Neighbor> {
        let n = store.slots();
        let scan = |range: std::ops::Range<usize>| {
            let mut top = TopK::new(k);
            for slot in range.map(|s| s as u32) {
                if accept(slot) {
                    let dist = self.metric.distance(query, store.vector(slot));
                    top.push(Neighbor { slot, dist });
                }
            }
            top
        };

        if n < PARALLEL_THRESHOLD {
            return scan(0..n).into_sorted();
        }

        (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| scan(c * CHUNK..((c + 1) * CHUNK).min(n)))
            .reduce(|| TopK::new(k), TopK::merge)
            .into_sorted()
    }
}
