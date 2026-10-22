use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write;

use quiver_storage::{
    VectorColumn,
    codec::{Reader, WriteExt},
};
use quiver_types::{Error, HnswParams, Metric, Result};

use crate::{Accept, Neighbor};

/// Hard cap on layer count; with m >= 2 reaching this is astronomically unlikely.
const MAX_LEVEL: usize = 16;

/// Approximate nearest-neighbour index over a layered proximity graph.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    params: HnswParams,
    metric: Metric,
    links: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    max_level: usize,
    level_mult: f64,
    rng: SplitMix64,
}

impl HnswIndex {
    pub fn new(params: HnswParams, metric: Metric) -> Self {
        Self {
            params,
            metric,
            links: Vec::new(),
            entry: None,
            max_level: 0,
            level_mult: 1.0 / (params.m as f64).ln(),
            rng: SplitMix64(params.seed),
        }
    }

    fn max_links(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m * 2
        } else {
            self.params.m
        }
    }

    fn random_level(&mut self) -> usize {
        // Uniform in (0, 1]; avoids ln(0).
        let u = 1.0 - self.rng.next_f64();
        ((-u.ln() * self.level_mult).floor() as usize).min(MAX_LEVEL)
    }

    /// The distance searches see, or while `building` the graph, the one
    /// the column builds indexes with.
    #[inline]
    fn dist(&self, query: &[f32], slot: u32, store: &VectorColumn, building: bool) -> f32 {
        if building {
            store.index_distance(self.metric, query, slot)
        } else {
            store.distance(self.metric, query, slot)
        }
    }

    /// Inserts `slot` (which must equal the number of nodes so far).
    pub fn add(&mut self, slot: u32, store: &VectorColumn) {
        assert_eq!(
            slot as usize,
            self.links.len(),
            "HNSW slots must be added in order"
        );

        let level = self.random_level();
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            self.entry = Some(slot);
            self.max_level = level;
            return;
        };

        let query = store.vector(slot);
        let query = &*query;
        let mut ep = Neighbor {
            slot: entry,
            dist: self.dist(query, entry, store, true),
        };

        for layer in (level + 1..=self.max_level).rev() {
            ep = self.greedy_closest(query, ep, layer, store, true);
        }

        let mut entry_points = vec![ep];

        for layer in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(
                query,
                &entry_points,
                self.params.ef_construction,
                layer,
                store,
                &|_| true,
                true,
            );
            let selected = self.select_neighbors(&candidates, self.params.m, store);
            self.links[slot as usize][layer] = selected.iter().map(|n| n.slot).collect();

            for nb in &selected {
                self.connect(nb.slot, slot, nb.dist, layer, store);
            }
            entry_points = candidates;
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(slot);
        }
    }

    fn connect(&mut self, from: u32, to: u32, dist: f32, layer: usize, store: &VectorColumn) {
        let max = self.max_links(layer);
        let metric = self.metric;
        let list = &mut self.links[from as usize][layer];
        if list.len() < max {
            list.push(to);
            return;
        }
        let base = store.vector(from);
        let mut candidates: Vec<Neighbor> = list
            .iter()
            .map(|&s| Neighbor {
                slot: s,
                dist: store.index_distance(metric, &base, s),
            })
            .chain(std::iter::once(Neighbor { slot: to, dist }))
            .collect();
        candidates.sort_unstable();
        let pruned = self.select_neighbors(&candidates, max, store);
        self.links[from as usize][layer] = pruned.iter().map(|n| n.slot).collect();
    }

    fn select_neighbors(&self, candidates: &[Neighbor], m: usize, store: &VectorColumn) -> Vec<Neighbor> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }
        let mut selected: Vec<Neighbor> = Vec::with_capacity(m);
        let mut pruned: Vec<Neighbor> = Vec::new();
        for &c in candidates {
            if selected.len() >= m {
                break;
            }
            let cv = store.vector(c.slot);
            let diverse = selected
                .iter()
                .all(|s| self.dist(&cv, s.slot, store, true) > c.dist);
            if diverse {
                selected.push(c);
            } else if !selected
                .iter()
                .any(|s| self.dist(&cv, s.slot, store, true) == 0.0)
            {
                // Exact duplicates of an already selected neighbour are dropped
                // rather than kept for backfill: a list full of duplicates would
                // otherwise crowd out every new node and disconnect the graph.
                pruned.push(c);
            }
        }
        let missing = m.saturating_sub(selected.len());
        selected.extend(pruned.into_iter().take(missing));
        selected
    }

    fn greedy_closest(
        &self,
        query: &[f32],
        mut best: Neighbor,
        layer: usize,
        store: &VectorColumn,
        building: bool,
    ) -> Neighbor {
        loop {
            let mut improved = false;
            for &nb in &self.links[best.slot as usize][layer] {
                let d = self.dist(query, nb, store, building);
                if d < best.dist {
                    best = Neighbor { slot: nb, dist: d };
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[Neighbor],
        ef: usize,
        layer: usize,
        store: &VectorColumn,
        accept: Accept<'_>,
        building: bool,
    ) -> Vec<Neighbor> {
        let mut visited = Visited::new(self.links.len());
        let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            if visited.insert(ep.slot) {
                candidates.push(Reverse(ep));
                if accept(ep.slot) {
                    results.push(ep);
                }
            }
        }

        while results.len() > ef {
            results.pop();
        }

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef && results.peek().is_some_and(|w| current.dist > w.dist) {
                break;
            }
            for &nb in &self.links[current.slot as usize][layer] {
                if !visited.insert(nb) {
                    continue;
                }
                let dist = self.dist(query, nb, store, building);
                let worst = results.peek().map_or(f32::INFINITY, |w| w.dist);
                if results.len() < ef || dist < worst {
                    let n = Neighbor { slot: nb, dist };
                    candidates.push(Reverse(n));
                    if accept(nb) {
                        results.push(n);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        results.into_sorted_vec()
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        store: &VectorColumn,
        accept: Accept<'_>,
    ) -> Vec<Neighbor> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let mut ep = Neighbor {
            slot: entry,
            dist: self.dist(query, entry, store, false),
        };
        for layer in (1..=self.max_level).rev() {
            ep = self.greedy_closest(query, ep, layer, store, false);
        }
        // A beam wider than the graph cannot find more, and `ef` sizes buffers.
        let ef = ef.unwrap_or(self.params.ef_search).max(k).min(self.links.len());
        let mut found = self.search_layer(query, &[ep], ef, 0, store, accept, false);
        found.truncate(k);
        found
    }

    pub(crate) fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        w.put_u8(self.entry.is_some() as u8)?;
        w.put_u32(self.entry.unwrap_or(0))?;
        w.put_u32(self.max_level as u32)?;
        w.put_u64(self.rng.0)?;
        w.put_u64(self.links.len() as u64)?;
        for node in &self.links {
            w.put_u8(node.len() as u8)?;
            for layer in node {
                w.put_u32(layer.len() as u32)?;
                for &n in layer {
                    w.put_u32(n)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn read_from(
        r: &mut Reader<'_>,
        params: HnswParams,
        metric: Metric,
        slots: usize,
    ) -> Result<Self> {
        let mut index = Self::new(params, metric);
        let has_entry = r.u8()? != 0;
        let entry = r.u32()?;
        index.max_level = r.u32()? as usize;
        index.rng = SplitMix64(r.u64()?);
        let nodes = r.u64()? as usize;
        if nodes != slots {
            return Err(Error::Corrupt(format!(
                "hnsw has {nodes} nodes but store has {slots} slots"
            )));
        }
        index.links.reserve_exact(nodes);
        for _ in 0..nodes {
            let levels = r.u8()? as usize;
            if levels == 0 || levels > MAX_LEVEL + 1 {
                return Err(Error::Corrupt(format!("hnsw node has {levels} layers")));
            }
            let mut node = Vec::with_capacity(levels);
            for _ in 0..levels {
                let count = r.u32()? as usize;
                let mut layer = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    let n = r.u32()?;
                    if n as usize >= nodes {
                        return Err(Error::Corrupt("hnsw edge points outside the graph".into()));
                    }
                    layer.push(n);
                }
                node.push(layer);
            }
            index.links.push(node);
        }
        // Searches index `links[n][layer]` without bounds checks of their own,
        // so every edge on a layer must lead to a node that has that layer.
        for node in &index.links {
            for (layer, neighbours) in node.iter().enumerate() {
                if neighbours.iter().any(|&n| index.links[n as usize].len() <= layer) {
                    return Err(Error::Corrupt("hnsw edge points to a missing layer".into()));
                }
            }
        }
        if has_entry != (nodes > 0) {
            return Err(Error::Corrupt(
                "hnsw entry point does not match node count".into(),
            ));
        }
        if has_entry {
            if entry as usize >= nodes {
                return Err(Error::Corrupt("hnsw entry point outside the graph".into()));
            }
            if index.links[entry as usize].len() != index.max_level + 1 {
                return Err(Error::Corrupt("hnsw entry point is not on the top layer".into()));
            }
            index.entry = Some(entry);
        } else if index.max_level != 0 {
            return Err(Error::Corrupt("empty hnsw graph has layers".into()));
        }
        Ok(index)
    }
}

/// Dense bitset of visited slots, allocated per search.
struct Visited(Vec<u64>);

impl Visited {
    fn new(n: usize) -> Self {
        Self(vec![0; n.div_ceil(64)])
    }

    /// Marks `slot` visited; returns `true` if it was not visited before.
    #[inline]
    fn insert(&mut self, slot: u32) -> bool {
        let (word, bit) = ((slot / 64) as usize, 1u64 << (slot % 64));
        let fresh = self.0[word] & bit == 0;
        self.0[word] |= bit;
        fresh
    }
}

/// Tiny deterministic PRNG (SplitMix64) for level assignment.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlatIndex, TopK};

    fn random_store(n: usize, dim: usize, seed: u64) -> VectorColumn {
        let mut rng = SplitMix64(seed);
        let mut store = VectorColumn::new(dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_f64() as f32 * 2.0 - 1.0).collect();
            store.push(&v);
        }
        store
    }

    #[test]
    fn recall_against_brute_force() {
        let (n, dim, k) = (3_000, 24, 10);
        let store = random_store(n, dim, 7);
        let mut hnsw = HnswIndex::new(HnswParams::default(), Metric::Euclidean);
        for slot in 0..n as u32 {
            hnsw.add(slot, &store);
        }
        let flat = FlatIndex::new(Metric::Euclidean);
        let queries = random_store(100, dim, 99);

        let mut hits = 0;
        for q in 0..100 {
            let query = &queries.vector(q);
            let truth: Vec<u32> = flat
                .search(query, k, &store, &|_| true)
                .iter()
                .map(|n| n.slot)
                .collect();
            let got = hnsw.search(query, k, Some(100), &store, &|_| true);
            hits += got.iter().filter(|n| truth.contains(&n.slot)).count();
        }
        let recall = hits as f64 / (100 * k) as f64;
        assert!(recall > 0.95, "recall@{k} too low: {recall}");
    }

    #[test]
    fn filtered_search_only_returns_accepted() {
        let store = random_store(1_000, 8, 3);
        let mut hnsw = HnswIndex::new(HnswParams::default(), Metric::Euclidean);
        for slot in 0..1_000 {
            hnsw.add(slot, &store);
        }
        let accept = |s: u32| s.is_multiple_of(50); // 2% selectivity
        let got = hnsw.search(&store.vector(1), 10, None, &store, &accept);
        assert_eq!(got.len(), 10);
        assert!(got.iter().all(|n| n.slot % 50 == 0));

        let mut truth = TopK::new(10);
        for s in (0..1_000).filter(|s| s % 50 == 0) {
            truth.push(Neighbor {
                slot: s,
                dist: Metric::Euclidean.distance(&store.vector(1), &store.vector(s)),
            });
        }
        let truth: Vec<u32> = truth.into_sorted().iter().map(|n| n.slot).collect();
        let overlap = got.iter().filter(|n| truth.contains(&n.slot)).count();
        assert!(overlap >= 9, "filtered recall too low: {overlap}/10");
    }

    #[test]
    fn roundtrip_persistence() {
        let store = random_store(200, 4, 1);
        let mut hnsw = HnswIndex::new(HnswParams::default(), Metric::Cosine);
        for slot in 0..200 {
            hnsw.add(slot, &store);
        }
        let mut buf = Vec::new();
        hnsw.write_to(&mut buf).unwrap();
        let restored =
            HnswIndex::read_from(&mut Reader::new(&buf), HnswParams::default(), Metric::Cosine, 200).unwrap();
        assert_eq!(restored.links, hnsw.links);
        assert_eq!(restored.entry, hnsw.entry);
        assert_eq!(restored.max_level, hnsw.max_level);
    }

    #[test]
    fn read_from_rejects_edges_to_missing_layers() {
        let store = random_store(50, 4, 1);
        let mut hnsw = HnswIndex::new(HnswParams::default(), Metric::Euclidean);
        for slot in 0..50 {
            hnsw.add(slot, &store);
        }
        // Link a layer-0-only node from an upper layer of the entry point.
        let low = (0..50).find(|&s| hnsw.links[s].len() == 1).unwrap() as u32;
        let entry = hnsw.entry.unwrap() as usize;
        hnsw.links[entry].push(vec![low]);
        hnsw.max_level += 1;
        let mut buf = Vec::new();
        hnsw.write_to(&mut buf).unwrap();
        let res = HnswIndex::read_from(
            &mut Reader::new(&buf),
            HnswParams::default(),
            Metric::Euclidean,
            50,
        );
        assert!(matches!(res, Err(Error::Corrupt(_))));
    }

    #[test]
    fn duplicate_vectors_do_not_disconnect_the_graph() {
        // 1800 copies of one vector plus 200 distinct points.
        let mut rng = SplitMix64(0);
        let mut store = VectorColumn::new(4);
        let mut hnsw = HnswIndex::new(HnswParams::default(), Metric::Euclidean);
        for i in 0..2_000u32 {
            let v: Vec<f32> = if i % 10 != 0 {
                vec![0.5; 4]
            } else {
                (0..4).map(|_| rng.next_f64() as f32).collect()
            };
            store.push(&v);
            hnsw.add(i, &store);
        }
        let filtered = hnsw.search(&[0.5; 4], 10, None, &store, &|s| s % 10 == 0);
        assert_eq!(filtered.len(), 10);
        let self_found = (0..2_000u32)
            .step_by(10)
            .filter(|&s| hnsw.search(&store.vector(s), 1, Some(200), &store, &|_| true)[0].dist == 0.0)
            .count();
        assert_eq!(self_found, 200);
    }

    #[test]
    fn empty_index_returns_nothing() {
        let store = VectorColumn::new(2);
        let hnsw = HnswIndex::new(HnswParams::default(), Metric::Cosine);
        assert!(hnsw.search(&[1.0, 0.0], 5, None, &store, &|_| true).is_empty());
    }
}
