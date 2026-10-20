use std::io::Write;

use quiver_storage::VectorStore;
use quiver_storage::codec::{Reader, WriteExt};
use quiver_types::{Error, KdTreeParams, Metric, Result};

use crate::{Accept, Neighbor, TopK};

const NIL: u32 = u32::MAX;
/// Don't rebalance more often than every this many inserts.
const MIN_INSERTS_BETWEEN_REBUILDS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Node {
    axis: u32,
    left: u32,
    right: u32,
}

#[derive(Debug, Clone)]
pub struct KdTreeIndex {
    params: KdTreeParams,
    metric: Metric,
    nodes: Vec<Node>,
    root: u32,
    depth: usize,
    inserts_since_rebuild: usize,
}

fn plane_bound(metric: Metric, diff: f32) -> f32 {
    match metric {
        Metric::Euclidean | Metric::Manhattan => diff.abs(),
        // Unit vectors: 1 - cos = ‖a - b‖² / 2 ≥ diff² / 2.
        Metric::Cosine => diff * diff / 2.0,
        // Far-side points differ from the query on this axis (unless the
        // query sits exactly on the plane).
        Metric::Hamming => {
            if diff == 0.0 {
                0.0
            } else {
                1.0
            }
        }
        // Rejected at configuration time; never prune.
        Metric::DotProduct => 0.0,
    }
}

impl KdTreeIndex {
    pub fn new(params: KdTreeParams, metric: Metric) -> Self {
        Self {
            params,
            metric,
            nodes: Vec::new(),
            root: NIL,
            depth: 0,
            inserts_since_rebuild: 0,
        }
    }

    pub fn add_batch(&mut self, slots: std::ops::Range<u32>, store: &VectorStore) {
        let count = slots.len();
        for slot in slots {
            self.insert(slot, store);
        }
        self.inserts_since_rebuild += count;
        let n = self.nodes.len();
        let limit = self.params.balance_factor as f64 * (n.max(2) as f64).log2();
        if self.depth as f64 > limit.max(8.0)
            && self.inserts_since_rebuild >= MIN_INSERTS_BETWEEN_REBUILDS.min(n)
        {
            self.rebuild(store);
        }
    }

    fn insert(&mut self, slot: u32, store: &VectorStore) {
        assert_eq!(
            slot as usize,
            self.nodes.len(),
            "kd-tree slots must be added in order"
        );
        let dim = store.dim() as u32;
        if self.root == NIL {
            self.nodes.push(Node {
                axis: 0,
                left: NIL,
                right: NIL,
            });
            self.root = slot;
            self.depth = 1;
            return;
        }
        let v = store.vector(slot);
        let (mut cur, mut depth) = (self.root, 1);
        loop {
            depth += 1;
            let node = self.nodes[cur as usize];
            let split = store.vector(cur)[node.axis as usize];
            let child = if v[node.axis as usize] < split {
                &mut self.nodes[cur as usize].left
            } else {
                &mut self.nodes[cur as usize].right
            };
            if *child == NIL {
                *child = slot;
                self.nodes.push(Node {
                    axis: (node.axis + 1) % dim,
                    left: NIL,
                    right: NIL,
                });
                break;
            }
            cur = *child;
        }
        self.depth = self.depth.max(depth);
    }

    /// Rebuilds a balanced tree over all nodes.
    fn rebuild(&mut self, store: &VectorStore) {
        let mut slots: Vec<u32> = (0..self.nodes.len() as u32).collect();
        let mut depth = 0;
        self.root = self.build(&mut slots, store, 1, &mut depth);
        self.depth = depth;
        self.inserts_since_rebuild = 0;
    }

    fn build(
        &mut self,
        slots: &mut [u32],
        store: &VectorStore,
        level: usize,
        depth: &mut usize,
    ) -> u32 {
        if slots.is_empty() {
            return NIL;
        }
        *depth = (*depth).max(level);
        let axis = widest_axis(slots, store);
        let mid = slots.len() / 2;
        slots.select_nth_unstable_by(mid, |&a, &b| {
            store.vector(a)[axis].total_cmp(&store.vector(b)[axis])
        });
        let node = slots[mid];
        let (left, rest) = slots.split_at_mut(mid);
        let right = &mut rest[1..];
        let left = self.build(left, store, level + 1, depth);
        let right = self.build(right, store, level + 1, depth);
        self.nodes[node as usize] = Node {
            axis: axis as u32,
            left,
            right,
        };
        node
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        store: &VectorStore,
        accept: Accept<'_>,
    ) -> Vec<Neighbor> {
        let mut top = TopK::new(k);
        if self.root == NIL {
            return Vec::new();
        }
        // (node, lower bound on the distance of anything in its subtree)
        let mut stack: Vec<(u32, f32)> = vec![(self.root, 0.0)];
        while let Some((cur, bound)) = stack.pop() {
            if bound > top.worst() {
                continue;
            }
            let v = store.vector(cur);
            if accept(cur) {
                top.push(Neighbor {
                    slot: cur,
                    dist: self.metric.distance(query, v),
                });
            }
            let node = self.nodes[cur as usize];
            let diff = query[node.axis as usize] - v[node.axis as usize];
            let (near, far) = if diff < 0.0 {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };
            // Push the far side first so the near side is explored first.
            if far != NIL {
                stack.push((far, bound.max(plane_bound(self.metric, diff))));
            }
            if near != NIL {
                stack.push((near, bound));
            }
        }
        top.into_sorted()
    }

    pub(crate) fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        w.put_u32(self.root)?;
        w.put_u32(self.depth as u32)?;
        w.put_u64(self.inserts_since_rebuild as u64)?;
        w.put_u64(self.nodes.len() as u64)?;
        for n in &self.nodes {
            w.put_u32(n.axis)?;
            w.put_u32(n.left)?;
            w.put_u32(n.right)?;
        }
        Ok(())
    }

    pub(crate) fn read_from(
        r: &mut Reader<'_>,
        params: KdTreeParams,
        metric: Metric,
        slots: usize,
        dim: usize,
    ) -> Result<Self> {
        let mut tree = Self::new(params, metric);
        tree.root = r.u32()?;
        tree.depth = r.u32()? as usize;
        tree.inserts_since_rebuild = r.u64()? as usize;
        let n = r.u64()? as usize;
        if n != slots {
            return Err(Error::Corrupt(format!(
                "kd-tree has {n} nodes but store has {slots} slots"
            )));
        }
        let valid = |x: u32| x == NIL || (x as usize) < n;
        if !valid(tree.root) || (n > 0) == (tree.root == NIL) {
            return Err(Error::Corrupt("kd-tree root out of range".into()));
        }
        tree.nodes.reserve_exact(n);
        for _ in 0..n {
            let node = Node {
                axis: r.u32()?,
                left: r.u32()?,
                right: r.u32()?,
            };
            if !valid(node.left) || !valid(node.right) {
                return Err(Error::Corrupt("kd-tree child out of range".into()));
            }
            if node.axis as usize >= dim {
                return Err(Error::Corrupt("kd-tree split axis out of range".into()));
            }
            tree.nodes.push(node);
        }
        tree.depth = tree.check_structure()?;
        Ok(tree)
    }

    fn check_structure(&self) -> Result<usize> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = Vec::new();
        if self.root != NIL {
            stack.push((self.root, 1));
        }
        let mut depth = 0;
        while let Some((cur, level)) = stack.pop() {
            if std::mem::replace(&mut seen[cur as usize], true) {
                return Err(Error::Corrupt(
                    "kd-tree node is linked more than once".into(),
                ));
            }
            depth = depth.max(level);
            let node = self.nodes[cur as usize];
            for child in [node.left, node.right] {
                if child != NIL {
                    stack.push((child, level + 1));
                }
            }
        }
        if seen.contains(&false) {
            return Err(Error::Corrupt("kd-tree does not reach every node".into()));
        }
        Ok(depth)
    }
}

fn widest_axis(slots: &[u32], store: &VectorStore) -> usize {
    let dim = store.dim();
    let mut lo = vec![f32::INFINITY; dim];
    let mut hi = vec![f32::NEG_INFINITY; dim];
    for &s in slots {
        for (d, &x) in store.vector(s).iter().enumerate() {
            lo[d] = lo[d].min(x);
            hi[d] = hi[d].max(x);
        }
    }
    (0..dim)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlatIndex, random_store};

    fn build(store: &VectorStore, metric: Metric) -> KdTreeIndex {
        let mut tree = KdTreeIndex::new(KdTreeParams::default(), metric);
        tree.add_batch(0..store.slots() as u32, store);
        tree
    }

    fn distances(found: &[Neighbor]) -> Vec<f32> {
        found.iter().map(|n| n.dist).collect()
    }

    #[test]
    fn matches_brute_force_for_every_supported_metric() {
        for metric in [
            Metric::Euclidean,
            Metric::Manhattan,
            Metric::Cosine,
            Metric::Hamming,
        ] {
            let mut store = random_store(2_000, 5, 11);
            if metric == Metric::Hamming {
                // Hamming is meant for discrete data.
                let mut q = VectorStore::new(5);
                for s in 0..store.slots() as u32 {
                    let v: Vec<f32> = store.vector(s).iter().map(|x| (x * 2.0).round()).collect();
                    q.push(s.to_string(), &v, None);
                }
                store = q;
            }
            if metric == Metric::Cosine {
                let mut q = VectorStore::new(5);
                for s in 0..store.slots() as u32 {
                    let mut v = store.vector(s).to_vec();
                    assert!(metric.prepare(&mut v));
                    q.push(s.to_string(), &v, None);
                }
                store = q;
            }
            let tree = build(&store, metric);
            let flat = FlatIndex::new(metric);
            for q in 0..50 {
                let query = store.vector(q * 7);
                let expected = flat.search(query, 10, &store, &|_| true);
                let got = tree.search(query, 10, &store, &|_| true);
                assert_eq!(
                    distances(&got),
                    distances(&expected),
                    "{metric:?} query {q}"
                );
            }
        }
    }

    #[test]
    fn filtered_search_is_exact() {
        let store = random_store(1_000, 3, 5);
        let tree = build(&store, Metric::Euclidean);
        let flat = FlatIndex::new(Metric::Euclidean);
        let accept = |s: u32| s % 7 == 3;
        let q = store.vector(0);
        assert_eq!(
            tree.search(q, 8, &store, &accept),
            flat.search(q, 8, &store, &accept)
        );
    }

    #[test]
    fn sorted_inserts_trigger_rebalancing() {
        let mut store = VectorStore::new(2);
        let mut tree = KdTreeIndex::new(KdTreeParams::default(), Metric::Euclidean);
        for i in 0..4_000u32 {
            let slot = store.push(i.to_string(), &[i as f32, 0.0], None);
            tree.add_batch(slot..slot + 1, &store);
        }
        let limit = 3.0 * 4_000f64.log2();
        assert!(
            (tree.depth as f64) <= limit + MIN_INSERTS_BETWEEN_REBUILDS as f64,
            "depth {}",
            tree.depth
        );
        let got = tree.search(&[1234.2, 0.0], 1, &store, &|_| true);
        assert_eq!(got[0].slot, 1234);
    }

    #[test]
    fn read_from_rejects_cycles_and_bad_axes() {
        let store = random_store(20, 3, 4);
        let tree = build(&store, Metric::Euclidean);
        let load = |t: &KdTreeIndex| {
            let mut buf = Vec::new();
            t.write_to(&mut buf).unwrap();
            KdTreeIndex::read_from(
                &mut Reader::new(&buf),
                KdTreeParams::default(),
                Metric::Euclidean,
                20,
                3,
            )
        };
        assert!(load(&tree).is_ok());

        let mut cyclic = tree.clone();
        let root = cyclic.root as usize;
        cyclic.nodes[root].left = cyclic.root;
        assert!(matches!(load(&cyclic), Err(Error::Corrupt(_))));

        let mut bad_axis = tree.clone();
        bad_axis.nodes[0].axis = 3;
        assert!(matches!(load(&bad_axis), Err(Error::Corrupt(_))));
    }

    #[test]
    fn empty_tree() {
        let store = VectorStore::new(2);
        let tree = KdTreeIndex::new(KdTreeParams::default(), Metric::Euclidean);
        assert!(tree.search(&[0.0, 0.0], 3, &store, &|_| true).is_empty());
    }
}
