use std::{cmp::Ordering, collections::BinaryHeap};

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
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
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
