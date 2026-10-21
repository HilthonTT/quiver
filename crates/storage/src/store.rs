use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::Bound;

use quiver_types::Metric;
use serde_json::Value;

/// Vectors, one after another, in one of two layouts.
#[derive(Debug, Clone)]
enum Vectors {
    F32(Vec<f32>),
    /// Scalar quantization: one code byte per component, decoded as
    /// `min + code * step` with `[min, step]` stored per vector.
    Sq8 {
        codes: Vec<u8>,
        ranges: Vec<[f32; 2]>,
    },
}

impl Vectors {
    fn push(&mut self, v: &[f32]) {
        match self {
            Vectors::F32(data) => data.extend_from_slice(v),
            Vectors::Sq8 { codes, ranges } => {
                let (min, step, code) = quantize(v);
                codes.extend(code);
                ranges.push([min, step]);
            }
        }
    }
}

fn encode(x: f32, min: f32, step: f32) -> u8 {
    if step > 0.0 {
        ((f64::from(x) - f64::from(min)) / f64::from(step))
            .round()
            .clamp(0.0, 255.0) as u8
    } else {
        0
    }
}

#[inline]
fn decode(code: u8, min: f32, step: f32) -> f32 {
    min + f32::from(code) * step
}

/// `[min, step]` for quantizing `v`. Of the step nearest `(max - min) / 255`
/// and its closest neighbouring floats, takes the one that reconstructs `v`
/// best. Decoded vectors thus get back exactly the step they were decoded
/// with, and checkpoints (which store decoded values) reload unchanged.
fn scalar_range(v: &[f32]) -> (f32, f32) {
    let Some(min) = v.iter().copied().reduce(f32::min) else {
        return (0.0, 0.0);
    };
    let max = v.iter().copied().fold(min, f32::max);
    // In f64: `max - min` can overflow f32 for finite inputs.
    let base = ((f64::from(max) - f64::from(min)) / 255.0) as f32;
    if base == 0.0 {
        return (min, 0.0);
    }
    let error = |step: f32| -> f64 {
        v.iter()
            .map(|&x| {
                let d = f64::from(decode(encode(x, min, step), min, step)) - f64::from(x);
                d * d
            })
            .sum()
    };
    let candidates = [
        base,
        base.next_down(),
        base.next_up(),
        base.next_down().next_down(),
        base.next_up().next_up(),
    ];
    let mut best = (base, error(base));
    for step in candidates
        .into_iter()
        .skip(1)
        .filter(|s| *s > 0.0 && s.is_finite())
    {
        let e = error(step);
        if e < best.1 {
            best = (step, e);
        }
    }
    (min, best.0)
}

/// Quantizes `v` to codes whose decoded values quantize to themselves, so
/// checkpoints (which store decoded values) reload exactly. Usually the first
/// encoding already is such a fixed point; rare ones get there in a round
/// or two, moving each value by far less than the quantization error.
fn quantize(v: &[f32]) -> (f32, f32, Vec<u8>) {
    let mut values = v.to_vec();
    let mut decoded = Vec::with_capacity(v.len());
    let mut last = (0.0, 0.0, Vec::new());
    for _ in 0..8 {
        let (min, step) = scalar_range(&values);
        let codes: Vec<u8> = values.iter().map(|&x| encode(x, min, step)).collect();
        decoded.clear();
        decoded.extend(codes.iter().map(|&c| decode(c, min, step)));
        let (min2, step2) = scalar_range(&decoded);
        let stable = decoded
            .iter()
            .all(|&x| decode(encode(x, min2, step2), min2, step2) == x);
        last = (min, step, codes);
        if stable {
            break;
        }
        std::mem::swap(&mut values, &mut decoded);
    }
    last
}

thread_local! {
    /// Decoding space for quantized distances, so searches do not allocate.
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone)]
pub struct VectorStore {
    dim: usize,
    data: Vectors,
    ids: Vec<String>,
    metadata: Vec<Option<Value>>,
    alive: Vec<bool>,
    /// Live ids, ordered, so pages can be served in id order. Rebuilt from
    /// `push`/`remove` on load, so it never needs to be persisted.
    id_to_slot: BTreeMap<String, u32>,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vectors::F32(Vec::new()),
            ids: Vec::new(),
            metadata: Vec::new(),
            alive: Vec::new(),
            id_to_slot: BTreeMap::new(),
        }
    }

    /// An empty store holding vectors the same way as this one.
    pub fn empty_like(&self) -> Self {
        let mut store = Self::new(self.dim);
        if self.is_quantized() {
            store.quantize();
        }
        store
    }

    /// Switches to scalar-quantized storage, re-encoding what is held.
    /// Decoded vectors re-encode to the same codes, so this is idempotent.
    pub fn quantize(&mut self) {
        let Vectors::F32(data) = &mut self.data else {
            return;
        };
        let data = std::mem::take(data);
        let mut q = Vectors::Sq8 {
            codes: Vec::with_capacity(data.len()),
            ranges: Vec::with_capacity(self.ids.len()),
        };
        for v in data.chunks_exact(self.dim.max(1)) {
            q.push(v);
        }
        self.data = q;
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.data, Vectors::Sq8 { .. })
    }

    /// Bytes of memory the vectors take, tombstones included.
    pub fn vector_bytes(&self) -> usize {
        match &self.data {
            Vectors::F32(data) => data.len() * 4,
            Vectors::Sq8 { codes, ranges } => codes.len() + ranges.len() * 8,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn slots(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    pub fn live(&self) -> usize {
        self.id_to_slot.len()
    }

    #[inline]
    pub fn tombstones(&self) -> usize {
        self.slots() - self.live()
    }

    /// The vector in `slot`; decoded (so allocated) when quantized.
    #[inline]
    pub fn vector(&self, slot: u32) -> Cow<'_, [f32]> {
        let start = slot as usize * self.dim;
        match &self.data {
            Vectors::F32(data) => Cow::Borrowed(&data[start..start + self.dim]),
            Vectors::Sq8 { codes, ranges } => {
                let mut out = Vec::with_capacity(self.dim);
                self.decode_into(codes, ranges, slot, &mut out);
                Cow::Owned(out)
            }
        }
    }

    fn decode_into(&self, codes: &[u8], ranges: &[[f32; 2]], slot: u32, out: &mut Vec<f32>) {
        let start = slot as usize * self.dim;
        let [min, step] = ranges[slot as usize];
        out.clear();
        out.extend(
            codes[start..start + self.dim]
                .iter()
                .map(|&c| decode(c, min, step)),
        );
    }

    /// `metric`'s distance from `query` to the vector in `slot`, without
    /// allocating for quantized vectors.
    #[inline]
    pub fn distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        match &self.data {
            Vectors::F32(data) => {
                let start = slot as usize * self.dim;
                metric.distance(query, &data[start..start + self.dim])
            }
            Vectors::Sq8 { codes, ranges } => SCRATCH.with_borrow_mut(|buf| {
                self.decode_into(codes, ranges, slot, buf);
                metric.distance(query, buf)
            }),
        }
    }

    #[inline]
    pub fn is_alive(&self, slot: u32) -> bool {
        self.alive[slot as usize]
    }

    #[inline]
    pub fn id(&self, slot: u32) -> &str {
        &self.ids[slot as usize]
    }

    #[inline]
    pub fn metadata(&self, slot: u32) -> Option<&Value> {
        self.metadata[slot as usize].as_ref()
    }

    /// Replaces a live slot's metadata in place; its vector and slot stay.
    pub fn set_metadata(&mut self, slot: u32, metadata: Option<Value>) {
        debug_assert!(self.is_alive(slot));
        self.metadata[slot as usize] = metadata;
    }

    pub fn slot_of(&self, id: &str) -> Option<u32> {
        self.id_to_slot.get(id).copied()
    }

    pub fn push(&mut self, id: String, vector: &[f32], metadata: Option<Value>) -> u32 {
        debug_assert_eq!(vector.len(), self.dim);
        self.remove(&id);
        let slot = u32::try_from(self.slots()).expect("collection exceeds u32::MAX slots");
        self.data.push(vector);
        self.id_to_slot.insert(id.clone(), slot);
        self.ids.push(id);
        self.metadata.push(metadata);
        self.alive.push(true);
        slot
    }

    pub fn push_tombstone(&mut self, id: String, vector: &[f32]) {
        self.data.push(vector);
        self.ids.push(id);
        self.metadata.push(None);
        self.alive.push(false);
    }

    pub fn remove(&mut self, id: &str) -> bool {
        match self.id_to_slot.remove(id) {
            Some(slot) => {
                self.alive[slot as usize] = false;
                self.metadata[slot as usize] = None;
                true
            }
            None => false,
        }
    }

    /// Live points with an id greater than `after` (all of them if `None`),
    /// in ascending byte-wise id order.
    pub fn live_after<'a>(&'a self, after: Option<&str>) -> impl Iterator<Item = (&'a str, u32)> + 'a {
        let start = after.map_or(Bound::Unbounded, Bound::Excluded);
        self.id_to_slot
            .range::<str, _>((start, Bound::Unbounded))
            .map(|(id, &slot)| (id.as_str(), slot))
    }

    pub fn live_slots(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.slots() as u32).filter(|&s| self.is_alive(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn push_get_and_replace() {
        let mut s = VectorStore::new(2);
        let a = s.push("a".into(), &[1.0, 2.0], Some(json!({"k": 1})));
        assert_eq!(*s.vector(a), [1.0, 2.0]);
        assert_eq!(s.metadata(a), Some(&json!({"k": 1})));

        let a2 = s.push("a".into(), &[3.0, 4.0], None);
        assert_ne!(a, a2);
        assert!(!s.is_alive(a));
        assert_eq!(s.slot_of("a"), Some(a2));
        assert_eq!((s.live(), s.tombstones()), (1, 1));
    }

    #[test]
    fn live_after_is_ordered_and_exclusive() {
        let mut s = VectorStore::new(1);
        for id in ["b", "a", "d", "c"] {
            s.push(id.into(), &[1.0], None);
        }
        s.remove("c");
        s.push("a".into(), &[2.0], None);
        let ids = |after| s.live_after(after).map(|(id, _)| id).collect::<Vec<_>>();
        assert_eq!(ids(None), ["a", "b", "d"]);
        assert_eq!(ids(Some("a")), ["b", "d"]);
        assert_eq!(ids(Some("bb")), ["d"]);
        assert!(ids(Some("d")).is_empty());
        assert_eq!(
            s.live_after(None).next().map(|(_, slot)| s.vector(slot)[0]),
            Some(2.0)
        );
    }

    #[test]
    fn quantized_vectors_stay_close_and_reencode_identically() {
        let vectors = [
            vec![0.1, -0.7, 0.3, 0.25],
            vec![5.0, 5.0, 5.0, 5.0],
            vec![-1e30, 1e30, 0.0, 3.0],
            vec![1e-8, -1e-8, 0.0, 2e-8],
        ];
        let mut s = VectorStore::new(4);
        for (i, v) in vectors.iter().enumerate() {
            s.push(i.to_string(), v, None);
        }
        let exact = s.clone();
        s.quantize();
        assert!(s.is_quantized());
        assert_eq!(s.vector_bytes(), 4 * (4 + 8));
        for (i, v) in vectors.iter().enumerate() {
            let got = s.vector(i as u32);
            let (lo, hi) = v
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            let tolerance = (f64::from(hi) - f64::from(lo)) / 510.0 * 1.001;
            for (a, b) in v.iter().zip(got.iter()) {
                assert!(
                    (f64::from(*a) - f64::from(*b)).abs() <= tolerance,
                    "{v:?} -> {got:?}"
                );
                assert!(b.is_finite());
            }
            let q = [0.2, 0.1, -0.3, 0.5];
            assert_eq!(
                s.distance(Metric::Euclidean, &q, i as u32),
                Metric::Euclidean.distance(&q, &got)
            );
        }

        // Writing decoded values back (as checkpoints do) changes nothing.
        let mut again = VectorStore::new(4);
        for slot in 0..s.slots() as u32 {
            again.push(slot.to_string(), &s.vector(slot), None);
        }
        again.quantize();
        for slot in 0..s.slots() as u32 {
            assert_eq!(again.vector(slot), s.vector(slot));
        }

        let mut fresh = s.empty_like();
        fresh.push("x".into(), exact.vector(0).as_ref(), None);
        assert!(fresh.is_quantized());
        assert_eq!(fresh.vector(0), s.vector(0));
    }

    #[test]
    fn reencoding_decoded_vectors_is_exact() {
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut misses = 0;
        let trials = 50_000;
        for _ in 0..trials {
            let scale = 10f64.powf(next() * 16.0 - 8.0);
            let offset = if next() < 0.3 {
                (next() - 0.5) * scale * 100.0
            } else {
                0.0
            };
            let v: Vec<f32> = (0..16)
                .map(|_| (offset + (next() - 0.5) * scale) as f32)
                .collect();
            let (min, step, codes) = quantize(&v);
            let decoded: Vec<f32> = codes.iter().map(|&c| decode(c, min, step)).collect();
            let (min2, step2) = scalar_range(&decoded);
            let again: Vec<f32> = decoded
                .iter()
                .map(|&x| decode(encode(x, min2, step2), min2, step2))
                .collect();
            misses += usize::from(again != decoded);
        }
        assert_eq!(misses, 0, "{misses} of {trials} vectors changed when re-encoded");
    }

    #[test]
    fn remove() {
        let mut s = VectorStore::new(1);
        s.push("a".into(), &[1.0], None);
        assert!(s.remove("a"));
        assert!(!s.remove("a"));
        assert_eq!(s.live(), 0);
        assert_eq!(s.live_slots().count(), 0);
    }
}
