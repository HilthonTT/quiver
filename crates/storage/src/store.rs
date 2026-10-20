use std::collections::BTreeMap;
use std::ops::Bound;

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct VectorStore {
    dim: usize,
    data: Vec<f32>,
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
            data: Vec::new(),
            ids: Vec::new(),
            metadata: Vec::new(),
            alive: Vec::new(),
            id_to_slot: BTreeMap::new(),
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

    #[inline]
    pub fn vector(&self, slot: u32) -> &[f32] {
        let start = slot as usize * self.dim;
        &self.data[start..start + self.dim]
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
        self.data.extend_from_slice(vector);
        self.id_to_slot.insert(id.clone(), slot);
        self.ids.push(id);
        self.metadata.push(metadata);
        self.alive.push(true);
        slot
    }

    pub fn push_tombstone(&mut self, id: String, vector: &[f32]) {
        self.data.extend_from_slice(vector);
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
        assert_eq!(s.vector(a), &[1.0, 2.0]);
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
    fn remove() {
        let mut s = VectorStore::new(1);
        s.push("a".into(), &[1.0], None);
        assert!(s.remove("a"));
        assert!(!s.remove("a"));
        assert_eq!(s.live(), 0);
        assert_eq!(s.live_slots().count(), 0);
    }
}
