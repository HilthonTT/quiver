use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::path::{Path, PathBuf};

use quiver_types::{CollectionConfig, Metric, Point, Quantization, Result, SparseVector, VectorConfig};
use serde_json::Value;

use crate::column::{ColumnSpec, VectorColumn};

/// What a [`VectorStore`] holds besides ids and metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StoreSpec {
    /// The main vectors.
    pub main: ColumnSpec,
    /// Named dense vectors, sorted by name.
    pub named: Vec<(String, ColumnSpec)>,
    /// Names of the sparse vectors, sorted.
    pub sparse: Vec<String>,
    /// Where on-disk columns keep their files.
    pub dir: Option<PathBuf>,
}

impl StoreSpec {
    /// Main vectors of dimension `dim`, in memory, and nothing else.
    pub fn new(dim: usize) -> Self {
        Self {
            main: ColumnSpec::new(dim),
            ..Self::default()
        }
    }

    /// The layout of a collection configured as `config` whose on-disk
    /// columns live in `dir`.
    pub fn for_config(config: &CollectionConfig, dir: Option<&Path>) -> Self {
        let column = |v: &VectorConfig| ColumnSpec {
            dim: v.dimension,
            quantization: v.quantization,
            on_disk: v.on_disk,
            datatype: v.datatype,
        };
        Self {
            main: column(&config.main_vector()),
            named: config
                .vectors
                .iter()
                .map(|(name, v)| (name.clone(), column(v)))
                .collect(),
            sparse: config.sparse_vectors.keys().cloned().collect(),
            dir: dir.map(Path::to_path_buf),
        }
    }

    /// The same layout with every column in memory.
    #[must_use]
    pub fn in_memory(mut self) -> Self {
        self.main.on_disk = false;
        for (_, c) in &mut self.named {
            c.on_disk = false;
        }
        self.dir = None;
        self
    }

    /// The same layout in memory only, keeping columns that are on disk
    /// as exact floats so they can be moved back without loss.
    #[must_use]
    pub fn staging(mut self) -> Self {
        for c in std::iter::once(&mut self.main).chain(self.named.iter_mut().map(|(_, c)| c)) {
            if c.on_disk {
                *c = ColumnSpec {
                    datatype: c.datatype,
                    ..ColumnSpec::new(c.dim)
                };
            }
        }
        self.dir = None;
        self
    }

    /// The main column's dimension.
    pub fn dim(&self) -> usize {
        self.main.dim
    }

    /// Name of the main column, for its file and in state files.
    pub const MAIN: &'static str = "";

    pub(crate) fn column_file_name(name: &str) -> &str {
        if name.is_empty() { "main" } else { name }
    }
}

/// Points: ids, metadata and vectors, one slot each. Deleting a point
/// leaves a tombstone in its slot until the collection is compacted.
#[derive(Debug)]
pub struct VectorStore {
    main: VectorColumn,
    /// Sorted by name.
    named: Vec<(String, VectorColumn)>,
    /// Sorted by name; empty vectors for points without one (and tombstones).
    sparse: Vec<(String, Vec<SparseVector>)>,
    dir: Option<PathBuf>,
    ids: Vec<String>,
    metadata: Vec<Option<Value>>,
    alive: Vec<bool>,
    /// Live ids, ordered, so pages can be served in id order. Rebuilt from
    /// `push`/`remove` on load, so it never needs to be persisted.
    id_to_slot: BTreeMap<String, u32>,
}

impl VectorStore {
    /// An in-memory store of main vectors only.
    pub fn new(dim: usize) -> Self {
        Self::from_columns(VectorColumn::new(dim), Vec::new(), &[], None)
    }

    /// An empty store laid out as `spec` says.
    pub fn with_spec(spec: &StoreSpec) -> Result<Self> {
        let dir = spec.dir.as_deref();
        let main = VectorColumn::create(spec.main, dir, StoreSpec::column_file_name(StoreSpec::MAIN))?;
        let named = spec
            .named
            .iter()
            .map(|(name, c)| Ok((name.clone(), VectorColumn::create(*c, dir, name)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_columns(main, named, &spec.sparse, spec.dir.clone()))
    }

    fn from_columns(
        main: VectorColumn,
        named: Vec<(String, VectorColumn)>,
        sparse: &[String],
        dir: Option<PathBuf>,
    ) -> Self {
        Self {
            main,
            named,
            sparse: sparse.iter().map(|name| (name.clone(), Vec::new())).collect(),
            dir,
            ids: Vec::new(),
            metadata: Vec::new(),
            alive: Vec::new(),
            id_to_slot: BTreeMap::new(),
        }
    }

    /// Assembles a store from decoded parts. A later live slot with the same
    /// id as an earlier one wins, as it would have through `push`.
    pub(crate) fn from_parts(
        rows: Vec<(bool, String, Option<Value>)>,
        main: VectorColumn,
        named: Vec<(String, VectorColumn)>,
        sparse: Vec<(String, Vec<SparseVector>)>,
        dir: Option<PathBuf>,
    ) -> Self {
        let mut store = Self::from_columns(main, named, &[], dir);
        store.sparse = sparse;
        for (alive, id, metadata) in rows {
            let slot = store.ids.len() as u32;
            if alive {
                if let Some(old) = store.id_to_slot.insert(id.clone(), slot) {
                    store.alive[old as usize] = false;
                    store.metadata[old as usize] = None;
                }
                store.metadata.push(metadata);
            } else {
                store.metadata.push(None);
            }
            store.ids.push(id);
            store.alive.push(alive);
        }
        let slots = store.ids.len();
        debug_assert!(store.columns().all(|(_, c)| c.slots() == slots));
        debug_assert!(store.sparse.iter().all(|(_, v)| v.len() == slots));
        store
    }

    /// The layout of this store.
    pub fn spec(&self) -> StoreSpec {
        StoreSpec {
            main: self.main.spec(),
            named: self.named.iter().map(|(n, c)| (n.clone(), c.spec())).collect(),
            sparse: self.sparse.iter().map(|(n, _)| n.clone()).collect(),
            dir: self.dir.clone(),
        }
    }

    /// An empty store holding vectors the same way as this one.
    pub fn empty_like(&self) -> Result<Self> {
        Self::with_spec(&self.spec())
    }

    /// Where on-disk columns keep their files.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Switches the main column to codes quantized as `quantization` says.
    pub fn quantize(&mut self, quantization: Quantization) {
        self.main.quantize(quantization);
    }

    pub fn is_quantized(&self) -> bool {
        self.main.is_quantized()
    }

    /// Bytes of memory the vectors of every column take, tombstones included.
    pub fn vector_bytes(&self) -> usize {
        self.columns().map(|(_, c)| c.vector_bytes()).sum()
    }

    /// The main column's dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.main.dim()
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

    /// The main vectors.
    pub fn main(&self) -> &VectorColumn {
        &self.main
    }

    /// The main column for `None`, else the named one.
    pub fn column(&self, using: Option<&str>) -> Option<&VectorColumn> {
        match using {
            None => Some(&self.main),
            Some(name) => self
                .named
                .binary_search_by(|(n, _)| n.as_str().cmp(name))
                .ok()
                .map(|i| &self.named[i].1),
        }
    }

    /// Replaces a column with one holding the same vectors differently, and
    /// returns the old one.
    pub fn replace_column(&mut self, using: Option<&str>, column: VectorColumn) -> VectorColumn {
        let slot = match using {
            None => &mut self.main,
            Some(name) => {
                let i = self
                    .named
                    .binary_search_by(|(n, _)| n.as_str().cmp(name))
                    .expect("replacing an existing column");
                &mut self.named[i].1
            }
        };
        assert_eq!(column.slots(), slot.slots(), "replacement column has every slot");
        std::mem::replace(slot, column)
    }

    /// Every dense column with its name ([`StoreSpec::MAIN`] for the main one).
    pub fn columns(&self) -> impl Iterator<Item = (&str, &VectorColumn)> {
        std::iter::once((StoreSpec::MAIN, &self.main)).chain(self.named.iter().map(|(n, c)| (n.as_str(), c)))
    }

    /// The named dense columns, sorted by name.
    pub fn named(&self) -> impl Iterator<Item = (&str, &VectorColumn)> {
        self.named.iter().map(|(n, c)| (n.as_str(), c))
    }

    /// Every slot's vector of the sparse column `name`.
    pub fn sparse(&self, name: &str) -> Option<&[SparseVector]> {
        self.sparse
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// The sparse columns, sorted by name.
    pub fn sparse_columns(&self) -> impl Iterator<Item = (&str, &[SparseVector])> {
        self.sparse.iter().map(|(n, v)| (n.as_str(), v.as_slice()))
    }

    /// The main vector in `slot`; decoded (so allocated) when quantized.
    #[inline]
    pub fn vector(&self, slot: u32) -> Cow<'_, [f32]> {
        self.main.vector(slot)
    }

    /// `metric`'s distance from `query` to the main vector in `slot`.
    #[inline]
    pub fn distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        self.main.distance(metric, query, slot)
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

    /// Replaces a live slot's metadata in place; its vectors and slot stay.
    pub fn set_metadata(&mut self, slot: u32, metadata: Option<Value>) {
        debug_assert!(self.is_alive(slot));
        self.metadata[slot as usize] = metadata;
    }

    pub fn slot_of(&self, id: &str) -> Option<u32> {
        self.id_to_slot.get(id).copied()
    }

    /// The point in `slot`, with all its vectors.
    pub fn point(&self, slot: u32) -> Point {
        Point {
            id: self.id(slot).to_owned(),
            vector: self.vector(slot).into_owned(),
            vectors: self
                .named
                .iter()
                .map(|(n, c)| (n.clone(), c.vector(slot).into_owned()))
                .collect(),
            sparse_vectors: self
                .sparse
                .iter()
                .filter(|(_, v)| !v[slot as usize].is_empty())
                .map(|(n, v)| (n.clone(), v[slot as usize].clone()))
                .collect(),
            metadata: self.metadata(slot).cloned(),
        }
    }

    /// Adds a point with main vectors only, replacing any live point with its id.
    pub fn push(&mut self, id: String, vector: &[f32], metadata: Option<Value>) -> u32 {
        debug_assert!(self.named.is_empty());
        let mut point = Point::new(id, Vec::new());
        point.metadata = metadata;
        self.push_with(point, vector)
    }

    /// Adds a point, replacing any live point with its id. It must carry
    /// every named vector of the store and no sparse vector it lacks.
    pub fn push_point(&mut self, mut point: Point) -> u32 {
        let vector = std::mem::take(&mut point.vector);
        self.push_with(point, &vector)
    }

    fn push_with(&mut self, point: Point, vector: &[f32]) -> u32 {
        let Point {
            id,
            mut vectors,
            mut sparse_vectors,
            metadata,
            ..
        } = point;
        self.remove(&id);
        let slot = u32::try_from(self.slots()).expect("collection exceeds u32::MAX slots");
        self.main.push(vector);
        for (name, column) in &mut self.named {
            let v = vectors.remove(name).expect("point has every named vector");
            column.push(&v);
        }
        debug_assert!(vectors.is_empty(), "point has unknown named vectors");
        for (name, column) in &mut self.sparse {
            column.push(sparse_vectors.remove(name).unwrap_or_default());
        }
        debug_assert!(sparse_vectors.is_empty(), "point has unknown sparse vectors");
        self.id_to_slot.insert(id.clone(), slot);
        self.ids.push(id);
        self.metadata.push(metadata);
        self.alive.push(true);
        slot
    }

    pub fn remove(&mut self, id: &str) -> bool {
        match self.id_to_slot.remove(id) {
            Some(slot) => {
                self.alive[slot as usize] = false;
                self.metadata[slot as usize] = None;
                for (_, column) in &mut self.sparse {
                    column[slot as usize] = SparseVector::default();
                }
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

    /// Makes on-disk columns durable.
    pub fn sync(&self) -> Result<()> {
        for (_, c) in self.columns() {
            c.sync()?;
        }
        Ok(())
    }

    /// The files of on-disk columns.
    pub fn disk_files(&self) -> Vec<&Path> {
        self.columns().filter_map(|(_, c)| c.disk_path()).collect()
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
    fn named_and_sparse_columns() {
        let config = CollectionConfig::new(2)
            .with_vector("title", VectorConfig::new(3))
            .with_sparse_vector("text", Default::default());
        let mut s = VectorStore::with_spec(&StoreSpec::for_config(&config, None)).unwrap();
        let p = Point::new("a", vec![1.0, 0.0])
            .with_vector("title", vec![0.0, 1.0, 2.0])
            .with_sparse_vector("text", SparseVector::new(vec![3], vec![1.5]));
        let slot = s.push_point(p.clone());
        assert_eq!(s.point(slot), p);
        assert_eq!(*s.column(Some("title")).unwrap().vector(slot), [0.0, 1.0, 2.0]);
        assert!(s.column(Some("nope")).is_none());
        assert_eq!(s.sparse("text").unwrap()[slot as usize].indices, [3]);

        let bare = Point::new("b", vec![0.0, 1.0]).with_vector("title", vec![1.0; 3]);
        let b = s.push_point(bare.clone());
        assert_eq!(s.point(b), bare, "missing sparse vectors stay missing");
        s.remove("a");
        assert!(s.sparse("text").unwrap()[slot as usize].is_empty());
        assert_eq!(s.spec(), StoreSpec::for_config(&config, None));
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
