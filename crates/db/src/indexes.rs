//! A collection's vector indexes: one per dense vector, one per sparse vector.

use std::collections::BTreeMap;

use quiver_index::{Index, SparseIndex};
use quiver_storage::codec::{Reader, WriteExt};
use quiver_storage::{VectorColumn, VectorStore};
use quiver_types::{CollectionConfig, Error, IndexConfig, Result, VectorConfig};

/// Marks a blob holding several indexes. Single-index blobs from before named
/// vectors start with the index's own tag, which is always smaller.
const TAG_SET: u8 = 0x10;

#[derive(Debug, Clone)]
pub(crate) struct Indexes {
    pub main: Index,
    pub named: BTreeMap<String, Index>,
    pub sparse: BTreeMap<String, SparseIndex>,
}

impl Indexes {
    pub fn new(config: &CollectionConfig) -> Self {
        Self {
            main: Index::new(&config.index, config.metric),
            named: config
                .vectors
                .iter()
                .map(|(name, v)| (name.clone(), Index::new(&v.index, v.metric)))
                .collect(),
            sparse: config
                .sparse_vectors
                .keys()
                .map(|name| (name.clone(), SparseIndex::default()))
                .collect(),
        }
    }

    /// Indexes everything in `store`.
    pub fn build(config: &CollectionConfig, store: &VectorStore) -> Self {
        let mut indexes = Self {
            sparse: BTreeMap::new(),
            ..Self::new(config)
        };
        indexes.add_dense(0..store.slots() as u32, store);
        indexes.build_sparse(config, store);
        indexes
    }

    fn build_sparse(&mut self, config: &CollectionConfig, store: &VectorStore) {
        self.sparse = config
            .sparse_vectors
            .keys()
            .map(|name| {
                let vectors = store
                    .sparse(name)
                    .expect("store has the configured sparse vectors");
                (name.clone(), SparseIndex::build(vectors, |s| store.is_alive(s)))
            })
            .collect();
    }

    /// The dense index `using` names, with its vectors.
    pub fn dense<'a>(&'a self, store: &'a VectorStore, using: Option<&str>) -> (&'a Index, &'a VectorColumn) {
        let index = match using {
            None => &self.main,
            Some(name) => &self.named[name],
        };
        let column = store.column(using).expect("store has the configured vectors");
        (index, column)
    }

    /// Makes a run of newly pushed slots searchable.
    pub fn add(&mut self, slots: std::ops::Range<u32>, store: &VectorStore) {
        self.add_dense(slots.clone(), store);
        for (name, index) in &mut self.sparse {
            let vectors = store
                .sparse(name)
                .expect("store has the configured sparse vectors");
            for slot in slots.clone() {
                index.add(slot, &vectors[slot as usize]);
            }
        }
    }

    fn add_dense(&mut self, slots: std::ops::Range<u32>, store: &VectorStore) {
        self.main.add_batch(slots.clone(), store.main());
        for (name, index) in &mut self.named {
            let column = store
                .column(Some(name))
                .expect("store has the configured vectors");
            index.add_batch(slots.clone(), column);
        }
    }

    /// Call before the point in `slot` is removed from `store`.
    pub fn remove(&mut self, slot: u32, store: &VectorStore) {
        for (name, index) in &mut self.sparse {
            let vectors = store
                .sparse(name)
                .expect("store has the configured sparse vectors");
            index.remove(&vectors[slot as usize]);
        }
    }

    /// Rebuilds sparse indexes that have collected many stale entries.
    pub fn maintain(&mut self, store: &VectorStore) {
        for (name, index) in &mut self.sparse {
            if index.needs_rebuild() {
                let vectors = store
                    .sparse(name)
                    .expect("store has the configured sparse vectors");
                *index = SparseIndex::build(vectors, |s| store.is_alive(s));
            }
        }
    }

    /// Serializes the dense indexes; sparse ones are rebuilt on load.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![TAG_SET];
        let mut put = |name: &str, index: &Index| {
            out.put_bytes(name.as_bytes())?;
            let blob = index.to_bytes();
            out.put_u64(blob.len() as u64)?;
            out.extend_from_slice(&blob);
            Ok::<_, std::io::Error>(())
        };
        put("", &self.main).expect("writing to a Vec cannot fail");
        for (name, index) in &self.named {
            put(name, index).expect("writing to a Vec cannot fail");
        }
        out
    }

    /// Restores indexes serialized by [`to_bytes`](Self::to_bytes) for
    /// `store`. Any that do not match `config` (it changed since they were
    /// saved) or are missing are rebuilt instead; other damage is an error.
    pub fn from_bytes(bytes: Option<&[u8]>, config: &CollectionConfig, store: &VectorStore) -> Result<Self> {
        let mut blobs = match bytes {
            Some(bytes) => split(bytes)?,
            None => BTreeMap::new(),
        };
        let mut load = |name: &str, v: &VectorConfig, column: &VectorColumn| -> Result<Index> {
            match blobs.remove(name) {
                Some(blob) if matches(&blob, &v.index) => {
                    Index::from_bytes(&blob, &v.index, v.metric, column)
                }
                stale => {
                    if stale.is_some() {
                        tracing::info!(
                            vector = name,
                            "stored index does not match the configuration; rebuilding"
                        );
                    }
                    Ok(Index::build(&v.index, v.metric, column))
                }
            }
        };
        let main = load("", &config.main_vector(), store.main())?;
        let mut named = BTreeMap::new();
        for (name, v) in &config.vectors {
            let column = store
                .column(Some(name))
                .ok_or_else(|| Error::Corrupt(format!("store lacks vector `{name}`")))?;
            named.insert(name.clone(), load(name, v, column)?);
        }
        let mut indexes = Self {
            main,
            named,
            sparse: BTreeMap::new(),
        };
        indexes.build_sparse(config, store);
        Ok(indexes)
    }
}

/// Whether a serialized index is of the kind `config` asks for.
fn matches(blob: &[u8], config: &IndexConfig) -> bool {
    let tag = match config {
        IndexConfig::Flat => 0,
        IndexConfig::Hnsw(_) => 1,
        IndexConfig::KdTree(_) => 2,
    };
    blob.first() == Some(&tag)
}

/// The serialized indexes in `bytes`, by vector name ("" for the main one).
fn split(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut out = BTreeMap::new();
    if bytes.first() != Some(&TAG_SET) {
        out.insert(String::new(), bytes.to_vec());
        return Ok(out);
    }
    let mut r = Reader::new(&bytes[1..]);
    while !r.is_empty() {
        let name = String::from_utf8(r.bytes()?.to_vec())
            .map_err(|_| Error::Corrupt("index name is not UTF-8".into()))?;
        let len = usize::try_from(r.u64()?).map_err(|_| Error::Corrupt("index too large".into()))?;
        let blob = r.take(len)?.to_vec();
        if out.insert(name, blob).is_some() {
            return Err(Error::Corrupt("index stored twice".into()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quiver_storage::StoreSpec;
    use quiver_types::{HnswParams, Point};

    fn store(config: &CollectionConfig) -> VectorStore {
        let mut s = VectorStore::with_spec(&StoreSpec::for_config(config, None)).unwrap();
        for i in 0..50 {
            let x = i as f32;
            s.push_point(Point::new(i.to_string(), vec![x, 1.0]).with_vector("t", vec![1.0, x, 2.0]));
        }
        s
    }

    #[test]
    fn roundtrip_and_tolerance() {
        let config = CollectionConfig::new(2)
            .index(IndexConfig::Flat)
            .with_vector("t", VectorConfig::new(3));
        let s = store(&config);
        let built = Indexes::build(&config, &s);
        let bytes = built.to_bytes();
        let back = Indexes::from_bytes(Some(&bytes), &config, &s).unwrap();
        assert_eq!(back.to_bytes(), bytes);

        // A changed index type is rebuilt, not an error.
        let mut changed = config.clone();
        changed.vectors.get_mut("t").unwrap().index = IndexConfig::Flat;
        changed.index = IndexConfig::Hnsw(HnswParams::default());
        let back = Indexes::from_bytes(Some(&bytes), &changed, &s).unwrap();
        assert!(matches!(back.main, Index::Hnsw(_)));
        assert!(matches!(back.named["t"], Index::Flat(_)));

        // A main-only blob from before named vectors still loads.
        let old = built.main.to_bytes();
        assert!(Indexes::from_bytes(Some(&old), &config, &s).is_ok());
        assert!(Indexes::from_bytes(Some(&[TAG_SET, 0, 0]), &config, &s).is_err());
    }
}
