use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::RwLock;
use quiver_index::Index;
use quiver_snapshot::{SnapshotInfo, SnapshotRegistry};
use quiver_storage::{
    Mutation, Op, Recovered, StorageEngine, StorageKind, StorageOptions, VectorStore, open_engine, state,
};
use quiver_types::{
    CollectionConfig, CollectionInfo, Error, MAX_ID_LEN, MAX_METADATA_DEPTH, Point, Result, SearchHit,
    SearchOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Compact once at least this many slots are tombstoned...
const COMPACT_MIN_TOMBSTONES: usize = 1_024;
/// ...and they make up at least this fraction of all slots.
const COMPACT_RATIO: f64 = 0.3;

pub(crate) const META_FILE: &str = "collection.json";
const ROCKSDB_DIR: &str = "rocksdb";

static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CollectionMeta {
    pub config: CollectionConfig,
    pub storage: StorageKind,
}

#[derive(Debug, Clone)]
pub(crate) struct Env {
    pub storage: StorageKind,
    pub storage_options: StorageOptions,
    pub snapshots: Option<Arc<SnapshotRegistry>>,
    pub snapshot_keep: usize,
}

#[derive(Debug)]
struct Inner {
    store: VectorStore,
    index: Index,
    engine: Box<dyn StorageEngine>,
}

#[derive(Debug)]
pub struct Collection {
    name: String,
    config: CollectionConfig,
    storage: StorageKind,
    env: Env,
    /// Unique per handle in this process, so a dropped collection and its
    /// same-named replacement are never mistaken for one another.
    instance: u64,
    generation: AtomicU64,
    /// Set (under the write lock) once the collection is dropped or replaced.
    /// Handles still held elsewhere must then stop touching the directory,
    /// which may already belong to a new collection of the same name.
    closed: AtomicBool,
    inner: RwLock<Inner>,
}

fn engine_dir(dir: &Path, kind: StorageKind) -> PathBuf {
    match kind {
        StorageKind::RocksDb => dir.join(ROCKSDB_DIR),
        _ => dir.to_path_buf(),
    }
}

impl Collection {
    pub(crate) fn create(name: &str, config: CollectionConfig, dir: Option<&Path>, env: Env) -> Result<Self> {
        config.validate()?;
        let storage = env.storage;

        let engine = match (storage.is_durable(), dir) {
            (true, Some(dir)) => {
                // A leftover directory (e.g. from an interrupted drop) must not
                // leak its old log or checkpoint into the new collection.
                if dir.exists() {
                    fs::remove_dir_all(dir)?;
                }
                fs::create_dir_all(dir)?;
                write_meta(
                    dir,
                    &CollectionMeta {
                        config: config.clone(),
                        storage,
                    },
                )?;
                open_engine(storage, Some(&engine_dir(dir, storage)), env.storage_options)?
            }
            _ => open_engine(StorageKind::Memory, None, env.storage_options)?,
        };

        Ok(Self::from_parts(
            name,
            config.clone(),
            env,
            Inner {
                store: VectorStore::new(config.dimension),
                index: Index::new(&config.index, config.metric),
                engine,
            },
        ))
    }

    pub(crate) fn open(name: &str, dir: &Path, env: Env) -> Result<Self> {
        let meta: CollectionMeta = serde_json::from_slice(&fs::read(dir.join(META_FILE))?)?;
        meta.config.validate()?;

        if !meta.storage.is_available() {
            return Err(Error::Unsupported(format!(
                "collection `{name}` uses the {} engine, which this build does not include",
                meta.storage
            )));
        }

        let mut engine = open_engine(
            meta.storage,
            Some(&engine_dir(dir, meta.storage)),
            env.storage_options,
        )?;

        let Recovered { store, index, replay } = engine.recover(meta.config.dimension)?;
        let index = match index {
            Some(bytes) => Index::from_bytes(&bytes, &meta.config.index, meta.config.metric, &store)?,
            None => Index::build(&meta.config.index, meta.config.metric, &store),
        };

        let mut inner = Inner { store, index, engine };
        let replayed = replay.len();
        for m in replay {
            match m {
                Mutation::Upsert { points } => apply_upsert(&mut inner, points),
                Mutation::Delete { ids } => {
                    apply_delete(&mut inner, &ids);
                }
            }
        }
        tracing::debug!(collection = name, storage = %meta.storage, points = inner.store.live(), replayed, "opened collection");

        let collection = Self::from_parts(
            name,
            meta.config,
            Env {
                storage: meta.storage,
                ..env
            },
            inner,
        );
        if replayed > 0 {
            // Fold the replayed log into a fresh checkpoint so the next start is fast.
            collection.checkpoint(&mut collection.inner.write())?;
        }
        Ok(collection)
    }

    pub(crate) fn snapshot_index(
        config: &CollectionConfig,
        store: &VectorStore,
        index: Option<&[u8]>,
    ) -> Result<Index> {
        config.validate()?;
        match index {
            Some(bytes) => Index::from_bytes(bytes, &config.index, config.metric, store),
            None => Ok(Index::build(&config.index, config.metric, store)),
        }
    }

    pub(crate) fn restore(
        name: &str,
        config: CollectionConfig,
        store: VectorStore,
        index: Index,
        dir: Option<&Path>,
        env: Env,
    ) -> Result<Self> {
        let collection = Self::create(name, config, dir, env)?;
        {
            let mut inner = collection.inner.write();
            inner.engine.import(&store, &index.to_bytes())?;
            inner.store = store;
            inner.index = index;
        }

        Ok(collection)
    }

    fn from_parts(name: &str, config: CollectionConfig, env: Env, inner: Inner) -> Self {
        Self {
            name: name.to_owned(),
            config,
            storage: inner.engine.kind(),
            env,
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            generation: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            inner: RwLock::new(inner),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    pub fn storage(&self) -> StorageKind {
        self.storage
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Identifies this collection's state: changes whenever it is written to,
    /// and differs between a collection and any later one of the same name.
    pub(crate) fn version(&self) -> (u64, u64) {
        (self.instance, self.generation())
    }

    pub fn len(&self) -> usize {
        self.inner.read().store.live()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn info(&self) -> CollectionInfo {
        let inner = self.inner.read();
        CollectionInfo {
            name: self.name.clone(),
            config: self.config.clone(),
            count: inner.store.live(),
            tombstones: inner.store.tombstones(),
        }
    }

    pub fn insert(&self, vector: Vec<f32>, metadata: Option<Value>) -> Result<String> {
        let mut point = Point::with_random_id(vector);
        point.metadata = metadata;
        let id = point.id.clone();
        self.upsert([point])?;
        Ok(id)
    }

    pub fn upsert(&self, points: impl IntoIterator<Item = Point>) -> Result<usize> {
        let mut points: Vec<Point> = points.into_iter().collect();
        for p in &mut points {
            self.validate_point(p)?;
        }
        if points.is_empty() {
            return Ok(0);
        }
        let n = points.len();

        let mut inner = self.inner.write();
        self.ensure_open()?;
        inner.engine.log(Op::Upsert { points: &points })?;
        apply_upsert(&mut inner, points);
        self.bump();
        self.maintain(&mut inner)?;
        Ok(n)
    }

    pub fn delete<S: AsRef<str>>(&self, ids: &[S]) -> Result<usize> {
        let ids: Vec<String> = ids.iter().map(|s| s.as_ref().to_owned()).collect();
        let mut inner = self.inner.write();
        self.ensure_open()?;
        if !ids.iter().any(|id| inner.store.slot_of(id).is_some()) {
            return Ok(0);
        }
        inner.engine.log(Op::Delete { ids: &ids })?;
        let removed = apply_delete(&mut inner, &ids);
        self.bump();
        self.maintain(&mut inner)?;
        Ok(removed)
    }

    pub fn get(&self, id: &str) -> Option<Point> {
        let inner = self.inner.read();
        inner.store.slot_of(id).map(|slot| point_at(&inner.store, slot))
    }

    pub fn get_many<S: AsRef<str>>(&self, ids: &[S]) -> Vec<Point> {
        let inner = self.inner.read();
        ids.iter()
            .filter_map(|id| inner.store.slot_of(id.as_ref()))
            .map(|slot| point_at(&inner.store, slot))
            .collect()
    }

    pub fn scroll(&self, offset: usize, limit: usize) -> Vec<Point> {
        let inner = self.inner.read();
        inner
            .store
            .live_slots()
            .skip(offset)
            .take(limit)
            .map(|slot| point_at(&inner.store, slot))
            .collect()
    }

    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.search_with(vector, k, &SearchOptions::default())
    }

    pub fn search_with(&self, vector: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        let mut query = vector.to_vec();
        self.validate_vector(&mut query)?;

        let inner = self.inner.read();
        let store = &inner.store;
        let accept = |slot: u32| {
            store.is_alive(slot)
                && opts
                    .filter
                    .as_ref()
                    .is_none_or(|f| f.matches(store.metadata(slot)))
        };
        let found = inner.index.search(&query, k, opts.ef, store, &accept);

        Ok(found
            .into_iter()
            .map(|n| SearchHit {
                id: store.id(n.slot).to_owned(),
                distance: n.dist,
                metadata: opts
                    .include_metadata
                    .then(|| store.metadata(n.slot).cloned())
                    .flatten(),
                vector: opts.include_vectors.then(|| store.vector(n.slot).to_vec()),
            })
            .collect())
    }

    pub fn flush(&self) -> Result<()> {
        self.checkpoint(&mut self.inner.write())
    }

    pub fn compact(&self) -> Result<()> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        self.compact_locked(&mut inner);
        self.bump();
        self.checkpoint(&mut inner)
    }

    pub fn create_snapshot(&self) -> Result<SnapshotInfo> {
        let registry = self
            .env
            .snapshots
            .as_ref()
            .ok_or_else(|| Error::Unsupported("this database has no snapshot directory".into()))?;
        let info = {
            let inner = self.inner.read();
            self.ensure_open()?;
            registry.create(
                &self.name,
                &self.config,
                &inner.store,
                Some(&inner.index.to_bytes()),
            )?
        };
        registry.prune(&self.name, self.env.snapshot_keep)?;
        Ok(info)
    }

    pub fn export_state(&self, w: impl std::io::Write) -> Result<()> {
        let inner = self.inner.read();
        state::encode(w, &inner.store, Some(&inner.index.to_bytes()))?;
        Ok(())
    }

    pub(crate) fn close(&self) {
        let _inner = self.inner.write();
        self.closed.store(true, Ordering::Release);
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::CollectionNotFound(self.name.clone()));
        }
        Ok(())
    }

    fn bump(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    fn validate_point(&self, p: &mut Point) -> Result<()> {
        if p.id.is_empty() || p.id.len() > MAX_ID_LEN {
            return Err(Error::invalid(format!(
                "point ids must be 1..={MAX_ID_LEN} bytes"
            )));
        }
        if p.metadata
            .as_ref()
            .is_some_and(|m| deeper_than(m, MAX_METADATA_DEPTH))
        {
            return Err(Error::invalid(format!(
                "point `{}`: metadata is nested more than {MAX_METADATA_DEPTH} levels deep",
                p.id
            )));
        }
        // `null` and absent metadata are stored the same way; normalise so a
        // point reads back identically however it was recovered.
        if p.metadata == Some(Value::Null) {
            p.metadata = None;
        }
        self.validate_vector(&mut p.vector)
            .map_err(|e| Error::invalid(format!("point `{}`: {e}", p.id)))
    }

    fn validate_vector(&self, v: &mut [f32]) -> Result<()> {
        if v.len() != self.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimension,
                actual: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid("vectors must not contain NaN or infinity"));
        }
        if !self.config.metric.prepare(v) {
            return Err(Error::invalid(
                "zero vectors cannot be used with the cosine metric",
            ));
        }
        // Keeps every metric's distance between two valid vectors finite in
        // f32. Checked after `prepare`, which already normalises cosine vectors.
        let squared_norm: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if squared_norm > f64::from(f32::MAX) / 4.0 {
            return Err(Error::invalid("vector magnitude too large"));
        }
        Ok(())
    }

    fn maintain(&self, inner: &mut Inner) -> Result<()> {
        let tombstones = inner.store.tombstones();
        if tombstones >= COMPACT_MIN_TOMBSTONES
            && tombstones as f64 >= inner.store.slots() as f64 * COMPACT_RATIO
        {
            self.compact_locked(inner);
        }
        if inner.engine.needs_checkpoint() {
            self.checkpoint(inner)?;
        }
        Ok(())
    }

    fn compact_locked(&self, inner: &mut Inner) {
        let old = &inner.store;
        let mut store = VectorStore::new(self.config.dimension);
        for slot in old.live_slots() {
            store.push(
                old.id(slot).to_owned(),
                old.vector(slot),
                old.metadata(slot).cloned(),
            );
        }
        let index = Index::build(&self.config.index, self.config.metric, &store);
        tracing::debug!(collection = %self.name, reclaimed = old.tombstones(), "compacted collection");
        inner.store = store;
        inner.index = index;
    }

    fn checkpoint(&self, inner: &mut Inner) -> Result<()> {
        if !self.storage.is_durable() || self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let index = inner.index.to_bytes();
        inner.engine.checkpoint(&inner.store, &index)
    }
}

fn apply_upsert(inner: &mut Inner, points: Vec<Point>) {
    let Inner { store, index, .. } = inner;
    let first = store.slots() as u32;
    for p in points {
        store.push(p.id, &p.vector, p.metadata);
    }
    index.add_batch(first..store.slots() as u32, store);
}

fn apply_delete(inner: &mut Inner, ids: &[String]) -> usize {
    ids.iter().filter(|id| inner.store.remove(id)).count()
}

fn point_at(store: &VectorStore, slot: u32) -> Point {
    Point {
        id: store.id(slot).to_owned(),
        vector: store.vector(slot).to_vec(),
        metadata: store.metadata(slot).cloned(),
    }
}

fn write_meta(dir: &Path, meta: &CollectionMeta) -> Result<()> {
    let path = dir.join(META_FILE);
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(meta)?)?;
    fs::File::open(&tmp)?.sync_all()?;
    fs::rename(&tmp, &path)?;
    state::sync_parent(&path)
}

/// Whether `v` nests arrays/objects more than `limit` levels deep. Recursion
/// stops at `limit`, so arbitrarily deep values cannot overflow the stack.
fn deeper_than(v: &Value, limit: usize) -> bool {
    match v {
        Value::Array(a) => limit == 0 || a.iter().any(|c| deeper_than(c, limit - 1)),
        Value::Object(o) => limit == 0 || o.values().any(|c| deeper_than(c, limit - 1)),
        _ => false,
    }
}
