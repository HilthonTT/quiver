use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::RwLock;
use quiver_index::{Index, Neighbor, TopK};
use quiver_snapshot::{SnapshotInfo, SnapshotRegistry};
use quiver_storage::{
    Mutation, Op, Recovered, StorageEngine, StorageKind, StorageOptions, VectorStore, open_engine, state,
};
use quiver_types::{
    CollectionConfig, CollectionInfo, Error, Filter, MAX_ID_LEN, MAX_METADATA_DEPTH, MetadataMode, Metric,
    Point, PointSelector, Result, SearchHit, SearchOptions, merge_patch,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::fields::FieldIndexes;

/// Compact once at least this many slots are tombstoned...
const COMPACT_MIN_TOMBSTONES: usize = 1_024;
/// ...and they make up at least this fraction of all slots.
const COMPACT_RATIO: f64 = 0.3;

pub(crate) const META_FILE: &str = "collection.json";
const ROCKSDB_DIR: &str = "rocksdb";
/// Filtered searches whose field-index candidates number at most this many
/// score every candidate exactly instead of searching the vector index.
const EXACT_SCAN_MAX_CANDIDATES: usize = 20_000;

static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(0);

/// One page of [`Collection::scroll`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScrollPage {
    /// Points in ascending id order.
    pub points: Vec<Point>,
    /// Pass as the next `cursor`; `None` when there are no further points.
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CollectionMeta {
    pub config: CollectionConfig,
    pub storage: StorageKind,
    /// Indexed metadata fields. The indexes themselves are rebuilt on open.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub field_indexes: Vec<String>,
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
    fields: FieldIndexes,
    engine: Box<dyn StorageEngine>,
}

#[derive(Debug)]
pub struct Collection {
    name: String,
    config: CollectionConfig,
    storage: StorageKind,
    /// Holds `collection.json`; `None` for collections that are not persisted.
    meta_dir: Option<PathBuf>,
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

        let meta_dir = dir.filter(|_| storage.is_durable()).map(Path::to_path_buf);
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
                        field_indexes: Vec::new(),
                    },
                )?;
                open_engine(storage, Some(&engine_dir(dir, storage)), env.storage_options)?
            }
            _ => open_engine(StorageKind::Memory, None, env.storage_options)?,
        };

        Ok(Self::from_parts(
            name,
            config.clone(),
            meta_dir,
            env,
            Inner {
                store: VectorStore::new(config.dimension),
                index: Index::new(&config.index, config.metric),
                fields: FieldIndexes::default(),
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

        FieldIndexes::validate(&meta.field_indexes)?;
        let mut inner = Inner {
            store,
            index,
            fields: FieldIndexes::default(),
            engine,
        };
        let replayed = replay.len();
        for m in replay {
            match m {
                Mutation::Upsert { points } => apply_upsert(&mut inner, points),
                Mutation::Delete { ids } => {
                    apply_delete(&mut inner, &ids);
                }
            }
        }
        inner.fields = FieldIndexes::build(&meta.field_indexes, &inner.store);
        tracing::debug!(collection = name, storage = %meta.storage, points = inner.store.live(), replayed, "opened collection");

        let collection = Self::from_parts(
            name,
            meta.config,
            Some(dir.to_path_buf()),
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
        check_restored_points(config, store)?;
        match index {
            Some(bytes) => Index::from_bytes(bytes, &config.index, config.metric, store),
            None => Ok(Index::build(&config.index, config.metric, store)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore(
        name: &str,
        config: CollectionConfig,
        field_indexes: &[String],
        store: VectorStore,
        index: Index,
        dir: Option<&Path>,
        env: Env,
    ) -> Result<Self> {
        FieldIndexes::validate(field_indexes)?;
        let collection = Self::create(name, config, dir, env)?;
        {
            let mut inner = collection.inner.write();
            inner.engine.import(&store, &index.to_bytes())?;
            inner.fields = FieldIndexes::build(field_indexes, &store);
            inner.store = store;
            inner.index = index;
            collection.write_field_indexes(&inner)?;
        }

        Ok(collection)
    }

    fn from_parts(
        name: &str,
        config: CollectionConfig,
        meta_dir: Option<PathBuf>,
        env: Env,
        inner: Inner,
    ) -> Self {
        Self {
            name: name.to_owned(),
            config,
            storage: inner.engine.kind(),
            meta_dir,
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
            field_indexes: inner.fields.names(),
        }
    }

    /// Indexes a metadata field (a dot-separated path, as in filters) so
    /// filters on it look at matching points only. Indexing a field that
    /// already has an index does nothing.
    pub fn create_field_index(&self, field: &str) -> Result<()> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let mut names = inner.fields.names();
        if names.iter().any(|n| n == field) {
            return Ok(());
        }
        names.push(field.to_owned());
        FieldIndexes::validate(&names)?;
        let built = FieldIndexes::build(&names, &inner.store);
        let previous = std::mem::replace(&mut inner.fields, built);
        if let Err(e) = self.write_field_indexes(&inner) {
            inner.fields = previous;
            return Err(e);
        }
        tracing::info!(collection = %self.name, field, "created field index");
        Ok(())
    }

    /// Drops a field index; returns whether there was one.
    pub fn drop_field_index(&self, field: &str) -> Result<bool> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let names: Vec<String> = inner.fields.names().into_iter().filter(|n| n != field).collect();
        if names.len() == inner.fields.names().len() {
            return Ok(false);
        }
        let built = FieldIndexes::build(&names, &inner.store);
        let previous = std::mem::replace(&mut inner.fields, built);
        if let Err(e) = self.write_field_indexes(&inner) {
            inner.fields = previous;
            return Err(e);
        }
        tracing::info!(collection = %self.name, field, "dropped field index");
        Ok(true)
    }

    /// Persists the list of indexed fields, for durable collections.
    fn write_field_indexes(&self, inner: &Inner) -> Result<()> {
        let Some(dir) = &self.meta_dir else {
            return Ok(());
        };
        write_meta(
            dir,
            &CollectionMeta {
                config: self.config.clone(),
                storage: self.storage,
                field_indexes: inner.fields.names(),
            },
        )
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

    /// Deletes every point whose metadata matches `filter`; returns how many.
    pub fn delete_where(&self, filter: &Filter) -> Result<usize> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let ids: Vec<String> = matching(&inner, filter)
            .map(|slot| inner.store.id(slot).to_owned())
            .collect();
        if ids.is_empty() {
            return Ok(0);
        }
        inner.engine.log(Op::Delete { ids: &ids })?;
        let removed = apply_delete(&mut inner, &ids);
        self.bump();
        self.maintain(&mut inner)?;
        Ok(removed)
    }

    /// Number of points, or of points whose metadata matches `filter`.
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        let inner = self.inner.read();
        let store = &inner.store;
        match filter {
            None => store.live(),
            Some(f) => matching(&inner, f).count(),
        }
    }

    /// Changes the metadata of the selected points without touching their
    /// vectors, and returns how many were updated (missing ids are skipped).
    ///
    /// Every new value is validated before anything changes, so a batch is
    /// applied completely or not at all.
    pub fn update_metadata(
        &self,
        selector: &PointSelector,
        metadata: Value,
        mode: MetadataMode,
    ) -> Result<usize> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let store = &inner.store;
        let mut slots: Vec<u32> = match selector {
            PointSelector::Ids(ids) => ids.iter().filter_map(|id| store.slot_of(id)).collect(),
            PointSelector::Filter(f) => matching(&inner, f).collect(),
        };
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Ok(0);
        }

        // Logged as an upsert of the whole points, so recovery needs nothing new.
        let updated = slots
            .iter()
            .map(|&slot| {
                let new = match mode {
                    MetadataMode::Replace => metadata.clone(),
                    MetadataMode::Merge => {
                        let mut value = store
                            .metadata(slot)
                            .filter(|m| m.is_object())
                            .cloned()
                            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                        merge_patch(&mut value, &metadata);
                        value
                    }
                };
                let id = store.id(slot);
                Ok(Point {
                    id: id.to_owned(),
                    vector: store.vector(slot).to_vec(),
                    metadata: checked_metadata(id, Some(new))?,
                })
            })
            .collect::<Result<Vec<Point>>>()?;

        inner.engine.log(Op::Upsert { points: &updated })?;
        let n = updated.len();
        // In place: the vectors are unchanged, so the index and slots stay valid.
        let Inner { store, fields, .. } = &mut *inner;
        for (slot, point) in slots.into_iter().zip(updated) {
            store.set_metadata(slot, point.metadata);
            fields.update(slot, store);
        }
        self.bump();
        self.maintain(&mut inner)?;
        Ok(n)
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

    /// Up to `limit` points with an id greater than `cursor` (from the start
    /// if `None`), in ascending byte-wise id order. Paging by id rather than
    /// position means deletes and updates between pages never cause points
    /// to be skipped or repeated.
    pub fn scroll(&self, cursor: Option<&str>, limit: usize) -> ScrollPage {
        let inner = self.inner.read();
        let mut live = inner.store.live_after(cursor);
        let points: Vec<Point> = live
            .by_ref()
            .take(limit)
            .map(|(_, slot)| point_at(&inner.store, slot))
            .collect();
        // Only hand out a cursor if it leads to at least one more point.
        let next_cursor = match (points.last(), live.next()) {
            (Some(last), Some(_)) => Some(last.id.clone()),
            _ => None,
        };
        ScrollPage { points, next_cursor }
    }

    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.search_with(vector, k, &SearchOptions::default())
    }

    pub fn search_with(&self, vector: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        let mut query = vector.to_vec();
        self.validate_vector(&mut query)?;
        let inner = self.inner.read();
        Ok(search_locked(&inner, self.config.metric, &query, k, opts, &[]))
    }

    /// Searches with the stored vector of point `id`, leaving that point out
    /// of the results.
    pub fn search_by_id(&self, id: &str, k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        let inner = self.inner.read();
        let slot = inner
            .store
            .slot_of(id)
            .ok_or_else(|| Error::PointNotFound(id.to_owned()))?;
        let mut query = inner.store.vector(slot).to_vec();
        self.validate_vector(&mut query)?;
        Ok(search_locked(
            &inner,
            self.config.metric,
            &query,
            k,
            opts,
            &[slot],
        ))
    }

    /// Finds points like the `positive` examples and unlike the `negative`
    /// ones. The query is the mean of the positives, pushed away from the
    /// mean of the negatives (`2·mean(pos) − mean(neg)`) when there are any.
    /// The examples themselves are left out of the results.
    pub fn recommend<S: AsRef<str>>(
        &self,
        positive: &[S],
        negative: &[S],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        if positive.is_empty() {
            return Err(Error::invalid("recommend needs at least one positive example"));
        }
        let inner = self.inner.read();
        let store = &inner.store;
        let slots = |ids: &[S]| {
            ids.iter()
                .map(|id| {
                    let id = id.as_ref();
                    store
                        .slot_of(id)
                        .ok_or_else(|| Error::PointNotFound(id.to_owned()))
                })
                .collect::<Result<Vec<u32>>>()
        };
        let (pos, neg) = (slots(positive)?, slots(negative)?);
        let mean = |slots: &[u32]| {
            let mut sum = vec![0.0f64; self.config.dimension];
            for &slot in slots {
                for (s, &x) in sum.iter_mut().zip(store.vector(slot)) {
                    *s += f64::from(x);
                }
            }
            sum.into_iter()
                .map(|s| s / slots.len() as f64)
                .collect::<Vec<f64>>()
        };
        let pos_mean = mean(&pos);
        let mut query: Vec<f32> = if neg.is_empty() {
            pos_mean.iter().map(|&x| x as f32).collect()
        } else {
            let neg_mean = mean(&neg);
            pos_mean
                .iter()
                .zip(&neg_mean)
                .map(|(&p, &n)| (2.0 * p - n) as f32)
                .collect()
        };
        self.validate_vector(&mut query)?;
        let mut exclude = pos;
        exclude.extend(neg);
        exclude.sort_unstable();
        exclude.dedup();
        Ok(search_locked(
            &inner,
            self.config.metric,
            &query,
            k,
            opts,
            &exclude,
        ))
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
            registry.create_with_fields(
                &self.name,
                &self.config,
                &inner.fields.names(),
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
        p.metadata = checked_metadata(&p.id, p.metadata.take())?;
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
        inner.fields.maintain(&inner.store);
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
        inner.fields = FieldIndexes::build(&inner.fields.names(), &store);
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

/// Runs a search over `store`, never returning the (sorted) `exclude` slots.
/// They are rejected inside the index so they do not use up any of the `k`.
fn search_locked(
    inner: &Inner,
    metric: Metric,
    query: &[f32],
    k: usize,
    opts: &SearchOptions,
    exclude: &[u32],
) -> Vec<SearchHit> {
    let (store, index) = (&inner.store, &inner.index);
    let candidates = opts.filter.as_ref().and_then(|f| inner.fields.candidates(f));
    let accept = |slot: u32| {
        store.is_alive(slot)
            && exclude.binary_search(&slot).is_err()
            && candidates.as_ref().is_none_or(|c| c.binary_search(&slot).is_ok())
            && opts
                .filter
                .as_ref()
                .is_none_or(|f| f.matches(store.metadata(slot)))
    };
    let neighbours = match &candidates {
        // Few enough to score them all: exact, and cheaper than a graph
        // walk that would mostly visit points the filter rejects.
        Some(c) if c.len() <= EXACT_SCAN_MAX_CANDIDATES => {
            let mut top = TopK::new(k.min(c.len()).max(1));
            for &slot in c {
                if accept(slot) {
                    top.push(Neighbor {
                        slot,
                        dist: metric.distance(query, store.vector(slot)),
                    });
                }
            }
            top.into_sorted()
        }
        _ => index.search(query, k, opts.ef, store, &accept),
    };
    neighbours
        .into_iter()
        .filter(|n| opts.max_distance.is_none_or(|max| n.dist <= max))
        .map(|n| SearchHit {
            id: store.id(n.slot).to_owned(),
            distance: n.dist,
            metadata: opts
                .include_metadata
                .then(|| store.metadata(n.slot).cloned())
                .flatten(),
            vector: opts.include_vectors.then(|| store.vector(n.slot).to_vec()),
        })
        .collect()
}

fn apply_upsert(inner: &mut Inner, points: Vec<Point>) {
    let Inner {
        store, index, fields, ..
    } = inner;
    let first = store.slots() as u32;
    let mut replaced = 0;
    for p in points {
        replaced += usize::from(store.slot_of(&p.id).is_some());
        store.push(p.id, &p.vector, p.metadata);
    }
    let added = first..store.slots() as u32;
    index.add_batch(added.clone(), store);
    fields.add(added, store);
    fields.removed(replaced);
}

fn apply_delete(inner: &mut Inner, ids: &[String]) -> usize {
    let removed = ids.iter().filter(|id| inner.store.remove(id)).count();
    inner.fields.removed(removed);
    removed
}

/// Live slots matching `filter`, in slot order, through a field index when
/// one applies.
fn matching<'a>(inner: &'a Inner, filter: &'a Filter) -> Box<dyn Iterator<Item = u32> + 'a> {
    let store = &inner.store;
    let matches = move |&slot: &u32| store.is_alive(slot) && filter.matches(store.metadata(slot));
    match inner.fields.candidates(filter) {
        Some(candidates) => Box::new(candidates.into_iter().filter(matches)),
        None => Box::new(store.live_slots().filter(matches)),
    }
}

/// Validates a point's metadata and normalises `null` to `None`: the two
/// are stored the same way, so a point reads back identically however it
/// was recovered.
fn checked_metadata(id: &str, metadata: Option<Value>) -> Result<Option<Value>> {
    if metadata
        .as_ref()
        .is_some_and(|m| deeper_than(m, MAX_METADATA_DEPTH))
    {
        return Err(Error::invalid(format!(
            "point `{id}`: metadata is nested more than {MAX_METADATA_DEPTH} levels deep"
        )));
    }
    Ok(metadata.filter(|m| !m.is_null()))
}

/// Holds a snapshot's points to the rules `validate_point` enforces on
/// upserts. Uploaded archives are untrusted, and their checksum only proves
/// they arrived intact, not that a Quiver server wrote them.
fn check_restored_points(config: &CollectionConfig, store: &VectorStore) -> Result<()> {
    let corrupt =
        |slot: u32, what: &str| Error::Corrupt(format!("snapshot point `{}` {what}", store.id(slot)));
    // Tombstones keep their vectors: the index still measures distances to them.
    for slot in 0..store.slots() as u32 {
        let v = store.vector(slot);
        if v.iter().any(|x| !x.is_finite()) {
            return Err(corrupt(slot, "has a NaN or infinite component"));
        }
        let squared_norm: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if squared_norm > f64::from(f32::MAX) / 4.0 {
            return Err(corrupt(slot, "has too large a magnitude"));
        }
        if config.metric == Metric::Cosine && squared_norm == 0.0 {
            return Err(corrupt(slot, "is a zero vector in a cosine collection"));
        }
    }
    for slot in store.live_slots() {
        if store.id(slot).is_empty() || store.id(slot).len() > MAX_ID_LEN {
            return Err(corrupt(
                slot,
                &format!("has an id outside 1..={MAX_ID_LEN} bytes"),
            ));
        }
        if store
            .metadata(slot)
            .is_some_and(|m| deeper_than(m, MAX_METADATA_DEPTH))
        {
            return Err(corrupt(slot, "has metadata nested too deeply"));
        }
    }
    Ok(())
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
