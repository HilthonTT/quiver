use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use quiver_index::{Index, Neighbor, TopK};
use quiver_snapshot::{SnapshotInfo, SnapshotRegistry};
use quiver_storage::column::remove_stale_disk_files;
use quiver_storage::{
    Mutation, Op, Recovered, StorageEngine, StorageKind, StorageOptions, StoreSpec, VectorColumn,
    VectorStore, open_engine, state,
};
use quiver_types::{
    CollectionConfig, CollectionInfo, CollectionUpdate, Error, Filter, Fusion, HybridQuery, IndexConfig,
    MAX_ID_LEN, MAX_METADATA_DEPTH, MetadataMode, Metric, Point, PointSelector, Quantization, QueryVector,
    Result, ScoredHit, SearchHit, SearchOptions, SparseVector, VectorConfig, merge_patch,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::expiry;
use crate::fields::FieldIndexes;
use crate::indexes::Indexes;

/// Compact once at least this many slots are tombstoned...
const COMPACT_MIN_TOMBSTONES: usize = 1_024;
/// ...and they make up at least this fraction of all slots.
const COMPACT_RATIO: f64 = 0.3;

pub(crate) const META_FILE: &str = "collection.json";
const ROCKSDB_DIR: &str = "rocksdb";
/// Filtered searches whose field-index candidates number at most this many
/// score every candidate exactly instead of searching the vector index.
const EXACT_SCAN_MAX_CANDIDATES: usize = 20_000;
/// Searches over scalar codes with exact vectors on disk fetch this many
/// times `k` candidates, then re-rank them by their exact distances.
const RESCORE_OVERSAMPLING: usize = 4;
/// The same for binary codes, which are far coarser.
const BINARY_RESCORE_OVERSAMPLING: usize = 20;

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
    config: CollectionConfig,
    store: VectorStore,
    indexes: Indexes,
    fields: FieldIndexes,
    engine: Box<dyn StorageEngine>,
}

impl Inner {
    /// Whether the point in `slot` is alive and has not expired by `now`.
    fn visible(&self, slot: u32, now: f64) -> bool {
        self.store.is_alive(slot) && !self.expired(slot, now)
    }

    /// Whether the point in `slot` has expired by `now`.
    fn expired(&self, slot: u32, now: f64) -> bool {
        self.config
            .expiry
            .as_ref()
            .is_some_and(|e| expiry::is_expired(e, self.store.metadata(slot), now))
    }

    /// The slot of the point `id`, unless it has expired by `now`.
    fn visible_slot(&self, id: &str, now: f64) -> Option<u32> {
        self.store.slot_of(id).filter(|&slot| !self.expired(slot, now))
    }

    /// Live points that have expired by `now`, in slot order.
    fn expired_slots(&self, now: f64) -> Vec<u32> {
        let Some(config) = &self.config.expiry else {
            return Vec::new();
        };
        let filter = expiry::expired_filter(config, now);
        live_matching(self, &filter)
            .filter(|&slot| self.expired(slot, now))
            .collect()
    }

    /// Number of points alive and not expired by `now`.
    fn visible_count(&self, now: f64) -> usize {
        self.store.live() - self.expired_slots(now).len()
    }
}

/// `names` with the field holding expiry times, which must be indexed.
fn with_expiry_field(config: &CollectionConfig, mut names: Vec<String>) -> Vec<String> {
    if let Some(e) = &config.expiry
        && !names.contains(&e.field)
    {
        names.push(e.field.clone());
    }
    names
}

#[derive(Debug)]
pub struct Collection {
    name: String,
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
        if config.uses_disk() && !storage.is_durable() {
            return Err(Error::invalid(
                "vectors can only be kept on disk with a durable storage engine",
            ));
        }

        let meta_dir = dir.filter(|_| storage.is_durable()).map(Path::to_path_buf);
        let field_indexes = with_expiry_field(&config, Vec::new());
        FieldIndexes::validate(&field_indexes)?;
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
                        field_indexes: field_indexes.clone(),
                    },
                )?;
                open_engine(storage, Some(&engine_dir(dir, storage)), env.storage_options)?
            }
            _ => open_engine(StorageKind::Memory, None, env.storage_options)?,
        };

        let store = VectorStore::with_spec(&StoreSpec::for_config(&config, meta_dir.as_deref()))?;
        Ok(Self::from_parts(
            name,
            meta_dir,
            env,
            Inner {
                indexes: Indexes::new(&config),
                fields: FieldIndexes::build(&field_indexes, &store),
                config,
                store,
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

        let Recovered { store, index, replay } =
            engine.recover(&StoreSpec::for_config(&meta.config, Some(dir)))?;
        let indexes = Indexes::from_bytes(index.as_deref(), &meta.config, &store)?;

        FieldIndexes::validate(&meta.field_indexes)?;
        let mut inner = Inner {
            config: meta.config,
            store,
            indexes,
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
        let field_indexes = with_expiry_field(&inner.config, meta.field_indexes);
        FieldIndexes::validate(&field_indexes)?;
        inner.fields = FieldIndexes::build(&field_indexes, &inner.store);
        tracing::debug!(collection = name, storage = %meta.storage, points = inner.store.live(), replayed, "opened collection");

        let collection = Self::from_parts(
            name,
            Some(dir.to_path_buf()),
            Env {
                storage: meta.storage,
                ..env
            },
            inner,
        );
        if replayed > 0 || collection.inner.read().config.uses_disk() {
            // Fold the replayed log into a fresh checkpoint so the next start
            // is fast, and let go of vector files no checkpoint refers to.
            collection.checkpoint(&mut collection.inner.write())?;
        }
        Ok(collection)
    }

    /// Checks a snapshot's contents and loads its indexes.
    pub(crate) fn snapshot_indexes(
        config: &CollectionConfig,
        store: &VectorStore,
        index: Option<&[u8]>,
    ) -> Result<Indexes> {
        config.validate()?;
        check_restored_points(config, store)?;
        Indexes::from_bytes(index, config, store)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore(
        name: &str,
        config: CollectionConfig,
        field_indexes: &[String],
        store: VectorStore,
        indexes: Indexes,
        dir: Option<&Path>,
        env: Env,
    ) -> Result<Self> {
        let field_indexes = with_expiry_field(&config, field_indexes.to_vec());
        FieldIndexes::validate(&field_indexes)?;
        let collection = Self::create(name, config, dir, env)?;
        {
            let mut inner = collection.inner.write();
            // Snapshots load into memory; lay the vectors out as configured.
            let store = relayout(store, &inner.store.spec())?;
            inner.engine.import(&store, &indexes.to_bytes())?;
            inner.fields = FieldIndexes::build(&field_indexes, &store);
            inner.store = store;
            inner.indexes = indexes;
            collection.write_meta_locked(&inner)?;
            collection.cleanup_disk_files(&inner)?;
        }

        Ok(collection)
    }

    /// Moves this collection to `name`, its files to `dir`, and returns the
    /// handle to use from now on. This handle is closed, as if dropped.
    pub(crate) fn relocate(&self, name: &str, dir: Option<&Path>) -> Result<Self> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let placeholder = || open_engine(StorageKind::Memory, None, StorageOptions::default());
        let meta_dir = match (&self.meta_dir, dir.filter(|_| self.storage.is_durable())) {
            (Some(old), Some(new)) => {
                self.checkpoint(&mut inner)?;
                // Close every file first: RocksDB (and Windows) cannot have
                // the directory moved from under open files.
                let spec = inner.store.spec();
                drop(std::mem::replace(&mut inner.engine, placeholder()?));
                drop(std::mem::replace(&mut inner.store, VectorStore::new(1)));
                let moved = (|| {
                    if new.exists() {
                        fs::remove_dir_all(new)?;
                    }
                    fs::rename(old, new)?;
                    state::sync_parent(new)
                })();
                let at = if moved.is_ok() { new } else { old };
                // Everything was checkpointed, so it comes back as it was,
                // though RocksDB may hand the points back in another order.
                let reopened = (|| {
                    let mut engine = open_engine(
                        self.storage,
                        Some(&engine_dir(at, self.storage)),
                        self.env.storage_options,
                    )?;
                    let Recovered { store, index, .. } = engine.recover(&StoreSpec {
                        dir: Some(at.to_path_buf()),
                        ..spec
                    })?;
                    let indexes = Indexes::from_bytes(index.as_deref(), &inner.config, &store)?;
                    Ok::<_, Error>((engine, store, indexes))
                })();
                let (engine, store, indexes) = match reopened {
                    Ok(parts) => parts,
                    Err(e) => {
                        // The files are intact, but this handle has nothing
                        // left to serve; reopening the database brings it back.
                        self.closed.store(true, Ordering::Release);
                        tracing::error!(collection = %self.name, dir = %at.display(), error = %e, "could not reopen collection after moving it");
                        return Err(e);
                    }
                };
                inner.fields = FieldIndexes::build(&inner.fields.names(), &store);
                inner.indexes = indexes;
                inner.engine = engine;
                inner.store = store;
                moved?;
                Some(new.to_path_buf())
            }
            _ => None,
        };
        self.closed.store(true, Ordering::Release);
        let parts = Inner {
            config: inner.config.clone(),
            store: std::mem::replace(&mut inner.store, VectorStore::new(1)),
            indexes: inner.indexes.clone(),
            fields: std::mem::take(&mut inner.fields),
            engine: std::mem::replace(&mut inner.engine, placeholder()?),
        };
        Ok(Self::from_parts(name, meta_dir, self.env.clone(), parts))
    }

    fn from_parts(name: &str, meta_dir: Option<PathBuf>, env: Env, inner: Inner) -> Self {
        Self {
            name: name.to_owned(),
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

    /// The collection's current configuration.
    pub fn config(&self) -> CollectionConfig {
        self.inner.read().config.clone()
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

    /// Number of points, not counting expired ones.
    pub fn len(&self) -> usize {
        self.inner.read().visible_count(expiry::now())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes of memory the vectors take, tombstones included.
    pub fn vector_bytes(&self) -> usize {
        self.inner.read().store.vector_bytes()
    }

    pub fn info(&self) -> CollectionInfo {
        let inner = self.inner.read();
        CollectionInfo {
            name: self.name.clone(),
            config: inner.config.clone(),
            count: inner.visible_count(expiry::now()),
            tombstones: inner.store.tombstones(),
            field_indexes: inner.fields.names(),
        }
    }

    /// Changes the collection's index, quantization or on-disk settings.
    ///
    /// Indexes whose settings changed (other than `ef_search`) are rebuilt,
    /// and vectors are moved to or from disk. That can take a while; reads
    /// carry on meanwhile, writes wait.
    pub fn update(&self, update: &CollectionUpdate) -> Result<CollectionInfo> {
        let inner = self.inner.upgradable_read();
        self.ensure_open()?;
        let old = &inner.config;
        let new = update.apply(old)?;
        if new.uses_disk() && !self.storage.is_durable() {
            return Err(Error::invalid(
                "vectors can only be kept on disk with a durable storage engine",
            ));
        }
        if &new == old {
            drop(inner);
            return Ok(self.info());
        }

        // Expiry times already stored must be numbers, and get indexed.
        let mut fields = None;
        if let Some(e) = new.expiry.as_ref().filter(|_| new.expiry != old.expiry) {
            for slot in inner.store.live_slots() {
                expiry::check(e, inner.store.id(slot), inner.store.metadata(slot))?;
            }
            let names = with_expiry_field(&new, inner.fields.names());
            if names != inner.fields.names() {
                FieldIndexes::validate(&names)?;
                fields = Some(FieldIndexes::build(&names, &inner.store));
            }
        }

        // Lay out the vectors anew where that changed.
        let mut columns = Vec::new();
        let dir = self.meta_dir.as_deref();
        let target = StoreSpec::for_config(&new, dir);
        let changed = std::iter::once((None, target.main))
            .chain(target.named.iter().map(|(n, c)| (Some(n.as_str()), *c)));
        for (using, spec) in changed {
            let column = inner
                .store
                .column(using)
                .expect("store has the configured vectors");
            if column.spec() != spec {
                let file = using.unwrap_or("main");
                columns.push((using.map(str::to_owned), column.convert(spec, dir, file)?));
            }
        }

        // Rebuild indexes whose settings changed; `ef_search` alone needs none.
        let mut indexes = inner.indexes.clone();
        let rebuilt =
            |o: &VectorConfig, n: &VectorConfig, index: &mut Index, column: &VectorColumn| -> Result<bool> {
                if o.index == n.index {
                    return Ok(false);
                }
                if same_but_ef_search(&o.index, &n.index) {
                    *index = Index::from_bytes(&index.to_bytes(), &n.index, n.metric, column)?;
                    return Ok(false);
                }
                *index = Index::build(&n.index, n.metric, column);
                Ok(true)
            };
        let mut rebuilds = Vec::new();
        if rebuilt(
            &old.main_vector(),
            &new.main_vector(),
            &mut indexes.main,
            inner.store.main(),
        )? {
            rebuilds.push(String::from("main"));
        }
        for (name, n) in &new.vectors {
            let column = inner
                .store
                .column(Some(name))
                .expect("store has the configured vectors");
            let index = indexes
                .named
                .get_mut(name)
                .expect("indexes cover the configured vectors");
            if rebuilt(&old.vectors[name], n, index, column)? {
                rebuilds.push(name.clone());
            }
        }

        let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
        let mut previous_columns = Vec::new();
        for (using, column) in columns {
            let old = inner.store.replace_column(using.as_deref(), column);
            previous_columns.push((using, old));
        }
        let previous_indexes = std::mem::replace(&mut inner.indexes, indexes);
        let previous_config = std::mem::replace(&mut inner.config, new);
        let previous_fields = fields.map(|f| std::mem::replace(&mut inner.fields, f));
        // Old vector files stay until both are saved, so going back works.
        let persisted = self
            .checkpoint_keeping_files(&mut inner)
            .and_then(|()| self.write_meta_locked(&inner));
        if let Err(e) = persisted {
            tracing::error!(collection = %self.name, error = %e, "could not save the new settings");
            for (using, column) in previous_columns {
                inner.store.replace_column(using.as_deref(), column);
            }
            inner.indexes = previous_indexes;
            inner.config = previous_config;
            if let Some(f) = previous_fields {
                inner.fields = f;
            }
            return Err(e);
        }
        drop(previous_columns);
        self.cleanup_disk_files(&inner)?;
        self.bump();
        tracing::info!(collection = %self.name, rebuilt = ?rebuilds, "updated collection settings");
        drop(inner);
        Ok(self.info())
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
        if let Err(e) = self.write_meta_locked(&inner) {
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
        if inner.config.expiry.as_ref().is_some_and(|e| e.field == field) {
            return Err(Error::invalid(format!(
                "`{field}` holds expiry times and stays indexed while expiry is on"
            )));
        }
        let names: Vec<String> = inner.fields.names().into_iter().filter(|n| n != field).collect();
        if names.len() == inner.fields.names().len() {
            return Ok(false);
        }
        let built = FieldIndexes::build(&names, &inner.store);
        let previous = std::mem::replace(&mut inner.fields, built);
        if let Err(e) = self.write_meta_locked(&inner) {
            inner.fields = previous;
            return Err(e);
        }
        tracing::info!(collection = %self.name, field, "dropped field index");
        Ok(true)
    }

    /// Persists the configuration and indexed fields, for durable collections.
    fn write_meta_locked(&self, inner: &Inner) -> Result<()> {
        let Some(dir) = &self.meta_dir else {
            return Ok(());
        };
        write_meta(
            dir,
            &CollectionMeta {
                config: inner.config.clone(),
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
        // What validation depends on (dimensions, metrics, the sets of
        // vectors) never changes, so the config may be read up front.
        let config = self.inner.read().config.clone();
        for p in &mut points {
            validate_point(&config, p)?;
        }
        if points.is_empty() {
            return Ok(0);
        }
        let n = points.len();

        let mut inner = self.inner.write();
        self.ensure_open()?;
        // Under the lock: expiry settings can change.
        if let Some(e) = &inner.config.expiry {
            let now = expiry::now();
            for p in &mut points {
                let metadata = expiry::stamp(e, &p.id, p.metadata.take(), now)?;
                p.metadata = checked_metadata(&p.id, metadata)?;
            }
        }
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
        // Expired points go too, but were already gone as far as callers know.
        let now = expiry::now();
        let mut visible: Vec<u32> = ids.iter().filter_map(|id| inner.visible_slot(id, now)).collect();
        visible.sort_unstable();
        visible.dedup();
        inner.engine.log(Op::Delete { ids: &ids })?;
        apply_delete(&mut inner, &ids);
        self.bump();
        self.maintain(&mut inner)?;
        Ok(visible.len())
    }

    /// Deletes every point whose metadata matches `filter`; returns how many.
    pub fn delete_where(&self, filter: &Filter) -> Result<usize> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let ids: Vec<String> = matching(&inner, filter, expiry::now())
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
        let now = expiry::now();
        match filter {
            None => inner.visible_count(now),
            Some(f) => matching(&inner, f, now).count(),
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
        let now = expiry::now();
        let mut slots: Vec<u32> = match selector {
            PointSelector::Ids(ids) => ids.iter().filter_map(|id| inner.visible_slot(id, now)).collect(),
            PointSelector::Filter(f) => matching(&inner, f, now).collect(),
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
                let mut point = store.point(slot);
                point.metadata = checked_metadata(&point.id, Some(new))?;
                if let Some(e) = &inner.config.expiry {
                    let stamped = expiry::stamp(e, &point.id, point.metadata.take(), now)?;
                    point.metadata = checked_metadata(&point.id, stamped)?;
                }
                Ok(point)
            })
            .collect::<Result<Vec<Point>>>()?;

        inner.engine.log(Op::Upsert { points: &updated })?;
        let n = updated.len();
        // In place: the vectors are unchanged, so the indexes and slots stay valid.
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
        inner
            .visible_slot(id, expiry::now())
            .map(|slot| inner.store.point(slot))
    }

    pub fn get_many<S: AsRef<str>>(&self, ids: &[S]) -> Vec<Point> {
        let inner = self.inner.read();
        let now = expiry::now();
        ids.iter()
            .filter_map(|id| inner.visible_slot(id.as_ref(), now))
            .map(|slot| inner.store.point(slot))
            .collect()
    }

    /// Up to `limit` points with an id greater than `cursor` (from the start
    /// if `None`), in ascending byte-wise id order. Paging by id rather than
    /// position means deletes and updates between pages never cause points
    /// to be skipped or repeated.
    pub fn scroll(&self, cursor: Option<&str>, limit: usize) -> ScrollPage {
        let inner = self.inner.read();
        let now = expiry::now();
        let mut live = inner
            .store
            .live_after(cursor)
            .filter(|&(_, slot)| !inner.expired(slot, now));
        let points: Vec<Point> = live
            .by_ref()
            .take(limit)
            .map(|(_, slot)| inner.store.point(slot))
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

    /// The `k` points nearest to `vector` in the dense vector `opts.using`.
    pub fn search_with(&self, vector: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        let inner = self.inner.read();
        let using = opts.using.as_deref();
        let config = inner.config.dense(using)?;
        let mut query = vector.to_vec();
        validate_vector(&config, &mut query)?;
        let found = search_dense(&inner, using, &query, k, opts, &[]);
        Ok(hits(&inner.store, found, opts))
    }

    /// Searches with the stored vector of point `id`, leaving that point out
    /// of the results.
    pub fn search_by_id(&self, id: &str, k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        let inner = self.inner.read();
        let using = opts.using.as_deref();
        let config = inner.config.dense(using)?;
        let slot = inner
            .visible_slot(id, expiry::now())
            .ok_or_else(|| Error::PointNotFound(id.to_owned()))?;
        let column = inner
            .store
            .column(using)
            .expect("store has the configured vectors");
        let mut query = column.vector(slot).to_vec();
        validate_vector(&config, &mut query)?;
        let found = search_dense(&inner, using, &query, k, opts, &[slot]);
        Ok(hits(&inner.store, found, opts))
    }

    /// The `k` points scoring highest against `vector` in the sparse vector
    /// `name`. Hits' distances are their negated scores (dot products,
    /// weighted as the vector's `modifier` says), so lower is better, as
    /// everywhere else; `opts.max_distance` applies to them likewise.
    pub fn search_sparse(
        &self,
        name: &str,
        vector: &SparseVector,
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        let mut query = vector.clone();
        query.normalize()?;
        let inner = self.inner.read();
        inner.config.sparse(name)?;
        let found = search_sparse(&inner, name, &query, k, opts);
        Ok(hits(&inner.store, found, opts))
    }

    /// Runs several searches, dense and sparse, and fuses their rankings.
    /// Returns the `k` best points by fused score.
    pub fn query(&self, query: &HybridQuery, k: usize) -> Result<Vec<ScoredHit>> {
        if query.prefetch.is_empty() {
            return Err(Error::invalid("a query needs at least one search to fuse"));
        }
        let inner = self.inner.read();
        let mut ranked = Vec::with_capacity(query.prefetch.len());
        for p in &query.prefetch {
            if !p.weight.is_finite() || p.weight < 0.0 {
                return Err(Error::invalid("search weights must be finite and not negative"));
            }
            if p.weight == 0.0 {
                // It would add nothing to any score.
                continue;
            }
            let opts = SearchOptions {
                using: None,
                filter: query.filter.clone(),
                ef: p.ef,
                include_vectors: false,
                include_metadata: false,
                max_distance: p.max_distance,
            };
            let found = match &p.query {
                QueryVector::Dense { vector, using } => {
                    let using = using.as_deref();
                    let mut v = vector.clone();
                    validate_vector(&inner.config.dense(using)?, &mut v)?;
                    search_dense(&inner, using, &v, p.limit, &opts, &[])
                }
                QueryVector::Sparse { vector, using } => {
                    inner.config.sparse(using)?;
                    let mut v = vector.clone();
                    v.normalize()?;
                    search_sparse(&inner, using, &v, p.limit, &opts)
                }
            };
            ranked.push((p.weight, found));
        }

        let Fusion::Rrf { k: rrf_k } = query.fusion;
        let mut scores: HashMap<u32, f64> = HashMap::new();
        for (weight, found) in &ranked {
            for (rank, n) in found.iter().enumerate() {
                *scores.entry(n.slot).or_default() +=
                    f64::from(*weight) / (f64::from(rrf_k) + rank as f64 + 1.0);
            }
        }
        let mut fused: Vec<(u32, f64)> = scores.into_iter().collect();
        fused.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        fused.truncate(k);

        let store = &inner.store;
        Ok(fused
            .into_iter()
            .map(|(slot, score)| {
                let (vector, vectors, sparse_vectors) = vectors_of(store, slot, query.include_vectors);
                ScoredHit {
                    id: store.id(slot).to_owned(),
                    score: score as f32,
                    metadata: query
                        .include_metadata
                        .then(|| store.metadata(slot).cloned())
                        .flatten(),
                    vector,
                    vectors,
                    sparse_vectors,
                }
            })
            .collect())
    }

    /// Finds points like the `positive` examples and unlike the `negative`
    /// ones, in the dense vector `opts.using`. The query is the mean of the
    /// positives, pushed away from the mean of the negatives
    /// (`2·mean(pos) − mean(neg)`) when there are any. The examples
    /// themselves are left out of the results.
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
        let using = opts.using.as_deref();
        let config = inner.config.dense(using)?;
        let store = &inner.store;
        let column = store.column(using).expect("store has the configured vectors");
        let now = expiry::now();
        let slots = |ids: &[S]| {
            ids.iter()
                .map(|id| {
                    let id = id.as_ref();
                    inner
                        .visible_slot(id, now)
                        .ok_or_else(|| Error::PointNotFound(id.to_owned()))
                })
                .collect::<Result<Vec<u32>>>()
        };
        let (pos, neg) = (slots(positive)?, slots(negative)?);
        let mean = |slots: &[u32]| {
            let mut sum = vec![0.0f64; config.dimension];
            for &slot in slots {
                for (s, &x) in sum.iter_mut().zip(column.vector(slot).iter()) {
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
        validate_vector(&config, &mut query)?;
        let mut exclude = pos;
        exclude.extend(neg);
        exclude.sort_unstable();
        exclude.dedup();
        let found = search_dense(&inner, using, &query, k, opts, &exclude);
        Ok(hits(store, found, opts))
    }

    pub fn flush(&self) -> Result<()> {
        self.checkpoint(&mut self.inner.write())
    }

    pub fn compact(&self) -> Result<()> {
        let mut inner = self.inner.write();
        self.ensure_open()?;
        self.compact_locked(&mut inner)?;
        self.bump();
        self.checkpoint(&mut inner)
    }

    /// Deletes the points that have expired; returns how many. Reads never
    /// see expired points anyway; this frees the space they take.
    pub fn purge_expired(&self) -> Result<usize> {
        // Look first, so sweeps that find nothing never block writers.
        if self.inner.read().expired_slots(expiry::now()).is_empty() {
            return Ok(0);
        }
        let mut inner = self.inner.write();
        self.ensure_open()?;
        let ids: Vec<String> = inner
            .expired_slots(expiry::now())
            .into_iter()
            .map(|slot| inner.store.id(slot).to_owned())
            .collect();
        if ids.is_empty() {
            return Ok(0);
        }
        inner.engine.log(Op::Delete { ids: &ids })?;
        let removed = apply_delete(&mut inner, &ids);
        self.bump();
        self.maintain(&mut inner)?;
        tracing::debug!(collection = %self.name, removed, "purged expired points");
        Ok(removed)
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
                &inner.config,
                &inner.fields.names(),
                &inner.store,
                Some(&inner.indexes.to_bytes()),
            )?
        };
        registry.prune(&self.name, self.env.snapshot_keep)?;
        Ok(info)
    }

    /// Writes the collection's points and indexes, every vector included.
    pub fn export_state(&self, w: impl std::io::Write) -> Result<()> {
        let inner = self.inner.read();
        state::encode(
            w,
            &inner.store,
            Some(&inner.indexes.to_bytes()),
            state::Columns::Inline,
        )?;
        Ok(())
    }

    pub(crate) fn close(&self) {
        let _inner = self.inner.write();
        self.closed.store(true, Ordering::Release);
    }

    /// Whether this handle was dropped, replaced or moved.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
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

    fn maintain(&self, inner: &mut Inner) -> Result<()> {
        inner.fields.maintain(&inner.store);
        inner.indexes.maintain(&inner.store);
        let tombstones = inner.store.tombstones();
        if tombstones >= COMPACT_MIN_TOMBSTONES
            && tombstones as f64 >= inner.store.slots() as f64 * COMPACT_RATIO
        {
            self.compact_locked(inner)?;
        }
        if inner.engine.needs_checkpoint() {
            self.checkpoint(inner)?;
        }
        Ok(())
    }

    fn compact_locked(&self, inner: &mut Inner) -> Result<()> {
        let old = &inner.store;
        let mut store = old.empty_like()?;
        for slot in old.live_slots() {
            store.push_point(old.point(slot));
        }
        let indexes = Indexes::build(&inner.config, &store);
        tracing::debug!(collection = %self.name, reclaimed = old.tombstones(), "compacted collection");
        inner.fields = FieldIndexes::build(&inner.fields.names(), &store);
        inner.store = store;
        inner.indexes = indexes;
        Ok(())
    }

    fn checkpoint(&self, inner: &mut Inner) -> Result<()> {
        self.checkpoint_keeping_files(inner)?;
        self.cleanup_disk_files(inner)
    }

    /// A checkpoint that leaves vector files it no longer refers to in place.
    fn checkpoint_keeping_files(&self, inner: &mut Inner) -> Result<()> {
        if !self.storage.is_durable() || self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let index = inner.indexes.to_bytes();
        inner.engine.checkpoint(&inner.store, &index)
    }

    /// Deletes vector files the last checkpoint no longer refers to: those
    /// of columns since compacted or moved.
    fn cleanup_disk_files(&self, inner: &Inner) -> Result<()> {
        match &self.meta_dir {
            Some(dir) => remove_stale_disk_files(dir, &inner.store.disk_files()),
            None => Ok(()),
        }
    }
}

/// Whether two index settings differ only in the HNSW search width.
fn same_but_ef_search(a: &IndexConfig, b: &IndexConfig) -> bool {
    match (a, b) {
        (IndexConfig::Hnsw(a), IndexConfig::Hnsw(b)) => {
            let mut a = *a;
            a.ef_search = b.ef_search;
            a == *b
        }
        _ => false,
    }
}

/// The accepted slots nearest to `query` in the dense vector `using`,
/// never returning the (sorted) `exclude` slots. They are rejected inside
/// the index so they do not use up any of the `k`.
fn search_dense(
    inner: &Inner,
    using: Option<&str>,
    query: &[f32],
    k: usize,
    opts: &SearchOptions,
    exclude: &[u32],
) -> Vec<Neighbor> {
    let store = &inner.store;
    let metric = inner.config.dense(using).expect("checked by the caller").metric;
    let (index, column) = inner.indexes.dense(store, using);
    let candidates = opts.filter.as_ref().and_then(|f| inner.fields.candidates(f));
    let accept = acceptor(inner, opts.filter.as_ref(), candidates.as_deref(), exclude);
    let mut neighbours = match &candidates {
        // Few enough to score them all: exact, and cheaper than a graph
        // walk that would mostly visit points the filter rejects.
        Some(c) if c.len() <= EXACT_SCAN_MAX_CANDIDATES => {
            let mut top = TopK::new(k.min(c.len()).max(1));
            for &slot in c {
                if accept(slot) {
                    top.push(Neighbor {
                        slot,
                        dist: column.exact_distance(metric, query, slot),
                    });
                }
            }
            top.into_sorted()
        }
        _ if column.rescores() => {
            // Search the codes in memory, then re-rank by the exact vectors.
            let oversampling = match column.quantization() {
                Quantization::Binary => BINARY_RESCORE_OVERSAMPLING,
                _ => RESCORE_OVERSAMPLING,
            };
            let wide = k.saturating_mul(oversampling);
            let mut found = index.search(query, wide, opts.ef.map(|ef| ef.max(wide)), column, &accept);
            for n in &mut found {
                n.dist = column.exact_distance(metric, query, n.slot);
            }
            found.sort_unstable();
            found.truncate(k);
            found
        }
        _ => index.search(query, k, opts.ef, column, &accept),
    };
    neighbours.retain(|n| opts.max_distance.is_none_or(|max| n.dist <= max));
    neighbours
}

/// The accepted slots scoring highest against `query` in the sparse vector `name`.
fn search_sparse(
    inner: &Inner,
    name: &str,
    query: &SparseVector,
    k: usize,
    opts: &SearchOptions,
) -> Vec<Neighbor> {
    let modifier = inner.config.sparse(name).expect("checked by the caller").modifier;
    let candidates = opts.filter.as_ref().and_then(|f| inner.fields.candidates(f));
    let accept = acceptor(inner, opts.filter.as_ref(), candidates.as_deref(), &[]);
    let mut found = inner.indexes.sparse[name].search(query, k, modifier, &accept);
    found.retain(|n| opts.max_distance.is_none_or(|max| n.dist <= max));
    found
}

/// Whether a slot may appear in results: alive, not expired, not excluded,
/// among the field-index candidates if any, and matching the filter.
fn acceptor<'a>(
    inner: &'a Inner,
    filter: Option<&'a Filter>,
    candidates: Option<&'a [u32]>,
    exclude: &'a [u32],
) -> impl Fn(u32) -> bool + Sync + 'a {
    let store = &inner.store;
    let now = expiry::now();
    move |slot: u32| {
        inner.visible(slot, now)
            && exclude.binary_search(&slot).is_err()
            && candidates.is_none_or(|c| c.binary_search(&slot).is_ok())
            && filter.is_none_or(|f| f.matches(store.metadata(slot)))
    }
}

fn vectors_of(
    store: &VectorStore,
    slot: u32,
    include: bool,
) -> (
    Option<Vec<f32>>,
    quiver_types::NamedVectors,
    quiver_types::SparseVectors,
) {
    if !include {
        return Default::default();
    }
    let p = store.point(slot);
    (Some(p.vector), p.vectors, p.sparse_vectors)
}

fn hits(store: &VectorStore, found: Vec<Neighbor>, opts: &SearchOptions) -> Vec<SearchHit> {
    found
        .into_iter()
        .map(|n| {
            let (vector, vectors, sparse_vectors) = vectors_of(store, n.slot, opts.include_vectors);
            SearchHit {
                id: store.id(n.slot).to_owned(),
                distance: n.dist,
                metadata: opts
                    .include_metadata
                    .then(|| store.metadata(n.slot).cloned())
                    .flatten(),
                vector,
                vectors,
                sparse_vectors,
            }
        })
        .collect()
}

fn apply_upsert(inner: &mut Inner, points: Vec<Point>) {
    let Inner {
        store,
        indexes,
        fields,
        ..
    } = inner;
    let first = store.slots() as u32;
    let mut replaced = 0;
    for p in points {
        if let Some(old) = store.slot_of(&p.id) {
            indexes.remove(old, store);
            replaced += 1;
        }
        store.push_point(p);
    }
    let added = first..store.slots() as u32;
    indexes.add(added.clone(), store);
    fields.add(added, store);
    fields.removed(replaced);
}

fn apply_delete(inner: &mut Inner, ids: &[String]) -> usize {
    let Inner { store, indexes, .. } = inner;
    let mut removed = 0;
    for id in ids {
        if let Some(slot) = store.slot_of(id) {
            indexes.remove(slot, store);
            store.remove(id);
            removed += 1;
        }
    }
    inner.fields.removed(removed);
    removed
}

/// Slots of points alive and not expired by `now` matching `filter`, in
/// slot order, through a field index when one applies.
fn matching<'a>(inner: &'a Inner, filter: &'a Filter, now: f64) -> impl Iterator<Item = u32> + 'a {
    live_matching(inner, filter).filter(move |&slot| !inner.expired(slot, now))
}

/// Live slots matching `filter`, in slot order, expired or not.
fn live_matching<'a>(inner: &'a Inner, filter: &'a Filter) -> Box<dyn Iterator<Item = u32> + 'a> {
    let store = &inner.store;
    let matches = move |&slot: &u32| store.is_alive(slot) && filter.matches(store.metadata(slot));
    match inner.fields.candidates(filter) {
        Some(candidates) => Box::new(candidates.into_iter().filter(matches)),
        None => Box::new(store.live_slots().filter(matches)),
    }
}

/// Checks a point against `config`, normalising its vectors in place.
fn validate_point(config: &CollectionConfig, p: &mut Point) -> Result<()> {
    if p.id.is_empty() || p.id.len() > MAX_ID_LEN {
        return Err(Error::invalid(format!(
            "point ids must be 1..={MAX_ID_LEN} bytes"
        )));
    }
    let id = p.id.clone();
    let invalid = |what: String| Error::invalid(format!("point `{id}`: {what}"));
    p.metadata = checked_metadata(&p.id, p.metadata.take())?;
    validate_vector(&config.main_vector(), &mut p.vector).map_err(|e| invalid(e.to_string()))?;
    if let Some(name) = p.vectors.keys().find(|n| !config.vectors.contains_key(*n)) {
        return Err(invalid(format!("the collection has no vector named `{name}`")));
    }
    for (name, v) in &config.vectors {
        let vector = p
            .vectors
            .get_mut(name)
            .ok_or_else(|| invalid(format!("vector `{name}` is missing")))?;
        validate_vector(v, vector).map_err(|e| invalid(format!("vector `{name}`: {e}")))?;
    }
    for (name, v) in &mut p.sparse_vectors {
        if !config.sparse_vectors.contains_key(name) {
            return Err(invalid(format!(
                "the collection has no sparse vector named `{name}`"
            )));
        }
        v.normalize()
            .map_err(|e| invalid(format!("sparse vector `{name}`: {e}")))?;
    }
    p.sparse_vectors.retain(|_, v| !v.is_empty());
    Ok(())
}

/// Checks a dense vector against its settings, normalising it for cosine.
fn validate_vector(config: &VectorConfig, v: &mut [f32]) -> Result<()> {
    if v.len() != config.dimension {
        return Err(Error::DimensionMismatch {
            expected: config.dimension,
            actual: v.len(),
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::invalid("vectors must not contain NaN or infinity"));
    }
    if !config.metric.prepare(v) {
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
    // Tombstones keep their vectors: the indexes still measure distances to them.
    for (name, column) in store.columns() {
        let using = (!name.is_empty()).then_some(name);
        let metric = config
            .dense(using)
            .map_err(|_| Error::Corrupt(format!("snapshot has unknown vector `{name}`")))?
            .metric;
        for slot in 0..store.slots() as u32 {
            let v = column.vector(slot);
            if v.iter().any(|x| !x.is_finite()) {
                return Err(corrupt(slot, "has a NaN or infinite component"));
            }
            let squared_norm: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if squared_norm > f64::from(f32::MAX) / 4.0 {
                return Err(corrupt(slot, "has too large a magnitude"));
            }
            if metric == Metric::Cosine && squared_norm == 0.0 {
                return Err(corrupt(slot, "is a zero vector in a cosine collection"));
            }
        }
    }
    for (_, vectors) in store.sparse_columns() {
        for (slot, v) in vectors.iter().enumerate() {
            let mut normalized = v.clone();
            if normalized.normalize().is_err() || normalized != *v {
                return Err(corrupt(slot as u32, "has an invalid sparse vector"));
            }
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

/// `store` with its columns held as `spec` says, slots unchanged.
fn relayout(mut store: VectorStore, spec: &StoreSpec) -> Result<VectorStore> {
    let targets =
        std::iter::once((None, spec.main)).chain(spec.named.iter().map(|(n, c)| (Some(n.as_str()), *c)));
    for (using, target) in targets {
        let column = store.column(using).expect("snapshot has the configured vectors");
        if column.spec() != target {
            let converted = column.convert(target, spec.dir.as_deref(), using.unwrap_or("main"))?;
            store.replace_column(using, converted);
        }
    }
    Ok(store)
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
