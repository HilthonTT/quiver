use std::collections::BTreeMap;
use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use quiver_snapshot::{SnapshotInfo, SnapshotRegistry};
use quiver_storage::{StorageKind, StorageOptions, state};
use quiver_types::auth::Access;
use quiver_types::{CollectionConfig, CollectionInfo, CollectionUpdate, Error, Result, validate_name};

use crate::Collection;
use crate::collection::{Env, LocalCollection, META_FILE, with_expiry_field};
use crate::fields::FieldIndexes;
use crate::metrics::Metrics;
use crate::remote::{ClusterBackend, ClusterCatalog, RemoteCollection};

const COLLECTIONS_DIR: &str = "collections";
const SNAPSHOTS_DIR: &str = "snapshots";
const LOCK_FILE: &str = "LOCK";
const ALIASES_FILE: &str = "aliases.json";

/// Collections and the aliases pointing at them, which share one namespace.
#[derive(Debug, Default)]
struct Catalog {
    collections: BTreeMap<String, Arc<Collection>>,
    /// Alias to collection name.
    aliases: BTreeMap<String, String>,
}

impl Catalog {
    fn get(&self, name: &str) -> Option<&Arc<Collection>> {
        self.collections
            .get(name)
            .or_else(|| self.aliases.get(name).and_then(|c| self.collections.get(c)))
    }

    /// Fails if `name` is taken by a collection or an alias.
    fn ensure_free(&self, name: &str) -> Result<()> {
        if self.collections.contains_key(name) {
            return Err(Error::CollectionExists(name.to_owned()));
        }
        if self.aliases.contains_key(name) {
            return Err(Error::AliasExists(name.to_owned()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbOptions {
    pub storage: StorageKind,
    pub fsync: bool,
    pub wal_checkpoint_bytes: u64,
    pub snapshot_dir: Option<PathBuf>,
    pub snapshot_keep: usize,
}

impl Default for DbOptions {
    fn default() -> Self {
        let storage = StorageOptions::default();
        Self {
            storage: StorageKind::File,
            fsync: storage.fsync,
            wal_checkpoint_bytes: storage.wal_checkpoint_bytes,
            snapshot_dir: None,
            snapshot_keep: 5,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestoreOptions {
    pub name: Option<String>,
    pub replace: bool,
}

#[derive(Debug)]
pub struct Database {
    root: Option<PathBuf>,
    options: DbOptions,
    env: Env,
    catalog: RwLock<Catalog>,
    metrics: Metrics,
    /// Set for a database whose collections live on a cluster's nodes.
    cluster: Option<Arc<dyn ClusterBackend>>,
    /// Held for the lifetime of the database to keep other processes out.
    _lock: Option<File>,
}

impl Database {
    pub fn in_memory() -> Self {
        Self::new(
            None,
            DbOptions {
                storage: StorageKind::Memory,
                ..DbOptions::default()
            },
        )
        .expect("in-memory databases need no I/O")
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, DbOptions::default())
    }

    pub fn open_with(path: impl AsRef<Path>, options: DbOptions) -> Result<Self> {
        Self::new(Some(path.as_ref().to_path_buf()), options)
    }

    pub fn new(root: Option<PathBuf>, options: DbOptions) -> Result<Self> {
        if !options.storage.is_available() {
            return Err(Error::Unsupported(format!(
                "the {} engine is not included in this build",
                options.storage
            )));
        }

        if root.is_none() && options.storage.is_durable() {
            return Err(Error::invalid(format!(
                "the {} storage engine needs a database directory",
                options.storage
            )));
        }

        let lock = match &root {
            Some(root) => Some(lock_dir(root)?),
            None => None,
        };
        let snapshot_dir = options
            .snapshot_dir
            .clone()
            .or_else(|| root.as_ref().map(|r| r.join(SNAPSHOTS_DIR)));
        let snapshots = snapshot_dir
            .map(SnapshotRegistry::open)
            .transpose()?
            .map(Arc::new);
        let env = Env {
            storage: options.storage,
            storage_options: StorageOptions {
                fsync: options.fsync,
                wal_checkpoint_bytes: options.wal_checkpoint_bytes,
            },
            snapshots,
            snapshot_keep: options.snapshot_keep,
        };

        let mut collections = BTreeMap::new();
        let mut aliases = BTreeMap::new();
        if let Some(root) = &root {
            let dir = root.join(COLLECTIONS_DIR);
            fs::create_dir_all(&dir)?;
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if !entry.file_type()?.is_dir()
                    || validate_name(&name).is_err()
                    || !entry.path().join(META_FILE).exists()
                {
                    continue;
                }
                let collection = LocalCollection::open(&name, &entry.path(), env.clone())?;
                collections.insert(name, Arc::new(Collection::local(collection)));
            }
            aliases = read_aliases(root)?;
            aliases.retain(|alias, target| {
                let ok = collections.contains_key(target) && !collections.contains_key(alias);
                if !ok {
                    tracing::warn!(alias, collection = %target, "dropping alias to a missing collection");
                }
                ok
            });
            tracing::info!(path = %root.display(), storage = %options.storage, collections = collections.len(), aliases = aliases.len(), "opened database");
        }

        Ok(Self {
            root,
            options,
            env,
            catalog: RwLock::new(Catalog { collections, aliases }),
            metrics: Metrics::default(),
            cluster: None,
            _lock: lock,
        })
    }

    /// A database whose collections are sharded across a cluster's nodes,
    /// which `backend` reaches. It keeps nothing in this process: every
    /// operation goes through the cluster. Snapshots are not available.
    pub fn distributed(backend: Arc<dyn ClusterBackend>) -> Self {
        let options = DbOptions {
            storage: backend.storage(),
            ..DbOptions::default()
        };
        Self {
            root: None,
            env: Env {
                storage: options.storage,
                storage_options: StorageOptions::default(),
                snapshots: None,
                snapshot_keep: options.snapshot_keep,
            },
            options,
            catalog: RwLock::default(),
            metrics: Metrics::default(),
            cluster: Some(backend),
            _lock: None,
        }
    }

    /// Whether this database's collections live on a cluster's nodes.
    pub fn is_distributed(&self) -> bool {
        self.cluster.is_some()
    }

    fn remote(
        &self,
        backend: &Arc<dyn ClusterBackend>,
        name: String,
        sharded: crate::ShardedCollection,
    ) -> Arc<Collection> {
        Arc::new(Collection::remote(RemoteCollection::new(
            name,
            sharded,
            Arc::clone(backend),
        )))
    }

    /// The cluster's catalog, for a distributed database.
    fn cluster_catalog(&self) -> Option<Result<ClusterCatalog>> {
        self.cluster.as_ref().map(|b| b.catalog())
    }

    pub fn path(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn options(&self) -> &DbOptions {
        &self.options
    }

    /// Request metrics, recorded by the REST and gRPC layers.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn collection_dir(&self, name: &str) -> Option<PathBuf> {
        self.root.as_ref().map(|r| r.join(COLLECTIONS_DIR).join(name))
    }

    pub fn create_collection(&self, name: &str, config: CollectionConfig) -> Result<Arc<Collection>> {
        validate_name(name)?;
        if let Some(backend) = &self.cluster {
            LocalCollection::check_creatable(&config, &[], backend.storage())?;
            backend.create_collection(name, &config, &[])?;
            return self.collection(name);
        }

        let mut catalog = self.catalog.write();
        catalog.ensure_free(name)?;
        let dir = self.collection_dir(name);
        let collection = Arc::new(Collection::local(LocalCollection::create(
            name,
            config,
            dir.as_deref(),
            self.env.clone(),
        )?));
        catalog
            .collections
            .insert(name.to_owned(), Arc::clone(&collection));
        Ok(collection)
    }

    /// Creates a collection with these metadata fields already indexed.
    pub fn create_collection_with_fields(
        &self,
        name: &str,
        config: CollectionConfig,
        field_indexes: &[String],
    ) -> Result<Arc<Collection>> {
        FieldIndexes::validate(&with_expiry_field(&config, field_indexes.to_vec()))?;
        if let Some(backend) = &self.cluster {
            validate_name(name)?;
            LocalCollection::check_creatable(&config, field_indexes, backend.storage())?;
            backend.create_collection(name, &config, field_indexes)?;
            return self.collection(name);
        }
        let collection = self.create_collection(name, config)?;
        for field in field_indexes {
            collection.create_field_index(field)?;
        }
        Ok(collection)
    }

    pub fn get_or_create_collection(&self, name: &str, config: CollectionConfig) -> Result<Arc<Collection>> {
        if let Ok(existing) = self.collection(name) {
            if existing.config() != config {
                return Err(Error::invalid(format!(
                    "collection `{name}` exists with a different configuration"
                )));
            }
            return Ok(existing);
        }
        match self.create_collection(name, config.clone()) {
            // Lost a race with a concurrent creator; return theirs.
            Err(Error::CollectionExists(_)) => self.get_or_create_collection(name, config),
            other => other,
        }
    }

    /// The collection called `name`, or the one the alias `name` points to.
    pub fn collection(&self, name: &str) -> Result<Arc<Collection>> {
        if let Some(backend) = &self.cluster {
            return match backend.resolve(name)? {
                Some((canonical, sharded)) => Ok(self.remote(backend, canonical, sharded)),
                None => Err(Error::CollectionNotFound(name.to_owned())),
            };
        }
        self.catalog
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))
    }

    pub fn collections(&self) -> Vec<Arc<Collection>> {
        if let Some(backend) = &self.cluster {
            return match backend.catalog() {
                Ok(catalog) => catalog
                    .collections
                    .into_iter()
                    .map(|(name, sharded)| self.remote(backend, name, sharded))
                    .collect(),
                Err(e) => {
                    tracing::warn!(error = %e, "cannot read the cluster's catalog");
                    Vec::new()
                }
            };
        }
        self.catalog.read().collections.values().cloned().collect()
    }

    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        self.collections().iter().map(|c| c.info()).collect()
    }

    /// Drops a collection, by its own name (not an alias), and every alias
    /// pointing to it.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        if let Some(backend) = &self.cluster {
            return backend.drop_collection(name);
        }
        let mut catalog = self.catalog.write();
        if catalog.aliases.contains_key(name) {
            return Err(Error::invalid(format!(
                "`{name}` is an alias; drop the collection by its own name"
            )));
        }
        let removed = catalog
            .collections
            .remove(name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))?;
        let before = catalog.aliases.len();
        catalog.aliases.retain(|_, target| target != name);
        if catalog.aliases.len() != before {
            self.write_aliases(&catalog.aliases)?;
        }
        self.remove_files(name, removed)
    }

    /// Changes a collection's settings; see [`Collection::update`].
    pub fn update_collection(&self, name: &str, update: &CollectionUpdate) -> Result<CollectionInfo> {
        self.collection(name)?.update(update)
    }

    /// Renames a collection (given by its own name, not an alias). Aliases
    /// follow it; snapshots keep the name they were taken under.
    pub fn rename_collection(&self, from: &str, to: &str) -> Result<Arc<Collection>> {
        validate_name(to)?;
        if let Some(backend) = &self.cluster {
            backend.rename_collection(from, to)?;
            return self.collection(to);
        }
        let mut catalog = self.catalog.write();
        if catalog.aliases.contains_key(from) {
            return Err(Error::invalid(format!(
                "`{from}` is an alias; rename the collection by its own name"
            )));
        }
        let old = catalog
            .collections
            .get(from)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(from.to_owned()))?;
        if from == to {
            return Ok(old);
        }
        catalog.ensure_free(to)?;
        let local = old.as_local().expect("a local database holds local collections");
        let renamed = match local.relocate(to, self.collection_dir(to).as_deref()) {
            Ok(renamed) => Arc::new(Collection::local(renamed)),
            Err(e) => {
                if local.is_closed() {
                    // Unusable until the database is reopened.
                    catalog.collections.remove(from);
                }
                return Err(e);
            }
        };
        catalog.collections.remove(from);
        catalog.collections.insert(to.to_owned(), Arc::clone(&renamed));
        let mut moved = false;
        for target in catalog.aliases.values_mut() {
            if target == from {
                *target = to.to_owned();
                moved = true;
            }
        }
        if moved {
            self.write_aliases(&catalog.aliases)?;
        }
        tracing::info!(from, to, "renamed collection");
        Ok(renamed)
    }

    /// Every alias, with the collection it points to.
    pub fn aliases(&self) -> BTreeMap<String, String> {
        match self.cluster_catalog() {
            Some(Ok(catalog)) => catalog.aliases,
            Some(Err(e)) => {
                tracing::warn!(error = %e, "cannot read the cluster's catalog");
                BTreeMap::new()
            }
            None => self.catalog.read().aliases.clone(),
        }
    }

    /// Names of the collections, and the aliases pointing to them.
    fn names(&self) -> Result<(Vec<String>, BTreeMap<String, String>)> {
        match self.cluster_catalog() {
            Some(catalog) => {
                let catalog = catalog?;
                Ok((catalog.collections.into_keys().collect(), catalog.aliases))
            }
            None => {
                let catalog = self.catalog.read();
                Ok((
                    catalog.collections.keys().cloned().collect(),
                    catalog.aliases.clone(),
                ))
            }
        }
    }

    /// Fails with [`Error::Forbidden`] unless `access` may use the collection
    /// `name` names. An alias is judged by the collection it points to, so a
    /// key scoped to `books` may also use an alias of `books`. Names that do
    /// not exist are judged as they are, so that a scoped key cannot learn
    /// whether a collection outside its scope exists.
    pub fn authorize(&self, access: &Access, name: &str) -> Result<()> {
        if !access.is_scoped() {
            return Ok(());
        }
        let (collections, aliases) = self.names()?;
        let effective = match (collections.iter().any(|c| c == name), aliases.get(name)) {
            (false, Some(target)) => target.as_str(),
            _ => name,
        };
        if access.covers(effective) {
            Ok(())
        } else {
            Err(Error::Forbidden(name.to_owned()))
        }
    }

    /// Fails unless `access` may create or claim the name `name` (a new
    /// collection, a rename target, a restore target or an alias). Aliases
    /// are not resolved: the name itself must be in scope.
    pub fn authorize_name(&self, access: &Access, name: &str) -> Result<()> {
        if access.covers(name) {
            Ok(())
        } else {
            Err(Error::Forbidden(name.to_owned()))
        }
    }

    /// Fails unless `access` may change or remove the alias `alias`: the
    /// alias's own name and the collection it points to now (if it exists)
    /// must both be in scope.
    pub fn authorize_alias(&self, access: &Access, alias: &str) -> Result<()> {
        if !access.is_scoped() {
            return Ok(());
        }
        self.authorize_name(access, alias)?;
        if let Some(target) = self.names()?.1.get(alias)
            && !access.covers(target)
        {
            return Err(Error::Forbidden(alias.to_owned()));
        }
        Ok(())
    }

    /// The collections `access` may see.
    pub fn list_collections_for(&self, access: &Access) -> Vec<CollectionInfo> {
        self.collections()
            .iter()
            .filter(|c| access.covers(c.name()))
            .map(|c| c.info())
            .collect()
    }

    /// The aliases `access` may see: those whose name and collection are both
    /// in scope.
    pub fn aliases_for(&self, access: &Access) -> BTreeMap<String, String> {
        let mut aliases = self.aliases();
        aliases.retain(|alias, target| access.covers(alias) && access.covers(target));
        aliases
    }

    /// Fails unless `access` may use the snapshot `id`, which is judged by
    /// the name of the collection it was taken of.
    pub fn authorize_snapshot(&self, access: &Access, id: &str) -> Result<()> {
        if !access.is_scoped() {
            return Ok(());
        }
        let info = self.snapshots()?.get(id)?;
        self.authorize_name(access, &info.manifest.collection)
    }

    /// Points `alias` at `collection` (by its own name), creating the alias
    /// or atomically moving it from the collection it pointed at before.
    pub fn set_alias(&self, alias: &str, collection: &str) -> Result<()> {
        validate_name(alias)?;
        if let Some(backend) = &self.cluster {
            return backend.set_alias(alias, collection);
        }
        let mut catalog = self.catalog.write();
        if catalog.collections.contains_key(alias) {
            return Err(Error::CollectionExists(alias.to_owned()));
        }
        if !catalog.collections.contains_key(collection) {
            return Err(match catalog.aliases.contains_key(collection) {
                true => Error::invalid(format!(
                    "`{collection}` is an alias; point aliases at collections"
                )),
                false => Error::CollectionNotFound(collection.to_owned()),
            });
        }
        let previous = catalog.aliases.insert(alias.to_owned(), collection.to_owned());
        if let Err(e) = self.write_aliases(&catalog.aliases) {
            match previous {
                Some(p) => catalog.aliases.insert(alias.to_owned(), p),
                None => catalog.aliases.remove(alias),
            };
            return Err(e);
        }
        tracing::info!(alias, collection, "set alias");
        Ok(())
    }

    /// Removes an alias; the collection it points to stays.
    pub fn delete_alias(&self, alias: &str) -> Result<()> {
        if let Some(backend) = &self.cluster {
            return backend.delete_alias(alias);
        }
        let mut catalog = self.catalog.write();
        let target = catalog
            .aliases
            .remove(alias)
            .ok_or_else(|| Error::AliasNotFound(alias.to_owned()))?;
        if let Err(e) = self.write_aliases(&catalog.aliases) {
            catalog.aliases.insert(alias.to_owned(), target);
            return Err(e);
        }
        Ok(())
    }

    fn write_aliases(&self, aliases: &BTreeMap<String, String>) -> Result<()> {
        let Some(root) = &self.root else {
            return Ok(());
        };
        let path = root.join(ALIASES_FILE);
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(aliases)?)?;
        File::open(&tmp)?.sync_all()?;
        fs::rename(&tmp, &path)?;
        state::sync_parent(&path)
    }

    fn remove_files(&self, name: &str, collection: Arc<Collection>) -> Result<()> {
        // Other threads may still hold this handle; stop them from writing
        // into a directory a new collection may soon reuse.
        if let Some(local) = collection.as_local() {
            local.close();
        }
        // Close our handles first; Windows refuses to delete open files.
        drop(collection);
        if let Some(dir) = self.collection_dir(name)
            && dir.exists()
        {
            // Remove the metadata first: if we crash part-way, what remains is
            // skipped on open instead of loading as a half-deleted collection.
            let meta = dir.join(META_FILE);
            if meta.exists() {
                fs::remove_file(&meta)?;
                state::sync_parent(&meta)?;
            }
            fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        for c in self.collections() {
            c.flush()?;
        }
        Ok(())
    }

    pub fn snapshots(&self) -> Result<&SnapshotRegistry> {
        if self.is_distributed() {
            return Err(Error::Unsupported(
                "snapshots are not available in cluster mode yet".into(),
            ));
        }
        self.env
            .snapshots
            .as_deref()
            .ok_or_else(|| Error::Unsupported("this database has no snapshot directory".into()))
    }

    pub fn create_snapshot(&self, collection: &str) -> Result<SnapshotInfo> {
        self.collection(collection)?.create_snapshot()
    }

    /// Snapshots, of one collection (by its name or an alias) or of all.
    pub fn list_snapshots(&self, collection: Option<&str>) -> Result<Vec<SnapshotInfo>> {
        // Snapshots are filed under collections' own names. Ones of dropped
        // collections are still listed under the name they were taken with.
        let resolved = collection.map(|name| {
            let catalog = self.catalog.read();
            match (catalog.collections.contains_key(name), catalog.aliases.get(name)) {
                (false, Some(target)) => target.clone(),
                _ => name.to_owned(),
            }
        });
        self.snapshots()?.list(resolved.as_deref())
    }

    pub fn delete_snapshot(&self, id: &str) -> Result<()> {
        self.snapshots()?.delete(id)
    }

    pub fn restore_snapshot(&self, id: &str, opts: RestoreOptions) -> Result<Arc<Collection>> {
        let loaded = self.snapshots()?.load(id)?;
        let collection = self.restore_loaded(loaded, opts)?;
        tracing::info!(snapshot = id, collection = %collection.name(), points = collection.len(), "restored snapshot");
        Ok(collection)
    }

    /// Replaces (or creates) the collection `name` with the contents of the
    /// archive at `path`, as [`Collection::export_archive`] writes it, and
    /// records that log entries up to `applied` are applied to it. This is
    /// how a shard's replica installs a snapshot of its replicated log.
    pub fn install_archive(
        &self,
        name: &str,
        path: &Path,
        applied: Option<crate::LogPosition>,
    ) -> Result<Arc<Collection>> {
        if self.is_distributed() {
            return Err(Error::Unsupported("archives install into local databases".into()));
        }
        let loaded = quiver_snapshot::read_archive(path)?;
        let collection = self.restore_loaded(
            loaded,
            RestoreOptions {
                name: Some(name.to_owned()),
                replace: true,
            },
        )?;
        if let Some(at) = applied {
            collection.mark_applied(at)?;
        }
        collection.flush()?;
        Ok(collection)
    }

    fn restore_loaded(
        &self,
        loaded: quiver_snapshot::Loaded,
        opts: RestoreOptions,
    ) -> Result<Arc<Collection>> {
        let name = opts.name.unwrap_or(loaded.manifest.collection);
        validate_name(&name)?;
        // Verify everything that can fail before an existing collection is
        // deleted for it: the index, and what creating the collection checks.
        let indexes = LocalCollection::snapshot_indexes(
            &loaded.manifest.config,
            &loaded.store,
            loaded.index.as_deref(),
        )?;
        LocalCollection::check_creatable(
            &loaded.manifest.config,
            &loaded.manifest.field_indexes,
            self.env.storage,
        )?;

        let mut catalog = self.catalog.write();
        if catalog.aliases.contains_key(&name) {
            return Err(Error::AliasExists(name));
        }
        if let Some(existing) = catalog.collections.remove(&name) {
            if !opts.replace {
                catalog.collections.insert(name.clone(), existing);
                return Err(Error::CollectionExists(name));
            }
            self.remove_files(&name, existing)?;
        }
        let dir = self.collection_dir(&name);
        let collection = Arc::new(Collection::local(LocalCollection::restore(
            &name,
            loaded.manifest.config,
            &loaded.manifest.field_indexes,
            loaded.store,
            indexes,
            dir.as_deref(),
            self.env.clone(),
        )?));
        catalog.collections.insert(name.clone(), Arc::clone(&collection));
        Ok(collection)
    }
}

fn read_aliases(root: &Path) -> Result<BTreeMap<String, String>> {
    match fs::read(root.join(ALIASES_FILE)) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(e.into()),
    }
}

fn lock_dir(root: &Path) -> Result<File> {
    fs::create_dir_all(root)?;
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join(LOCK_FILE))?;

    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(Error::invalid(format!(
            "database at {} is already open in another process",
            root.display()
        ))),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}
