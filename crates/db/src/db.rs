use std::collections::BTreeMap;
use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use vectordb_snapshot::{SnapshotInfo, SnapshotRegistry};
use vectordb_storage::{StorageKind, StorageOptions};
use vectordb_types::{CollectionConfig, CollectionInfo, Error, Result, validate_name};

use crate::collection::{Collection, Env, META_FILE};

const COLLECTIONS_DIR: &str = "collections";
const SNAPSHOTS_DIR: &str = "snapshots";
const LOCK_FILE: &str = "LOCK";

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
    collections: RwLock<BTreeMap<String, Arc<Collection>>>,
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
                let collection = Collection::open(&name, &entry.path(), env.clone())?;
                collections.insert(name, Arc::new(collection));
            }
            tracing::info!(path = %root.display(), storage = %options.storage, collections = collections.len(), "opened database");
        }

        Ok(Self {
            root,
            options,
            env,
            collections: RwLock::new(collections),
            _lock: lock,
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn options(&self) -> &DbOptions {
        &self.options
    }

    fn collection_dir(&self, name: &str) -> Option<PathBuf> {
        self.root
            .as_ref()
            .map(|r| r.join(COLLECTIONS_DIR).join(name))
    }

    pub fn create_collection(
        &self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<Arc<Collection>> {
        validate_name(name)?;

        let mut collections = self.collections.write();
        if collections.contains_key(name) {
            return Err(Error::CollectionExists(name.to_owned()));
        }
        let dir = self.collection_dir(name);
        let collection = Arc::new(Collection::create(
            name,
            config,
            dir.as_deref(),
            self.env.clone(),
        )?);
        collections.insert(name.to_owned(), Arc::clone(&collection));
        Ok(collection)
    }

    pub fn get_or_create_collection(
        &self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<Arc<Collection>> {
        if let Ok(existing) = self.collection(name) {
            if existing.config() != &config {
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

    pub fn collection(&self, name: &str) -> Result<Arc<Collection>> {
        self.collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))
    }

    pub fn collections(&self) -> Vec<Arc<Collection>> {
        self.collections.read().values().cloned().collect()
    }

    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        self.collections.read().values().map(|c| c.info()).collect()
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let mut collections = self.collections.write();
        let removed = collections
            .remove(name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))?;
        self.remove_files(name, removed)
    }

    fn remove_files(&self, name: &str, collection: Arc<Collection>) -> Result<()> {
        // Other threads may still hold this handle; stop them from writing
        // into a directory a new collection may soon reuse.
        collection.close();
        // Close our handles first; Windows refuses to delete open files.
        drop(collection);
        if let Some(dir) = self.collection_dir(name) {
            if dir.exists() {
                fs::remove_dir_all(dir)?;
            }
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
        self.env
            .snapshots
            .as_deref()
            .ok_or_else(|| Error::Unsupported("this database has no snapshot directory".into()))
    }

    pub fn create_snapshot(&self, collection: &str) -> Result<SnapshotInfo> {
        self.collection(collection)?.create_snapshot()
    }

    pub fn list_snapshots(&self, collection: Option<&str>) -> Result<Vec<SnapshotInfo>> {
        self.snapshots()?.list(collection)
    }

    pub fn delete_snapshot(&self, id: &str) -> Result<SnapshotInfo> {
        self.snapshots()?.delete(id)
    }

    pub fn restore_snapshot(&self, id: &str, opts: RestoreOptions) -> Result<Arc<Collection>> {
        let loaded = self.snapshots()?.load(id)?;
        let name = opts.name.unwrap_or(loaded.manifest.collection);
        validate_name(&name)?;
        // Verify the index before an existing collection is deleted for it.
        let index = Collection::snapshot_index(
            &loaded.manifest.config,
            &loaded.store,
            loaded.index.as_deref(),
        )?;

        let mut collections = self.collections.write();
        if let Some(existing) = collections.remove(&name) {
            if !opts.replace {
                collections.insert(name.clone(), existing);
                return Err(Error::CollectionExists(name));
            }
            self.remove_files(&name, existing)?;
        }
        let dir = self.collection_dir(&name);
        let collection = Arc::new(Collection::restore(
            &name,
            loaded.manifest.config,
            loaded.store,
            index,
            dir.as_deref(),
            self.env.clone(),
        )?);
        collections.insert(name.clone(), Arc::clone(&collection));
        tracing::info!(snapshot = id, collection = %name, points = collection.len(), "restored snapshot");
        Ok(collection)
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
