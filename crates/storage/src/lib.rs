pub mod codec;
mod file;
mod memory;
#[cfg(feature = "rocksdb")]
mod rocks;
pub mod state;
mod store;
mod wal;

use std::path::Path;

use quiver_types::{Error, Point, Result};
use serde::{Deserialize, Serialize};

pub use file::FileEngine;
pub use memory::MemoryEngine;
#[cfg(feature = "rocksdb")]
pub use rocks::RocksDbEngine;
pub use store::VectorStore;

/// A mutation, as handed to [`StorageEngine::log`].
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op<'a> {
    Upsert { points: &'a [Point] },
    Delete { ids: &'a [String] },
}

/// An owned mutation, as read back from a log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Mutation {
    Upsert { points: Vec<Point> },
    Delete { ids: Vec<String> },
}

#[derive(Debug)]
pub struct Recovered {
    pub store: VectorStore,
    pub index: Option<Vec<u8>>,
    pub replay: Vec<Mutation>,
}

impl Recovered {
    pub fn empty(dim: usize) -> Self {
        Self {
            store: VectorStore::new(dim),
            index: None,
            replay: Vec::new(),
        }
    }
}

/// How a collection's data is made durable.
///
/// The database calls these methods with the collection's write lock held:
/// [`recover`](Self::recover) once on open, [`log`](Self::log) before applying
/// each mutation in memory, and [`checkpoint`](Self::checkpoint) whenever
/// [`needs_checkpoint`](Self::needs_checkpoint) says so (and on flush).
pub trait StorageEngine: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> StorageKind;

    fn recover(&mut self, dim: usize) -> Result<Recovered>;

    fn log(&mut self, op: Op<'_>) -> Result<()>;

    fn needs_checkpoint(&self) -> bool;

    fn checkpoint(&mut self, store: &VectorStore, index: &[u8]) -> Result<()>;

    fn import(&mut self, store: &VectorStore, index: &[u8]) -> Result<()>;
}

/// Which storage engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageKind {
    /// Nothing is persisted.
    #[serde(alias = "inmemory", alias = "in_memory")]
    Memory,
    /// Quiver's own WAL + snapshot files (the default).
    #[default]
    File,
    /// RocksDB (requires the `rocksdb` cargo feature).
    #[serde(alias = "rocks_db", alias = "rocks")]
    RocksDb,
}

impl StorageKind {
    /// Lowercase name, as accepted on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            StorageKind::Memory => "memory",
            StorageKind::File => "file",
            StorageKind::RocksDb => "rocksdb",
        }
    }

    /// Whether data survives a restart.
    pub fn is_durable(self) -> bool {
        self != StorageKind::Memory
    }

    /// Whether this binary was built with support for this engine.
    pub fn is_available(self) -> bool {
        self != StorageKind::RocksDb || cfg!(feature = "rocksdb")
    }
}

impl std::str::FromStr for StorageKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        serde_json::from_value(serde_json::Value::String(s.to_ascii_lowercase()))
            .map_err(|_| format!("unknown storage engine `{s}` (expected memory, file or rocksdb)"))
    }
}

impl std::fmt::Display for StorageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Durability and maintenance settings for the file and RocksDB engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageOptions {
    pub fsync: bool,
    pub wal_checkpoint_bytes: u64,
}

impl Default for StorageOptions {
    fn default() -> Self {
        Self {
            fsync: false,
            wal_checkpoint_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Opens an engine of the given kind. `dir` is required for durable engines.
pub fn open_engine(
    kind: StorageKind,
    dir: Option<&Path>,
    options: StorageOptions,
) -> Result<Box<dyn StorageEngine>> {
    let need_dir =
        || dir.ok_or_else(|| Error::invalid(format!("the {kind} storage engine needs a directory")));
    match kind {
        StorageKind::Memory => Ok(Box::new(MemoryEngine)),
        StorageKind::File => Ok(Box::new(FileEngine::open(need_dir()?, options)?)),
        #[cfg(feature = "rocksdb")]
        StorageKind::RocksDb => Ok(Box::new(RocksDbEngine::open(need_dir()?, options)?)),
        #[cfg(not(feature = "rocksdb"))]
        StorageKind::RocksDb => Err(Error::Unsupported(
            "this build has no RocksDB support; rebuild with `--features rocksdb`".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn points(n: usize) -> Vec<Point> {
        (0..n)
            .map(|i| Point::new(format!("p{i}"), vec![i as f32, 1.0]).with_metadata(json!({"i": i})))
            .collect()
    }

    /// Exercises the contract every durable engine must honour.
    fn durable_engine_contract(kind: StorageKind) {
        let dir = tempfile::tempdir().unwrap();
        let open = || open_engine(kind, Some(dir.path()), StorageOptions::default()).unwrap();
        {
            let mut e = open();
            assert_eq!(e.recover(2).unwrap().store.live(), 0);
            e.log(Op::Upsert { points: &points(5) }).unwrap();
            e.log(Op::Delete { ids: &["p1".into()] }).unwrap();
        }
        // Rebuild the expected state the way the database would.
        let mut e = open();
        let rec = e.recover(2).unwrap();
        let mut store = rec.store;
        for m in rec.replay {
            match m {
                Mutation::Upsert { points } => {
                    for p in points {
                        store.push(p.id, &p.vector, p.metadata);
                    }
                }
                Mutation::Delete { ids } => ids.iter().for_each(|id| {
                    store.remove(id);
                }),
            }
        }
        assert_eq!(store.live(), 4);
        assert!(store.slot_of("p1").is_none());
        let p3 = store.slot_of("p3").unwrap();
        assert_eq!(*store.vector(p3), [3.0, 1.0]);
        assert_eq!(store.metadata(p3), Some(&json!({"i": 3})));

        // Import replaces everything.
        let mut fresh = VectorStore::new(2);
        fresh.push("only".into(), &[9.0, 9.0], None);
        e.import(&fresh, b"idx").unwrap();
        drop(e);
        let rec = open().recover(2).unwrap();
        assert!(rec.replay.is_empty());
        assert_eq!(rec.store.live(), 1);
        assert!(rec.store.slot_of("only").is_some());
    }

    #[test]
    fn file_engine_contract() {
        durable_engine_contract(StorageKind::File);
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn rocksdb_engine_contract() {
        durable_engine_contract(StorageKind::RocksDb);
    }

    #[test]
    fn storage_kind_parsing() {
        assert_eq!("RocksDB".parse::<StorageKind>().unwrap(), StorageKind::RocksDb);
        assert_eq!("inmemory".parse::<StorageKind>().unwrap(), StorageKind::Memory);
        assert!("mongo".parse::<StorageKind>().is_err());
    }
}
