pub mod codec;
pub mod column;
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

pub use column::{ColumnSpec, VectorColumn};
pub use file::FileEngine;
pub use memory::MemoryEngine;
#[cfg(feature = "rocksdb")]
pub use rocks::RocksDbEngine;
pub use store::{StoreSpec, VectorStore};

/// The position of an entry in a replicated log: its index, and the term
/// and node of the leader that wrote it. Ordered by index first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LogPosition {
    pub index: u64,
    pub term: u64,
    pub node: u64,
}

impl LogPosition {
    pub(crate) fn to_bytes(self) -> [u8; 24] {
        let mut out = [0; 24];
        out[..8].copy_from_slice(&self.index.to_le_bytes());
        out[8..16].copy_from_slice(&self.term.to_le_bytes());
        out[16..].copy_from_slice(&self.node.to_le_bytes());
        out
    }

    pub(crate) fn from_bytes(b: &[u8]) -> Option<Self> {
        let word = |i: usize| Some(u64::from_le_bytes(b.get(i * 8..i * 8 + 8)?.try_into().ok()?));
        (b.len() == 24).then_some(())?;
        Some(Self {
            index: word(0)?,
            term: word(1)?,
            node: word(2)?,
        })
    }
}

/// A mutation, as handed to [`StorageEngine::log`].
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op<'a> {
    Upsert {
        points: &'a [Point],
    },
    Delete {
        ids: &'a [String],
    },
    /// Marks every change logged so far as the effect of the replicated log
    /// up to and including the entry at `at`. Changes nothing by itself.
    Applied {
        at: LogPosition,
    },
}

/// An owned mutation, as read back from a log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Mutation {
    Upsert { points: Vec<Point> },
    Delete { ids: Vec<String> },
    Applied { at: LogPosition },
}

#[derive(Debug)]
pub struct Recovered {
    pub store: VectorStore,
    pub index: Option<Vec<u8>>,
    /// Upserts and deletes to apply on top of `store`, in order.
    pub replay: Vec<Mutation>,
    /// The last [`Op::Applied`] position persisted, if any was ever logged.
    pub applied: Option<LogPosition>,
}

impl Recovered {
    pub fn empty(spec: &StoreSpec) -> Result<Self> {
        Ok(Self {
            store: VectorStore::with_spec(spec)?,
            index: None,
            replay: Vec::new(),
            applied: None,
        })
    }
}

/// How a collection's data is made durable.
///
/// The database calls these methods with the collection's write lock held:
/// [`recover`](Self::recover) once on open, [`log`](Self::log) before applying
/// each mutation in memory, and [`checkpoint`](Self::checkpoint) whenever
/// [`needs_checkpoint`](Self::needs_checkpoint) says so (and on flush).
///
/// [`Op::Applied`] marks must survive checkpoints and imports: the last one
/// logged is what [`Recovered::applied`] reports after a restart, and it must
/// never claim changes that were not persisted with it.
pub trait StorageEngine: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> StorageKind;

    /// Loads what was persisted into a store laid out as `spec` says.
    fn recover(&mut self, spec: &StoreSpec) -> Result<Recovered>;

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
            assert_eq!(e.recover(&StoreSpec::new(2)).unwrap().store.live(), 0);
            e.log(Op::Upsert { points: &points(5) }).unwrap();
            e.log(Op::Delete { ids: &["p1".into()] }).unwrap();
        }
        // Rebuild the expected state the way the database would.
        let mut e = open();
        let rec = e.recover(&StoreSpec::new(2)).unwrap();
        let mut store = rec.store;
        for m in rec.replay {
            match m {
                Mutation::Upsert { points } => {
                    for p in points {
                        store.push_point(p);
                    }
                }
                Mutation::Delete { ids } => ids.iter().for_each(|id| {
                    store.remove(id);
                }),
                Mutation::Applied { .. } => {}
            }
        }
        assert_eq!(store.live(), 4);
        assert!(store.slot_of("p1").is_none());
        let p3 = store.slot_of("p3").unwrap();
        assert_eq!(*store.vector(p3), [3.0, 1.0]);
        assert_eq!(store.metadata(p3), Some(&json!({"i": 3})));
        assert_eq!(rec.applied, None);

        // Import replaces everything.
        let mut fresh = VectorStore::new(2);
        fresh.push("only".into(), &[9.0, 9.0], None);
        e.import(&fresh, b"idx").unwrap();
        drop(e);
        let rec = open().recover(&StoreSpec::new(2)).unwrap();
        assert!(rec.replay.is_empty());
        assert_eq!(rec.store.live(), 1);
        assert!(rec.store.slot_of("only").is_some());
    }

    /// Named, sparse and on-disk vectors survive every durable engine.
    fn rich_points_contract(kind: StorageKind) {
        use quiver_types::{CollectionConfig, SparseVector, VectorConfig};
        let dir = tempfile::tempdir().unwrap();
        let config = CollectionConfig::new(2)
            .on_disk(true)
            .with_vector("title", VectorConfig::new(3))
            .with_sparse_vector("text", Default::default());
        let spec = StoreSpec::for_config(&config, Some(dir.path()));
        let p = Point::new("a", vec![1.0, 2.0])
            .with_vector("title", vec![3.0, 4.0, 5.0])
            .with_sparse_vector("text", SparseVector::new(vec![7], vec![0.5]))
            .with_metadata(json!({"k": 1}));
        let engine_dir = dir.path().join("engine");
        let open = || open_engine(kind, Some(&engine_dir), StorageOptions::default()).unwrap();
        {
            let mut e = open();
            let rec = e.recover(&spec).unwrap();
            e.log(Op::Upsert {
                points: std::slice::from_ref(&p),
            })
            .unwrap();
            let mut store = rec.store;
            store.push_point(p.clone());
            e.checkpoint(&store, b"").unwrap();
        }
        let rec = open().recover(&spec).unwrap();
        let mut store = rec.store;
        for m in rec.replay {
            if let Mutation::Upsert { points } = m {
                points.into_iter().for_each(|p| {
                    store.push_point(p);
                });
            }
        }
        let slot = store.slot_of("a").unwrap();
        assert_eq!(store.point(slot), p);
        assert!(store.main().is_on_disk());
    }

    #[test]
    fn rich_points_survive_the_file_engine() {
        rich_points_contract(StorageKind::File);
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn rich_points_survive_rocksdb() {
        rich_points_contract(StorageKind::RocksDb);
    }

    /// The last applied mark survives restarts, checkpoints and imports.
    fn at(index: u64) -> LogPosition {
        LogPosition {
            index,
            term: 2,
            node: 5,
        }
    }

    fn applied_contract(kind: StorageKind) {
        let dir = tempfile::tempdir().unwrap();
        let open = || open_engine(kind, Some(dir.path()), StorageOptions::default()).unwrap();
        let spec = StoreSpec::new(2);
        {
            let mut e = open();
            e.recover(&spec).unwrap();
            e.log(Op::Upsert { points: &points(2) }).unwrap();
            e.log(Op::Applied { at: at(7) }).unwrap();
            e.log(Op::Delete { ids: &["p0".into()] }).unwrap();
            e.log(Op::Applied { at: at(8) }).unwrap();
        }
        let mut e = open();
        let rec = e.recover(&spec).unwrap();
        assert_eq!(rec.applied, Some(at(8)));
        assert!(rec.replay.iter().all(|m| !matches!(m, Mutation::Applied { .. })));
        let mut store = rec.store;
        for m in rec.replay {
            match m {
                Mutation::Upsert { points } => points.into_iter().for_each(|p| {
                    store.push_point(p);
                }),
                Mutation::Delete { ids } => ids.iter().for_each(|id| {
                    store.remove(id);
                }),
                Mutation::Applied { .. } => unreachable!(),
            }
        }
        e.checkpoint(&store, b"").unwrap();
        drop(e);
        let mut e = open();
        let rec = e.recover(&spec).unwrap();
        assert_eq!(rec.applied, Some(at(8)));
        assert_eq!(rec.store.live() + rec.replay.len(), 1);
        e.log(Op::Applied { at: at(9) }).unwrap();
        e.import(&VectorStore::new(2), b"").unwrap();
        drop(e);
        assert_eq!(open().recover(&spec).unwrap().applied, Some(at(9)));
    }

    #[test]
    fn file_engine_keeps_applied_marks() {
        applied_contract(StorageKind::File);
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn rocksdb_keeps_applied_marks() {
        applied_contract(StorageKind::RocksDb);
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
