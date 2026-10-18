use serde::{Deserialize, Serialize};
use vectordb_types::{Error, Point, Result};

use crate::store::VectorStore;

pub mod codec;
mod memory;
mod state;
mod store;
mod wal;

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
    /// SagittaDB's own WAL + snapshot files (the default).
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
