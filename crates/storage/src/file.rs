use std::path::{Path, PathBuf};

use vectordb_types::Result;

use crate::wal::Wal;
use crate::{
    Mutation, Op, Recovered, StorageEngine, StorageKind, StorageOptions, VectorStore, state,
};

const STATE_FILE: &str = "state.bin";
const WAL_FILE: &str = "wal.log";

/// Durable engine built on VectorDB's own WAL and snapshot files.
///
/// Restarts load the last checkpoint, including the serialized index,
/// and replay the WAL on top: no index rebuild is needed.
#[derive(Debug)]
pub struct FileEngine {
    dir: PathBuf,
    wal: Wal,
    pending: Vec<Mutation>,
    options: StorageOptions,
}

impl FileEngine {
    pub fn open(dir: &Path, options: StorageOptions) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let (wal, pending) = Wal::open(&dir.join(WAL_FILE), options.fsync)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            wal,
            pending,
            options,
        })
    }
}

impl StorageEngine for FileEngine {
    fn kind(&self) -> StorageKind {
        StorageKind::File
    }

    fn recover(&mut self, dim: usize) -> Result<Recovered> {
        let (store, index) = state::read_file(&self.dir.join(STATE_FILE), dim)?
            .unwrap_or((VectorStore::new(dim), None));
        Ok(Recovered {
            store,
            index,
            replay: std::mem::take(&mut self.pending),
        })
    }

    fn log(&mut self, op: Op<'_>) -> Result<()> {
        self.wal.append(op)
    }

    fn needs_checkpoint(&self) -> bool {
        self.wal.size() >= self.options.wal_checkpoint_bytes
    }

    fn checkpoint(&mut self, store: &VectorStore, index: &[u8]) -> Result<()> {
        state::write_file(&self.dir.join(STATE_FILE), store, Some(index))?;
        // Only drop the log once the snapshot that covers it is durable.
        self.wal.reset()
    }

    fn import(&mut self, store: &VectorStore, index: &[u8]) -> Result<()> {
        self.checkpoint(store, index)
    }
}
