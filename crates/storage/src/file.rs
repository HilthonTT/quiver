use std::path::{Path, PathBuf};

use quiver_types::Result;

use crate::wal::Wal;
use crate::{
    LogPosition, Mutation, Op, Recovered, StorageEngine, StorageKind, StorageOptions, StoreSpec, VectorStore,
    state,
};

const STATE_FILE: &str = "state.bin";
const WAL_FILE: &str = "wal.log";
/// The last applied mark as of the checkpoint in `STATE_FILE`.
const APPLIED_FILE: &str = "applied";

/// Durable engine built on Quiver's own WAL and snapshot files.
///
/// Restarts load the last checkpoint, including the serialized index,
/// and replay the WAL on top: no index rebuild is needed.
#[derive(Debug)]
pub struct FileEngine {
    dir: PathBuf,
    wal: Wal,
    pending: Vec<Mutation>,
    options: StorageOptions,
    applied: Option<LogPosition>,
}

impl FileEngine {
    pub fn open(dir: &Path, options: StorageOptions) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let (wal, mut pending) = Wal::open(&dir.join(WAL_FILE), options.fsync)?;
        let mut applied = read_applied(&dir.join(APPLIED_FILE))?;
        // Marks in the log are newer than the file's, which a crash between
        // writing it and clearing the log can leave behind.
        pending.retain(|m| match m {
            Mutation::Applied { at } => {
                applied = applied.max(Some(*at));
                false
            }
            _ => true,
        });
        Ok(Self {
            dir: dir.to_path_buf(),
            wal,
            pending,
            options,
            applied,
        })
    }
}

fn read_applied(path: &Path) -> Result<Option<LogPosition>> {
    match std::fs::read(path) {
        Ok(bytes) => LogPosition::from_bytes(&bytes)
            .map(Some)
            .ok_or_else(|| quiver_types::Error::Corrupt(format!("{} is not a log position", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn write_applied(path: &Path, at: LogPosition) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        std::io::Write::write_all(&mut f, &at.to_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    state::sync_parent(path)
}

impl StorageEngine for FileEngine {
    fn kind(&self) -> StorageKind {
        StorageKind::File
    }

    fn recover(&mut self, spec: &StoreSpec) -> Result<Recovered> {
        let (store, index) = match state::read_file(&self.dir.join(STATE_FILE), spec)? {
            Some(state) => state,
            None => (VectorStore::with_spec(spec)?, None),
        };
        Ok(Recovered {
            store,
            index,
            replay: std::mem::take(&mut self.pending),
            applied: self.applied,
        })
    }

    fn log(&mut self, op: Op<'_>) -> Result<()> {
        self.wal.append(op)?;
        if let Op::Applied { at } = op {
            self.applied = Some(at);
        }
        Ok(())
    }

    fn needs_checkpoint(&self) -> bool {
        self.wal.size() >= self.options.wal_checkpoint_bytes
    }

    fn checkpoint(&mut self, store: &VectorStore, index: &[u8]) -> Result<()> {
        state::write_file(&self.dir.join(STATE_FILE), store, Some(index))?;
        // The mark goes after the state it describes, and before the log
        // that also holds it is cleared: at every point a crash leaves a
        // mark no newer than the state that survives.
        if let Some(applied) = self.applied {
            write_applied(&self.dir.join(APPLIED_FILE), applied)?;
        }
        // Only drop the log once the snapshot that covers it is durable.
        self.wal.reset()
    }

    fn import(&mut self, store: &VectorStore, index: &[u8]) -> Result<()> {
        self.checkpoint(store, index)
    }
}
