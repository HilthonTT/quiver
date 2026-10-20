use std::path::Path;

use quiver_types::{Error, Result};
use rocksdb::{DB, Direction, IteratorMode, Options, WriteBatch, WriteOptions};

use crate::codec::{Reader, WriteExt};
use crate::{Op, Recovered, StorageEngine, StorageKind, StorageOptions, VectorStore, rocks};

const PREFIX: &[u8] = b"p/";

/// Durable engine backed by RocksDB.
pub struct RocksDbEngine {
    db: DB,
    options: StorageOptions,
}

impl std::fmt::Debug for RocksDbEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksDbEngine")
            .field("path", &self.db.path())
            .finish_non_exhaustive()
    }
}

fn storage_err(e: rocksdb::Error) -> Error {
    Error::Storage(e.into_string())
}

fn key(id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(PREFIX.len() + id.len());
    k.extend_from_slice(PREFIX);
    k.extend_from_slice(id.as_bytes());
    k
}

fn encode_value(vector: &[f32], metadata: Option<&serde_json::Value>) -> Result<Vec<u8>> {
    let mut v = Vec::with_capacity(vector.len() * 4 + 16);
    v.put_f32s(vector)?;
    match metadata {
        Some(m) => v.put_bytes(&serde_json::to_vec(m)?)?,
        None => v.put_u32(0)?,
    }
    Ok(v)
}

impl RocksDbEngine {
    /// Opens (or creates) a RocksDB database in `dir`.
    pub fn open(dir: &Path, options: StorageOptions) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let db = DB::open(&opts, dir).map_err(storage_err)?;
        Ok(Self { db, options })
    }

    fn write(&self, batch: WriteBatch) -> Result<()> {
        let mut wo = WriteOptions::default();
        wo.set_sync(self.options.fsync);
        self.db.write_opt(batch, &wo).map_err(storage_err)
    }

    fn put_store(&self, batch: &mut WriteBatch, store: &VectorStore) -> Result<()> {
        for slot in store.live_slots() {
            batch.put(
                key(store.id(slot)),
                encode_value(store.vector(slot), store.metadata(slot))?,
            );
        }
        Ok(())
    }
}

impl StorageEngine for RocksDbEngine {
    fn kind(&self) -> StorageKind {
        StorageKind::RocksDb
    }

    fn recover(&mut self, dim: usize) -> Result<Recovered> {
        let mut store = VectorStore::new(dim);
        let mut vector = Vec::with_capacity(dim);
        for item in self.db.iterator(IteratorMode::From(PREFIX, Direction::Forward)) {
            let (k, v) = item.map_err(storage_err)?;
            let Some(id) = k.strip_prefix(PREFIX) else {
                break;
            };
            let id = std::str::from_utf8(id)
                .map_err(|_| Error::Corrupt("RocksDB key is not valid UTF-8".into()))?;
            let mut r = Reader::new(&v);
            r.f32s_into(dim, &mut vector)?;
            let meta = r.bytes()?;
            let metadata = if meta.is_empty() {
                None
            } else {
                Some(serde_json::from_slice(meta)?)
            };
            store.push(id.to_owned(), &vector, metadata);
        }
        Ok(Recovered {
            store,
            index: None,
            replay: Vec::new(),
        })
    }

    fn log(&mut self, op: Op<'_>) -> Result<()> {
        let mut batch = WriteBatch::default();
        match op {
            Op::Upsert { points } => {
                for p in points {
                    batch.put(key(&p.id), encode_value(&p.vector, p.metadata.as_ref())?);
                }
            }
            Op::Delete { ids } => {
                for id in ids {
                    batch.delete(key(id));
                }
            }
        }
        self.write(batch)
    }

    fn needs_checkpoint(&self) -> bool {
        false
    }

    fn checkpoint(&mut self, _store: &VectorStore, _index: &[u8]) -> Result<()> {
        self.db.flush().map_err(storage_err)
    }

    fn import(&mut self, store: &VectorStore, _index: &[u8]) -> Result<()> {
        let mut batch = WriteBatch::default();
        let mut end = PREFIX.to_vec();
        *end.last_mut().expect("non-empty prefix") += 1;
        batch.delete_range(PREFIX, end.as_slice());
        self.put_store(&mut batch, store)?;
        self.write(batch)?;
        self.db.flush().map_err(storage_err)
    }
}
