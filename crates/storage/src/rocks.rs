use std::path::Path;

use std::borrow::Cow;

use quiver_types::{Error, NamedVectors, Point, Result, SparseVectors};
use rocksdb::{DB, Direction, IteratorMode, Options, WriteBatch, WriteOptions};
use serde::{Deserialize, Serialize};

use crate::codec::{Reader, WriteExt};
use crate::{Op, Recovered, StorageEngine, StorageKind, StorageOptions, StoreSpec, VectorStore};

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

/// The main vector, the metadata, then (if there are any) the named and
/// sparse vectors as JSON. Values from before named vectors end after the
/// metadata.
fn encode_value(p: &Point) -> Result<Vec<u8>> {
    let mut v = Vec::with_capacity(p.vector.len() * 4 + 16);
    v.put_f32s(&p.vector)?;
    match &p.metadata {
        Some(m) => v.put_bytes(&serde_json::to_vec(m)?)?,
        None => v.put_u32(0)?,
    }
    if !p.vectors.is_empty() || !p.sparse_vectors.is_empty() {
        let extra = Extra {
            vectors: Cow::Borrowed(&p.vectors),
            sparse_vectors: Cow::Borrowed(&p.sparse_vectors),
        };
        v.put_bytes(&serde_json::to_vec(&extra)?)?;
    }
    Ok(v)
}

#[derive(Serialize, Deserialize)]
struct Extra<'a> {
    #[serde(default)]
    vectors: Cow<'a, NamedVectors>,
    #[serde(default)]
    sparse_vectors: Cow<'a, SparseVectors>,
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
            batch.put(key(store.id(slot)), encode_value(&store.point(slot))?);
        }
        Ok(())
    }
}

/// Points must match the collection exactly, or pushing them would panic.
fn check_layout(p: &Point, spec: &StoreSpec) -> Result<()> {
    let named_ok = p.vectors.len() == spec.named.len()
        && spec
            .named
            .iter()
            .all(|(name, c)| p.vectors.get(name).is_some_and(|v| v.len() == c.dim));
    let sparse_ok = p.sparse_vectors.keys().all(|name| spec.sparse.contains(name));
    if named_ok && sparse_ok {
        Ok(())
    } else {
        Err(Error::Corrupt(format!(
            "stored point `{}` does not match the collection's vectors",
            p.id
        )))
    }
}

impl StorageEngine for RocksDbEngine {
    fn kind(&self) -> StorageKind {
        StorageKind::RocksDb
    }

    fn recover(&mut self, spec: &StoreSpec) -> Result<Recovered> {
        let mut store = VectorStore::with_spec(spec)?;
        let dim = spec.dim();
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
            let extra: Extra<'_> = if r.is_empty() {
                Extra {
                    vectors: Cow::Owned(NamedVectors::new()),
                    sparse_vectors: Cow::Owned(SparseVectors::new()),
                }
            } else {
                serde_json::from_slice(r.bytes()?)?
            };
            let point = Point {
                id: id.to_owned(),
                vector: std::mem::take(&mut vector),
                vectors: extra.vectors.into_owned(),
                sparse_vectors: extra.sparse_vectors.into_owned(),
                metadata,
            };
            check_layout(&point, spec)?;
            store.push_point(point);
            vector = Vec::with_capacity(dim);
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
                    batch.put(key(&p.id), encode_value(p)?);
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
