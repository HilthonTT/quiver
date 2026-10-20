use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use quiver_types::{Error, Result};

use crate::codec::{Reader, WriteExt};
use crate::store::VectorStore;

const MAGIC: &[u8; 8] = b"VECTSNAP";
const VERSION: u32 = 2;
const NO_INDEX: u64 = u64::MAX;

pub type State = (VectorStore, Option<Vec<u8>>);

struct CrcWriter<W> {
    inner: W,
    hasher: crc32fast::Hasher,
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

pub fn encode<W: Write>(w: W, store: &VectorStore, index: Option<&[u8]>) -> Result<W> {
    let mut w = CrcWriter {
        inner: w,
        hasher: crc32fast::Hasher::new(),
    };

    w.write_all(MAGIC)?;
    w.put_u32(VERSION)?;
    w.put_u32(store.dim() as u32)?;
    w.put_u64(store.slots() as u64)?;
    for slot in 0..store.slots() as u32 {
        w.put_u8(store.is_alive(slot) as u8)?;
        w.put_bytes(store.id(slot).as_bytes())?;
        w.put_f32s(store.vector(slot))?;
        match store.metadata(slot) {
            Some(meta) => w.put_bytes(&serde_json::to_vec(meta)?)?,
            None => w.put_u32(0)?,
        }
    }

    match index {
        Some(blob) => {
            w.put_u64(blob.len() as u64)?;
            w.write_all(blob)?;
        }
        None => w.put_u64(NO_INDEX)?,
    }
    let crc = w.hasher.finalize();
    let mut inner = w.inner;
    inner.put_u32(crc)?;
    Ok(inner)
}

pub fn decode(buf: &[u8], dim: usize) -> Result<State> {
    if buf.len() < MAGIC.len() + 4 {
        return Err(Error::Corrupt("state file too short".into()));
    }

    let (body, crc) = buf.split_at(buf.len() - 4);
    if crc32fast::hash(body) != u32::from_le_bytes(crc.try_into().expect("4 bytes")) {
        return Err(Error::Corrupt("state checksum mismatch".into()));
    }

    let mut r = Reader::new(body);
    if r.take(8)? != MAGIC {
        return Err(Error::Corrupt("not a Quiver state file".into()));
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(Error::Corrupt(format!(
            "unsupported state version {version}"
        )));
    }
    let stored_dim = r.u32()? as usize;
    if stored_dim != dim {
        return Err(Error::Corrupt(format!(
            "state dimension {stored_dim} does not match collection dimension {dim}"
        )));
    }

    let slots = r.u64()? as usize;
    let mut store = VectorStore::new(dim);
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..slots {
        let alive = r.u8()? != 0;
        let id = String::from_utf8(r.bytes()?.to_vec())
            .map_err(|_| Error::Corrupt("record id is not valid UTF-8".into()))?;
        r.f32s_into(dim, &mut vector)?;
        let meta = r.bytes()?;
        if alive {
            let metadata = if meta.is_empty() {
                None
            } else {
                Some(serde_json::from_slice(meta)?)
            };
            store.push(id, &vector, metadata);
        } else {
            store.push_tombstone(id, &vector);
        }
    }

    let index = match r.u64()? {
        NO_INDEX => None,
        len => Some(r.take(len as usize)?.to_vec()),
    };
    if !r.is_empty() {
        return Err(Error::Corrupt("trailing bytes in state file".into()));
    }
    Ok((store, index))
}

pub fn write_file(path: &Path, store: &VectorStore, index: Option<&[u8]>) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let writer = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    let file = encode(writer, store, index)?
        .into_inner()
        .map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)?;
    sync_parent(path)
}

pub fn read_file(path: &Path, dim: usize) -> Result<Option<State>> {
    match fs::read(path) {
        Ok(buf) => decode(&buf, dim).map(Some).map_err(|e| match e {
            Error::Corrupt(msg) => Error::Corrupt(format!("{}: {msg}", path.display())),
            other => other,
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> VectorStore {
        let mut s = VectorStore::new(2);
        s.push("a".into(), &[1.0, 2.0], Some(json!({"k": 1})));
        s.push("b".into(), &[3.0, 4.0], None);
        s.remove("a");
        s
    }

    #[test]
    fn roundtrip_with_and_without_index() {
        for index in [None, Some(&b"graph"[..])] {
            let bytes = encode(Vec::new(), &sample(), index).unwrap();
            let (store, blob) = decode(&bytes, 2).unwrap();
            assert_eq!((store.slots(), store.live()), (2, 1));
            assert_eq!(store.vector(1), &[3.0, 4.0]);
            assert_eq!(blob.as_deref(), index);
        }
    }

    #[test]
    fn detects_corruption_and_wrong_dimension() {
        let mut bytes = encode(Vec::new(), &sample(), None).unwrap();
        assert!(matches!(decode(&bytes, 3), Err(Error::Corrupt(_))));
        bytes[20] ^= 0xFF;
        assert!(matches!(decode(&bytes, 2), Err(Error::Corrupt(_))));
    }
}
