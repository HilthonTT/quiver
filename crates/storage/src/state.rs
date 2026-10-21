use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use std::collections::BTreeMap;

use quiver_types::{Error, MAX_SPARSE_NNZ, Result, SparseVector};

use crate::codec::{Reader, WriteExt};
use crate::column::{ColumnSpec, VectorColumn};
use crate::store::{StoreSpec, VectorStore};

/// First bytes of every encoded state, so other formats can be told apart cheaply.
pub const MAGIC: &[u8; 8] = b"VECTSNAP";
const VERSION: u32 = 3;
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

/// How [`encode`] writes on-disk columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Columns {
    /// As references to their files, which must outlive the state: for
    /// checkpoints next to the files.
    Reference,
    /// Every vector inline, so the state stands alone: for snapshots.
    Inline,
}

const COLUMN_INLINE: u8 = 0;
const COLUMN_FILE: u8 = 1;

pub fn encode<W: Write>(w: W, store: &VectorStore, index: Option<&[u8]>, columns: Columns) -> Result<W> {
    let mut w = CrcWriter {
        inner: w,
        hasher: crc32fast::Hasher::new(),
    };

    w.write_all(MAGIC)?;
    w.put_u32(VERSION)?;
    let slots = store.slots();
    w.put_u64(slots as u64)?;
    for slot in 0..slots as u32 {
        w.put_u8(store.is_alive(slot) as u8)?;
        w.put_bytes(store.id(slot).as_bytes())?;
        match store.metadata(slot) {
            Some(meta) => w.put_bytes(&serde_json::to_vec(meta)?)?,
            None => w.put_u32(0)?,
        }
    }

    let dense: Vec<_> = store.columns().collect();
    w.put_u32(dense.len() as u32)?;
    for (name, column) in dense {
        w.put_bytes(name.as_bytes())?;
        w.put_u32(column.dim() as u32)?;
        match (column.disk_path(), columns) {
            (Some(path), Columns::Reference) => {
                w.put_u8(COLUMN_FILE)?;
                let file = path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .expect("column files have UTF-8 names");
                w.put_bytes(file.as_bytes())?;
            }
            _ => {
                w.put_u8(COLUMN_INLINE)?;
                for slot in 0..slots as u32 {
                    w.put_f32s(&column.vector(slot))?;
                }
            }
        }
    }

    let sparse: Vec<_> = store.sparse_columns().collect();
    w.put_u32(sparse.len() as u32)?;
    for (name, vectors) in sparse {
        w.put_bytes(name.as_bytes())?;
        for v in vectors {
            w.put_u32(v.len() as u32)?;
            for &i in &v.indices {
                w.put_u32(i)?;
            }
            w.put_f32s(&v.values)?;
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

/// Decodes a state into a store laid out as `spec` says. Columns the state
/// keeps in files are looked up in `spec.dir`.
pub fn decode(buf: &[u8], spec: &StoreSpec) -> Result<State> {
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
    let (store, index) = match r.u32()? {
        2 => decode_v2(&mut r, spec)?,
        VERSION => decode_v3(&mut r, spec)?,
        version => return Err(Error::Corrupt(format!("unsupported state version {version}"))),
    };
    if !r.is_empty() {
        return Err(Error::Corrupt("trailing bytes in state file".into()));
    }
    Ok((store, index))
}

fn read_index(r: &mut Reader<'_>) -> Result<Option<Vec<u8>>> {
    Ok(match r.u64()? {
        NO_INDEX => None,
        len => Some(
            r.take(usize::try_from(len).map_err(|_| Error::Corrupt("index too large".into()))?)?
                .to_vec(),
        ),
    })
}

fn read_metadata(bytes: &[u8]) -> Result<Option<serde_json::Value>> {
    Ok(if bytes.is_empty() {
        None
    } else {
        Some(serde_json::from_slice(bytes)?)
    })
}

fn read_id(r: &mut Reader<'_>) -> Result<String> {
    String::from_utf8(r.bytes()?.to_vec()).map_err(|_| Error::Corrupt("record id is not valid UTF-8".into()))
}

fn check_dim(name: &str, stored: usize, expected: usize) -> Result<()> {
    if stored != expected {
        let which = if name.is_empty() {
            String::from("main vector")
        } else {
            format!("vector `{name}`")
        };
        return Err(Error::Corrupt(format!(
            "state {which} dimension {stored} does not match the collection's {expected}"
        )));
    }
    Ok(())
}

/// Version 2: one vector per point, interleaved with ids and metadata.
fn decode_v2(r: &mut Reader<'_>, spec: &StoreSpec) -> Result<State> {
    if !spec.named.is_empty() {
        return Err(Error::Corrupt("state predates named vectors".into()));
    }
    let dim = spec.dim();
    check_dim("", r.u32()? as usize, dim)?;
    let slots = r.u64()? as usize;
    let dir = spec.dir.as_deref();
    let mut main = VectorColumn::create(spec.main, dir, StoreSpec::column_file_name(StoreSpec::MAIN))?;
    let mut rows = Vec::with_capacity(slots.min(1 << 20));
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..slots {
        let alive = r.u8()? != 0;
        let id = read_id(r)?;
        r.f32s_into(dim, &mut vector)?;
        let meta = r.bytes()?;
        main.push(&vector);
        rows.push((alive, id, if alive { read_metadata(meta)? } else { None }));
    }
    let sparse = spec
        .sparse
        .iter()
        .map(|n| (n.clone(), vec![SparseVector::default(); slots]))
        .collect();
    let index = read_index(r)?;
    Ok((
        VectorStore::from_parts(rows, main, Vec::new(), sparse, spec.dir.clone()),
        index,
    ))
}

fn decode_v3(r: &mut Reader<'_>, spec: &StoreSpec) -> Result<State> {
    let slots = usize::try_from(r.u64()?).map_err(|_| Error::Corrupt("too many slots".into()))?;
    let mut rows = Vec::with_capacity(slots.min(1 << 20));
    for _ in 0..slots {
        let alive = r.u8()? != 0;
        let id = read_id(r)?;
        let meta = r.bytes()?;
        rows.push((alive, id, if alive { read_metadata(meta)? } else { None }));
    }

    let count = r.u32()? as usize;
    if count != spec.named.len() + 1 {
        return Err(Error::Corrupt(format!(
            "state has {count} dense vectors, the collection {}",
            spec.named.len() + 1
        )));
    }
    let expected =
        std::iter::once((StoreSpec::MAIN, spec.main)).chain(spec.named.iter().map(|(n, c)| (n.as_str(), *c)));
    let mut main = None;
    let mut named = Vec::with_capacity(spec.named.len());
    for (name, column_spec) in expected {
        let stored =
            std::str::from_utf8(r.bytes()?).map_err(|_| Error::Corrupt("vector name is not UTF-8".into()))?;
        if stored != name {
            return Err(Error::Corrupt(format!(
                "state has vector `{stored}` where the collection has `{name}`"
            )));
        }
        check_dim(name, r.u32()? as usize, column_spec.dim)?;
        let column = read_column(r, column_spec, name, slots, spec.dir.as_deref())?;
        if name.is_empty() {
            main = Some(column);
        } else {
            named.push((name.to_owned(), column));
        }
    }

    let count = r.u32()? as usize;
    let mut stored_sparse = BTreeMap::new();
    for _ in 0..count {
        let name = String::from_utf8(r.bytes()?.to_vec())
            .map_err(|_| Error::Corrupt("vector name is not UTF-8".into()))?;
        if !spec.sparse.contains(&name) {
            return Err(Error::Corrupt(format!(
                "state has sparse vector `{name}`, which the collection lacks"
            )));
        }
        let mut vectors = Vec::with_capacity(slots.min(1 << 20));
        for _ in 0..slots {
            let nnz = r.u32()? as usize;
            if nnz > MAX_SPARSE_NNZ {
                return Err(Error::Corrupt("sparse vector too large".into()));
            }
            let mut indices = Vec::with_capacity(nnz);
            for _ in 0..nnz {
                indices.push(r.u32()?);
            }
            let mut values = Vec::with_capacity(nnz);
            r.f32s_into(nnz, &mut values)?;
            vectors.push(SparseVector { indices, values });
        }
        stored_sparse.insert(name, vectors);
    }
    // Sparse vectors added to the collection since start out empty.
    let sparse = spec
        .sparse
        .iter()
        .map(|name| {
            let vectors = stored_sparse
                .remove(name)
                .unwrap_or_else(|| vec![SparseVector::default(); slots]);
            (name.clone(), vectors)
        })
        .collect();

    let index = read_index(r)?;
    let main = main.expect("the main column comes first");
    Ok((
        VectorStore::from_parts(rows, main, named, sparse, spec.dir.clone()),
        index,
    ))
}

fn read_column(
    r: &mut Reader<'_>,
    spec: ColumnSpec,
    name: &str,
    slots: usize,
    dir: Option<&Path>,
) -> Result<VectorColumn> {
    let file_name = StoreSpec::column_file_name(name);
    match r.u8()? {
        COLUMN_INLINE => {
            let mut column = VectorColumn::create(spec, dir, file_name)?;
            let mut vector = Vec::with_capacity(spec.dim);
            for _ in 0..slots {
                r.f32s_into(spec.dim, &mut vector)?;
                column.push(&vector);
            }
            Ok(column)
        }
        COLUMN_FILE => {
            let file = std::str::from_utf8(r.bytes()?)
                .map_err(|_| Error::Corrupt("file name is not UTF-8".into()))?;
            // Only ever a bare file name next to the state.
            if !crate::column::is_disk_file(file) || file.contains(['/', '\\']) {
                return Err(Error::Corrupt(format!("invalid vector file name `{file}`")));
            }
            let dir = dir
                .ok_or_else(|| Error::Corrupt("state refers to a vector file but has no directory".into()))?;
            let adopted = VectorColumn::adopt(
                ColumnSpec {
                    on_disk: true,
                    ..spec
                },
                &dir.join(file),
                slots,
            )?;
            if spec.on_disk {
                Ok(adopted)
            } else {
                // Moved off disk since: load it into memory.
                adopted.convert(spec, Some(dir), file_name)
            }
        }
        kind => Err(Error::Corrupt(format!("unknown column storage {kind}"))),
    }
}

/// Writes a checkpoint, after making the on-disk columns it refers to durable.
pub fn write_file(path: &Path, store: &VectorStore, index: Option<&[u8]>) -> Result<()> {
    store.sync()?;
    let tmp = path.with_extension("tmp");
    let writer = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    let file = encode(writer, store, index, Columns::Reference)?
        .into_inner()
        .map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)?;
    sync_parent(path)
}

pub fn read_file(path: &Path, spec: &StoreSpec) -> Result<Option<State>> {
    match fs::read(path) {
        Ok(buf) => decode(&buf, spec).map(Some).map_err(|e| match e {
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
    use quiver_types::{CollectionConfig, Point, VectorConfig};
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
            let bytes = encode(Vec::new(), &sample(), index, Columns::Inline).unwrap();
            let (store, blob) = decode(&bytes, &StoreSpec::new(2)).unwrap();
            assert_eq!((store.slots(), store.live()), (2, 1));
            assert_eq!(*store.vector(1), [3.0, 4.0]);
            assert_eq!(blob.as_deref(), index);
        }
    }

    #[test]
    fn detects_corruption_and_wrong_dimension() {
        let mut bytes = encode(Vec::new(), &sample(), None, Columns::Inline).unwrap();
        assert!(matches!(
            decode(&bytes, &StoreSpec::new(3)),
            Err(Error::Corrupt(_))
        ));
        bytes[20] ^= 0xFF;
        assert!(matches!(
            decode(&bytes, &StoreSpec::new(2)),
            Err(Error::Corrupt(_))
        ));
    }

    /// A version 2 state, as 0.2.0 wrote it.
    fn v2(store: &VectorStore, index: Option<&[u8]>) -> Vec<u8> {
        let mut w = Vec::new();
        w.write_all(MAGIC).unwrap();
        w.put_u32(2).unwrap();
        w.put_u32(store.dim() as u32).unwrap();
        w.put_u64(store.slots() as u64).unwrap();
        for slot in 0..store.slots() as u32 {
            w.put_u8(store.is_alive(slot) as u8).unwrap();
            w.put_bytes(store.id(slot).as_bytes()).unwrap();
            w.put_f32s(&store.vector(slot)).unwrap();
            match store.metadata(slot) {
                Some(m) => w.put_bytes(&serde_json::to_vec(m).unwrap()).unwrap(),
                None => w.put_u32(0).unwrap(),
            }
        }
        match index {
            Some(blob) => {
                w.put_u64(blob.len() as u64).unwrap();
                w.write_all(blob).unwrap();
            }
            None => w.put_u64(NO_INDEX).unwrap(),
        }
        let crc = crc32fast::hash(&w);
        w.put_u32(crc).unwrap();
        w
    }

    #[test]
    fn reads_version_2() {
        let (store, blob) = decode(&v2(&sample(), Some(b"g")), &StoreSpec::new(2)).unwrap();
        assert_eq!((store.slots(), store.live()), (2, 1));
        assert_eq!(store.point(1), sample().point(1));
        assert_eq!(blob.as_deref(), Some(&b"g"[..]));
        let mut with_sparse = StoreSpec::new(2);
        with_sparse.sparse.push("text".into());
        assert!(decode(&v2(&sample(), None), &with_sparse).is_ok());
        let named = StoreSpec::for_config(
            &CollectionConfig::new(2).with_vector("t", VectorConfig::new(1)),
            None,
        );
        assert!(decode(&v2(&sample(), None), &named).is_err());
    }

    fn rich_config() -> CollectionConfig {
        CollectionConfig::new(2)
            .with_vector("title", VectorConfig::new(3))
            .with_sparse_vector("text", Default::default())
    }

    fn rich_store(spec: &StoreSpec) -> VectorStore {
        let mut s = VectorStore::with_spec(spec).unwrap();
        for i in 0..4 {
            let mut p =
                Point::new(format!("p{i}"), vec![i as f32, 1.0]).with_vector("title", vec![i as f32; 3]);
            if i % 2 == 0 {
                p = p.with_sparse_vector("text", SparseVector::new(vec![i, 9], vec![1.0, 0.5]));
            }
            s.push_point(p);
        }
        s.remove("p1");
        s
    }

    #[test]
    fn named_and_sparse_vectors_roundtrip() {
        let spec = StoreSpec::for_config(&rich_config(), None);
        let store = rich_store(&spec);
        let bytes = encode(Vec::new(), &store, None, Columns::Inline).unwrap();
        let (back, _) = decode(&bytes, &spec).unwrap();
        for slot in store.live_slots() {
            assert_eq!(back.point(slot), store.point(slot));
        }
        assert_eq!(back.live(), 3);

        // Sparse vectors added to the config later start out empty.
        let mut more = spec.clone();
        more.sparse.push("zzz".into());
        let (back, _) = decode(&bytes, &more).unwrap();
        assert!(back.sparse("zzz").unwrap().iter().all(SparseVector::is_empty));
        // But vectors the config lacks are corruption.
        let mut fewer = spec.clone();
        fewer.sparse.clear();
        assert!(decode(&bytes, &fewer).is_err());
        assert!(decode(&bytes, &StoreSpec::new(2)).is_err());
    }

    #[test]
    fn on_disk_columns_are_referenced_by_checkpoints_and_inlined_in_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = rich_config();
        config.on_disk = true;
        config.vectors.get_mut("title").unwrap().on_disk = true;
        let spec = StoreSpec::for_config(&config, Some(dir.path()));
        let store = rich_store(&spec);
        assert_eq!(store.disk_files().len(), 2);

        let path = dir.path().join("state.bin");
        write_file(&path, &store, None).unwrap();
        assert!(fs::metadata(&path).unwrap().len() < 300, "vectors are not copied");
        let (back, _) = read_file(&path, &spec).unwrap().unwrap();
        assert_eq!(back.disk_files(), store.disk_files(), "the files are adopted");
        for slot in store.live_slots() {
            assert_eq!(back.point(slot), store.point(slot));
        }

        // A snapshot needs no files, and loads into memory.
        let inline = encode(Vec::new(), &store, None, Columns::Inline).unwrap();
        let (loaded, _) = decode(&inline, &spec.clone().in_memory()).unwrap();
        assert!(loaded.disk_files().is_empty());
        assert_eq!(loaded.point(3), store.point(3));

        // Moving a column off disk loads its file.
        let mut off = spec.clone();
        off.main.on_disk = false;
        let (moved, _) = read_file(&path, &off).unwrap().unwrap();
        assert!(!moved.main().is_on_disk());
        assert_eq!(moved.point(2), store.point(2));
    }
}
