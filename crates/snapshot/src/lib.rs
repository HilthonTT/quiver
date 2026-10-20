use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::Compression;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

use quiver_storage::{VectorStore, state};
pub use quiver_types::snapshot::{Manifest, SnapshotInfo};
use quiver_types::{CollectionConfig, Error, Result, validate_name};

/// Version of the archive layout. Bumped on incompatible changes.
pub const FORMAT_VERSION: u32 = 1;

const MANIFEST: &str = "manifest.json";
const DATA: &str = "data.bin";
const EXTENSION: &str = ".tar.gz";
/// Refuse to load archives whose data claims to be larger than this.
const MAX_DATA_BYTES: u64 = 64 << 30;
/// Reserve at most this much for `data.bin` before reading it.
const PREALLOCATE_MAX: u64 = 64 << 20;

#[derive(Debug)]
pub struct Loaded {
    /// The manifest.
    pub manifest: Manifest,
    /// The vector store.
    pub store: VectorStore,
    /// The serialized index, if one was saved.
    pub index: Option<Vec<u8>>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn append(
    builder: &mut tar::Builder<impl Write>,
    name: &str,
    bytes: &[u8],
    mtime: u64,
) -> Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(mtime);
    header.set_cksum();
    builder.append_data(&mut header, name, bytes)?;
    Ok(())
}

pub fn write_archive(
    path: &Path,
    id: &str,
    collection: &str,
    config: &CollectionConfig,
    store: &VectorStore,
    index: Option<&[u8]>,
) -> Result<Manifest> {
    let data = state::encode(Vec::new(), store, index)?;
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        id: id.to_owned(),
        collection: collection.to_owned(),
        created_at_ms: now_ms(),
        quiver_version: env!("CARGO_PKG_VERSION").to_owned(),
        config: config.clone(),
        points: store.live(),
        data_bytes: data.len() as u64,
        data_sha256: sha256_hex(&data),
    };

    let tmp = path.with_extension("partial");
    let file = BufWriter::new(File::create(&tmp)?);
    let mut builder = tar::Builder::new(GzEncoder::new(file, Compression::fast()));
    let mtime = manifest.created_at_ms / 1000;

    append(
        &mut builder,
        MANIFEST,
        &serde_json::to_vec_pretty(&manifest)?,
        mtime,
    )?;

    append(&mut builder, DATA, &data, mtime)?;
    let file = builder
        .into_inner()?
        .finish()?
        .into_inner()
        .map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)?;
    state::sync_parent(path)?;
    Ok(manifest)
}

fn check_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.format_version != FORMAT_VERSION {
        return Err(Error::Corrupt(format!(
            "unsupported snapshot format version {}",
            manifest.format_version
        )));
    }

    if uuid::Uuid::parse_str(&manifest.id).is_err() {
        return Err(Error::Corrupt("snapshot id is not a UUID".into()));
    }

    // Uploaded archives are untrusted, and the name ends up in file names
    // and response headers.
    validate_name(&manifest.collection)
        .map_err(|_| Error::Corrupt("snapshot has an invalid collection name".into()))?;

    manifest.config.validate()
}

pub fn read_manifest(path: &Path) -> Result<Manifest> {
    let mut archive = tar::Archive::new(GzDecoder::new(BufReader::new(File::open(path)?)));

    for entry in archive.entries()? {
        let entry = entry?;
        if entry.path()?.as_os_str() == MANIFEST {
            let manifest: Manifest = serde_json::from_reader(entry)?;
            check_manifest(&manifest)?;
            return Ok(manifest);
        }
    }

    Err(Error::Corrupt(format!(
        "{} has no manifest",
        path.display()
    )))
}

pub fn read_archive(path: &Path) -> Result<Loaded> {
    let mut archive = tar::Archive::new(GzDecoder::new(BufReader::new(File::open(path)?)));
    let mut manifest: Option<Manifest> = None;
    let mut data: Option<Vec<u8>> = None;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let name = entry.path()?.to_string_lossy().into_owned();
        // Duplicates would let `read_manifest` (first entry wins) and this
        // function disagree about what the archive contains.
        let duplicate = || Error::Corrupt(format!("snapshot contains `{name}` twice"));

        match name.as_str() {
            MANIFEST => {
                if manifest.is_some() {
                    return Err(duplicate());
                }
                manifest = Some(serde_json::from_reader(&mut entry)?);
            }
            DATA => {
                if data.is_some() {
                    return Err(duplicate());
                }
                if entry.size() > MAX_DATA_BYTES {
                    return Err(Error::Corrupt("snapshot data is implausibly large".into()));
                }
                // The size comes from an untrusted header: grow as data
                // actually arrives rather than reserving it all up front.
                let mut buf = Vec::with_capacity(entry.size().min(PREALLOCATE_MAX) as usize);
                entry.read_to_end(&mut buf)?;
                data = Some(buf);
            }
            _ => {}
        }
    }

    let manifest = manifest.ok_or_else(|| Error::Corrupt("snapshot has no manifest".into()))?;
    let data = data.ok_or_else(|| Error::Corrupt("snapshot has no data".into()))?;
    check_manifest(&manifest)?;
    if data.len() as u64 != manifest.data_bytes || sha256_hex(&data) != manifest.data_sha256 {
        return Err(Error::Corrupt(format!(
            "snapshot {} failed its checksum",
            manifest.id
        )));
    }
    let (store, index) = state::decode(&data, manifest.config.dimension)?;
    if store.live() != manifest.points {
        return Err(Error::Corrupt(format!(
            "snapshot {} holds {} points but its manifest says {}",
            manifest.id,
            store.live(),
            manifest.points
        )));
    }
    Ok(Loaded {
        manifest,
        store,
        index,
    })
}

/// A directory of snapshot archives.
#[derive(Debug, Clone)]
pub struct SnapshotRegistry {
    dir: PathBuf,
}

impl SnapshotRegistry {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir);
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self, id: &str) -> Result<PathBuf> {
        let uuid = uuid::Uuid::parse_str(id).map_err(|_| Error::SnapshotNotFound(id.to_owned()))?;
        Ok(self.dir.join(format!("{}{EXTENSION}", uuid.hyphenated())))
    }

    fn existing_path(&self, id: &str) -> Result<PathBuf> {
        let path = self.path(id)?;
        if path.is_file() {
            Ok(path)
        } else {
            Err(Error::SnapshotNotFound(id.to_owned()))
        }
    }

    fn info(path: &Path) -> Result<SnapshotInfo> {
        Ok(SnapshotInfo {
            manifest: read_manifest(path)?,
            archive_bytes: fs::metadata(path)?.len(),
        })
    }

    pub fn create(
        &self,
        collection: &str,
        config: &CollectionConfig,
        store: &VectorStore,
        index: Option<&[u8]>,
    ) -> Result<SnapshotInfo> {
        let id = uuid::Uuid::new_v4().hyphenated().to_string();
        let path = self.path(&id)?;
        write_archive(&path, &id, collection, config, store, index)?;

        tracing::info!(snapshot = %id, collection, points = store.live(), "created snapshot");
        Self::info(&path)
    }

    pub fn list(&self, collection: Option<&str>) -> Result<Vec<SnapshotInfo>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            let is_archive = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(EXTENSION));
            if !is_archive {
                continue;
            }
            match Self::info(&path) {
                Ok(info) if collection.is_none_or(|c| c == info.manifest.collection) => {
                    out.push(info)
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "skipping unreadable snapshot")
                }
            }
        }
        out.sort_by(|a, b| {
            (a.manifest.created_at_ms, &a.manifest.id)
                .cmp(&(b.manifest.created_at_ms, &b.manifest.id))
        });
        Ok(out)
    }

    pub fn get(&self, id: &str) -> Result<SnapshotInfo> {
        Self::info(&self.existing_path(id)?)
    }

    pub fn load(&self, id: &str) -> Result<Loaded> {
        read_archive(&self.existing_path(id)?)
    }

    pub fn delete(&self, id: &str) -> Result<SnapshotInfo> {
        let path = self.existing_path(id)?;
        let info = Self::info(&path)?;
        fs::remove_file(&path)?;
        tracing::info!(snapshot = %id, "deleted snapshot");
        Ok(info)
    }

    pub fn staging_path(&self) -> PathBuf {
        self.dir
            .join(format!("import-{}.partial", uuid::Uuid::new_v4()))
    }

    pub fn import(&self, mut reader: impl Read) -> Result<SnapshotInfo> {
        let tmp = self.staging_path();
        let staged = (|| {
            let mut file = File::create(&tmp)?;
            std::io::copy(&mut reader, &mut file)?;
            file.sync_all()?;
            Ok(())
        })();
        match staged {
            Ok(()) => self.import_staged(&tmp),
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    pub fn import_staged(&self, staged: &Path) -> Result<SnapshotInfo> {
        let result = (|| {
            let manifest = read_archive(staged)?.manifest;
            let target = self.path(&manifest.id)?;
            if target.exists() {
                return Err(Error::invalid(format!(
                    "snapshot {} already exists",
                    manifest.id
                )));
            }
            fs::rename(staged, &target)?;
            state::sync_parent(&target)?;
            tracing::info!(snapshot = %manifest.id, collection = %manifest.collection, "imported snapshot");
            Self::info(&target)
        })();
        let _ = fs::remove_file(staged);
        result
    }

    pub fn prune(&self, collection: &str, keep: usize) -> Result<Vec<String>> {
        if keep == 0 {
            return Ok(Vec::new());
        }
        let all = self.list(Some(collection))?;
        let excess = all.len().saturating_sub(keep);
        let mut deleted = Vec::with_capacity(excess);
        for info in &all[..excess] {
            self.delete(&info.manifest.id)?;
            deleted.push(info.manifest.id.clone());
        }
        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store(n: usize) -> VectorStore {
        let mut s = VectorStore::new(3);
        for i in 0..n {
            s.push(
                format!("p{i}"),
                &[i as f32, 1.0, 2.0],
                Some(json!({"i": i})),
            );
        }
        s
    }

    fn registry() -> (tempfile::TempDir, SnapshotRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = SnapshotRegistry::open(dir.path().join("snapshots")).unwrap();
        (dir, reg)
    }

    #[test]
    fn create_list_load_delete() {
        let (_dir, reg) = registry();
        let config = CollectionConfig::new(3);
        let info = reg
            .create("docs", &config, &store(10), Some(b"idx"))
            .unwrap();
        assert_eq!(info.manifest.points, 10);
        assert_eq!(info.manifest.config, config);

        let loaded = reg.load(&info.manifest.id).unwrap();
        assert_eq!(loaded.store.live(), 10);
        assert_eq!(loaded.index.as_deref(), Some(&b"idx"[..]));
        assert_eq!(loaded.store.metadata(4), Some(&json!({"i": 4})));

        assert_eq!(reg.list(None).unwrap().len(), 1);
        assert!(reg.list(Some("other")).unwrap().is_empty());
        reg.delete(&info.manifest.id).unwrap();
        assert!(matches!(
            reg.get(&info.manifest.id),
            Err(Error::SnapshotNotFound(_))
        ));
    }

    #[test]
    fn prune_keeps_newest() {
        let (_dir, reg) = registry();
        let config = CollectionConfig::new(3);
        let ids: Vec<String> = (0..4)
            .map(|i| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                reg.create("docs", &config, &store(i), None)
                    .unwrap()
                    .manifest
                    .id
            })
            .collect();
        reg.create("other", &config, &store(1), None).unwrap();
        let deleted = reg.prune("docs", 2).unwrap();
        assert_eq!(deleted, ids[..2]);
        let left: Vec<String> = reg
            .list(Some("docs"))
            .unwrap()
            .into_iter()
            .map(|i| i.manifest.id)
            .collect();
        assert_eq!(left, ids[2..]);
        assert_eq!(reg.list(Some("other")).unwrap().len(), 1);
    }

    #[test]
    fn tampered_archives_are_rejected() {
        let (_dir, reg) = registry();
        let info = reg
            .create("docs", &CollectionConfig::new(3), &store(5), None)
            .unwrap();
        let path = reg.path(&info.manifest.id).unwrap();

        // Rewrite the archive with a modified data.bin but the old manifest.
        let mut manifest = info.manifest.clone();
        let tampered = state::encode(Vec::new(), &store(6), None).unwrap();
        manifest.data_bytes = tampered.len() as u64;
        let file = File::create(&path).unwrap();
        let mut b = tar::Builder::new(GzEncoder::new(file, Compression::fast()));
        append(&mut b, MANIFEST, &serde_json::to_vec(&manifest).unwrap(), 0).unwrap();
        append(&mut b, DATA, &tampered, 0).unwrap();
        b.into_inner().unwrap().finish().unwrap();

        assert!(matches!(
            reg.load(&info.manifest.id),
            Err(Error::Corrupt(_))
        ));
    }

    #[test]
    fn import_roundtrip_and_duplicates() {
        let (_dir, a) = registry();
        let (_dir2, b) = registry();
        let info = a
            .create("docs", &CollectionConfig::new(3), &store(3), None)
            .unwrap();
        let bytes = fs::read(a.path(&info.manifest.id).unwrap()).unwrap();
        let imported = b.import(&bytes[..]).unwrap();
        assert_eq!(imported.manifest, info.manifest);
        assert!(b.import(&bytes[..]).is_err(), "duplicate id");
        assert!(b.import(&b"not a tarball"[..]).is_err());
        assert_eq!(
            b.list(None).unwrap().len(),
            1,
            "failed imports leave nothing behind"
        );
    }

    #[test]
    fn ids_cannot_escape_the_directory() {
        let (_dir, reg) = registry();
        for bad in ["../etc/passwd", "a/b", "", "not-a-uuid"] {
            assert!(
                matches!(reg.get(bad), Err(Error::SnapshotNotFound(_))),
                "{bad}"
            );
        }
    }
}
