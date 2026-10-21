use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use flate2::Compression;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

use quiver_storage::{VectorStore, state};
pub use quiver_types::snapshot::{Manifest, SnapshotInfo};
use quiver_types::{CollectionConfig, Error, MAX_FIELD_INDEXES, Result, validate_field_path, validate_name};

/// Version of the archive layout. Bumped on incompatible changes.
pub const FORMAT_VERSION: u32 = 1;

const MANIFEST: &str = "manifest.json";
const DATA: &str = "data.bin";
const EXTENSION: &str = ".tar.gz";
/// Refuse to load archives whose data claims to be larger than this.
const MAX_DATA_BYTES: u64 = 64 << 30;
/// Reserve at most this much for `data.bin` before reading it.
const PREALLOCATE_MAX: u64 = 64 << 20;
/// Real manifests are well under a kilobyte.
const MAX_MANIFEST_BYTES: u64 = 1 << 20;
/// Partial files untouched for this long are left over from a crash, not
/// an upload or snapshot still in progress (those keep writing).
const STALE_PARTIAL_AGE: Duration = Duration::from_secs(60 * 60);

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
    hex(&Sha256::digest(bytes))
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn append(builder: &mut tar::Builder<impl Write>, name: &str, bytes: &[u8], mtime: u64) -> Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(mtime);
    header.set_cksum();
    builder.append_data(&mut header, name, bytes)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn write_archive(
    path: &Path,
    id: &str,
    collection: &str,
    config: &CollectionConfig,
    field_indexes: &[String],
    store: &VectorStore,
    index: Option<&[u8]>,
    created_at_ms: u64,
) -> Result<Manifest> {
    let data = state::encode(Vec::new(), store, index)?;
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        id: id.to_owned(),
        collection: collection.to_owned(),
        created_at_ms,
        quiver_version: env!("CARGO_PKG_VERSION").to_owned(),
        config: config.clone(),
        points: store.live(),
        data_bytes: data.len() as u64,
        data_sha256: sha256_hex(&data),
        field_indexes: field_indexes.to_vec(),
    };

    let tmp = path.with_extension("partial");
    let written = (|| {
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
        state::sync_parent(path)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written.map(|()| manifest)
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

    if manifest.field_indexes.len() > MAX_FIELD_INDEXES {
        return Err(Error::Corrupt("snapshot has too many field indexes".into()));
    }
    for field in &manifest.field_indexes {
        validate_field_path(field)
            .map_err(|_| Error::Corrupt("snapshot has an invalid field index".into()))?;
    }

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

    Err(Error::Corrupt(format!("{} has no manifest", path.display())))
}

/// Walks an archive's entries. The manifest must come first and be valid;
/// `on_data` then gets a reader over exactly the `data_bytes` it declares.
/// Checking this before touching the data keeps a hostile archive from
/// making us decompress or buffer more than its manifest admits to.
fn walk_archive(
    path: &Path,
    mut on_data: impl FnMut(&Manifest, &mut dyn Read) -> Result<()>,
) -> Result<Manifest> {
    let mut archive = tar::Archive::new(GzDecoder::new(BufReader::new(File::open(path)?)));
    let mut manifest: Option<Manifest> = None;
    let mut seen_data = false;

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
                if entry.size() > MAX_MANIFEST_BYTES {
                    return Err(Error::Corrupt("snapshot manifest is implausibly large".into()));
                }
                let parsed: Manifest = serde_json::from_reader((&mut entry).take(MAX_MANIFEST_BYTES))?;
                check_manifest(&parsed)?;
                manifest = Some(parsed);
            }
            DATA => {
                if seen_data {
                    return Err(duplicate());
                }
                let m = manifest
                    .as_ref()
                    .ok_or_else(|| Error::Corrupt("snapshot data comes before its manifest".into()))?;
                if m.data_bytes > MAX_DATA_BYTES {
                    return Err(Error::Corrupt("snapshot data is implausibly large".into()));
                }
                if entry.size() != m.data_bytes {
                    return Err(Error::Corrupt(format!(
                        "snapshot {} data is {} bytes but its manifest says {}",
                        m.id,
                        entry.size(),
                        m.data_bytes
                    )));
                }
                seen_data = true;
                on_data(m, &mut (&mut entry).take(m.data_bytes))?;
            }
            _ => {}
        }
    }

    let manifest = manifest.ok_or_else(|| Error::Corrupt("snapshot has no manifest".into()))?;
    if !seen_data {
        return Err(Error::Corrupt("snapshot has no data".into()));
    }
    Ok(manifest)
}

/// Checks an archive's manifest and data checksum by streaming the data,
/// without holding it in memory (it may be far larger than the archive).
pub fn verify_archive(path: &Path) -> Result<Manifest> {
    let mut digest = None;
    let manifest = walk_archive(path, |m, data| {
        let mut magic = [0u8; 8];
        data.read_exact(&mut magic)
            .map_err(|_| Error::Corrupt(format!("snapshot {} data is truncated", m.id)))?;
        if &magic != state::MAGIC {
            return Err(Error::Corrupt(format!(
                "snapshot {} data is not a Quiver state",
                m.id
            )));
        }
        let mut hasher = Sha256::new();
        hasher.update(magic);
        let mut len = magic.len() as u64;
        let mut buf = vec![0u8; 64 << 10];
        loop {
            let n = data.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            len += n as u64;
        }
        digest = Some((len, hex(&hasher.finalize())));
        Ok(())
    })?;
    match digest {
        Some((len, sha)) if len == manifest.data_bytes && sha == manifest.data_sha256 => Ok(manifest),
        _ => Err(Error::Corrupt(format!(
            "snapshot {} failed its checksum",
            manifest.id
        ))),
    }
}

pub fn read_archive(path: &Path) -> Result<Loaded> {
    let mut data = Vec::new();
    let manifest = walk_archive(path, |m, reader| {
        // Grow as data actually arrives rather than trusting the size up front.
        data.reserve(m.data_bytes.min(PREALLOCATE_MAX) as usize);
        reader.read_to_end(&mut data)?;
        Ok(())
    })?;
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
    /// Serialises changes to the directory, and holds the last `created_at_ms`
    /// handed out so that snapshots taken within one millisecond still sort
    /// oldest-first (retention relies on that order).
    last_created_ms: Arc<Mutex<u64>>,
}

impl SnapshotRegistry {
    /// Opens (creating if needed) a snapshot directory, removing `*.partial`
    /// files that interrupted snapshots or uploads left behind.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let mut removed = 0usize;
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            let stale = || {
                fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|age| age >= STALE_PARTIAL_AGE)
            };
            // Another process sharing this directory may still be writing a
            // fresh one; leftovers from a crash are removed once they age.
            if path.extension().is_some_and(|e| e == "partial") && path.is_file() && stale() {
                match fs::remove_file(&path) {
                    Ok(()) => removed += 1,
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "cannot remove stale partial snapshot")
                    }
                }
            }
        }
        if removed > 0 {
            tracing::info!(dir = %dir.display(), removed, "removed stale partial snapshot files");
        }
        Ok(Self {
            dir,
            last_created_ms: Arc::default(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, u64> {
        self.last_created_ms
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
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
        self.create_with_fields(collection, config, &[], store, index)
    }

    /// Like [`Self::create`], also recording the collection's indexed fields.
    pub fn create_with_fields(
        &self,
        collection: &str,
        config: &CollectionConfig,
        field_indexes: &[String],
        store: &VectorStore,
        index: Option<&[u8]>,
    ) -> Result<SnapshotInfo> {
        let id = uuid::Uuid::new_v4().hyphenated().to_string();
        let path = self.path(&id)?;
        let mut last = self.lock();
        let created_at_ms = now_ms().max(*last + 1);
        let manifest = write_archive(
            &path,
            &id,
            collection,
            config,
            field_indexes,
            store,
            index,
            created_at_ms,
        )?;
        *last = created_at_ms;

        tracing::info!(snapshot = %id, collection, points = store.live(), "created snapshot");
        Ok(SnapshotInfo {
            manifest,
            archive_bytes: fs::metadata(&path)?.len(),
        })
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
                Ok(info) if collection.is_none_or(|c| c == info.manifest.collection) => out.push(info),
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "skipping unreadable snapshot")
                }
            }
        }
        out.sort_by(|a, b| {
            (a.manifest.created_at_ms, &a.manifest.id).cmp(&(b.manifest.created_at_ms, &b.manifest.id))
        });
        Ok(out)
    }

    pub fn get(&self, id: &str) -> Result<SnapshotInfo> {
        Self::info(&self.existing_path(id)?)
    }

    pub fn load(&self, id: &str) -> Result<Loaded> {
        read_archive(&self.existing_path(id)?)
    }

    /// Deletes a snapshot, even one whose archive is no longer readable.
    pub fn delete(&self, id: &str) -> Result<()> {
        let _guard = self.lock();
        self.delete_locked(id)
    }

    fn delete_locked(&self, id: &str) -> Result<()> {
        fs::remove_file(self.existing_path(id)?)?;
        tracing::info!(snapshot = %id, "deleted snapshot");
        Ok(())
    }

    pub fn staging_path(&self) -> PathBuf {
        self.dir.join(format!("import-{}.partial", uuid::Uuid::new_v4()))
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
            // Streams the data rather than decoding it, so memory stays flat
            // however far a (possibly hostile) upload decompresses.
            let manifest = verify_archive(staged)?;
            let target = self.path(&manifest.id)?;
            let _guard = self.lock();
            if target.exists() {
                return Err(Error::invalid(format!("snapshot {} already exists", manifest.id)));
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
        let _guard = self.lock();
        let all = self.list(Some(collection))?;
        let excess = all.len().saturating_sub(keep);
        let mut deleted = Vec::with_capacity(excess);
        for info in &all[..excess] {
            self.delete_locked(&info.manifest.id)?;
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
            s.push(format!("p{i}"), &[i as f32, 1.0, 2.0], Some(json!({"i": i})));
        }
        s
    }

    fn registry() -> (tempfile::TempDir, SnapshotRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = SnapshotRegistry::open(dir.path().join("snapshots")).unwrap();
        (dir, reg)
    }

    #[test]
    fn prune_keeps_the_newest_even_within_one_millisecond() {
        let (_dir, reg) = registry();
        let config = CollectionConfig::new(3);
        for _ in 0..200 {
            let older = reg.create("docs", &config, &store(0), None).unwrap().manifest;
            let newest = reg.create("docs", &config, &store(0), None).unwrap().manifest;
            assert!(newest.created_at_ms > older.created_at_ms);
            reg.prune("docs", 1).unwrap();
            assert!(
                reg.get(&newest.id).is_ok(),
                "prune deleted the snapshot just created"
            );
        }
    }

    #[test]
    fn open_removes_only_stale_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let two_hours_ago = SystemTime::now() - Duration::from_secs(2 * 3600);
        for name in ["import-1234.partial", "abcd.tar.partial"] {
            let path = dir.path().join(name);
            fs::write(&path, b"junk").unwrap();
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(two_hours_ago)
                .unwrap();
        }
        // Fresh partial files may belong to another process still writing them.
        fs::write(dir.path().join("import-live.partial"), b"in progress").unwrap();
        fs::write(dir.path().join("notes.txt"), b"keep").unwrap();

        SnapshotRegistry::open(dir.path()).unwrap();
        let mut left: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["import-live.partial", "notes.txt"]);
    }

    #[test]
    fn corrupt_archive_can_be_deleted() {
        let (_dir, reg) = registry();
        let info = reg
            .create("docs", &CollectionConfig::new(3), &store(1), None)
            .unwrap();
        let path = reg.path(&info.manifest.id).unwrap();
        fs::write(&path, b"garbage").unwrap();
        reg.delete(&info.manifest.id).unwrap();
        assert!(!path.exists());
        assert!(matches!(
            reg.delete(&info.manifest.id),
            Err(Error::SnapshotNotFound(_))
        ));
    }

    #[test]
    fn create_list_load_delete() {
        let (_dir, reg) = registry();
        let config = CollectionConfig::new(3);
        let info = reg.create("docs", &config, &store(10), Some(b"idx")).unwrap();
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
                std::thread::sleep(Duration::from_millis(2));
                reg.create("docs", &config, &store(i), None).unwrap().manifest.id
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

        assert!(matches!(reg.load(&info.manifest.id), Err(Error::Corrupt(_))));
    }

    /// Builds an archive from raw entries, to forge hostile uploads.
    fn forged(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
        for (name, bytes) in entries {
            append(&mut builder, name, bytes, 0).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    fn manifest_for(data: &[u8]) -> Vec<u8> {
        let (_dir, reg) = registry();
        let mut manifest = reg
            .create("docs", &CollectionConfig::new(3), &store(0), None)
            .unwrap()
            .manifest;
        manifest.data_bytes = data.len() as u64;
        manifest.data_sha256 = sha256_hex(data);
        serde_json::to_vec(&manifest).unwrap()
    }

    #[test]
    fn hostile_uploads_are_rejected_before_their_data_is_buffered() {
        let (_dir, reg) = registry();
        let corrupt = |bytes: Vec<u8>| matches!(reg.import(&bytes[..]), Err(Error::Corrupt(_)));

        // A checksum-valid but meaningless payload (e.g. a run of zeros that
        // compresses to almost nothing) is not a Quiver state.
        let zeros = vec![0u8; 1 << 20];
        assert!(corrupt(forged(&[
            (MANIFEST, &manifest_for(&zeros)),
            (DATA, &zeros)
        ])));

        // Data whose size disagrees with the manifest.
        let mut data = state::encode(Vec::new(), &store(2), None).unwrap();
        let manifest = manifest_for(&data);
        data.push(0);
        assert!(corrupt(forged(&[(MANIFEST, &manifest), (DATA, &data)])));

        // Data before the manifest, so its size cannot be checked first.
        let data = state::encode(Vec::new(), &store(2), None).unwrap();
        assert!(corrupt(forged(&[
            (DATA, &data),
            (MANIFEST, &manifest_for(&data))
        ])));

        assert!(reg.list(None).unwrap().is_empty());
        assert!(
            fs::read_dir(reg.dir()).unwrap().next().is_none(),
            "staging file left behind"
        );
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
            assert!(matches!(reg.get(bad), Err(Error::SnapshotNotFound(_))), "{bad}");
        }
    }
}
