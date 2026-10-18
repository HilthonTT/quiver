use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use vectordb_storage::{VectorStore, state};
use vectordb_types::Result;
pub use vectordb_types::snapshot::{Manifest, SnapshotInfo};

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
