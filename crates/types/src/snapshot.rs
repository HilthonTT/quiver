use serde::{Deserialize, Serialize};

use crate::config::CollectionConfig;

/// Describes one snapshot. Stored as `manifest.json` inside the archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Archive layout version.
    pub format_version: u32,
    /// Unique id (UUID v4); also the archive's file name.
    pub id: String,
    /// Name of the collection the snapshot was taken from.
    pub collection: String,
    /// When the snapshot was taken, in milliseconds since the Unix epoch.
    pub created_at_ms: u64,
    /// Version of Quiver that wrote the snapshot.
    pub quiver_version: String,
    /// The collection's configuration.
    pub config: CollectionConfig,
    /// Number of live points.
    pub points: usize,
    /// Size of `data.bin` in bytes.
    pub data_bytes: u64,
    /// Lowercase hex SHA-256 of `data.bin`.
    pub data_sha256: String,
}

/// A snapshot in the registry: its manifest plus the archive's size on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotInfo {
    /// The archive's manifest.
    #[serde(flatten)]
    pub manifest: Manifest,
    /// Size of the compressed archive in bytes.
    pub archive_bytes: u64,
}
