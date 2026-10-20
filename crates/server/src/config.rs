use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use quiver_db::auth::{ApiKey, KeyStore, Role};
use quiver_db::{DbOptions, StorageKind};

/// Quiver: a vector database server with REST and gRPC APIs.
#[derive(Debug, Clone, Parser)]
#[command(version, about, max_term_width = 100)]
pub struct Config {
    /// Address both APIs bind to.
    #[arg(long, env = "QUIVER_HOST", default_value = "127.0.0.1")]
    pub host: IpAddr,

    /// REST API port.
    #[arg(long, alias = "port", env = "QUIVER_HTTP_PORT", default_value_t = 6900)]
    pub http_port: u16,

    /// gRPC API port.
    #[arg(long, env = "QUIVER_GRPC_PORT", default_value_t = 6901)]
    pub grpc_port: u16,

    /// Don't start the REST API.
    #[arg(long, env = "QUIVER_DISABLE_HTTP")]
    pub disable_http: bool,

    /// Don't start the gRPC API.
    #[arg(long, env = "QUIVER_DISABLE_GRPC")]
    pub disable_grpc: bool,

    /// Directory holding collections, snapshots and the lock file.
    #[arg(long, env = "QUIVER_DATA_DIR", default_value = "./quiver-data")]
    pub data_dir: PathBuf,

    /// Storage engine for new collections: `file` (WAL + snapshots), `rocksdb`
    /// (needs a build with `--features rocksdb`) or `memory`.
    #[arg(long, env = "QUIVER_STORAGE", default_value = "file")]
    pub storage: StorageKind,

    /// Keep everything in memory and touch no disk at all (no snapshots
    /// unless --snapshot-dir is given).
    #[arg(long, env = "QUIVER_IN_MEMORY")]
    pub in_memory: bool,

    /// fsync after every write (slower, power-loss safe).
    #[arg(long, env = "QUIVER_FSYNC")]
    pub fsync: bool,

    /// JSON file of API keys: `[{"name": "...", "role": "readwrite"|"readonly", "key": "..."}]`.
    #[arg(long, env = "QUIVER_KEYS_FILE")]
    pub keys_file: Option<PathBuf>,

    /// A single read-write API key (combined with --keys-file if both are set).
    #[arg(long, env = "QUIVER_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// A single read-only API key.
    #[arg(long, env = "QUIVER_READONLY_API_KEY", hide_env_values = true)]
    pub readonly_api_key: Option<String>,

    /// Where snapshot archives are stored (default: DATA_DIR/snapshots).
    #[arg(long, env = "QUIVER_SNAPSHOT_DIR")]
    pub snapshot_dir: Option<PathBuf>,

    /// Snapshot every changed collection this often, in seconds (0 = never).
    #[arg(long, env = "QUIVER_SNAPSHOT_INTERVAL", default_value_t = 0)]
    pub snapshot_interval: u64,

    /// Keep at most this many snapshots per collection (0 = keep all).
    #[arg(long, env = "QUIVER_SNAPSHOT_KEEP", default_value_t = 5)]
    pub snapshot_keep: usize,

    /// Maximum REST request body size in MiB.
    #[arg(long, env = "QUIVER_MAX_BODY_MB", default_value_t = 64)]
    pub max_body_mb: usize,

    /// Allow cross-origin REST requests from any origin.
    #[arg(long, env = "QUIVER_CORS")]
    pub cors: bool,
}

impl Config {
    /// Checks combinations that clap cannot express.
    pub fn validate(&self) -> Result<(), String> {
        if self.disable_http && self.disable_grpc {
            return Err("--disable-http and --disable-grpc cannot both be set".into());
        }
        if !self.storage.is_available() {
            return Err(format!(
                "the {} storage engine is not included in this build (rebuild with `--features rocksdb`)",
                self.storage
            ));
        }
        if self.snapshot_interval > 0 && self.in_memory && self.snapshot_dir.is_none() {
            return Err("--snapshot-interval needs a snapshot directory; set --snapshot-dir".into());
        }
        Ok(())
    }

    /// The REST API address.
    pub fn http_addr(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.http_port)
    }

    /// The gRPC API address.
    pub fn grpc_addr(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.grpc_port)
    }

    /// Database root, or `None` with `--in-memory`.
    pub fn root(&self) -> Option<PathBuf> {
        (!self.in_memory).then(|| self.data_dir.clone())
    }

    /// Options for [`quiver_db::Database::new`].
    pub fn db_options(&self) -> DbOptions {
        DbOptions {
            storage: if self.in_memory {
                StorageKind::Memory
            } else {
                self.storage
            },
            fsync: self.fsync,
            snapshot_dir: self.snapshot_dir.clone(),
            snapshot_keep: self.snapshot_keep,
            ..DbOptions::default()
        }
    }

    /// Periodic snapshot interval, if enabled.
    pub fn snapshot_every(&self) -> Option<Duration> {
        (self.snapshot_interval > 0).then(|| Duration::from_secs(self.snapshot_interval))
    }

    /// Builds the key store from --keys-file, --api-key and --readonly-api-key.
    pub fn key_store(&self) -> Result<KeyStore, String> {
        let mut keys = match &self.keys_file {
            Some(path) => KeyStore::from_file(path)
                .map_err(|e| format!("cannot load keys file {}: {e}", path.display()))?,
            None => KeyStore::default(),
        };
        let extra = [
            ("--api-key", Role::ReadWrite, &self.api_key),
            ("--readonly-api-key", Role::ReadOnly, &self.readonly_api_key),
        ];
        for (name, role, key) in extra {
            if let Some(key) = key {
                keys.push(ApiKey {
                    name: name.trim_start_matches('-').into(),
                    role,
                    key: key.clone(),
                })
                .map_err(|e| e.to_string())?;
            }
        }
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Config {
        Config::try_parse_from(std::iter::once("quiver").chain(args.iter().copied())).unwrap()
    }

    #[test]
    fn defaults() {
        let c = parse(&[]);
        assert_eq!(c.http_addr().to_string(), "127.0.0.1:6900");
        assert_eq!(c.grpc_addr().to_string(), "127.0.0.1:6901");
        assert_eq!(c.db_options().storage, StorageKind::File);
        assert!(c.snapshot_every().is_none());
        assert!(c.key_store().unwrap().is_empty());
        assert!(c.validate().is_ok());
    }

    #[test]
    fn in_memory_and_storage() {
        let c = parse(&["--in-memory"]);
        assert!(c.root().is_none());
        assert_eq!(c.db_options().storage, StorageKind::Memory);
        assert_eq!(
            parse(&["--storage", "memory"]).db_options().storage,
            StorageKind::Memory
        );
        assert!(Config::try_parse_from(["quiver", "--storage", "mongo"]).is_err());
    }

    #[test]
    fn port_alias_and_validation() {
        assert_eq!(parse(&["--port", "1234"]).http_port, 1234);
        assert!(parse(&["--disable-http", "--disable-grpc"]).validate().is_err());
        assert!(
            parse(&["--in-memory", "--snapshot-interval", "60"])
                .validate()
                .is_err()
        );
        assert!(
            parse(&[
                "--in-memory",
                "--snapshot-interval",
                "60",
                "--snapshot-dir",
                "/tmp/x"
            ])
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn keys_are_merged() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("keys.json");
        std::fs::write(&file, r#"[{"name": "ci", "role": "readwrite", "key": "k1"}]"#).unwrap();
        let file = file.to_str().unwrap();
        let c = parse(&["--keys-file", file, "--readonly-api-key", "k2"]);
        let keys = c.key_store().unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys.find("k2").unwrap().role, Role::ReadOnly);
        assert!(
            parse(&["--keys-file", file, "--api-key", "k1"])
                .key_store()
                .is_err(),
            "duplicate"
        );
        assert!(parse(&["--keys-file", "/nonexistent.json"]).key_store().is_err());
    }
}
