use std::collections::HashMap;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::Database;

/// A thread running a task every `interval` until stopped.
#[derive(Debug)]
struct Periodic {
    stop: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<JoinHandle<()>>,
}

impl Periodic {
    fn start(name: &str, interval: Duration, mut task: impl FnMut() + Send + 'static) -> Self {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                loop {
                    {
                        let (lock, cv) = &*signal;
                        let mut stopped = lock.lock();
                        if !*stopped {
                            cv.wait_for(&mut stopped, interval);
                        }
                        if *stopped {
                            return;
                        }
                    }
                    task();
                }
            })
            .unwrap_or_else(|e| panic!("spawn {name} thread: {e}"));

        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn shutdown(&mut self) {
        let (lock, cv) = &*self.stop;
        *lock.lock() = true;
        cv.notify_all();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Periodic {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Snapshots every collection that changed since its last snapshot, every
/// `interval`.
#[derive(Debug)]
pub struct SnapshotScheduler(Periodic);

impl SnapshotScheduler {
    pub fn start(db: Arc<Database>, interval: Duration) -> Self {
        let mut seen = HashMap::new();
        Self(Periodic::start("quiver-snapshots", interval, move || {
            snapshot_changed(&db, &mut seen);
        }))
    }

    pub fn stop(mut self) {
        self.0.shutdown();
    }
}

/// Deletes expired points from every collection, every `interval`.
#[derive(Debug)]
pub struct ExpirySweeper(Periodic);

impl ExpirySweeper {
    pub fn start(db: Arc<Database>, interval: Duration) -> Self {
        Self(Periodic::start("quiver-expiry", interval, move || {
            purge_expired(&db);
        }))
    }

    pub fn stop(mut self) {
        self.0.shutdown();
    }
}

/// Deletes expired points from every collection where expiry is on;
/// returns how many.
pub(crate) fn purge_expired(db: &Database) -> usize {
    let mut purged = 0;
    for c in db.collections() {
        match c.purge_expired() {
            Ok(n) => purged += n,
            Err(e) => tracing::warn!(collection = c.name(), error = %e, "purging expired points failed"),
        }
    }
    purged
}

pub(crate) fn snapshot_changed(db: &Database, seen: &mut HashMap<String, (u64, u64)>) {
    let collections = db.collections();
    seen.retain(|name, _| collections.iter().any(|c| c.name() == name));
    for c in collections {
        let version = c.version();
        if seen.get(c.name()) == Some(&version) {
            continue;
        }
        match c.create_snapshot() {
            Ok(info) => {
                tracing::debug!(collection = c.name(), snapshot = %info.manifest.id, "scheduled snapshot");
                seen.insert(c.name().to_owned(), version);
            }
            Err(e) => {
                tracing::warn!(collection = c.name(), error = %e, "scheduled snapshot failed")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CollectionConfig, DbOptions, Point};

    #[test]
    fn snapshots_only_changed_collections() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open_with(dir.path(), DbOptions::default()).unwrap();
        let a = db.create_collection("a", CollectionConfig::new(2)).unwrap();
        db.create_collection("b", CollectionConfig::new(2)).unwrap();
        a.upsert([Point::new("x", vec![1.0, 0.0])]).unwrap();

        let mut seen = HashMap::new();
        snapshot_changed(&db, &mut seen);
        assert_eq!(db.list_snapshots(None).unwrap().len(), 2);

        snapshot_changed(&db, &mut seen);
        assert_eq!(db.list_snapshots(None).unwrap().len(), 2, "nothing changed");

        a.upsert([Point::new("y", vec![0.0, 1.0])]).unwrap();
        snapshot_changed(&db, &mut seen);
        assert_eq!(db.list_snapshots(Some("a")).unwrap().len(), 2);
        assert_eq!(db.list_snapshots(Some("b")).unwrap().len(), 1);
    }

    #[test]
    fn recreated_collection_is_snapshotted() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open_with(dir.path(), DbOptions::default()).unwrap();
        let a = db.create_collection("a", CollectionConfig::new(2)).unwrap();
        a.upsert([Point::new("old", vec![1.0, 0.0])]).unwrap();
        let mut seen = HashMap::new();
        snapshot_changed(&db, &mut seen);

        // Same name, same number of changes: a different collection all the same.
        drop(a);
        db.drop_collection("a").unwrap();
        let a = db.create_collection("a", CollectionConfig::new(2)).unwrap();
        a.upsert([Point::new("new", vec![0.0, 1.0])]).unwrap();
        snapshot_changed(&db, &mut seen);
        assert_eq!(db.list_snapshots(Some("a")).unwrap().len(), 2);
    }

    #[test]
    fn start_and_stop() {
        let db = Arc::new(Database::in_memory());
        let s = SnapshotScheduler::start(Arc::clone(&db), Duration::from_secs(3600));
        s.stop(); // must return promptly, not after an hour
        ExpirySweeper::start(db, Duration::from_secs(3600)).stop();
    }

    #[test]
    fn sweeps_expired_points() {
        let db = Arc::new(Database::in_memory());
        let config = CollectionConfig::new(2).expiry(crate::ExpiryConfig::new("exp"));
        let a = db.create_collection("a", config).unwrap();
        let plain = db.create_collection("plain", CollectionConfig::new(2)).unwrap();
        a.upsert([
            Point::new("old", vec![1.0, 0.0]).with_metadata(serde_json::json!({"exp": 1})),
            Point::new("new", vec![0.0, 1.0]).with_metadata(serde_json::json!({"exp": 1e12})),
        ])
        .unwrap();
        plain.upsert([Point::new("x", vec![1.0, 0.0])]).unwrap();
        assert_eq!(a.info().tombstones, 0);
        assert_eq!(purge_expired(&db), 1);
        assert_eq!(a.info().tombstones, 1);
        assert_eq!(purge_expired(&db), 0);

        a.upsert([Point::new("later", vec![1.0, 1.0]).with_metadata(serde_json::json!({"exp": 2}))])
            .unwrap();
        let sweeper = ExpirySweeper::start(Arc::clone(&db), Duration::from_millis(10));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while a.info().tombstones < 2 {
            assert!(std::time::Instant::now() < deadline, "the sweeper never ran");
            std::thread::sleep(Duration::from_millis(10));
        }
        sweeper.stop();
        assert_eq!(plain.len(), 1);
    }
}
