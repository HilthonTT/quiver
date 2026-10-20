use std::collections::HashMap;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::Database;

#[derive(Debug)]
pub struct SnapshotScheduler {
    stop: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<JoinHandle<()>>,
}

impl SnapshotScheduler {
    pub fn start(db: Arc<Database>, interval: Duration) -> Self {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("quiver-snapshots".into())
            .spawn(move || {
                let mut seen = HashMap::new();
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
                    snapshot_changed(&db, &mut seen);
                }
            })
            .expect("spawn snapshot thread");

        Self {
            stop,
            handle: Some(handle),
        }
    }

    pub fn stop(mut self) {
        self.shutdown();
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

impl Drop for SnapshotScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
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
        let s = SnapshotScheduler::start(db, Duration::from_secs(3600));
        s.stop(); // must return promptly, not after an hour
    }
}
