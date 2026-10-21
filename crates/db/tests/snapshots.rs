use quiver_db::{
    CollectionConfig, Database, DbOptions, Error, IndexConfig, Metric, Point, RestoreOptions, StorageKind,
};
use serde_json::json;

fn seed(db: &Database, name: &str, n: usize) {
    let c = db
        .create_collection(name, CollectionConfig::new(3).metric(Metric::Euclidean))
        .unwrap();
    c.upsert(
        (0..n).map(|i| Point::new(format!("p{i}"), vec![i as f32, 0.5, -1.0]).with_metadata(json!({"i": i}))),
    )
    .unwrap();
}

#[test]
fn snapshot_and_restore_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    seed(&db, "docs", 100);
    let snap = db.create_snapshot("docs").unwrap();
    assert_eq!(snap.manifest.points, 100);

    // Diverge, then restore under a new name and in place.
    let docs = db.collection("docs").unwrap();
    docs.delete(&["p1", "p2"]).unwrap();
    let copy = db
        .restore_snapshot(
            &snap.manifest.id,
            RestoreOptions {
                name: Some("docs_copy".into()),
                replace: false,
            },
        )
        .unwrap();
    assert_eq!(copy.len(), 100);
    assert_eq!(copy.search(&[42.0, 0.5, -1.0], 1).unwrap()[0].id, "p42");

    let err = db
        .restore_snapshot(&snap.manifest.id, RestoreOptions::default())
        .unwrap_err();
    assert!(matches!(err, Error::CollectionExists(_)));
    let restored = db
        .restore_snapshot(
            &snap.manifest.id,
            RestoreOptions {
                name: None,
                replace: true,
            },
        )
        .unwrap();
    assert_eq!(restored.len(), 100);
    assert_eq!(docs.len(), 98, "old handles keep their (dropped) data");

    // Restored collections are durable.
    drop((docs, copy, restored));
    drop(db);
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("docs").unwrap().len(), 100);
    assert_eq!(db.collection("docs_copy").unwrap().len(), 100);
}

#[test]
fn retention_keeps_last_k() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with(
        dir.path(),
        DbOptions {
            snapshot_keep: 2,
            ..Default::default()
        },
    )
    .unwrap();
    seed(&db, "docs", 3);
    for _ in 0..4 {
        db.create_snapshot("docs").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(db.list_snapshots(Some("docs")).unwrap().len(), 2);
}

#[test]
fn snapshots_move_between_storage_engines_and_databases() {
    let a_dir = tempfile::tempdir().unwrap();
    let b_dir = tempfile::tempdir().unwrap();
    let a = Database::open(a_dir.path()).unwrap();
    seed(&a, "docs", 50);
    let snap = a.create_snapshot("docs").unwrap();
    let archive = std::fs::read(a.snapshots().unwrap().path(&snap.manifest.id).unwrap()).unwrap();

    // An in-memory database with a snapshot directory can import and restore it.
    let b = Database::new(
        None,
        DbOptions {
            storage: StorageKind::Memory,
            snapshot_dir: Some(b_dir.path().to_path_buf()),
            ..Default::default()
        },
    )
    .unwrap();
    b.snapshots().unwrap().import(&archive[..]).unwrap();
    let docs = b
        .restore_snapshot(&snap.manifest.id, RestoreOptions::default())
        .unwrap();
    assert_eq!(docs.storage(), StorageKind::Memory);
    assert_eq!(docs.get("p7").unwrap().metadata, Some(json!({"i": 7})));
}

#[test]
fn snapshots_preserve_kd_tree_collections() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = db
        .create_collection(
            "kd",
            CollectionConfig::new(2)
                .metric(Metric::Manhattan)
                .index(IndexConfig::KdTree(Default::default())),
        )
        .unwrap();
    c.upsert((0..500).map(|i| Point::new(i.to_string(), vec![i as f32, (i % 7) as f32])))
        .unwrap();
    let snap = c.create_snapshot().unwrap();
    db.drop_collection("kd").unwrap();
    let restored = db
        .restore_snapshot(&snap.manifest.id, RestoreOptions::default())
        .unwrap();
    assert_eq!(restored.config().metric, Metric::Manhattan);
    assert_eq!(restored.search(&[250.0, 5.0], 1).unwrap()[0].id, "250");
}

#[test]
fn in_memory_databases_have_no_snapshots_by_default() {
    let db = Database::in_memory();
    seed(&db, "docs", 1);
    assert!(matches!(db.create_snapshot("docs"), Err(Error::Unsupported(_))));
}

#[test]
fn memory_storage_with_a_path_persists_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let opts = DbOptions {
        storage: StorageKind::Memory,
        ..Default::default()
    };
    {
        let db = Database::open_with(dir.path(), opts.clone()).unwrap();
        seed(&db, "docs", 5);
        db.create_snapshot("docs").unwrap();
    }
    let db = Database::open_with(dir.path(), opts).unwrap();
    assert!(db.list_collections().is_empty());
    assert_eq!(
        db.list_snapshots(None).unwrap().len(),
        1,
        "but snapshots are on disk"
    );
}

#[cfg(feature = "rocksdb")]
#[test]
fn rocksdb_collections_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let opts = DbOptions {
        storage: StorageKind::RocksDb,
        ..Default::default()
    };
    {
        let db = Database::open_with(dir.path(), opts.clone()).unwrap();
        seed(&db, "docs", 200);
        db.collection("docs").unwrap().delete(&["p3"]).unwrap();
    }
    // Reopening with a different default engine still uses RocksDB for this collection.
    let db = Database::open(dir.path()).unwrap();
    let docs = db.collection("docs").unwrap();
    assert_eq!(docs.storage(), StorageKind::RocksDb);
    assert_eq!(docs.len(), 199);
    assert_eq!(docs.search(&[150.0, 0.5, -1.0], 1).unwrap()[0].id, "p150");
}

#[test]
fn concurrent_snapshots_with_retention_all_succeed() {
    let dir = tempfile::tempdir().unwrap();
    let options = DbOptions {
        snapshot_keep: 1,
        fsync: false,
        ..DbOptions::default()
    };
    let db = std::sync::Arc::new(Database::open_with(dir.path(), options).unwrap());
    seed(&db, "docs", 10);
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let db = std::sync::Arc::clone(&db);
            std::thread::spawn(move || {
                let c = db.collection("docs").unwrap();
                (0..20).filter_map(|_| c.create_snapshot().err()).count()
            })
        })
        .collect();
    let failures: usize = workers.into_iter().map(|w| w.join().unwrap()).sum();
    assert_eq!(failures, 0);
    assert_eq!(db.list_snapshots(Some("docs")).unwrap().len(), 1);
}

#[test]
fn forged_snapshots_with_invalid_points_are_not_restored() {
    use quiver_db::snapshot::write_archive;
    use quiver_db::storage::VectorStore;

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    seed(&db, "docs", 5);
    let config = CollectionConfig::new(3).metric(Metric::Cosine);
    let deep = (0..100).fold(json!(1), |v, _| json!([v]));
    let forgeries: [(&str, [f32; 3], Option<serde_json::Value>); 5] = [
        ("nan", [f32::NAN, 0.0, 1.0], None),
        ("huge", [f32::MAX, f32::MAX, 0.0], None),
        ("zero", [0.0, 0.0, 0.0], None),
        (
            &"x".repeat(quiver_db::types::MAX_ID_LEN + 1),
            [1.0, 0.0, 0.0],
            None,
        ),
        ("deep", [1.0, 0.0, 0.0], Some(deep)),
    ];
    for (id, vector, metadata) in forgeries {
        let mut store = VectorStore::new(3);
        store.push(id.to_owned(), &vector, metadata);
        let snap_id = uuid_v4();
        let path = db.snapshots().unwrap().path(&snap_id).unwrap();
        write_archive(&path, &snap_id, "docs", &config, &store, None, 0).unwrap();

        let err = db
            .restore_snapshot(
                &snap_id,
                RestoreOptions {
                    name: None,
                    replace: true,
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "{id}: {err:?}");
        assert_eq!(
            db.collection("docs").unwrap().len(),
            5,
            "{id}: existing collection was replaced"
        );
    }
}

/// A random (version 4) UUID, without depending on the `uuid` crate.
fn uuid_v4() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let h = format!("{n:032x}");
    format!(
        "{}-{}-4{}-8{}-{}",
        &h[0..8],
        &h[8..12],
        &h[13..16],
        &h[17..20],
        &h[20..32]
    )
}
