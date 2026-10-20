use quiver_db::{CollectionConfig, Database, DbOptions, Error, IndexConfig, Metric, Point};
use serde_json::json;

/// Deterministic pseudo-random vector, unique per `i`.
fn vec_for(i: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let mut z = (i as u64) << 16 | d as u64;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 + 0.01
        })
        .collect()
}

fn points(range: std::ops::Range<usize>, dim: usize) -> Vec<Point> {
    range
        .map(|i| Point::new(format!("p{i}"), vec_for(i, dim)).with_metadata(json!({ "i": i })))
        .collect()
}

#[test]
fn data_survives_reopen_via_wal_replay() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(8)).unwrap();
        c.upsert(points(0..100, 8)).unwrap();
        c.delete(&["p3", "p4"]).unwrap();
        // No flush: everything must come back from the WAL.
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("docs").unwrap();
    assert_eq!(c.len(), 98);
    assert!(c.get("p3").is_none());
    assert_eq!(c.get("p5").unwrap().metadata, Some(json!({ "i": 5 })));

    let hits = c.search(&vec_for(42, 8), 1).unwrap();
    assert_eq!(hits[0].id, "p42");
}

#[test]
fn data_survives_reopen_via_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        for (name, index) in [("flat", IndexConfig::Flat), ("hnsw", IndexConfig::default())] {
            let c = db
                .create_collection(
                    name,
                    CollectionConfig::new(8).metric(Metric::Euclidean).index(index),
                )
                .unwrap();
            c.upsert(points(0..200, 8)).unwrap();
        }
        db.flush().unwrap();
        // Writes after the snapshot land in the WAL.
        db.collection("hnsw")
            .unwrap()
            .upsert(points(200..210, 8))
            .unwrap();
    }
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("flat").unwrap().len(), 200);
    let hnsw = db.collection("hnsw").unwrap();
    assert_eq!(hnsw.len(), 210);
    assert_eq!(hnsw.config().metric, Metric::Euclidean);
    assert_eq!(hnsw.search(&vec_for(205, 8), 1).unwrap()[0].id, "p205");
}

#[test]
fn torn_wal_write_loses_only_the_last_batch() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("collections/docs/wal.log");
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(4)).unwrap();
        c.upsert(points(0..10, 4)).unwrap();
        c.upsert(points(10..20, 4)).unwrap();
    }
    let len = std::fs::metadata(&wal).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal)
        .unwrap()
        .set_len(len - 10)
        .unwrap();

    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("docs").unwrap().len(), 10);
}

fn nested(depth: usize) -> serde_json::Value {
    let mut v = json!(1);
    for _ in 0..depth {
        v = json!([v]);
    }
    v
}

#[test]
fn overly_nested_metadata_is_rejected_and_later_writes_survive() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(2)).unwrap();
        c.upsert([Point::new("a", vec![1.0, 0.5])]).unwrap();
        // Deep enough that the WAL or checkpoint could no longer be read back.
        for depth in [126, 200] {
            let deep = Point::new("deep", vec![0.5, 1.0]).with_metadata(nested(depth));
            assert!(matches!(c.upsert([deep]), Err(Error::InvalidArgument(_))));
        }
        c.upsert([Point::new("ok", vec![0.5, 1.0]).with_metadata(nested(32))])
            .unwrap();
        c.upsert([Point::new("b", vec![0.3, 0.7])]).unwrap();
        c.flush().unwrap();
        c.upsert([Point::new("c", vec![0.2, 0.9])]).unwrap();
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("docs").unwrap();
    assert_eq!(c.len(), 4);
    assert_eq!(c.get("ok").unwrap().metadata, Some(nested(32)));
}

#[test]
fn null_metadata_reads_back_the_same_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(2)).unwrap();
        c.upsert([Point::new("a", vec![1.0, 0.5]).with_metadata(serde_json::Value::Null)])
            .unwrap();
        assert_eq!(c.get("a").unwrap().metadata, None);
    }
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("docs").unwrap().get("a").unwrap().metadata, None);
}

#[test]
fn recreating_a_half_dropped_collection_starts_empty() {
    let dir = tempfile::tempdir().unwrap();
    let cdir = dir.path().join("collections/docs");
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(2)).unwrap();
        c.upsert([Point::new("old", vec![1.0, 0.5])]).unwrap();
    }
    // A drop interrupted after removing the metadata but before the data files.
    std::fs::remove_file(cdir.join("collection.json")).unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(2)).unwrap();
        assert_eq!(c.len(), 0);
        c.upsert([Point::new("new", vec![0.3, 0.7])]).unwrap();
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("docs").unwrap();
    assert!(c.get("old").is_none());
    assert_eq!(c.len(), 1);
}

#[test]
fn corrupt_snapshot_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(4)).unwrap();
        c.upsert(points(0..10, 4)).unwrap();
        c.flush().unwrap();
    }
    let snap = dir.path().join("collections/docs/state.bin");
    let mut bytes = std::fs::read(&snap).unwrap();
    bytes[20] ^= 0xFF;
    std::fs::write(&snap, bytes).unwrap();
    assert!(matches!(Database::open(dir.path()), Err(Error::Corrupt(_))));
}

#[test]
fn second_process_cannot_open_locked_database() {
    let dir = tempfile::tempdir().unwrap();
    let _db = Database::open(dir.path()).unwrap();
    assert!(Database::open(dir.path()).is_err());
}

#[test]
fn drop_collection_removes_files() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    db.create_collection("tmp", CollectionConfig::new(4)).unwrap();
    db.drop_collection("tmp").unwrap();
    assert!(!dir.path().join("collections/tmp").exists());
    drop(db);
    assert!(Database::open(dir.path()).unwrap().list_collections().is_empty());
}

#[test]
fn compaction_preserves_data_and_reclaims_tombstones() {
    let dir = tempfile::tempdir().unwrap();
    let opts = DbOptions {
        wal_checkpoint_bytes: 64 * 1024,
        ..Default::default()
    };
    let db = Database::open_with(dir.path(), opts.clone()).unwrap();
    let c = db.create_collection("docs", CollectionConfig::new(4)).unwrap();
    c.upsert(points(0..3_000, 4)).unwrap();
    let ids: Vec<String> = (0..2_000).map(|i| format!("p{i}")).collect();
    c.delete(&ids).unwrap();

    let info = c.info();
    assert_eq!(info.count, 1_000);
    assert_eq!(info.tombstones, 0, "auto-compaction should have run");
    assert_eq!(c.search(&vec_for(2_500, 4), 1).unwrap()[0].id, "p2500");

    drop(c);
    drop(db);
    let db = Database::open_with(dir.path(), opts).unwrap();
    assert_eq!(db.collection("docs").unwrap().len(), 1_000);
}

#[test]
fn stale_handles_cannot_write_into_a_recreated_collection() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let old = db.create_collection("c", CollectionConfig::new(2)).unwrap();
    old.upsert([Point::new("old", vec![1.0, 0.0])]).unwrap();
    db.drop_collection("c").unwrap();

    let new = db.create_collection("c", CollectionConfig::new(2)).unwrap();
    new.upsert([Point::new("new", vec![0.0, 1.0])]).unwrap();

    // The dropped handle must not log to, or checkpoint over, the new files.
    assert!(matches!(
        old.upsert([Point::new("ghost", vec![1.0, 1.0])]),
        Err(Error::CollectionNotFound(_))
    ));
    old.flush().unwrap();
    drop((old, new, db));

    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.len(), 1);
    assert!(c.get("new").is_some());
}

#[test]
fn scroll_cursor_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let cursor = {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", CollectionConfig::new(8)).unwrap();
        c.upsert(points(0..30, 8)).unwrap();
        c.flush().unwrap();
        // Some changes only in the WAL, so recovery replays them.
        c.delete(&["p25"]).unwrap();
        c.upsert(points(30..35, 8)).unwrap();
        let page = c.scroll(None, 10);
        assert_eq!(page.points.last().unwrap().id, "p17");
        page.next_cursor.unwrap()
    };
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("docs").unwrap();
    let page = c.scroll(Some(&cursor), 100);
    let ids: Vec<String> = page.points.into_iter().map(|p| p.id).collect();
    let mut expected: Vec<String> = (0..35)
        .map(|i| format!("p{i}"))
        .filter(|id| id.as_str() > "p17" && id != "p25")
        .collect();
    expected.sort();
    assert_eq!(ids, expected);
    assert_eq!(page.next_cursor, None);
}
