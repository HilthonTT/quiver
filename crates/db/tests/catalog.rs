//! Renaming collections and aliases.

use quiver_db::{CollectionConfig, Database, DbOptions, Error, Point, RestoreOptions, StorageKind};
use serde_json::json;

fn points(n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| Point::new(format!("p{i}"), vec![i as f32 + 1.0, 1.0]).with_metadata(json!({ "i": i })))
        .collect()
}

#[test]
fn renamed_collections_keep_their_data() {
    let mut engines = vec![None, Some(StorageKind::File)];
    if StorageKind::RocksDb.is_available() {
        engines.push(Some(StorageKind::RocksDb));
    }
    for storage in engines {
        let dir = tempfile::tempdir().unwrap();
        let open = || match storage {
            None => Database::in_memory(),
            Some(storage) => Database::open_with(
                dir.path(),
                DbOptions {
                    storage,
                    ..Default::default()
                },
            )
            .unwrap(),
        };
        let db = open();
        let old = db.create_collection("v1", CollectionConfig::new(2)).unwrap();
        old.upsert(points(50)).unwrap();
        old.create_field_index("i").unwrap();
        db.set_alias("live", "v1").unwrap();

        let renamed = db.rename_collection("v1", "v2").unwrap();
        assert_eq!(renamed.name(), "v2");
        assert_eq!(renamed.len(), 50);
        assert_eq!(renamed.info().field_indexes, ["i"]);
        assert_eq!(renamed.search(&[3.0, 1.0], 1).unwrap()[0].id, "p2");
        assert!(matches!(db.collection("v1"), Err(Error::CollectionNotFound(_))));
        // The old handle is closed, like a dropped collection's.
        assert!(old.upsert(points(1)).is_err());
        assert_eq!(db.aliases()["live"], "v2", "{storage:?}: aliases follow");
        renamed.upsert([Point::new("new", vec![9.0, 9.0])]).unwrap();

        if storage.is_some() {
            drop((old, renamed, db));
            let db = open();
            let c = db.collection("v2").unwrap();
            assert_eq!(c.len(), 51, "{storage:?}");
            assert!(db.collection("v1").is_err());
            assert_eq!(db.collection("live").unwrap().name(), "v2");
            assert!(!dir.path().join("collections/v1").exists());
        }
    }
}

#[test]
fn rename_checks_names() {
    let db = Database::in_memory();
    db.create_collection("a", CollectionConfig::new(2)).unwrap();
    db.create_collection("b", CollectionConfig::new(2)).unwrap();
    db.set_alias("x", "a").unwrap();
    assert!(matches!(
        db.rename_collection("a", "b"),
        Err(Error::CollectionExists(_))
    ));
    assert!(matches!(
        db.rename_collection("a", "x"),
        Err(Error::AliasExists(_))
    ));
    assert!(matches!(
        db.rename_collection("nope", "c"),
        Err(Error::CollectionNotFound(_))
    ));
    assert!(db.rename_collection("x", "c").is_err(), "not by alias");
    assert!(db.rename_collection("a", "bad name").is_err());
    assert_eq!(db.rename_collection("a", "a").unwrap().name(), "a");
}

#[test]
fn aliases_resolve_switch_and_persist() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let blue = db.create_collection("blue", CollectionConfig::new(2)).unwrap();
        let green = db.create_collection("green", CollectionConfig::new(2)).unwrap();
        blue.upsert(points(3)).unwrap();
        green.upsert(points(5)).unwrap();

        db.set_alias("prod", "blue").unwrap();
        assert_eq!(db.collection("prod").unwrap().len(), 3);
        // Re-pointing is atomic: no moment without a `prod`.
        db.set_alias("prod", "green").unwrap();
        assert_eq!(db.collection("prod").unwrap().len(), 5);
        db.set_alias("staging", "blue").unwrap();

        // Aliases and collections share one namespace.
        assert!(matches!(
            db.create_collection("prod", CollectionConfig::new(2)),
            Err(Error::AliasExists(_))
        ));
        assert!(matches!(
            db.set_alias("blue", "green"),
            Err(Error::CollectionExists(_))
        ));
        assert!(matches!(
            db.set_alias("x", "nope"),
            Err(Error::CollectionNotFound(_))
        ));
        assert!(db.set_alias("x", "prod").is_err(), "no aliases of aliases");
        assert!(db.set_alias("bad name", "blue").is_err());
        assert!(db.drop_collection("prod").is_err(), "not by alias");
    }
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.aliases().len(), 2);
    assert_eq!(db.collection("prod").unwrap().name(), "green");

    db.delete_alias("staging").unwrap();
    assert!(matches!(db.delete_alias("staging"), Err(Error::AliasNotFound(_))));
    assert!(db.collection("blue").is_ok(), "the collection stays");

    // Dropping a collection takes its aliases with it.
    db.drop_collection("green").unwrap();
    assert!(db.aliases().is_empty());
    drop(db);
    assert!(Database::open(dir.path()).unwrap().aliases().is_empty());
}

#[test]
fn snapshots_cannot_restore_over_an_alias() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = db.create_collection("docs", CollectionConfig::new(2)).unwrap();
    c.upsert(points(2)).unwrap();
    let snap = c.create_snapshot().unwrap();
    db.set_alias("live", "docs").unwrap();
    let opts = RestoreOptions {
        name: Some("live".into()),
        replace: true,
    };
    assert!(matches!(
        db.restore_snapshot(&snap.manifest.id, opts),
        Err(Error::AliasExists(_))
    ));
}

#[test]
fn searches_on_a_renamed_handle_fail_cleanly() {
    use quiver_db::{SearchOptions, SparseVector, SparseVectorConfig, VectorConfig};
    let db = Database::in_memory();
    let config = CollectionConfig::new(2)
        .with_vector("v", VectorConfig::new(2))
        .with_sparse_vector("s", SparseVectorConfig::default());
    let old = db.create_collection("a", config).unwrap();
    old.upsert([Point::new("p", vec![1.0, 0.0])
        .with_vector("v", vec![1.0, 0.0])
        .with_sparse_vector("s", SparseVector::new(vec![1], vec![1.0]))])
        .unwrap();
    db.rename_collection("a", "b").unwrap();

    let not_found = |r: quiver_db::Result<Vec<quiver_db::SearchHit>>| {
        assert!(matches!(r, Err(Error::CollectionNotFound(_))), "{r:?}");
    };
    not_found(old.search(&[1.0, 0.0], 1));
    not_found(old.search_with(&[1.0, 0.0], 1, &SearchOptions::default().using("v")));
    not_found(old.search_by_id("p", 1, &SearchOptions::default()));
    not_found(old.search_sparse(
        "s",
        &SparseVector::new(vec![1], vec![1.0]),
        1,
        &SearchOptions::default(),
    ));
}
