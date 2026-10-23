use quiver_db::{
    CollectionConfig, CollectionUpdate, Database, Error, ExpiryConfig, Filter, MetadataMode, Point,
    PointSelector, RestoreOptions, SearchOptions,
};
use serde_json::json;

/// Far in the past and far in the future, as Unix seconds.
const PAST: u64 = 1_000;
const FUTURE: u64 = 100_000_000_000;

fn point(id: &str, x: f32, expires: Option<u64>) -> Point {
    let p = Point::new(id, vec![x, 1.0]);
    match expires {
        Some(at) => p.with_metadata(json!({"kind": "doc", "exp": at})),
        None => p.with_metadata(json!({"kind": "doc"})),
    }
}

fn config() -> CollectionConfig {
    CollectionConfig::new(2).expiry(ExpiryConfig::new("exp"))
}

fn ids(mut ids: Vec<String>) -> Vec<String> {
    ids.sort();
    ids
}

#[test]
fn expired_points_disappear_from_every_read() {
    let db = Database::in_memory();
    let c = db.create_collection("c", config()).unwrap();
    c.upsert([
        point("gone", 1.0, Some(PAST)),
        point("kept", 2.0, Some(FUTURE)),
        point("forever", 3.0, None),
    ])
    .unwrap();

    assert_eq!(c.len(), 2);
    assert_eq!(c.info().count, 2);
    assert_eq!(c.info().field_indexes, ["exp"], "the expiry field is indexed");
    assert!(c.get("gone").is_none());
    assert_eq!(c.get_many(&["gone", "kept"]).len(), 1);
    let page = c.scroll(None, 10);
    assert_eq!(
        page.points.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        ["forever", "kept"]
    );
    let page = c.scroll(None, 1);
    assert_eq!(page.points[0].id, "forever");
    assert_eq!(page.next_cursor.as_deref(), Some("forever"));
    assert_eq!(c.scroll(Some("forever"), 1).next_cursor, None);

    let hits = |found: Vec<quiver_db::SearchHit>| ids(found.into_iter().map(|h| h.id).collect());
    assert_eq!(hits(c.search(&[1.0, 1.0], 10).unwrap()), ["forever", "kept"]);
    let filter = Filter::parse(&json!({"kind": "doc"})).unwrap();
    let opts = SearchOptions {
        filter: Some(filter.clone()),
        ..Default::default()
    };
    assert_eq!(
        hits(c.search_with(&[1.0, 1.0], 10, &opts).unwrap()),
        ["forever", "kept"]
    );
    assert!(matches!(
        c.search_by_id("gone", 5, &SearchOptions::default()),
        Err(Error::PointNotFound(_))
    ));
    assert!(matches!(
        c.recommend(&["gone"], &[], 5, &SearchOptions::default()),
        Err(Error::PointNotFound(_))
    ));
    assert_eq!(c.count(Some(&filter)), 2);

    // Writes treat expired points as already gone.
    let selector = PointSelector::Ids(vec!["gone".into(), "kept".into()]);
    assert_eq!(
        c.update_metadata(&selector, json!({"seen": true}), MetadataMode::Merge)
            .unwrap(),
        1
    );
    assert_eq!(c.delete_where(&filter).unwrap(), 2);
    assert_eq!(c.info().tombstones, 2);
    // `gone` is still stored until purged, but deleting it removes nothing visible.
    assert_eq!(c.delete(&["gone"]).unwrap(), 0);
    assert_eq!(c.info().tombstones, 3);
    assert_eq!(c.purge_expired().unwrap(), 0);
}

#[test]
fn expiry_times_must_be_numbers() {
    let db = Database::in_memory();
    let c = db.create_collection("c", config()).unwrap();
    let bad = Point::new("a", vec![1.0, 1.0]).with_metadata(json!({"exp": "tomorrow"}));
    assert!(matches!(c.upsert([bad]), Err(Error::InvalidArgument(_))));
    c.upsert([point("a", 1.0, Some(FUTURE))]).unwrap();
    let selector = PointSelector::Ids(vec!["a".into()]);
    assert!(
        c.update_metadata(&selector, json!({"exp": "soon"}), MetadataMode::Merge)
            .is_err()
    );
    assert_eq!(
        c.get("a").unwrap().metadata,
        Some(json!({"kind": "doc", "exp": FUTURE}))
    );
    // Moving the expiry time into the past expires the point.
    c.update_metadata(&selector, json!({"exp": PAST}), MetadataMode::Merge)
        .unwrap();
    assert!(c.get("a").is_none());
    // The field stays indexed while expiry is on.
    assert!(c.drop_field_index("exp").is_err());
}

#[test]
fn default_ttl_stamps_points_written_without_an_expiry_time() {
    let db = Database::in_memory();
    let config = CollectionConfig::new(2).expiry(ExpiryConfig::new("meta.exp").default_ttl(3600));
    let c = db.create_collection("c", config).unwrap();
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    c.upsert([
        Point::new("bare", vec![1.0, 0.0]),
        Point::new("given", vec![0.0, 1.0]).with_metadata(json!({"meta": {"exp": FUTURE}})),
    ])
    .unwrap();
    let exp = |id: &str| {
        c.get(id).unwrap().metadata.unwrap()["meta"]["exp"]
            .as_u64()
            .unwrap()
    };
    assert!(
        (before + 3600..=before + 3602).contains(&exp("bare")),
        "{}",
        exp("bare")
    );
    assert_eq!(exp("given"), FUTURE);

    // Replacing the metadata restarts the clock; merging keeps the time.
    let selector = PointSelector::Ids(vec!["given".into()]);
    c.update_metadata(&selector, json!({"k": 1}), MetadataMode::Merge)
        .unwrap();
    assert_eq!(exp("given"), FUTURE);
    c.update_metadata(&selector, json!({"k": 2}), MetadataMode::Replace)
        .unwrap();
    assert!(exp("given") < FUTURE);
    assert!(
        c.upsert([Point::new("odd", vec![1.0, 1.0]).with_metadata(json!(["not", "an", "object"]))])
            .is_err()
    );
}

#[test]
fn expiry_can_be_turned_on_and_off() {
    let db = Database::in_memory();
    let c = db.create_collection("c", CollectionConfig::new(2)).unwrap();
    c.upsert([point("old", 1.0, Some(PAST)), point("new", 2.0, Some(FUTURE))])
        .unwrap();
    let odd = Point::new("odd", vec![3.0, 1.0]).with_metadata(json!({"exp": "never"}));
    c.upsert([odd]).unwrap();
    assert_eq!(c.len(), 3);

    let on = CollectionUpdate {
        expiry: Some(Some(ExpiryConfig::new("exp"))),
        ..Default::default()
    };
    assert!(c.update(&on).is_err(), "`odd` has no valid expiry time");
    c.delete(&["odd"]).unwrap();
    let info = c.update(&on).unwrap();
    assert_eq!(info.count, 1);
    assert_eq!(info.field_indexes, ["exp"]);
    assert!(c.get("old").is_none());

    let off = CollectionUpdate {
        expiry: Some(None),
        ..Default::default()
    };
    assert_eq!(c.update(&off).unwrap().count, 2, "unpurged points come back");
    assert!(c.drop_field_index("exp").unwrap());
}

#[test]
fn expiry_survives_restarts_and_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot = {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("c", config()).unwrap();
        c.upsert([point("gone", 1.0, Some(PAST)), point("kept", 2.0, Some(FUTURE))])
            .unwrap();
        c.create_snapshot().unwrap().manifest.id
    };
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.config(), config());
    assert_eq!(c.len(), 1);
    assert_eq!(c.info().field_indexes, ["exp"]);
    assert_eq!(c.purge_expired().unwrap(), 1);
    drop(c);
    drop(db);

    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!((c.len(), c.info().tombstones), (1, 1), "the purge was logged");
    let restored = db
        .restore_snapshot(
            &snapshot,
            RestoreOptions {
                name: Some("copy".into()),
                replace: false,
            },
        )
        .unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored.purge_expired().unwrap(), 1);
}

#[test]
fn field_index_limit_counts_the_expiry_field_before_creating() {
    let db = Database::in_memory();
    let fields: Vec<String> = (0..quiver_db::types::MAX_FIELD_INDEXES)
        .map(|i| format!("f{i}"))
        .collect();
    assert!(db.create_collection_with_fields("c", config(), &fields).is_err());
    assert!(
        db.collection("c").is_err(),
        "a failed create must not leave the collection behind"
    );
    db.create_collection_with_fields("c", config(), &fields[1..])
        .unwrap();
}
