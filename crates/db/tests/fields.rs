//! Field indexes must never change a result, only how it is computed.

use quiver_db::{
    CollectionConfig, Database, Filter, IndexConfig, MetadataMode, Metric, Point, PointSelector,
    RestoreOptions, SearchOptions,
};
use serde_json::{Value, json};

const FIELDS: [&str; 4] = ["genre", "year", "tags", "stats.views"];

fn points(n: usize) -> Vec<Point> {
    let genres = ["scifi", "fantasy", "horror", "crime"];
    (0..n)
        .map(|i| {
            let mut md = json!({
                "genre": genres[i % 4],
                "year": 1900 + (i * 7) % 120,
                "tags": [format!("t{}", i % 5), format!("t{}", i % 3)],
                "stats": {"views": (i * 13) % 50},
            });
            // Some points lack fields or hold other types.
            match i % 9 {
                0 => drop(md.as_object_mut().unwrap().remove("genre")),
                1 => md["year"] = json!(format!("{}", 1900 + i)),
                2 => md["genre"] = json!(["scifi", "horror"]),
                3 => md["year"] = json!(1950.5),
                _ => {}
            }
            let v = vec![
                (i as f32 * 0.37).sin(),
                (i as f32 * 0.11).cos(),
                (i % 17) as f32 / 17.0,
            ];
            Point::new(format!("p{i}"), v).with_metadata(md)
        })
        .collect()
}

fn filters() -> Vec<Value> {
    vec![
        json!({"genre": "scifi"}),
        json!({"genre": {"$in": ["fantasy", "crime"]}}),
        json!({"year": {"$gte": 1950, "$lt": 1990}}),
        json!({"year": {"$gt": "1950"}}),
        json!({"year": 1950.5}),
        json!({"tags": "t2"}),
        json!({"stats.views": {"$lte": 5}}),
        json!({"genre": "horror", "stats.views": {"$gt": 20}}),
        json!({"$or": [{"genre": "crime"}, {"tags": "t4"}]}),
        json!({"$or": [{"genre": "crime"}, {"missing": 1}]}),
        json!({"genre": {"$ne": "scifi"}}),
        json!({"genre": {"$exists": false}}),
        json!({"genre": {"$exists": true}, "year": {"$lt": 1930}}),
        json!({"$not": {"tags": "t1"}}),
        json!({"$and": [{"genre": "fantasy"}, {"$not": {"year": {"$gt": 2000}}}]}),
    ]
}

fn pair(index: IndexConfig) -> (Database, Database) {
    let (plain, indexed) = (Database::in_memory(), Database::in_memory());
    let config = CollectionConfig::new(3).metric(Metric::Euclidean).index(index);
    plain
        .create_collection("c", config.clone())
        .unwrap()
        .upsert(points(3000))
        .unwrap();
    let fields: Vec<String> = FIELDS.iter().map(|f| f.to_string()).collect();
    indexed
        .create_collection_with_fields("c", config, &fields)
        .unwrap()
        .upsert(points(3000))
        .unwrap();
    (plain, indexed)
}

fn assert_same(plain: &Database, indexed: &Database, stage: &str) {
    let (a, b) = (plain.collection("c").unwrap(), indexed.collection("c").unwrap());
    for f in filters() {
        let filter = Filter::parse(&f).unwrap();
        assert_eq!(
            a.count(Some(&filter)),
            b.count(Some(&filter)),
            "{stage}: count {f}"
        );
        let opts = SearchOptions {
            filter: Some(filter),
            ..Default::default()
        };
        let q = [0.3, -0.2, 0.5];
        assert_eq!(
            a.search_with(&q, 25, &opts).unwrap(),
            b.search_with(&q, 25, &opts).unwrap(),
            "{stage}: search {f}"
        );
    }
}

#[test]
fn indexed_and_unindexed_collections_agree() {
    let (plain, indexed) = pair(IndexConfig::Flat);
    assert_eq!(indexed.collection("c").unwrap().info().field_indexes, FIELDS);
    assert_same(&plain, &indexed, "fresh");

    for db in [&plain, &indexed] {
        let c = db.collection("c").unwrap();
        c.update_metadata(
            &PointSelector::Filter(Filter::parse(&json!({"genre": "crime"})).unwrap()),
            json!({"genre": "scifi", "stats": {"views": 1}}),
            MetadataMode::Merge,
        )
        .unwrap();
        let f = Filter::parse(&json!({"year": {"$lt": 1920}})).unwrap();
        assert!(c.delete_where(&f).unwrap() > 0);
        c.upsert(points(200).into_iter().map(|mut p| {
            p.metadata = Some(json!({"genre": "fantasy", "year": 2024}));
            p
        }))
        .unwrap();
    }
    assert_same(&plain, &indexed, "after updates");

    for db in [&plain, &indexed] {
        db.collection("c").unwrap().compact().unwrap();
    }
    assert_same(&plain, &indexed, "after compaction");
}

#[test]
fn selective_filters_get_exact_results_from_hnsw_collections() {
    let (exact, _) = pair(IndexConfig::Flat);
    let (_, hnsw) = pair(IndexConfig::Hnsw(Default::default()));
    let opts = SearchOptions {
        filter: Some(Filter::parse(&json!({"genre": "horror", "stats.views": 7})).unwrap()),
        ..Default::default()
    };
    let q = [0.9, 0.1, 0.2];
    let want = exact.collection("c").unwrap().search_with(&q, 10, &opts).unwrap();
    assert!(!want.is_empty());
    assert_eq!(
        hnsw.collection("c").unwrap().search_with(&q, 10, &opts).unwrap(),
        want
    );
}

#[test]
fn field_indexes_persist_and_travel_with_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot_id = {
        let db = Database::open(dir.path()).unwrap();
        let c = db
            .create_collection("c", CollectionConfig::new(3).metric(Metric::Euclidean))
            .unwrap();
        c.upsert(points(100)).unwrap();
        c.create_field_index("genre").unwrap();
        c.create_field_index("genre").unwrap(); // no-op
        c.create_field_index("year").unwrap();
        assert!(c.drop_field_index("year").unwrap());
        assert!(!c.drop_field_index("year").unwrap());
        assert!(c.create_field_index("$bad").is_err());
        c.create_snapshot().unwrap().manifest.id
    };

    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.info().field_indexes, ["genre"]);
    let f = Filter::parse(&json!({"genre": "scifi"})).unwrap();
    let before = c.count(Some(&f));
    assert!(before > 0);

    let restored = db
        .restore_snapshot(
            &snapshot_id,
            RestoreOptions {
                name: Some("copy".into()),
                replace: false,
            },
        )
        .unwrap();
    assert_eq!(restored.info().field_indexes, ["genre"]);
    assert_eq!(restored.count(Some(&f)), before);
    drop((c, restored, db));

    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("copy").unwrap().info().field_indexes, ["genre"]);
}
