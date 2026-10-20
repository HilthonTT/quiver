use serde_json::json;
use quiver_db::{
    CollectionConfig, Database, Error, Filter, IndexConfig, Metric, Point, SearchOptions,
};

fn db_with(metric: Metric, index: IndexConfig) -> std::sync::Arc<quiver_db::Collection> {
    Database::in_memory()
        .create_collection("c", CollectionConfig::new(2).metric(metric).index(index))
        .unwrap()
}

#[test]
fn upsert_replaces_existing_ids() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    c.upsert([Point::new("a", vec![0.0, 0.0])]).unwrap();
    c.upsert([Point::new("a", vec![5.0, 5.0]).with_metadata(json!({"v": 2}))])
        .unwrap();
    assert_eq!(c.len(), 1);
    let a = c.get("a").unwrap();
    assert_eq!(a.vector, vec![5.0, 5.0]);
    assert_eq!(a.metadata, Some(json!({"v": 2})));
    assert_eq!(c.search(&[5.0, 5.0], 10).unwrap().len(), 1);
}

#[test]
fn later_duplicates_in_a_batch_win() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    c.upsert([
        Point::new("a", vec![1.0, 1.0]),
        Point::new("a", vec![2.0, 2.0]),
    ])
    .unwrap();
    assert_eq!(c.len(), 1);
    assert_eq!(c.get("a").unwrap().vector, vec![2.0, 2.0]);
}

#[test]
fn invalid_batches_are_rejected_atomically() {
    let c = db_with(Metric::Cosine, IndexConfig::default());
    let err = c
        .upsert([
            Point::new("ok", vec![1.0, 0.0]),
            Point::new("bad", vec![1.0]),
        ])
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    assert!(c.is_empty());

    assert!(
        c.upsert([Point::new("z", vec![0.0, 0.0])]).is_err(),
        "zero vector under cosine"
    );
    assert!(c.upsert([Point::new("n", vec![f32::NAN, 1.0])]).is_err());
    assert!(c.upsert([Point::new("", vec![1.0, 1.0])]).is_err());
    assert!(matches!(
        c.search(&[1.0, 2.0, 3.0], 1),
        Err(Error::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));
}

#[test]
fn results_are_ordered_by_distance_for_every_metric_and_index() {
    for metric in [
        Metric::Cosine,
        Metric::Euclidean,
        Metric::DotProduct,
        Metric::Manhattan,
    ] {
        for index in [
            IndexConfig::Flat,
            IndexConfig::default(),
            IndexConfig::KdTree(Default::default()),
        ] {
            if metric == Metric::DotProduct && matches!(index, IndexConfig::KdTree(_)) {
                continue; // rejected by config validation
            }
            let c = db_with(metric, index);
            c.upsert([
                Point::new("east", vec![1.0, 0.0]),
                Point::new("north", vec![0.0, 1.0]),
                Point::new("northeast", vec![0.7, 0.7]),
            ])
            .unwrap();
            let hits = c.search(&[1.0, 0.1], 3).unwrap();
            assert_eq!(hits[0].id, "east", "{metric:?} {index:?}");
            assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
        }
    }
}

#[test]
fn deleted_points_never_appear_in_results() {
    let c = db_with(Metric::Euclidean, IndexConfig::default());
    c.upsert((0..50).map(|i| Point::new(i.to_string(), vec![i as f32, 0.0])))
        .unwrap();
    assert_eq!(c.delete(&["0", "1", "missing"]).unwrap(), 2);
    let hits = c.search(&[0.0, 0.0], 5).unwrap();
    assert_eq!(hits[0].id, "2");
    assert!(hits.iter().all(|h| h.id != "0" && h.id != "1"));
}

#[test]
fn filtered_search() {
    let c = db_with(Metric::Euclidean, IndexConfig::default());
    c.upsert((0..200).map(|i| {
        Point::new(i.to_string(), vec![i as f32, 0.0])
            .with_metadata(json!({"parity": if i % 2 == 0 { "even" } else { "odd" }, "i": i}))
    }))
    .unwrap();

    let filter = Filter::parse(&json!({"parity": "odd", "i": {"$gte": 100}})).unwrap();
    let opts = SearchOptions::default()
        .filter(filter)
        .include_vectors(true);
    let hits = c.search_with(&[0.0, 0.0], 3, &opts).unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, ["101", "103", "105"]);
    assert_eq!(hits[0].vector.as_deref(), Some(&[101.0, 0.0][..]));
    assert_eq!(hits[0].metadata.as_ref().unwrap()["parity"], "odd");
}

#[test]
fn search_options_can_omit_metadata() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    c.upsert([Point::new("a", vec![0.0, 0.0]).with_metadata(json!({"x": 1}))])
        .unwrap();
    let opts = SearchOptions::default().include_metadata(false);
    assert!(
        c.search_with(&[0.0, 0.0], 1, &opts).unwrap()[0]
            .metadata
            .is_none()
    );
}

#[test]
fn scroll_pages_through_points() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    c.upsert((0..10).map(|i| Point::new(i.to_string(), vec![i as f32, 0.0])))
        .unwrap();
    c.delete(&["0"]).unwrap();
    let page: Vec<String> = c.scroll(2, 3).into_iter().map(|p| p.id).collect();
    assert_eq!(page, ["3", "4", "5"]);
}

#[test]
fn collection_management() {
    let db = Database::in_memory();
    db.create_collection("a", CollectionConfig::new(2)).unwrap();
    assert!(matches!(
        db.create_collection("a", CollectionConfig::new(2)),
        Err(Error::CollectionExists(_))
    ));
    assert!(
        db.create_collection("bad name", CollectionConfig::new(2))
            .is_err()
    );
    assert!(
        db.create_collection("zero", CollectionConfig::new(0))
            .is_err()
    );

    assert!(
        db.get_or_create_collection("a", CollectionConfig::new(2))
            .is_ok()
    );
    assert!(
        db.get_or_create_collection("a", CollectionConfig::new(3))
            .is_err()
    );

    assert_eq!(db.list_collections().len(), 1);
    db.drop_collection("a").unwrap();
    assert!(matches!(
        db.collection("a"),
        Err(Error::CollectionNotFound(_))
    ));
}

#[test]
fn concurrent_readers_and_writers() {
    let c = db_with(Metric::Euclidean, IndexConfig::default());
    std::thread::scope(|s| {
        for t in 0..4 {
            let c = &c;
            s.spawn(move || {
                for i in 0..250 {
                    c.upsert([Point::new(format!("{t}-{i}"), vec![t as f32, i as f32])])
                        .unwrap();
                    c.search(&[0.0, 0.0], 5).unwrap();
                }
            });
        }
    });
    assert_eq!(c.len(), 1_000);
}

#[test]
fn insert_generates_ids() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    let id = c.insert(vec![1.0, 2.0], Some(json!({"k": 1}))).unwrap();
    assert_eq!(id.len(), 36);
    assert_eq!(c.get(&id).unwrap().metadata, Some(json!({"k": 1})));
}

#[test]
fn hamming_counts_differing_components() {
    let c = Database::in_memory()
        .create_collection("bits", CollectionConfig::new(4).metric(Metric::Hamming))
        .unwrap();
    c.upsert([
        Point::new("a", vec![1.0, 0.0, 1.0, 0.0]),
        Point::new("b", vec![1.0, 1.0, 1.0, 1.0]),
    ])
    .unwrap();
    let hits = c.search(&[1.0, 0.0, 1.0, 1.0], 2).unwrap();
    assert_eq!((hits[0].id.as_str(), hits[0].distance), ("a", 1.0));
    assert_eq!((hits[1].id.as_str(), hits[1].distance), ("b", 1.0));
}

#[test]
fn kd_tree_is_exact() {
    let db = Database::in_memory();
    let config = CollectionConfig::new(3).metric(Metric::Euclidean);
    let flat = db
        .create_collection("flat", config.clone().index(IndexConfig::Flat))
        .unwrap();
    let kd = db
        .create_collection("kd", config.index(IndexConfig::KdTree(Default::default())))
        .unwrap();
    let points: Vec<Point> = (0..2_000)
        .map(|i| {
            let f = i as f32;
            Point::new(
                i.to_string(),
                vec![(f * 0.37).sin(), (f * 0.11).cos(), (f * 0.07).sin()],
            )
        })
        .collect();
    flat.upsert(points.clone()).unwrap();
    kd.upsert(points).unwrap();
    kd.delete(&["5", "6"]).unwrap();
    flat.delete(&["5", "6"]).unwrap();
    for q in [[0.0, 0.0, 0.0], [0.5, -0.5, 0.2], [1.0, 1.0, 1.0]] {
        let a: Vec<f32> = flat
            .search(&q, 10)
            .unwrap()
            .iter()
            .map(|h| h.distance)
            .collect();
        let b: Vec<f32> = kd
            .search(&q, 10)
            .unwrap()
            .iter()
            .map(|h| h.distance)
            .collect();
        assert_eq!(a, b);
    }
}
