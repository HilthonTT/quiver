use quiver_db::{CollectionConfig, Database, Error, Filter, IndexConfig, Metric, Point, SearchOptions};
use serde_json::json;

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
    c.upsert([Point::new("a", vec![1.0, 1.0]), Point::new("a", vec![2.0, 2.0])])
        .unwrap();
    assert_eq!(c.len(), 1);
    assert_eq!(c.get("a").unwrap().vector, vec![2.0, 2.0]);
}

#[test]
fn invalid_batches_are_rejected_atomically() {
    let c = db_with(Metric::Cosine, IndexConfig::default());
    let err = c
        .upsert([Point::new("ok", vec![1.0, 0.0]), Point::new("bad", vec![1.0])])
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
    let opts = SearchOptions::default().filter(filter).include_vectors(true);
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
fn scroll_pages_through_points_in_id_order() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    // Inserted out of order: pages follow ids, not insertion.
    c.upsert([9, 3, 0, 7, 1, 8, 2, 6, 4, 5].map(|i| Point::new(i.to_string(), vec![i as f32, 0.0])))
        .unwrap();
    c.delete(&["0"]).unwrap();
    let page = c.scroll(Some("2"), 3);
    let ids: Vec<String> = page.points.into_iter().map(|p| p.id).collect();
    assert_eq!(ids, ["3", "4", "5"]);
    assert_eq!(page.next_cursor.as_deref(), Some("5"));

    let first = c.scroll(None, 2);
    assert_eq!(first.points[0].id, "1");
    // The last page hands out no cursor, even when it is exactly full.
    let last = c.scroll(Some("6"), 3);
    assert_eq!(last.points.len(), 3);
    assert_eq!(last.next_cursor, None);
    assert!(c.scroll(Some("9"), 3).points.is_empty());
    assert_eq!(c.scroll(None, 0).next_cursor, None, "limit 0 never loops");
}

/// Pages through `c` `limit` at a time, calling `between` with the page
/// number and the last id seen after each page.
fn scroll_all(c: &quiver_db::Collection, limit: usize, mut between: impl FnMut(usize, &str)) -> Vec<String> {
    let (mut seen, mut cursor, mut page_no) = (Vec::<String>::new(), None::<String>, 0);
    loop {
        let page = c.scroll(cursor.as_deref(), limit);
        seen.extend(page.points.into_iter().map(|p| p.id));
        between(page_no, seen.last().map_or("", String::as_str));
        page_no += 1;
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return seen,
        }
    }
}

#[test]
fn scroll_never_skips_or_repeats_points_changed_between_pages() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    let id = |i: usize| format!("p{i:03}");
    let ids: Vec<String> = (0..100).map(id).collect();
    c.upsert(ids.iter().map(|id| Point::new(id.clone(), vec![1.0, 0.0])))
        .unwrap();

    let mut deleted_ahead = Vec::new();
    let seen = scroll_all(&c, 10, |_, last| {
        let n: usize = last[1..].parse().unwrap();
        // Delete one point already seen and one still ahead, and update
        // (re-insert, which moves it to a new slot) one on each side.
        let (behind, ahead) = (id(n - 3), id(n + 4));
        c.delete(&[behind, ahead.clone()]).unwrap();
        deleted_ahead.push(ahead);
        let updated = [n - 1, n + 7].into_iter().filter(|&i| i < 100).map(id);
        c.upsert(updated.map(|id| Point::new(id, vec![0.0, 1.0])))
            .unwrap();
    });

    let expected: Vec<&String> = ids.iter().filter(|id| !deleted_ahead.contains(id)).collect();
    assert_eq!(
        seen.iter().collect::<Vec<_>>(),
        expected,
        "every surviving point exactly once"
    );
}

#[test]
fn scroll_cursor_survives_compaction() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    c.upsert((0..50).map(|i| Point::new(format!("p{i:02}"), vec![i as f32, 0.0])))
        .unwrap();
    let seen = scroll_all(&c, 7, |page, _| {
        if page == 1 {
            let gone: Vec<String> = (0..50).step_by(2).map(|i| format!("p{i:02}")).collect();
            c.delete(&gone).unwrap();
            c.compact().unwrap();
            assert_eq!(c.info().tombstones, 0);
        }
    });
    let mut expected: Vec<String> = (0..14).map(|i| format!("p{i:02}")).collect();
    expected.extend((14..50).filter(|i| i % 2 == 1).map(|i| format!("p{i:02}")));
    assert_eq!(seen, expected);
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
    assert!(db.create_collection("zero", CollectionConfig::new(0)).is_err());

    assert!(db.get_or_create_collection("a", CollectionConfig::new(2)).is_ok());
    assert!(
        db.get_or_create_collection("a", CollectionConfig::new(3))
            .is_err()
    );

    assert_eq!(db.list_collections().len(), 1);
    db.drop_collection("a").unwrap();
    assert!(matches!(db.collection("a"), Err(Error::CollectionNotFound(_))));
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
        let a: Vec<f32> = flat.search(&q, 10).unwrap().iter().map(|h| h.distance).collect();
        let b: Vec<f32> = kd.search(&q, 10).unwrap().iter().map(|h| h.distance).collect();
        assert_eq!(a, b);
    }
}

#[test]
fn vectors_whose_distances_would_overflow_are_rejected() {
    let c = db_with(Metric::Euclidean, IndexConfig::Flat);
    for v in [[3e19, 0.0], [-3e19, 0.0], [2e19, 0.0]] {
        let err = c.upsert([Point::new("p", v.to_vec())]).unwrap_err();
        assert!(err.to_string().contains("vector magnitude too large"), "{err}");
        let err = c.search(&v, 1).unwrap_err();
        assert!(err.to_string().contains("vector magnitude too large"), "{err}");
    }
    assert!(c.is_empty());
    c.upsert([Point::new("ok", vec![1e18, -1e18])]).unwrap();
    assert_eq!(
        c.search(&[-1e18, 1e18], 1).unwrap()[0].distance,
        2.0 * 2f32.sqrt() * 1e18
    );

    // Cosine normalises first, so large components are fine there.
    let cosine = db_with(Metric::Cosine, IndexConfig::Flat);
    cosine.upsert([Point::new("big", vec![2e19, 0.0])]).unwrap();
    assert_eq!(cosine.search(&[3e19, 0.0], 1).unwrap()[0].id, "big");
}
