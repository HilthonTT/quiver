use quiver_db::{
    CollectionConfig, CollectionUpdate, Database, DbOptions, Error, Filter, HybridQuery, IndexConfig, Metric,
    Point, Prefetch, Quantization, QueryVector, RestoreOptions, SearchOptions, SparseModifier, SparseVector,
    SparseVectorConfig, StorageKind, VectorConfig,
};
use serde_json::json;

/// Deterministic pseudo-random vector, unique per `(i, salt)`.
fn vec_for(i: usize, dim: usize, salt: u64) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let mut z = (i as u64) << 20 | (d as u64) << 4 | salt;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn sparse(pairs: &[(u32, f32)]) -> SparseVector {
    SparseVector::new(
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}

/// A main vector of 4, a `title` vector of 3 and a `text` sparse vector.
fn config() -> CollectionConfig {
    CollectionConfig::new(4)
        .metric(Metric::Euclidean)
        .with_vector(
            "title",
            VectorConfig::new(3)
                .metric(Metric::Euclidean)
                .index(IndexConfig::Flat),
        )
        .with_sparse_vector("text", SparseVectorConfig::default())
}

/// Point `i` has the term `i % 7` and, for even `i`, the term 100.
fn point(i: usize) -> Point {
    let mut terms = vec![(i as u32 % 7, 1.0 + i as f32 / 100.0)];
    if i.is_multiple_of(2) {
        terms.push((100, 0.5));
    }
    Point::new(format!("p{i}"), vec_for(i, 4, 1))
        .with_vector("title", vec_for(i, 3, 2))
        .with_sparse_vector("text", sparse(&terms))
        .with_metadata(json!({ "i": i }))
}

#[test]
fn named_vectors_are_searched_separately() {
    let db = Database::in_memory();
    let c = db.create_collection("docs", config()).unwrap();
    c.upsert((0..200).map(point)).unwrap();

    let by_main = c
        .search_with(&vec_for(5, 4, 1), 1, &SearchOptions::default())
        .unwrap();
    assert_eq!(by_main[0].id, "p5");
    let by_title = c
        .search_with(&vec_for(9, 3, 2), 1, &SearchOptions::default().using("title"))
        .unwrap();
    assert_eq!(by_title[0].id, "p9");
    assert_eq!(by_title[0].distance, 0.0);

    // Each vector space has its own dimension.
    assert!(matches!(
        c.search_with(&vec_for(9, 3, 2), 1, &SearchOptions::default()),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(
        c.search_with(&[0.0; 3], 1, &SearchOptions::default().using("nope"))
            .is_err()
    );
    assert!(
        c.search_with(&[0.0; 3], 1, &SearchOptions::default().using("text"))
            .is_err()
    );

    // By id and recommendations use the named vector too.
    let near = c
        .search_by_id("p9", 3, &SearchOptions::default().using("title"))
        .unwrap();
    assert!(near.iter().all(|h| h.id != "p9"));
    let rec = c
        .recommend(&["p9"], &[], 1, &SearchOptions::default().using("title"))
        .unwrap();
    assert_eq!(rec[0].id, near[0].id);

    // Vectors come back with the point and with hits that ask for them.
    let p = c.get("p3").unwrap();
    assert_eq!(p, {
        let mut want = point(3);
        want.sparse_vectors.get_mut("text").unwrap().normalize().unwrap();
        want
    });
    let hit = &c
        .search_with(
            &vec_for(3, 4, 1),
            1,
            &SearchOptions::default().include_vectors(true),
        )
        .unwrap()[0];
    assert_eq!(hit.vectors["title"], vec_for(3, 3, 2));
    assert_eq!(hit.sparse_vectors["text"], p.sparse_vectors["text"]);
}

#[test]
fn points_must_match_the_vectors_configured() {
    let db = Database::in_memory();
    let c = db.create_collection("docs", config()).unwrap();
    let bad = [
        Point::new("a", vec![0.0; 4]),
        point(1).with_vector("title", vec![0.0; 2]),
        point(1).with_vector("other", vec![0.0; 3]),
        point(1).with_sparse_vector("other", sparse(&[(1, 1.0)])),
        point(1).with_sparse_vector("text", sparse(&[(1, 1.0), (1, 2.0)])),
        point(1).with_sparse_vector("text", SparseVector::new(vec![1], vec![])),
    ];
    for p in bad {
        let err = c.upsert([p.clone()]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{p:?}: {err}");
    }
    assert!(c.is_empty());

    // Sparse vectors are optional, and come back sorted.
    let mut p = point(2);
    p.sparse_vectors.clear();
    c.upsert([
        p,
        point(3).with_sparse_vector("text", sparse(&[(9, 1.0), (2, 3.0)])),
    ])
    .unwrap();
    assert!(c.get("p2").unwrap().sparse_vectors.is_empty());
    assert_eq!(
        c.get("p3").unwrap().sparse_vectors["text"],
        sparse(&[(2, 3.0), (9, 1.0)])
    );
}

#[test]
fn sparse_search_scores_dot_products() {
    let db = Database::in_memory();
    let c = db.create_collection("docs", config()).unwrap();
    c.upsert((0..50).map(point)).unwrap();

    let q = sparse(&[(3, 1.0)]);
    let hits = c
        .search_sparse("text", &q, 100, &SearchOptions::default())
        .unwrap();
    // Exactly the points with term 3, best (largest weight) first.
    let want: Vec<String> = (0..50)
        .filter(|i| i % 7 == 3)
        .rev()
        .map(|i| format!("p{i}"))
        .collect();
    assert_eq!(hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>(), want);
    let top = c.get(&hits[0].id).unwrap();
    assert_eq!(-hits[0].distance, top.sparse_vectors["text"].dot(&q));

    // Filters and deletes apply.
    let filter = Filter::parse(&json!({"i": {"$lt": 20}})).unwrap();
    let filtered = c
        .search_sparse("text", &q, 100, &SearchOptions::default().filter(filter))
        .unwrap();
    assert_eq!(filtered.len(), 3);
    c.delete(&["p45"]).unwrap();
    let after = c.search_sparse("text", &q, 1, &SearchOptions::default()).unwrap();
    assert_eq!(after[0].id, "p38");
    // Replacing a point drops its old terms.
    c.upsert([point(38).with_sparse_vector("text", sparse(&[(1, 1.0)]))])
        .unwrap();
    let after = c.search_sparse("text", &q, 1, &SearchOptions::default()).unwrap();
    assert_eq!(after[0].id, "p31");

    assert!(
        c.search_sparse("title", &q, 1, &SearchOptions::default())
            .is_err()
    );
    assert!(
        c.search_sparse("text", &sparse(&[(1, f32::NAN)]), 1, &SearchOptions::default())
            .is_err()
    );
}

#[test]
fn idf_weights_rare_terms_up() {
    let db = Database::in_memory();
    let idf = SparseVectorConfig {
        modifier: SparseModifier::Idf,
    };
    let make = |name: &str, config: SparseVectorConfig| {
        let c = db
            .create_collection(name, CollectionConfig::new(1).with_sparse_vector("text", config))
            .unwrap();
        // Term 1 is in every point, term 2 in one.
        let mut points: Vec<Point> = (0..20)
            .map(|i| {
                Point::new(format!("filler{i}"), vec![1.0]).with_sparse_vector("text", sparse(&[(1, 1.0)]))
            })
            .collect();
        points.push(Point::new("common", vec![1.0]).with_sparse_vector("text", sparse(&[(1, 2.0)])));
        points.push(Point::new("rare", vec![1.0]).with_sparse_vector("text", sparse(&[(2, 1.0)])));
        c.upsert(points).unwrap();
        c
    };
    let q = sparse(&[(1, 1.0), (2, 1.0)]);
    let plain = make("plain", SparseVectorConfig::default());
    let top = plain
        .search_sparse("text", &q, 1, &SearchOptions::default())
        .unwrap();
    assert_eq!(top[0].id, "common");
    let weighted = make("idf", idf);
    let top = weighted
        .search_sparse("text", &q, 1, &SearchOptions::default())
        .unwrap();
    assert_eq!(top[0].id, "rare");
}

#[test]
fn hybrid_queries_fuse_rankings() {
    let db = Database::in_memory();
    let c = db.create_collection("docs", config()).unwrap();
    c.upsert((0..100).map(point)).unwrap();

    let dense = Prefetch::new(
        QueryVector::Dense {
            vector: vec_for(10, 4, 1),
            using: None,
        },
        20,
    );
    let keyword = Prefetch::new(
        QueryVector::Sparse {
            vector: sparse(&[(10 % 7, 1.0)]),
            using: "text".into(),
        },
        20,
    );
    let hits = c
        .query(&HybridQuery::new(vec![dense.clone(), keyword.clone()]), 5)
        .unwrap();
    // p10 ranks first by vector and well by keyword.
    assert_eq!(hits[0].id, "p10");
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(hits[0].score <= 2.0 / 61.0 + 1e-6);

    // Weights steer the fusion; zero ignores a search.
    let mut only_keyword = HybridQuery::new(vec![
        Prefetch {
            weight: 0.0,
            ..dense.clone()
        },
        keyword.clone(),
    ]);
    only_keyword.include_vectors = true;
    let fused = c.query(&only_keyword, 3).unwrap();
    let ranked = c
        .search_sparse("text", &sparse(&[(3, 1.0)]), 3, &SearchOptions::default())
        .unwrap();
    assert_eq!(
        fused.iter().map(|h| &h.id).collect::<Vec<_>>(),
        ranked.iter().map(|h| &h.id).collect::<Vec<_>>()
    );
    assert!(fused[0].vector.is_some() && !fused[0].vectors.is_empty());

    // The filter applies to every search.
    let mut filtered = HybridQuery::new(vec![dense, keyword]);
    filtered.filter = Some(Filter::parse(&json!({"i": {"$gte": 50}})).unwrap());
    let hits = c.query(&filtered, 10).unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits.iter()
            .all(|h| h.metadata.as_ref().unwrap()["i"].as_u64().unwrap() >= 50)
    );

    assert!(c.query(&HybridQuery::new(vec![]), 5).is_err());
}

#[test]
fn named_and_sparse_vectors_persist_and_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", config()).unwrap();
        c.upsert((0..100).map(point)).unwrap();
        db.flush().unwrap();
        c.upsert((100..120).map(point)).unwrap();
        c.delete(&["p7"]).unwrap();
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("docs").unwrap();
    assert_eq!(c.len(), 119);
    let mut want = point(110);
    want.sparse_vectors.get_mut("text").unwrap().normalize().unwrap();
    assert_eq!(c.get("p110").unwrap(), want);
    let hits = c
        .search_sparse("text", &sparse(&[(0, 1.0)]), 1, &SearchOptions::default())
        .unwrap();
    assert_eq!(hits[0].id, "p119");

    let snap = c.create_snapshot().unwrap();
    let restored = db
        .restore_snapshot(
            &snap.manifest.id,
            RestoreOptions {
                name: Some("copy".into()),
                replace: false,
            },
        )
        .unwrap();
    assert_eq!(restored.config(), config());
    assert_eq!(restored.get("p110").unwrap(), want);
    let title = restored
        .search_with(&vec_for(50, 3, 2), 1, &SearchOptions::default().using("title"))
        .unwrap();
    assert_eq!(title[0].id, "p50");
}

fn on_disk_config() -> CollectionConfig {
    let mut c = config().on_disk(true);
    c.vectors.get_mut("title").unwrap().on_disk = true;
    c
}

fn vector_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir.join("collections/docs"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("vectors-")
        })
        .count()
}

#[test]
fn on_disk_vectors_take_no_memory_and_survive_restarts() {
    for storage in [StorageKind::File, StorageKind::RocksDb] {
        if !storage.is_available() {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let open = || {
            Database::open_with(
                dir.path(),
                DbOptions {
                    storage,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        {
            let db = open();
            let c = db.create_collection("docs", on_disk_config()).unwrap();
            c.upsert((0..300).map(point)).unwrap();
            assert_eq!(c.vector_bytes(), 0, "{storage}");
            db.flush().unwrap();
            c.upsert((300..320).map(point)).unwrap();
            c.delete(&["p1"]).unwrap();
        }
        let db = open();
        let c = db.collection("docs").unwrap();
        assert_eq!(c.len(), 319);
        assert_eq!(c.vector_bytes(), 0);
        assert_eq!(vector_files(dir.path()), 2, "{storage}: stale files are removed");
        let hits = c.search(&vec_for(310, 4, 1), 1).unwrap();
        assert_eq!(hits[0].id, "p310");
        let hits = c
            .search_with(&vec_for(42, 3, 2), 1, &SearchOptions::default().using("title"))
            .unwrap();
        assert_eq!(hits[0].id, "p42");

        // Compaction writes new files and drops the old ones.
        c.delete(&(0..250).map(|i| format!("p{i}")).collect::<Vec<_>>())
            .unwrap();
        c.compact().unwrap();
        assert_eq!(vector_files(dir.path()), 2);
        assert_eq!(c.get("p300").unwrap().vector, vec_for(300, 4, 1));
        drop(c);
        drop(db);
        let db = open();
        assert_eq!(db.collection("docs").unwrap().len(), 70);
    }
}

#[test]
fn on_disk_vectors_need_a_durable_engine() {
    let db = Database::in_memory();
    assert!(matches!(
        db.create_collection("docs", on_disk_config()),
        Err(Error::InvalidArgument(_))
    ));
    let c = db.create_collection("docs", config()).unwrap();
    let update = CollectionUpdate {
        on_disk: Some(true),
        ..Default::default()
    };
    assert!(c.update(&update).is_err());
}

#[test]
fn quantized_codes_with_exact_vectors_on_disk_rescore() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let dim = 32;
    let base = CollectionConfig::new(dim)
        .metric(Metric::Euclidean)
        .index(IndexConfig::Flat);
    let exact = db.create_collection("exact", base.clone()).unwrap();
    let disk = db
        .create_collection("disk", base.quantization(Quantization::Scalar).on_disk(true))
        .unwrap();
    let points: Vec<Point> = (0..2_000)
        .map(|i| Point::new(format!("p{i}"), vec_for(i, dim, 3)))
        .collect();
    exact.upsert(points.clone()).unwrap();
    disk.upsert(points).unwrap();
    assert_eq!(disk.vector_bytes(), 2_000 * (dim + 8));

    for q in 0..20 {
        let query = vec_for(10_000 + q, dim, 4);
        let want = exact.search(&query, 10).unwrap();
        let got = disk.search(&query, 10).unwrap();
        let common = got.iter().filter(|h| want.iter().any(|w| w.id == h.id)).count();
        assert!(common >= 9, "recall {common}/10");
        // Distances are exact, from the vectors on disk.
        for h in &got {
            if let Some(w) = want.iter().find(|w| w.id == h.id) {
                assert_eq!(w.distance, h.distance);
            }
        }
    }
    assert_eq!(
        disk.get("p5").unwrap().vector,
        vec_for(5, dim, 3),
        "returned vectors are exact"
    );
}

#[test]
fn on_disk_collections_snapshot_and_restore() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let mut config = on_disk_config();
    config.quantization = Quantization::Scalar;
    let c = db.create_collection("docs", config.clone()).unwrap();
    c.upsert((0..100).map(point)).unwrap();
    let snap = c.create_snapshot().unwrap();
    drop(c);
    db.drop_collection("docs").unwrap();
    let restored = db
        .restore_snapshot(&snap.manifest.id, RestoreOptions::default())
        .unwrap();
    assert_eq!(restored.config(), config);
    assert_eq!(
        restored.get("p4").unwrap().vector,
        vec_for(4, 4, 1),
        "exact after the round trip"
    );
    assert_eq!(restored.vector_bytes(), 100 * (4 + 8));
    assert_eq!(vector_files(dir.path()), 2);
}

#[test]
fn updates_rebuild_indexes_and_move_vectors() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("docs", config()).unwrap();
        c.upsert((0..300).map(point)).unwrap();
        let before = c.search(&vec_for(7, 4, 1), 5).unwrap();

        let update: CollectionUpdate = serde_json::from_value(json!({
            "index": {"type": "flat"},
            "on_disk": true,
            "vectors": {"title": {"index": {"type": "hnsw", "m": 8}, "quantization": "scalar"}},
            "sparse_vectors": {"text": {"modifier": "idf"}},
        }))
        .unwrap();
        let info = c.update(&update).unwrap();
        assert_eq!(info.config.index, IndexConfig::Flat);
        assert!(info.config.on_disk);
        assert_eq!(info.config.sparse_vectors["text"].modifier, SparseModifier::Idf);
        assert_eq!(c.search(&vec_for(7, 4, 1), 5).unwrap(), before, "flat is exact");
        assert_eq!(vector_files(dir.path()), 1);
        let title = c
            .search_with(&vec_for(7, 3, 2), 1, &SearchOptions::default().using("title"))
            .unwrap();
        assert_eq!(title[0].id, "p7");
        c.upsert((300..310).map(point)).unwrap();
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("docs").unwrap();
    assert!(c.config().on_disk);
    assert_eq!(c.config().vectors["title"].quantization, Quantization::Scalar);
    assert_eq!(c.len(), 310);
    assert_eq!(c.get("p305").unwrap().vector, vec_for(305, 4, 1));

    // ...and back into memory.
    let back = CollectionUpdate {
        on_disk: Some(false),
        ..Default::default()
    };
    c.update(&back).unwrap();
    assert_eq!(vector_files(dir.path()), 0);
    assert_eq!(c.get("p305").unwrap().vector, vec_for(305, 4, 1));

    // Dimensions, metrics and the set of vectors cannot change.
    for bad in [
        json!({"dimension": 8}),
        json!({"metric": "cosine"}),
        json!({"vectors": {"body": {"on_disk": true}}}),
        json!({"index": {"type": "kd_tree"}, "quantization": "scalar"}),
    ] {
        let parsed = serde_json::from_value::<CollectionUpdate>(bad.clone());
        assert!(parsed.map_or(true, |u| c.update(&u).is_err()), "{bad}");
    }
}

#[test]
fn ef_search_changes_keep_the_graph() {
    let db = Database::in_memory();
    let c = db.create_collection("docs", CollectionConfig::new(4)).unwrap();
    c.upsert((0..500).map(|i| Point::new(format!("p{i}"), vec_for(i, 4, 1))))
        .unwrap();
    let before = c.search(&vec_for(3, 4, 1), 10).unwrap();
    let update: CollectionUpdate = serde_json::from_value(json!({"index": {"ef_search": 200}})).unwrap();
    let info = c.update(&update).unwrap();
    let IndexConfig::Hnsw(p) = info.config.index else {
        panic!()
    };
    assert_eq!(p.ef_search, 200);
    let after = c.search(&vec_for(3, 4, 1), 10).unwrap();
    assert_eq!(after[0], before[0]);
}
