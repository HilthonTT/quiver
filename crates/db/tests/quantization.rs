use quiver_db::{CollectionConfig, Database, IndexConfig, Metric, Point, Quantization, RestoreOptions};

fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn points(vs: &[Vec<f32>], first_id: usize) -> Vec<Point> {
    vs.iter()
        .enumerate()
        .map(|(i, v)| Point::new(format!("p{}", first_id + i), v.clone()))
        .collect()
}

#[test]
fn quantized_search_keeps_recall_high() {
    let dim = 64;
    let data = vectors(3000, dim, 1);
    for metric in [
        Metric::Cosine,
        Metric::Euclidean,
        Metric::DotProduct,
        Metric::Manhattan,
    ] {
        let db = Database::in_memory();
        let base = CollectionConfig::new(dim).metric(metric).index(IndexConfig::Flat);
        let exact = db.create_collection("exact", base.clone()).unwrap();
        let quant = db
            .create_collection("quant", base.quantization(Quantization::Scalar))
            .unwrap();
        exact.upsert(points(&data, 0)).unwrap();
        quant.upsert(points(&data, 0)).unwrap();

        let (mut found, mut total) = (0, 0);
        for q in vectors(50, dim, 2) {
            let want: Vec<String> = exact.search(&q, 10).unwrap().into_iter().map(|h| h.id).collect();
            let got = quant.search(&q, 10).unwrap();
            found += got.iter().filter(|h| want.contains(&h.id)).count();
            total += want.len();
        }
        let recall = found as f64 / total as f64;
        assert!(recall >= 0.9, "{metric:?}: recall {recall}");
    }
}

#[test]
fn quantized_vectors_survive_restarts_and_snapshots_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let data = vectors(200, 16, 3);
    let config = CollectionConfig::new(16)
        .metric(Metric::Euclidean)
        .quantization(Quantization::Scalar);
    let (before, snapshot) = {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("q", config.clone()).unwrap();
        c.upsert(points(&data[..100], 0)).unwrap();
        c.flush().unwrap(); // half from the checkpoint, half replayed from the log
        c.upsert(points(&data[100..], 100)).unwrap();
        let before: Vec<Point> = (0..200).map(|i| c.get(&format!("p{i}")).unwrap()).collect();
        // Decoded, so close to but not the same as what was written.
        assert_ne!(before[0].vector, data[0]);
        for (p, v) in before.iter().zip(&data) {
            for (a, b) in p.vector.iter().zip(v) {
                assert!((a - b).abs() < 2.0 / 255.0, "{a} vs {b}");
            }
        }
        (before, c.create_snapshot().unwrap().manifest.id)
    };

    for round in 0..3 {
        let db = Database::open(dir.path()).unwrap();
        let c = db.collection("q").unwrap();
        assert_eq!(c.config().quantization, Quantization::Scalar);
        for p in &before {
            assert_eq!(&c.get(&p.id).unwrap(), p, "round {round}");
        }
        c.compact().unwrap();
    }

    let db = Database::open(dir.path()).unwrap();
    let restored = db
        .restore_snapshot(
            &snapshot,
            RestoreOptions {
                name: Some("copy".into()),
                replace: false,
            },
        )
        .unwrap();
    assert_eq!(restored.config(), config);
    for p in &before {
        assert_eq!(&restored.get(&p.id).unwrap(), p);
    }
}

/// Recall@10 of `quant` against `exact` over 50 queries.
fn recall(exact: &quiver_db::Collection, quant: &quiver_db::Collection, dim: usize) -> f64 {
    let (mut found, mut total) = (0, 0);
    for q in vectors(50, dim, 2) {
        let want = exact.search(&q, 10).unwrap();
        let got = quant.search(&q, 10).unwrap();
        for h in &got {
            if let Some(w) = want.iter().find(|w| w.id == h.id) {
                found += 1;
                assert_eq!(w.distance, h.distance, "distances are exact");
            }
        }
        total += want.len();
    }
    found as f64 / total as f64
}

#[test]
fn binary_search_rescores_to_high_recall() {
    let dim = 256;
    let data = vectors(2000, dim, 1);
    // Every metric through the flat index; the graph walk with cosine.
    let cases = [
        (Metric::Cosine, IndexConfig::Flat),
        (Metric::Euclidean, IndexConfig::Flat),
        (Metric::DotProduct, IndexConfig::Flat),
        (Metric::Manhattan, IndexConfig::Flat),
        (Metric::Cosine, IndexConfig::default()),
    ];
    for (metric, index) in cases {
        let db = Database::in_memory();
        let base = CollectionConfig::new(dim).metric(metric).index(IndexConfig::Flat);
        let exact = db.create_collection("exact", base.clone()).unwrap();
        let quant = db
            .create_collection("quant", base.index(index).quantization(Quantization::Binary))
            .unwrap();
        exact.upsert(points(&data, 0)).unwrap();
        quant.upsert(points(&data, 0)).unwrap();
        assert_eq!(quant.get("p7").unwrap().vector, exact.get("p7").unwrap().vector);
        let recall = recall(&exact, &quant, dim);
        assert!(recall >= 0.95, "{metric:?} {}: recall {recall}", index.kind());
    }
}

#[test]
fn binary_codes_survive_restarts_settings_changes_and_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let dim = 128;
    let data = vectors(400, dim, 3);
    let config = CollectionConfig::new(dim)
        .metric(Metric::Cosine)
        .quantization(Quantization::Binary)
        .on_disk(true);
    let snapshot = {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("b", config.clone()).unwrap();
        c.upsert(points(&data[..200], 0)).unwrap();
        c.flush().unwrap(); // half from the checkpoint, half replayed from the log
        c.upsert(points(&data[200..], 200)).unwrap();
        // A bit per component plus a scale, nothing else in memory.
        assert_eq!(c.vector_bytes(), 400 * (dim / 8 + 4));
        c.create_snapshot().unwrap().manifest.id
    };

    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("b").unwrap();
    assert_eq!(c.config(), config);
    let exact = db
        .create_collection("exact", CollectionConfig::new(dim).index(IndexConfig::Flat))
        .unwrap();
    exact.upsert(points(&data, 0)).unwrap();
    let stored = |c: &quiver_db::Collection| c.get("p300").unwrap().vector;
    assert_eq!(stored(&c), stored(&exact), "vectors are kept exactly");
    assert!(recall(&exact, &c, dim) >= 0.95);

    // Into memory, with the exact vectors beside the codes...
    let update = quiver_db::CollectionUpdate {
        on_disk: Some(false),
        ..Default::default()
    };
    c.update(&update).unwrap();
    assert_eq!(c.vector_bytes(), 400 * (dim * 4 + dim / 8 + 4));
    assert_eq!(stored(&c), stored(&exact));
    assert!(recall(&exact, &c, dim) >= 0.95);
    // ...and off: the vectors were never rounded.
    let update = quiver_db::CollectionUpdate {
        quantization: Some(Quantization::None),
        ..Default::default()
    };
    c.update(&update).unwrap();
    assert_eq!(stored(&c), stored(&exact));
    assert!(recall(&exact, &c, dim) >= 0.99);

    let restored = db
        .restore_snapshot(
            &snapshot,
            RestoreOptions {
                name: Some("copy".into()),
                replace: false,
            },
        )
        .unwrap();
    assert_eq!(restored.config(), config);
    assert_eq!(stored(&restored), stored(&exact));
}

#[test]
fn unsupported_combinations_are_rejected() {
    let db = Database::in_memory();
    let q = CollectionConfig::new(3).quantization(Quantization::Scalar);
    assert!(
        db.create_collection("a", q.clone().metric(Metric::Hamming))
            .is_err()
    );
    assert!(
        db.create_collection(
            "b",
            q.metric(Metric::Euclidean)
                .index(IndexConfig::KdTree(Default::default()))
        )
        .is_err()
    );
    let b = CollectionConfig::new(3).quantization(Quantization::Binary);
    assert!(
        db.create_collection("c", b.clone().metric(Metric::Hamming))
            .is_err()
    );
    assert!(
        db.create_collection(
            "d",
            b.metric(Metric::Euclidean)
                .index(IndexConfig::KdTree(Default::default()))
        )
        .is_err()
    );
}
