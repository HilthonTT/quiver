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
}
