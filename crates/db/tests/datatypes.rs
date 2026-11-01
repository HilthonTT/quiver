use quiver_db::{
    CollectionConfig, CollectionUpdate, Database, Datatype, Error, IndexConfig, Metric, Point, Quantization,
    RestoreOptions, VectorConfig,
};

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

/// Whole numbers from 0 to 255.
fn byte_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    vectors(n, dim, seed)
        .into_iter()
        .map(|v| v.into_iter().map(|x| ((x + 1.0) * 127.5).floor()).collect())
        .collect()
}

fn points(vs: &[Vec<f32>]) -> Vec<Point> {
    vs.iter()
        .enumerate()
        .map(|(i, v)| Point::new(format!("p{i}"), v.clone()))
        .collect()
}

fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn ids(hits: &[quiver_db::SearchHit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

#[test]
fn float16_rounds_vectors_and_halves_the_memory() {
    let dim = 64;
    let data = vectors(500, dim, 1);
    for metric in [
        Metric::Cosine,
        Metric::Euclidean,
        Metric::DotProduct,
        Metric::Manhattan,
    ] {
        let db = Database::in_memory();
        let base = CollectionConfig::new(dim).metric(metric).index(IndexConfig::Flat);
        let exact = db.create_collection("exact", base.clone()).unwrap();
        let halves = db
            .create_collection("halves", base.datatype(Datatype::Float16))
            .unwrap();
        exact.upsert(points(&data)).unwrap();
        halves.upsert(points(&data)).unwrap();
        assert_eq!(halves.vector_bytes() * 2, exact.vector_bytes(), "{metric:?}");

        // Stored vectors are the rounded ones (normalised first for cosine).
        let stored = halves.get("p3").unwrap().vector;
        let want: Vec<f32> = exact.get("p3").unwrap().vector.into_iter().map(half).collect();
        assert_eq!(stored, want, "{metric:?}");

        // Distances are the exact distances to the rounded vectors.
        let q = &vectors(1, dim, 9)[0];
        let hits = halves.search(q, 5).unwrap();
        let mut query = q.clone();
        metric.prepare(&mut query);
        for h in &hits {
            let v = halves.get(&h.id).unwrap().vector;
            let want = metric.distance(&query, &v);
            assert!(
                (h.distance - want).abs() < 1e-4,
                "{metric:?}: {} vs {want}",
                h.distance
            );
        }
        // And rank nearly as the exact vectors do.
        let overlap = ids(&hits)
            .iter()
            .filter(|id| ids(&exact.search(q, 5).unwrap()).contains(id))
            .count();
        assert!(overlap >= 4, "{metric:?}: {overlap}");
    }
}

#[test]
fn uint8_stores_bytes_exactly() {
    let dim = 48;
    let data = byte_vectors(300, dim, 2);
    for metric in [
        Metric::Euclidean,
        Metric::DotProduct,
        Metric::Manhattan,
        Metric::Hamming,
    ] {
        let db = Database::in_memory();
        let base = CollectionConfig::new(dim).metric(metric).index(IndexConfig::Flat);
        let exact = db.create_collection("exact", base.clone()).unwrap();
        let bytes = db
            .create_collection("bytes", base.datatype(Datatype::Uint8))
            .unwrap();
        exact.upsert(points(&data)).unwrap();
        bytes.upsert(points(&data)).unwrap();
        assert_eq!(bytes.vector_bytes() * 4, exact.vector_bytes(), "{metric:?}");
        assert_eq!(bytes.get("p7").unwrap().vector, data[7]);

        // The same vectors, so the same results, whatever the query.
        for q in vectors(5, dim, 3) {
            let q: Vec<f32> = q.into_iter().map(|x| x * 300.0).collect();
            let want = exact.search(&q, 10).unwrap();
            let got = bytes.search(&q, 10).unwrap();
            for (a, b) in want.iter().zip(&got) {
                assert!(
                    (a.distance - b.distance).abs() <= 1e-3 * a.distance.abs().max(1.0),
                    "{metric:?}"
                );
            }
        }
    }
}

#[test]
fn values_a_datatype_cannot_hold_are_refused() {
    let db = Database::in_memory();
    let bytes = db
        .create_collection(
            "bytes",
            CollectionConfig::new(2)
                .metric(Metric::Euclidean)
                .datatype(Datatype::Uint8),
        )
        .unwrap();
    for bad in [[1.5, 2.0], [-1.0, 0.0], [256.0, 0.0]] {
        let err = bytes.upsert([Point::new("a", bad.to_vec())]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{bad:?}: {err}");
    }
    bytes.upsert([Point::new("a", vec![0.0, 255.0])]).unwrap();
    // Queries are floats, not bytes.
    let hits = bytes.search(&[0.5, -3.0], 1).unwrap();
    assert_eq!(hits[0].id, "a");

    let halves = db
        .create_collection(
            "halves",
            CollectionConfig::new(2)
                .metric(Metric::Euclidean)
                .datatype(Datatype::Float16),
        )
        .unwrap();
    assert!(halves.upsert([Point::new("a", vec![70000.0, 0.0])]).is_err());
    halves.upsert([Point::new("a", vec![65504.0, 0.1])]).unwrap();
    assert_eq!(halves.get("a").unwrap().vector, [65504.0, half(0.1)]);
    assert_eq!(bytes.len() + halves.len(), 2);
}

#[test]
fn unsupported_combinations_are_rejected() {
    let db = Database::in_memory();
    let refused = [
        CollectionConfig::new(4).datatype(Datatype::Uint8), // cosine by default
        CollectionConfig::new(4)
            .metric(Metric::Euclidean)
            .datatype(Datatype::Float16)
            .quantization(Quantization::Scalar),
        CollectionConfig::new(4).with_vector(
            "v",
            VectorConfig::new(4)
                .metric(Metric::Cosine)
                .datatype(Datatype::Uint8),
        ),
    ];
    for config in refused {
        let err = db.create_collection("x", config.clone()).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{config:?}: {err}");
    }

    // Binary quantization works with compact datatypes; scalar cannot be added.
    let c = db
        .create_collection(
            "ok",
            CollectionConfig::new(4)
                .datatype(Datatype::Float16)
                .quantization(Quantization::Binary),
        )
        .unwrap();
    let to_scalar = CollectionUpdate {
        quantization: Some(Quantization::Scalar),
        ..Default::default()
    };
    assert!(c.update(&to_scalar).is_err());
    assert_eq!(c.config().quantization, Quantization::Binary);
}

#[test]
fn compact_datatypes_survive_restarts_moves_and_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let dim = 32;
    let floats = vectors(200, dim, 4);
    let bytes = byte_vectors(200, dim, 5);
    let config = CollectionConfig::new(dim)
        .metric(Metric::Euclidean)
        .datatype(Datatype::Float16)
        .on_disk(true)
        .with_vector(
            "b",
            VectorConfig::new(dim)
                .metric(Metric::DotProduct)
                .datatype(Datatype::Uint8)
                .quantization(Quantization::Binary),
        );
    let make = |range: std::ops::Range<usize>| -> Vec<Point> {
        range
            .map(|i| Point::new(format!("p{i}"), floats[i].clone()).with_vector("b", bytes[i].clone()))
            .collect()
    };

    let (before, snapshot) = {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("c", config.clone()).unwrap();
        c.upsert(make(0..100)).unwrap();
        c.flush().unwrap(); // half from the checkpoint, half replayed from the log
        c.upsert(make(100..200)).unwrap();
        let before: Vec<Point> = (0..200).map(|i| c.get(&format!("p{i}")).unwrap()).collect();
        assert_eq!(
            before[5].vector,
            floats[5].iter().copied().map(half).collect::<Vec<_>>()
        );
        assert_eq!(before[5].vectors["b"], bytes[5]);
        (before, c.create_snapshot().unwrap().manifest.id)
    };
    // The main vectors' file holds two bytes per component.
    let files: Vec<_> = walk(dir.path())
        .into_iter()
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("vectors-"))
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(files[0].to_str().unwrap().ends_with(".f16"));
    assert_eq!(
        std::fs::metadata(&files[0]).unwrap().len(),
        (200 * dim * 2) as u64
    );

    let check = |c: &quiver_db::Collection, what: &str| {
        for p in &before {
            assert_eq!(&c.get(&p.id).unwrap(), p, "{what}");
        }
        let q = &floats[17];
        assert_eq!(c.search(q, 1).unwrap()[0].id, "p17", "{what}");
    };

    for round in 0..2 {
        let db = Database::open(dir.path()).unwrap();
        let c = db.collection("c").unwrap();
        assert_eq!(c.config(), config);
        check(&c, &format!("reopened {round}"));
        c.compact().unwrap();
    }

    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    // Off disk and back: the datatype goes along.
    for on_disk in [false, true] {
        c.update(&CollectionUpdate {
            on_disk: Some(on_disk),
            ..Default::default()
        })
        .unwrap();
        check(&c, &format!("on_disk={on_disk}"));
    }
    c.update(&CollectionUpdate {
        on_disk: Some(false),
        ..Default::default()
    })
    .unwrap();
    // Half-width main vectors plus bytes and their binary codes, in memory.
    assert_eq!(c.vector_bytes(), 200 * (dim * 2 + dim + dim / 8 + 4));

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
    check(&restored, "restored");
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}
