use std::sync::Arc;

use quiver_db::auth::{ApiKey, KeyStore, Role};
use quiver_db::{CollectionConfig, Database, IndexConfig, Point};
use quiver_grpc::pb;
use quiver_grpc::pb::quiver_client::QuiverClient;
use tonic::transport::Channel;
use tonic::{Code, Request};

/// Serves `db` on an ephemeral port and connects a client to it.
async fn start(db: Arc<Database>, keys: KeyStore) -> QuiverClient<Channel> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(quiver_grpc::serve(
        listener,
        db,
        Arc::new(keys),
        None,
        std::future::pending(),
    ));
    QuiverClient::connect(format!("http://{addr}")).await.unwrap()
}

#[tokio::test]
async fn serves_over_tls() {
    use tonic::transport::{Certificate, ClientTlsConfig};
    let key = rcgen::KeyPair::generate().unwrap();
    let cert = rcgen::CertificateParams::new(vec!["localhost".into()])
        .unwrap()
        .self_signed(&key)
        .unwrap();
    let tls = quiver_grpc::tls_config(cert.pem().as_bytes(), key.serialize_pem().as_bytes(), None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(quiver_grpc::serve(
        listener,
        Arc::new(Database::in_memory()),
        Arc::new(KeyStore::default()),
        Some(tls),
        std::future::pending(),
    ));

    let channel = Channel::from_shared(format!("https://localhost:{port}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(cert.pem()))
                .domain_name("localhost"),
        )
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = QuiverClient::new(channel);
    client.health(pb::HealthRequest {}).await.unwrap();

    // Plain-text clients get nowhere.
    let plain = QuiverClient::connect(format!("http://127.0.0.1:{port}")).await;
    if let Ok(mut plain) = plain {
        assert!(plain.health(pb::HealthRequest {}).await.is_err());
    }
}

fn keys() -> KeyStore {
    let key = |name: &str, role, key: &str| ApiKey {
        name: name.into(),
        role,
        key: key.into(),
    };
    KeyStore::new(vec![
        key("admin", Role::ReadWrite, "rw-secret"),
        key("viewer", Role::ReadOnly, "ro-secret"),
    ])
    .unwrap()
}

fn with_auth<T>(msg: T, header: &'static str, value: &str) -> Request<T> {
    let mut req = Request::new(msg);
    req.metadata_mut().insert(header, value.parse().unwrap());
    req
}

fn seeded() -> Arc<Database> {
    let db = Arc::new(Database::in_memory());
    db.create_collection("c", CollectionConfig::new(2).index(IndexConfig::Flat))
        .unwrap()
        .upsert([Point::new("a", vec![1.0, 0.0])])
        .unwrap();
    db
}

fn create(name: &str) -> pb::CreateCollectionRequest {
    pb::CreateCollectionRequest {
        name: name.into(),
        dimension: 2,
        ..Default::default()
    }
}

#[tokio::test]
async fn health_is_public_but_other_rpcs_need_keys() {
    let mut client = start(seeded(), keys()).await;

    let health = client.health(pb::HealthRequest {}).await.unwrap();
    assert_eq!(health.get_ref().status, "ok");

    let err = client
        .list_collections(pb::ListCollectionsRequest {})
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);

    // Writes: no key, a wrong key and a read-only key all fail.
    let err = client.create_collection(create("x")).await.unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);
    let err = client
        .create_collection(with_auth(create("x"), "x-api-key", "wrong"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);
    let err = client
        .create_collection(with_auth(create("x"), "x-api-key", "ro-secret"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::PermissionDenied);
    let err = client
        .delete_points(with_auth(
            pb::DeletePointsRequest {
                collection: "c".into(),
                ids: vec!["a".into()],
                ..Default::default()
            },
            "authorization",
            "Bearer ro-secret",
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::PermissionDenied);
    let err = client
        .update_metadata(with_auth(
            pb::UpdateMetadataRequest {
                collection: "c".into(),
                ids: vec!["a".into()],
                metadata_json: "{}".into(),
                ..Default::default()
            },
            "x-api-key",
            "ro-secret",
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::PermissionDenied);

    // The new read RPCs need a key too, and a read-only one is enough.
    let count = || pb::CountRequest {
        collection: "c".into(),
        ..Default::default()
    };
    let recommend = || pb::RecommendRequest {
        collection: "c".into(),
        positive: vec!["a".into()],
        ..Default::default()
    };
    assert_eq!(
        client.count(count()).await.unwrap_err().code(),
        Code::Unauthenticated
    );
    assert_eq!(
        client.recommend(recommend()).await.unwrap_err().code(),
        Code::Unauthenticated
    );
    assert_eq!(
        client
            .update_metadata(pb::UpdateMetadataRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    let res = client
        .count(with_auth(count(), "x-api-key", "ro-secret"))
        .await
        .unwrap();
    assert_eq!(res.get_ref().count, 1);
    client
        .recommend(with_auth(recommend(), "x-api-key", "ro-secret"))
        .await
        .unwrap();

    client
        .create_collection(with_auth(create("x"), "authorization", "Bearer rw-secret"))
        .await
        .unwrap();
}

#[tokio::test]
async fn bearer_scheme_is_case_insensitive() {
    let mut client = start(seeded(), keys()).await;
    for auth in ["bearer ro-secret", "BEARER   ro-secret  "] {
        let req = with_auth(pb::ListCollectionsRequest {}, "authorization", auth);
        assert!(client.list_collections(req).await.is_ok(), "{auth}");
    }
    let req = with_auth(pb::ListCollectionsRequest {}, "authorization", "Basic ro-secret");
    assert_eq!(
        client.list_collections(req).await.unwrap_err().code(),
        Code::Unauthenticated
    );
}

#[tokio::test]
async fn null_filter_means_no_filter() {
    let mut client = start(seeded(), KeyStore::default()).await;
    let res = client
        .search(pb::SearchRequest {
            collection: "c".into(),
            vector: vec![1.0, 0.0],
            filter_json: "null".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(res.get_ref().hits.len(), 1);
}

#[tokio::test]
async fn id_lists_are_bounded() {
    let mut client = start(seeded(), KeyStore::default()).await;
    let too_many = vec!["a".to_owned(); 10_001];

    let err = client
        .get_points(pb::GetPointsRequest {
            collection: "c".into(),
            ids: too_many.clone(),
            with_vectors: true,
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
    let err = client
        .delete_points(pb::DeletePointsRequest {
            collection: "c".into(),
            ids: too_many.clone(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    let res = client
        .get_points(pb::GetPointsRequest {
            collection: "c".into(),
            ids: too_many[1..].to_vec(),
            with_vectors: false,
        })
        .await
        .unwrap();
    assert_eq!(res.get_ref().points.len(), 10_000);
}

#[tokio::test]
async fn reads_report_missing_collections() {
    let mut client = start(seeded(), KeyStore::default()).await;
    let err = client
        .get_collection(pb::GetCollectionRequest { name: "nope".into() })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    let err = client
        .get_points(pb::GetPointsRequest {
            collection: "nope".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    let list = client
        .list_collections(pb::ListCollectionsRequest {})
        .await
        .unwrap();
    assert_eq!(list.get_ref().collections[0].count, 1);
}

#[tokio::test]
async fn scroll_follows_cursors_in_id_order() {
    let db = seeded();
    db.collection("c")
        .unwrap()
        .upsert(["d", "b", "c"].map(|id| Point::new(id, vec![0.0, 1.0])))
        .unwrap();
    let mut client = start(db, KeyStore::default()).await;
    let scroll = |cursor: &str, limit| pb::ScrollRequest {
        collection: "c".into(),
        cursor: cursor.into(),
        limit,
        with_vectors: false,
    };

    let first = client.scroll(scroll("", 2)).await.unwrap().into_inner();
    let ids: Vec<_> = first.points.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["a", "b"]);
    assert_eq!(first.next_cursor.as_deref(), Some("b"));

    let last = client.scroll(scroll("b", 2)).await.unwrap().into_inner();
    let ids: Vec<_> = last.points.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["c", "d"]);
    assert_eq!(last.next_cursor, None, "no cursor after the last page");

    let err = client.scroll(scroll("", 10_001)).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

/// A collection `c` of points `x0`..`x9` at `(i, 0)`, with `{"i": i, "even": bool}`.
fn line() -> Arc<Database> {
    let db = Arc::new(Database::in_memory());
    db.create_collection(
        "c",
        CollectionConfig::new(2)
            .metric(quiver_db::Metric::Euclidean)
            .index(IndexConfig::Flat),
    )
    .unwrap()
    .upsert((0..10).map(|i| {
        Point::new(format!("x{i}"), vec![i as f32, 0.0])
            .with_metadata(serde_json::json!({ "i": i, "even": i % 2 == 0 }))
    }))
    .unwrap();
    db
}

fn ids(hits: &[pb::SearchHit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

#[tokio::test]
async fn count_update_metadata_and_delete_by_filter() {
    let db = line();
    let mut client = start(Arc::clone(&db), KeyStore::default()).await;
    let count = |filter: &str| pb::CountRequest {
        collection: "c".into(),
        filter_json: filter.into(),
    };
    assert_eq!(client.count(count("")).await.unwrap().get_ref().count, 10);
    assert_eq!(
        client
            .count(count(r#"{"even": true}"#))
            .await
            .unwrap()
            .get_ref()
            .count,
        5
    );

    let res = client
        .update_metadata(pb::UpdateMetadataRequest {
            collection: "c".into(),
            filter_json: r#"{"i": {"$lt": 3}}"#.into(),
            metadata_json: r#"{"low": true, "even": null}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(res.get_ref().updated, 3);
    let c = db.collection("c").unwrap();
    assert_eq!(
        c.get("x1").unwrap().metadata,
        Some(serde_json::json!({"i": 1, "low": true}))
    );
    let res = client
        .update_metadata(pb::UpdateMetadataRequest {
            collection: "c".into(),
            ids: vec!["x1".into(), "missing".into()],
            metadata_json: "null".into(),
            mode: pb::MetadataMode::Replace.into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(res.get_ref().updated, 1);
    assert_eq!(c.get("x1").unwrap().metadata, None);

    for bad in [
        pb::UpdateMetadataRequest {
            collection: "c".into(),
            ids: vec!["x1".into()],
            ..Default::default()
        },
        pb::UpdateMetadataRequest {
            collection: "c".into(),
            ids: vec!["x1".into()],
            filter_json: "{}".into(),
            metadata_json: "{}".into(),
            ..Default::default()
        },
        pb::UpdateMetadataRequest {
            collection: "c".into(),
            ids: vec!["x1".into()],
            metadata_json: "{}".into(),
            mode: 7,
            ..Default::default()
        },
    ] {
        let err = client.update_metadata(bad).await.unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    let delete = |filter: &str| pb::DeletePointsRequest {
        collection: "c".into(),
        filter_json: filter.into(),
        ..Default::default()
    };
    let err = client.delete_points(delete("{}")).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("drop the collection"));
    let mut both = delete(r#"{"even": true}"#);
    both.ids = vec!["x1".into()];
    let err = client.delete_points(both).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
    let res = client.delete_points(delete(r#"{"even": true}"#)).await.unwrap();
    // x0 and x2 lost `even` above, leaving x4, x6 and x8.
    assert_eq!(res.get_ref().deleted, 3);
    assert_eq!(client.count(count("")).await.unwrap().get_ref().count, 7);
}

#[tokio::test]
async fn search_by_id_recommend_and_max_distance() {
    let mut client = start(line(), KeyStore::default()).await;
    let res = client
        .search(pb::SearchRequest {
            collection: "c".into(),
            id: "x4".into(),
            k: 2,
            ..Default::default()
        })
        .await
        .unwrap();
    let mut found = ids(&res.get_ref().hits);
    found.sort_unstable();
    assert_eq!(found, ["x3", "x5"], "the point itself is left out");

    let res = client
        .search(pb::SearchRequest {
            collection: "c".into(),
            vector: vec![0.0, 0.0],
            max_distance: Some(1.5),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(ids(&res.get_ref().hits), ["x0", "x1"]);

    let err = client
        .search(pb::SearchRequest {
            collection: "c".into(),
            id: "nope".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    let err = client
        .search(pb::SearchRequest {
            collection: "c".into(),
            id: "x1".into(),
            vector: vec![0.0, 0.0],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    let res = client
        .search_batch(pb::SearchBatchRequest {
            searches: vec![pb::SearchRequest {
                collection: "c".into(),
                id: "x0".into(),
                k: 1,
                ..Default::default()
            }],
        })
        .await
        .unwrap();
    assert_eq!(ids(&res.get_ref().results[0].hits), ["x1"]);

    let res = client
        .recommend(pb::RecommendRequest {
            collection: "c".into(),
            positive: vec!["x4".into()],
            negative: vec!["x2".into()],
            k: 3,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(ids(&res.get_ref().hits)[0], "x6");
    let err = client
        .recommend(pb::RecommendRequest {
            collection: "c".into(),
            positive: vec!["gone".into()],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("gone"));
    let err = client
        .recommend(pb::RecommendRequest {
            collection: "c".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

fn with_snapshots(dir: &std::path::Path) -> Arc<Database> {
    let db = Database::new(
        None,
        quiver_db::DbOptions {
            storage: quiver_db::StorageKind::Memory,
            snapshot_dir: Some(dir.to_path_buf()),
            ..Default::default()
        },
    )
    .unwrap();
    db.create_collection("c", CollectionConfig::new(2).index(IndexConfig::Flat))
        .unwrap()
        .upsert((0..500).map(|i| Point::new(format!("p{i}"), vec![i as f32, 1.0])))
        .unwrap();
    Arc::new(db)
}

#[tokio::test]
async fn snapshots_download_and_upload_in_chunks() {
    let (dir_a, dir_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let a = with_snapshots(dir_a.path());
    let snap = a.create_snapshot("c").unwrap();
    let mut client_a = start(a, keys()).await;

    // Read-only keys may download.
    let mut stream = client_a
        .download_snapshot(with_auth(
            pb::DownloadSnapshotRequest {
                id: snap.manifest.id.clone(),
            },
            "x-api-key",
            "ro-secret",
        ))
        .await
        .unwrap()
        .into_inner();
    let mut archive = Vec::new();
    while let Some(chunk) = stream.message().await.unwrap() {
        archive.extend_from_slice(&chunk.data);
    }
    assert_eq!(archive.len() as u64, snap.archive_bytes);

    let got = client_a
        .get_snapshot(with_auth(
            pb::GetSnapshotRequest {
                id: snap.manifest.id.clone(),
            },
            "x-api-key",
            "ro-secret",
        ))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(got.points, 500);
    let info = client_a
        .info(with_auth(pb::InfoRequest {}, "x-api-key", "ro-secret"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        (info.collections, info.points, info.snapshots_enabled),
        (1, 500, true)
    );
    assert_eq!(info.storage, "memory");

    let missing = client_a
        .download_snapshot(with_auth(
            pb::DownloadSnapshotRequest {
                id: "00000000-0000-4000-8000-000000000000".into(),
            },
            "x-api-key",
            "ro-secret",
        ))
        .await
        .unwrap_err();
    assert_eq!(missing.code(), Code::NotFound);

    let b = with_snapshots(dir_b.path());
    let mut client_b = start(Arc::clone(&b), keys()).await;
    let chunks = |bytes: Vec<u8>| {
        let chunks: Vec<pb::SnapshotChunk> = bytes
            .chunks(1000)
            .map(|c| pb::SnapshotChunk { data: c.to_vec() })
            .collect();
        tokio_stream::iter(chunks)
    };

    // Uploading needs a read-write key.
    let denied = client_b
        .upload_snapshot(with_auth(chunks(archive.clone()), "x-api-key", "ro-secret"))
        .await
        .unwrap_err();
    assert_eq!(denied.code(), Code::PermissionDenied);

    let info = client_b
        .upload_snapshot(with_auth(chunks(archive.clone()), "x-api-key", "rw-secret"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.id, snap.manifest.id);
    assert_eq!(info.points, 500);
    assert_eq!(b.list_snapshots(None).unwrap().len(), 1);

    let bad = client_b
        .upload_snapshot(with_auth(
            chunks(b"not an archive".to_vec()),
            "x-api-key",
            "rw-secret",
        ))
        .await
        .unwrap_err();
    assert_eq!(bad.code(), Code::InvalidArgument);
    assert_eq!(
        std::fs::read_dir(dir_b.path()).unwrap().count(),
        1,
        "failed uploads leave nothing behind"
    );
}

#[tokio::test]
async fn rpcs_are_recorded_in_metrics() {
    let db = seeded();
    let mut client = start(Arc::clone(&db), KeyStore::default()).await;
    client.health(pb::HealthRequest {}).await.unwrap();
    let missing = client
        .get_collection(pb::GetCollectionRequest { name: "nope".into() })
        .await
        .unwrap_err();
    assert_eq!(missing.code(), Code::NotFound);

    let text = db.metrics().render(&db);
    for line in [
        "quiver_requests_total{api=\"grpc\",route=\"/quiver.v1.Quiver/Health\",status=\"0\"} 1",
        "quiver_requests_total{api=\"grpc\",route=\"/quiver.v1.Quiver/GetCollection\",status=\"5\"} 1",
    ] {
        assert!(text.contains(line), "missing `{line}` in:\n{text}");
    }
}

#[tokio::test]
async fn quantization_and_field_indexes_over_grpc() {
    let db = Arc::new(Database::in_memory());
    let mut client = start(Arc::clone(&db), KeyStore::default()).await;
    let info = client
        .create_collection(pb::CreateCollectionRequest {
            name: "q".into(),
            dimension: 2,
            quantization: pb::Quantization::Scalar.into(),
            field_indexes: vec!["genre".into()],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.quantization, i32::from(pb::Quantization::Scalar));
    assert_eq!(info.field_indexes, ["genre"]);

    let field = |f: &str| pb::FieldIndexRequest {
        collection: "q".into(),
        field: f.into(),
    };
    let info = client
        .create_field_index(field("year"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.field_indexes, ["genre", "year"]);
    let info = client
        .drop_field_index(field("genre"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.field_indexes, ["year"]);
    assert_eq!(
        client.drop_field_index(field("genre")).await.unwrap_err().code(),
        Code::NotFound
    );
    assert_eq!(
        client.create_field_index(field("$bad")).await.unwrap_err().code(),
        Code::InvalidArgument
    );
    assert_eq!(
        db.collection("q").unwrap().config().quantization,
        quiver_db::Quantization::Scalar
    );
}

#[tokio::test]
async fn named_sparse_and_hybrid_search() {
    let db = Arc::new(Database::in_memory());
    let mut client = start(Arc::clone(&db), KeyStore::default()).await;
    let mut vectors = std::collections::HashMap::new();
    vectors.insert(
        "title".to_owned(),
        pb::VectorParams {
            dimension: 2,
            metric: pb::Metric::Euclidean as i32,
            index: Some(pb::IndexConfig {
                kind: Some(pb::index_config::Kind::Flat(pb::FlatIndex {})),
            }),
            ..Default::default()
        },
    );
    let mut sparse_vectors = std::collections::HashMap::new();
    sparse_vectors.insert(
        "text".to_owned(),
        pb::SparseVectorParams {
            modifier: pb::SparseModifier::Idf as i32,
        },
    );
    let info = client
        .create_collection(pb::CreateCollectionRequest {
            name: "docs".into(),
            dimension: 2,
            metric: pb::Metric::Euclidean as i32,
            vectors,
            sparse_vectors,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.vectors["title"].dimension, 2);
    assert_eq!(
        info.sparse_vectors["text"].modifier,
        pb::SparseModifier::Idf as i32
    );

    let point = |id: &str, v: [f32; 2], title: [f32; 2], term: u32| pb::Point {
        id: id.into(),
        vector: v.to_vec(),
        vectors: [(
            "title".to_owned(),
            pb::DenseVector {
                values: title.to_vec(),
            },
        )]
        .into(),
        sparse_vectors: [(
            "text".to_owned(),
            pb::SparseVector {
                indices: vec![term],
                values: vec![1.0],
            },
        )]
        .into(),
        ..Default::default()
    };
    client
        .upsert(pb::UpsertRequest {
            collection: "docs".into(),
            points: vec![
                point("a", [0.0, 0.0], [1.0, 1.0], 1),
                point("b", [1.0, 1.0], [0.0, 0.0], 2),
            ],
        })
        .await
        .unwrap();

    let hits = client
        .search(pb::SearchRequest {
            collection: "docs".into(),
            vector: vec![0.0, 0.0],
            using: "title".into(),
            with_vectors: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner()
        .hits;
    assert_eq!(hits[0].id, "b");
    assert_eq!(hits[0].vectors["title"].values, [0.0, 0.0]);
    assert_eq!(hits[0].sparse_vectors["text"].indices, [2]);

    let hits = client
        .search(pb::SearchRequest {
            collection: "docs".into(),
            sparse: Some(pb::SparseVector {
                indices: vec![1],
                values: vec![1.0],
            }),
            using: "text".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner()
        .hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");

    let fused = client
        .query(pb::QueryRequest {
            collection: "docs".into(),
            prefetch: vec![
                pb::Prefetch {
                    vector: vec![1.0, 1.0],
                    ..Default::default()
                },
                pb::Prefetch {
                    sparse: Some(pb::SparseVector {
                        indices: vec![2],
                        values: vec![1.0],
                    }),
                    using: "text".into(),
                    ..Default::default()
                },
            ],
            k: 1,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner()
        .hits;
    assert_eq!(fused[0].id, "b");
    assert!((fused[0].score - 2.0 / 61.0).abs() < 1e-6);

    for bad in [
        pb::SearchRequest {
            collection: "docs".into(),
            sparse: Some(pb::SparseVector::default()),
            ..Default::default()
        },
        pb::SearchRequest {
            collection: "docs".into(),
            vector: vec![0.0, 0.0],
            id: "a".into(),
            ..Default::default()
        },
    ] {
        let err = client.search(bad).await.unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }
}

#[tokio::test]
async fn update_rename_and_aliases() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("v1", CollectionConfig::new(2)).unwrap();
    let mut client = start(Arc::clone(&db), keys()).await;
    fn rw<T>(msg: T) -> Request<T> {
        with_auth(msg, "x-api-key", "rw-secret")
    }
    fn ro<T>(msg: T) -> Request<T> {
        with_auth(msg, "x-api-key", "ro-secret")
    }

    let info = client
        .update_collection(rw(pb::UpdateCollectionRequest {
            name: "v1".into(),
            update_json: r#"{"index": {"type": "flat"}}"#.into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(matches!(
        info.index.unwrap().kind,
        Some(pb::index_config::Kind::Flat(_))
    ));
    let err = client
        .update_collection(rw(pb::UpdateCollectionRequest {
            name: "v1".into(),
            update_json: r#"{"dimension": 3}"#.into(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    client
        .set_alias(rw(pb::SetAliasRequest {
            alias: "live".into(),
            collection: "v1".into(),
        }))
        .await
        .unwrap();
    let info = client
        .rename_collection(rw(pb::RenameCollectionRequest {
            name: "v1".into(),
            new_name: "v2".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.name, "v2");
    let aliases = client
        .list_aliases(ro(pb::ListAliasesRequest {}))
        .await
        .unwrap()
        .into_inner()
        .aliases;
    assert_eq!(
        aliases,
        [pb::Alias {
            alias: "live".into(),
            collection: "v2".into()
        }]
    );
    let got = client
        .get_collection(rw(pb::GetCollectionRequest { name: "live".into() }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(got.name, "v2");

    // Read-only keys cannot change settings or aliases.
    let denied = client
        .delete_alias(ro(pb::DeleteAliasRequest { alias: "live".into() }))
        .await
        .unwrap_err();
    assert_eq!(denied.code(), Code::PermissionDenied);
    client
        .delete_alias(rw(pb::DeleteAliasRequest { alias: "live".into() }))
        .await
        .unwrap();
    let err = client
        .delete_alias(rw(pb::DeleteAliasRequest { alias: "live".into() }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
}
