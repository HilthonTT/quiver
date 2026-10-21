//! Drives the client against a real HTTP server on an ephemeral port.

use std::sync::Arc;

use quiver_client::{Client, ClientError, SearchRequest};
use quiver_db::auth::{ApiKey, KeyStore, Role};
use quiver_db::{Database, DbOptions, StorageKind};
use quiver_http::HttpOptions;
use quiver_types::{CollectionConfig, PointInput};
use serde_json::json;

struct Server {
    url: String,
    _dir: tempfile::TempDir,
}

async fn start(keys: KeyStore) -> Server {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::new(
        None,
        DbOptions {
            storage: StorageKind::Memory,
            snapshot_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        },
    )
    .unwrap();
    let options = HttpOptions {
        keys: Arc::new(keys),
        ..Default::default()
    };
    let router = quiver_http::router(Arc::new(db), options);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(quiver_http::serve(listener, router, std::future::pending()));
    Server { url, _dir: dir }
}

fn point(id: &str, vector: Vec<f32>) -> PointInput {
    serde_json::from_value(json!({ "id": id, "vector": vector, "metadata": { "tag": id } })).unwrap()
}

#[tokio::test]
async fn talks_to_a_tls_server_with_a_private_ca() {
    use quiver_client::TlsOptions;
    let key = rcgen::KeyPair::generate().unwrap();
    let cert = rcgen::CertificateParams::new(vec!["localhost".into()])
        .unwrap()
        .self_signed(&key)
        .unwrap();
    let config =
        quiver_http::tls_config(cert.pem().as_bytes(), key.serialize_pem().as_bytes(), None).unwrap();
    let router = quiver_http::router(Arc::new(Database::in_memory()), HttpOptions::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("https://localhost:{}", listener.local_addr().unwrap().port());
    tokio::spawn(quiver_http::serve_tls(
        listener,
        config,
        router,
        std::future::pending(),
    ));

    assert!(Client::new(&url).unwrap().info().await.is_err(), "untrusted");
    let tls = TlsOptions {
        ca_pem: Some(cert.pem().into_bytes()),
        ..Default::default()
    };
    let info = Client::with_tls(&url, &tls).unwrap().info().await.unwrap();
    assert_eq!(info.collections, 0);
    let bad = TlsOptions {
        ca_pem: Some(b"not a certificate".to_vec()),
        ..Default::default()
    };
    assert!(Client::with_tls(&url, &bad).is_err());
}

#[tokio::test]
async fn full_lifecycle() {
    let server = start(KeyStore::default()).await;
    let c = Client::new(&server.url).unwrap();

    let created = c
        .create_collection("docs", &CollectionConfig::new(2))
        .await
        .unwrap();
    assert_eq!(created.name, "docs");
    assert_eq!(c.list_collections().await.unwrap().len(), 1);

    let res = c
        .upsert(
            "docs",
            vec![point("a", vec![0.0, 1.0]), point("b", vec![1.0, 0.0])],
        )
        .await
        .unwrap();
    assert_eq!(res.upserted, 2);
    assert_eq!(res.ids, ["a", "b"]);

    let info = c.info().await.unwrap();
    assert_eq!((info.collections, info.points), (1, 2));
    assert_eq!(info.storage, "memory");

    let page = c.scroll("docs", None, 1, false).await.unwrap();
    assert_eq!(page.points[0].id, "a");
    assert_eq!(page.next_cursor.as_deref(), Some("a"));
    let page = c.scroll("docs", Some("a"), 1, false).await.unwrap();
    assert_eq!(page.points[0].id, "b");
    assert_eq!(page.next_cursor, None, "no cursor after the last page");

    let hits = c
        .search("docs", &SearchRequest::new(vec![0.0, 1.0], 1))
        .await
        .unwrap();
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[0].metadata, Some(json!({ "tag": "a" })));

    let mut filtered = SearchRequest::new(vec![0.0, 1.0], 2);
    filtered.filter = Some(json!({ "tag": "b" }));
    let hits = c.search("docs", &filtered).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "b");

    assert_eq!(
        c.get_point("docs", "a").await.unwrap().unwrap().vector,
        [0.0, 1.0]
    );
    assert!(c.get_point("docs", "missing").await.unwrap().is_none());
    let err = c.get_point("typo", "missing").await.unwrap_err();
    assert!(
        matches!(err, ClientError::Api { status, .. } if status == 404),
        "{err}"
    );

    let snap = c.create_snapshot("docs").await.unwrap();
    assert_eq!(snap.manifest.points, 2);
    assert_eq!(c.list_snapshots(Some("docs")).await.unwrap().len(), 1);

    assert_eq!(
        c.delete_points("docs", &["a".into(), "zz".into()]).await.unwrap(),
        1
    );

    let restored = c
        .restore_snapshot(&snap.manifest.id, Some("docs2"), false)
        .await
        .unwrap();
    assert_eq!((restored.name.as_str(), restored.count), ("docs2", 2));

    c.delete_snapshot(&snap.manifest.id).await.unwrap();
    c.drop_collection("docs").await.unwrap();
    let err = c.get_collection("docs").await.unwrap_err();
    assert!(
        matches!(err, ClientError::Api { ref code, .. } if code == "not_found"),
        "{err}"
    );
}

#[tokio::test]
async fn data_operations_round_trip() {
    use quiver_client::{MetadataMode, RecommendRequest, Selector};
    use quiver_types::{IndexConfig, Metric};

    let server = start(KeyStore::default()).await;
    let c = Client::new(&server.url).unwrap();
    let config = CollectionConfig::new(2)
        .metric(Metric::Euclidean)
        .index(IndexConfig::Flat);
    c.create_collection("line", &config).await.unwrap();
    let points = (0..6)
        .map(|i| {
            serde_json::from_value(json!({
                "id": format!("x{i}"), "vector": [i as f32, 0.0], "metadata": { "i": i, "even": i % 2 == 0 }
            }))
            .unwrap()
        })
        .collect();
    c.upsert("line", points).await.unwrap();

    assert_eq!(c.count("line", None).await.unwrap(), 6);
    assert_eq!(c.count("line", Some(&json!({"even": true}))).await.unwrap(), 3);

    let ids = Selector::Ids(vec!["x1".into(), "missing".into()]);
    let n = c
        .update_metadata("line", &ids, &json!({"tag": "t", "i": null}), MetadataMode::Merge)
        .await
        .unwrap();
    assert_eq!(n, 1);
    let x1 = c.get_point("line", "x1").await.unwrap().unwrap();
    assert_eq!(x1.metadata, Some(json!({"even": false, "tag": "t"})));
    let odd = Selector::Filter(json!({"even": false}));
    let n = c
        .update_metadata("line", &odd, &json!({"odd": true}), MetadataMode::Replace)
        .await
        .unwrap();
    assert_eq!(n, 3);

    let mut by_id = SearchRequest::by_id("x2", 2);
    let hits = c.search("line", &by_id).await.unwrap();
    assert!(hits.iter().all(|h| h.id != "x2") && hits.len() == 2);
    by_id.max_distance = Some(0.5);
    assert!(c.search("line", &by_id).await.unwrap().is_empty());

    let mut rec = RecommendRequest::new(vec!["x2".into()], 1);
    rec.negative = vec!["x0".into()];
    let hits = c.recommend("line", &rec).await.unwrap();
    assert_eq!(hits[0].id, "x4", "2·x2 − x0 = (4, 0)");
    let err = c
        .recommend("line", &RecommendRequest::new(vec!["nope".into()], 1))
        .await
        .unwrap_err();
    assert!(
        matches!(err, ClientError::Api { ref code, .. } if code == "point_not_found"),
        "{err}"
    );

    assert_eq!(c.delete_where("line", &json!({"odd": true})).await.unwrap(), 3);
    assert_eq!(c.count("line", None).await.unwrap(), 3);
    let err = c.delete_where("line", &json!({})).await.unwrap_err();
    assert!(
        matches!(err, ClientError::Api { status, .. } if status == 400),
        "{err}"
    );
}

#[tokio::test]
async fn path_unsafe_point_ids_round_trip() {
    let server = start(KeyStore::default()).await;
    let c = Client::new(&server.url).unwrap();
    c.create_collection("docs", &CollectionConfig::new(2))
        .await
        .unwrap();
    // Dot-segments, and names the server's fixed `points/...` routes take.
    for id in [".", "..", "...", "lookup", "delete", "count", "metadata"] {
        c.upsert("docs", vec![point(id, vec![1.0, 0.0])]).await.unwrap();
        let got = c.get_point("docs", id).await.unwrap();
        assert_eq!(got.map(|p| p.id).as_deref(), Some(id));
    }
    assert!(c.get_point("docs", "count2").await.unwrap().is_none());
}

#[tokio::test]
async fn api_keys_are_sent() {
    let keys = KeyStore::new(vec![ApiKey {
        name: "admin".into(),
        role: Role::ReadWrite,
        key: "s3cret".into(),
    }])
    .unwrap();
    let server = start(keys).await;

    let anonymous = Client::new(&server.url).unwrap();
    let err = anonymous.list_collections().await.unwrap_err();
    assert!(
        matches!(err, ClientError::Api { status, .. } if status == 401),
        "{err}"
    );

    let authed = Client::new(&server.url).unwrap().with_api_key("s3cret");
    assert!(authed.list_collections().await.unwrap().is_empty());
}

#[tokio::test]
async fn unreachable_server_is_a_transport_error() {
    let c = Client::new("http://127.0.0.1:9").unwrap();
    assert!(matches!(c.info().await, Err(ClientError::Http(_))));
}

#[tokio::test]
async fn batch_lookup_maintenance_and_snapshot_transfer() {
    let (a, b) = (start(KeyStore::default()).await, start(KeyStore::default()).await);
    let (ca, cb) = (Client::new(&a.url).unwrap(), Client::new(&b.url).unwrap());
    ca.create_collection("docs", &CollectionConfig::new(2))
        .await
        .unwrap();
    ca.upsert(
        "docs",
        vec![
            point("a", vec![0.0, 1.0]),
            point("b", vec![1.0, 0.0]),
            point("c", vec![1.0, 1.0]),
        ],
    )
    .await
    .unwrap();

    let ids = ["c".to_owned(), "missing".to_owned(), "a".to_owned()];
    let points = ca.get_points("docs", &ids, false).await.unwrap();
    let got: Vec<&str> = points.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(got, ["c", "a"]);
    assert!(points[0].vector.is_empty());

    let batch = ca
        .search_batch(
            "docs",
            &[
                SearchRequest::new(vec![0.0, 1.0], 1),
                SearchRequest::by_id("b", 2),
            ],
        )
        .await
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0][0].id, "a");
    assert!(batch[1].iter().all(|h| h.id != "b"));

    let info = ca.create_field_index("docs", "tag").await.unwrap();
    assert_eq!(info.field_indexes, ["tag"]);
    assert_eq!(ca.count("docs", Some(&json!({"tag": "a"}))).await.unwrap(), 1);
    assert!(
        ca.drop_field_index("docs", "tag")
            .await
            .unwrap()
            .field_indexes
            .is_empty()
    );
    assert!(ca.drop_field_index("docs", "tag").await.is_err());

    ca.delete_points("docs", &["c".to_owned()]).await.unwrap();
    ca.flush("docs").await.unwrap();
    assert_eq!(ca.compact("docs").await.unwrap().count, 2);

    let snap = ca.create_snapshot("docs").await.unwrap();
    assert_eq!(ca.get_snapshot(&snap.manifest.id).await.unwrap(), snap);
    let archive = ca.download_snapshot(&snap.manifest.id).await.unwrap();
    assert_eq!(archive.len() as u64, snap.archive_bytes);

    let uploaded = cb.upload_snapshot(archive).await.unwrap();
    assert_eq!(uploaded.manifest, snap.manifest);
    let restored = cb.restore_snapshot(&snap.manifest.id, None, false).await.unwrap();
    assert_eq!(restored.count, 2);

    let bad = cb.upload_snapshot(b"junk".to_vec()).await.unwrap_err();
    assert!(
        matches!(bad, ClientError::Api { status, .. } if status == 400),
        "{bad:?}"
    );
}
