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

    let page = c.scroll("docs", 0, 1, false).await.unwrap();
    assert_eq!(page.points.len(), 1);
    assert_eq!(page.next_offset, Some(1));

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
