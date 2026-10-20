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
        std::future::pending(),
    ));
    QuiverClient::connect(format!("http://{addr}")).await.unwrap()
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
            },
            "authorization",
            "Bearer ro-secret",
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::PermissionDenied);

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
