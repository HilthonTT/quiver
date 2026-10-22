use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use quiver_db::auth::{ApiKey, KeyStore, Role};
use quiver_db::{Database, DbOptions, StorageKind};
use quiver_http::{HttpOptions, router};
use serde_json::{Value, json};
use tower::ServiceExt;

struct TestApp {
    app: Router,
    key: Option<&'static str>,
    _dir: tempfile::TempDir,
}

impl TestApp {
    fn new() -> Self {
        Self::with_options(HttpOptions::default())
    }

    /// An in-memory database whose snapshots go to a temporary directory.
    fn with_options(options: HttpOptions) -> Self {
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
        Self {
            app: router(Arc::new(db), options),
            key: None,
            _dir: dir,
        }
    }

    async fn raw(&self, req: Request<Body>) -> axum::response::Response {
        self.app.clone().oneshot(req).await.unwrap()
    }

    async fn call(&self, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        if let Some(key) = self.key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let req = match body {
            Some(b) => req
                .header("content-type", "application/json")
                .body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap();
        let res = self.app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, value)
    }

    async fn post(&self, uri: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::POST, uri, Some(body)).await
    }

    async fn get(&self, uri: &str) -> (StatusCode, Value) {
        self.call(Method::GET, uri, None).await
    }

    async fn seeded() -> Self {
        let app = Self::new();
        let (status, _) = app
            .post(
                "/collections",
                json!({"name": "books", "dimension": 2, "metric": "euclidean", "index": {"type": "hnsw", "m": 8}}),
            )
            .await;
        assert_eq!(status, StatusCode::CREATED);
        let (status, body) = app
            .post(
                "/collections/books/points",
                json!({"points": [
                    {"id": "a", "vector": [0.0, 0.0], "metadata": {"genre": "sf"}},
                    {"id": "b", "vector": [1.0, 0.0], "metadata": {"genre": "fantasy"}},
                    {"id": "c", "vector": [5.0, 5.0], "metadata": {"genre": "sf"}}
                ]}),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["upserted"], 3);
        app
    }
}

#[tokio::test]
async fn health() {
    let (status, body) = TestApp::new().get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn collection_lifecycle() {
    let app = TestApp::seeded().await;

    let (status, info) = app.get("/collections/books").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(info["count"], 3);
    assert_eq!(info["dimension"], 2);
    assert_eq!(info["metric"], "euclidean");
    assert_eq!(info["index"]["m"], 8);

    let (status, body) = app.get("/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["collections"].as_array().unwrap().len(), 1);

    let (status, body) = app
        .post("/collections", json!({"name": "books", "dimension": 2}))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "already_exists");

    let (status, _) = app.call(Method::DELETE, "/collections/books", None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, body) = app.get("/collections/books").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
}

#[tokio::test]
async fn search_with_filter() {
    let app = TestApp::seeded().await;
    let (status, body) = app
        .post("/collections/books/search", json!({"vector": [1.0, 0.1], "k": 2}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<&str> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["b", "a"]);

    let (_, body) = app
        .post(
            "/collections/books/search",
            json!({"vector": [1.0, 0.1], "k": 5, "filter": {"genre": "sf"}, "with_vectors": true}),
        )
        .await;
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["id"], "a");
    assert_eq!(results[0]["vector"], json!([0.0, 0.0]));
    assert_eq!(results[0]["metadata"]["genre"], "sf");
}

#[tokio::test]
async fn batch_search() {
    let app = TestApp::seeded().await;
    let (status, body) = app
        .post(
            "/collections/books/search/batch",
            json!({"searches": [{"vector": [0.0, 0.0], "k": 1}, {"vector": [5.0, 5.0], "k": 1}]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"][0][0]["id"], "a");
    assert_eq!(body["results"][1][0]["id"], "c");
}

#[tokio::test]
async fn point_crud() {
    let app = TestApp::seeded().await;

    let (status, body) = app.get("/collections/books/points/a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        json!({"id": "a", "vector": [0.0, 0.0], "metadata": {"genre": "sf"}})
    );

    let (_, body) = app.get("/collections/books/points/a?with_vectors=false").await;
    assert!(body.get("vector").is_none());

    let (_, body) = app
        .post(
            "/collections/books/points/lookup",
            json!({"ids": ["a", "zzz", "c"]}),
        )
        .await;
    assert_eq!(body["points"].as_array().unwrap().len(), 2);

    let (_, body) = app.get("/collections/books/points?limit=2").await;
    assert_eq!(body["points"].as_array().unwrap().len(), 2);
    assert_eq!(body["next_cursor"], "b");
    let (_, body) = app.get("/collections/books/points?cursor=b&limit=2").await;
    assert_eq!(body["points"].as_array().unwrap().len(), 1);
    assert_eq!(body["next_cursor"], Value::Null);

    let (_, body) = app
        .post("/collections/books/points/delete", json!({"ids": ["a", "nope"]}))
        .await;
    assert_eq!(body["deleted"], 1);
    let (status, _) = app
        .call(Method::DELETE, "/collections/books/points/b", None)
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = app
        .call(Method::DELETE, "/collections/books/points/b", None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, info) = app.get("/collections/books").await;
    assert_eq!(info["count"], 1);
}

#[tokio::test]
async fn scroll_pages_by_cursor_in_id_order() {
    let app = TestApp::seeded().await;
    let page = |body: &Value| -> Vec<String> {
        body["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["id"].as_str().unwrap().to_owned())
            .collect()
    };

    let (status, body) = app.get("/collections/books/points?limit=1").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        (page(&body), &body["next_cursor"]),
        (vec!["a".to_owned()], &json!("a"))
    );
    // A page that exactly reaches the end hands out no cursor.
    let (_, body) = app.get("/collections/books/points?cursor=a&limit=2").await;
    assert_eq!(page(&body), ["b", "c"]);
    assert_eq!(body["next_cursor"], Value::Null);
    // The cursor need not be a live id.
    let (_, body) = app.get("/collections/books/points?cursor=aa").await;
    assert_eq!(page(&body), ["b", "c"]);

    let (status, _) = app.get("/collections/books/points?limit=10001").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = app.get("/collections/books/points?offset=1").await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "offset is no longer accepted: {body}"
    );
}

#[tokio::test]
async fn missing_points_and_collections_have_distinct_codes() {
    let app = TestApp::seeded().await;
    let (status, body) = app.get("/collections/books/points/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "point_not_found");
    let (status, body) = app
        .call(Method::DELETE, "/collections/books/points/nope", None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "point_not_found");

    let (status, body) = app.get("/collections/typo/points/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
}

#[tokio::test]
async fn validation_errors_are_json() {
    let app = TestApp::seeded().await;
    let cases = [
        (
            "/collections/books/search",
            json!({"vector": [1.0]}),
            "dimension_mismatch",
        ),
        (
            "/collections/books/search",
            json!({"vector": [1.0, 1.0], "k": 0}),
            "invalid_argument",
        ),
        (
            "/collections/books/search",
            json!({"vector": [1.0, 1.0], "filter": {"$bogus": 1}}),
            "invalid_filter",
        ),
        (
            "/collections/books/search",
            json!({"vectr": [1.0, 1.0]}),
            "invalid_body",
        ),
        (
            "/collections",
            json!({"name": "bad name!", "dimension": 2}),
            "invalid_argument",
        ),
        (
            "/collections",
            json!({"name": "x", "dimension": 2, "metric": "chebyshev"}),
            "invalid_body",
        ),
    ];
    for (uri, body, code) in cases {
        let (status, res) = app.post(uri, body.clone()).await;
        assert!(status.is_client_error(), "{body} -> {status}");
        assert_eq!(res["error"]["code"], code, "{body} -> {res}");
    }

    let (status, res) = app.get("/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(res["error"]["code"], "not_found");
}

fn keys() -> Arc<KeyStore> {
    let key = |name: &str, role, key: &str| ApiKey {
        name: name.into(),
        role,
        key: key.into(),
    };
    Arc::new(
        KeyStore::new(vec![
            key("admin", Role::ReadWrite, "rw-secret"),
            key("viewer", Role::ReadOnly, "ro-secret"),
        ])
        .unwrap(),
    )
}

#[tokio::test]
async fn api_keys_and_roles_are_enforced() {
    let mut app = TestApp::with_options(HttpOptions {
        keys: keys(),
        ..Default::default()
    });

    assert_eq!(
        app.get("/health").await.0,
        StatusCode::OK,
        "health is always public"
    );
    assert_eq!(app.get("/").await.0, StatusCode::OK, "root is always public");
    let (status, body) = app.get("/collections").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");

    app.key = Some("wrong");
    assert_eq!(app.get("/collections").await.0, StatusCode::UNAUTHORIZED);

    app.key = Some("ro-secret");
    assert_eq!(app.get("/collections").await.0, StatusCode::OK);
    let (status, body) = app
        .post("/collections", json!({"name": "x", "dimension": 2}))
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["code"], "forbidden");

    app.key = Some("rw-secret");
    let (status, _) = app
        .post("/collections", json!({"name": "x", "dimension": 2}))
        .await;
    assert_eq!(status, StatusCode::CREATED);

    app.key = Some("ro-secret");
    let (status, _) = app
        .post("/collections/x/search", json!({"vector": [1.0, 0.0]}))
        .await;
    assert_eq!(status, StatusCode::OK, "read-only keys can search");
    assert_eq!(
        app.call(Method::DELETE, "/collections/x", None).await.0,
        StatusCode::FORBIDDEN
    );
}

#[tokio::test]
async fn missing_ids_are_generated() {
    let app = TestApp::seeded().await;
    let (status, body) = app
        .post(
            "/collections/books/points",
            json!({"points": [{"vector": [2.0, 2.0]}, {"id": "named", "vector": [3.0, 3.0]}]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids = body["ids"].as_array().unwrap();
    assert_eq!(ids[0].as_str().unwrap().len(), 36);
    assert_eq!(ids[1], "named");
    let (status, _) = app
        .get(&format!("/collections/books/points/{}", ids[0].as_str().unwrap()))
        .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn info_endpoint() {
    let app = TestApp::seeded().await;
    let (status, body) = app.get("/info").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["collections"], 1);
    assert_eq!(body["points"], 3);
    assert_eq!(body["storage"], "memory");
    assert_eq!(body["snapshots_enabled"], true);
}

#[tokio::test]
async fn snapshot_lifecycle() {
    let app = TestApp::seeded().await;

    let (status, snap) = app.post("/collections/books/snapshots", json!({})).await;
    assert_eq!(status, StatusCode::CREATED, "{snap}");
    let id = snap["id"].as_str().unwrap().to_owned();
    assert_eq!(snap["collection"], "books");
    assert_eq!(snap["points"], 3);

    let (_, list) = app.get("/snapshots").await;
    assert_eq!(list["snapshots"].as_array().unwrap().len(), 1);
    let (_, list) = app.get("/snapshots?collection=nope").await;
    assert!(list["snapshots"].as_array().unwrap().is_empty());
    let (_, list) = app.get("/collections/books/snapshots").await;
    assert_eq!(list["snapshots"][0]["id"], id.as_str());

    // Restore under a new name.
    let (status, info) = app
        .post(
            &format!("/snapshots/{id}/restore"),
            json!({"collection": "books_copy"}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{info}");
    assert_eq!(info["count"], 3);
    // Restoring over an existing collection needs `replace`.
    let (status, _) = app
        .call(Method::POST, &format!("/snapshots/{id}/restore"), None)
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (status, _) = app
        .post(&format!("/snapshots/{id}/restore"), json!({"replace": true}))
        .await;
    assert_eq!(status, StatusCode::OK);

    // Download, delete, upload again.
    let res = app
        .raw(
            Request::get(format!("/snapshots/{id}/download"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "application/gzip");
    let archive = res.into_body().collect().await.unwrap().to_bytes();
    assert!(archive.len() > 20);

    assert_eq!(
        app.call(Method::DELETE, &format!("/snapshots/{id}"), None)
            .await
            .0,
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        app.get(&format!("/snapshots/{id}")).await.0,
        StatusCode::NOT_FOUND
    );

    let res = app
        .raw(
            Request::post("/snapshots")
                .header("content-type", "application/gzip")
                .body(Body::from(archive))
                .unwrap(),
        )
        .await;
    assert_eq!(res.status(), StatusCode::CREATED);
    assert_eq!(app.get(&format!("/snapshots/{id}")).await.0, StatusCode::OK);

    let res = app
        .raw(Request::post("/snapshots").body(Body::from("garbage")).unwrap())
        .await;
    assert!(res.status().is_client_error() || res.status().is_server_error());
    assert_eq!(
        app.get("/snapshots").await.1["snapshots"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn snapshots_disabled_is_reported() {
    let app = Router::new().merge(router(Arc::new(Database::in_memory()), HttpOptions::default()));
    let t = TestApp {
        app,
        key: None,
        _dir: tempfile::tempdir().unwrap(),
    };
    t.post("/collections", json!({"name": "a", "dimension": 2})).await;
    let (status, body) = t.post("/collections/a/snapshots", json!({})).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["code"], "unsupported");
}

#[tokio::test]
async fn new_metrics_and_kd_tree() {
    let app = TestApp::new();
    let (status, body) = app
        .post(
            "/collections",
            json!({"name": "kd", "dimension": 2, "metric": "manhattan", "index": {"type": "kd_tree"}}),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["index"]["type"], "kd_tree");
    let (status, body) = app
        .post(
            "/collections",
            json!({"name": "bad", "dimension": 2, "metric": "dot_product", "index": {"type": "kd_tree"}}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    app.post(
        "/collections/kd/points",
        json!({"points": [{"id": "a", "vector": [0, 0]}, {"id": "b", "vector": [3, 4]}]}),
    )
    .await;
    let (_, res) = app
        .post("/collections/kd/search", json!({"vector": [3, 3]}))
        .await;
    assert_eq!(res["results"][0]["id"], "b");
    assert_eq!(res["results"][0]["distance"], 1.0);
    assert_eq!(res["results"][1]["distance"], 6.0);
}

fn leftover_files(dir: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

async fn status_and_json(res: axum::response::Response) -> (StatusCode, Value) {
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn snapshot_upload_limit_is_enforced() {
    let app = TestApp::with_options(HttpOptions {
        max_upload_bytes: Some(1024),
        ..Default::default()
    });
    // Declared too large up front.
    let res = app
        .raw(
            Request::post("/snapshots")
                .body(Body::from(vec![0u8; 1 << 20]))
                .unwrap(),
        )
        .await;
    let (status, body) = status_and_json(res).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["code"], "payload_too_large");

    // Streamed without a length, crossing the limit midway.
    let chunks = (0..8).map(|_| Ok::<_, std::io::Error>(axum::body::Bytes::from(vec![0u8; 512])));
    let res = app
        .raw(
            Request::post("/snapshots")
                .body(Body::from_stream(futures_util::stream::iter(chunks)))
                .unwrap(),
        )
        .await;
    let (status, body) = status_and_json(res).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert!(
        leftover_files(app._dir.path()).is_empty(),
        "staging files must be removed"
    );
}

#[tokio::test]
async fn invalid_snapshot_upload_is_a_client_error() {
    let app = TestApp::new();
    let res = app
        .raw(Request::post("/snapshots").body(Body::from("garbage")).unwrap())
        .await;
    let (status, body) = status_and_json(res).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_snapshot");
    assert!(leftover_files(app._dir.path()).is_empty());
}

#[tokio::test]
async fn duplicate_snapshot_upload_is_a_client_error() {
    let app = TestApp::seeded().await;
    let (_, snap) = app.post("/collections/books/snapshots", json!({})).await;
    let id = snap["id"].as_str().unwrap();
    let res = app
        .raw(
            Request::get(format!("/snapshots/{id}/download"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    let archive = res.into_body().collect().await.unwrap().to_bytes();
    let res = app
        .raw(Request::post("/snapshots").body(Body::from(archive)).unwrap())
        .await;
    let (status, body) = status_and_json(res).await;
    assert!(status.is_client_error(), "{status} {body}");
}

#[tokio::test]
async fn upload_staging_file_is_removed_when_the_request_is_dropped() {
    let app = TestApp::new();
    // One chunk, then the client stalls and the request future is dropped.
    let stream = futures_util::StreamExt::chain(
        futures_util::stream::once(async {
            Ok::<_, std::io::Error>(axum::body::Bytes::from_static(b"partial archive"))
        }),
        futures_util::stream::pending(),
    );
    let req = Request::post("/snapshots")
        .body(Body::from_stream(stream))
        .unwrap();
    let pending = app.app.clone().oneshot(req);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(200), pending)
            .await
            .is_err()
    );
    assert!(leftover_files(app._dir.path()).is_empty());
}

#[tokio::test]
async fn empty_snapshot_collection_filter_lists_everything() {
    let app = TestApp::seeded().await;
    app.post("/collections/books/snapshots", json!({})).await;
    let (status, list) = app.get("/snapshots?collection=").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["snapshots"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn bearer_scheme_is_case_insensitive() {
    let app = TestApp::with_options(HttpOptions {
        keys: keys(),
        ..Default::default()
    });
    let get = |auth: &'static str| {
        app.raw(
            Request::get("/collections")
                .header("authorization", auth)
                .body(Body::empty())
                .unwrap(),
        )
    };
    for auth in ["bearer ro-secret", "BEARER   ro-secret  ", "Bearer\tro-secret"] {
        assert_eq!(get(auth).await.status(), StatusCode::OK, "{auth}");
    }
    for auth in ["Basic ro-secret", "Bearerro-secret", "bearer wrong"] {
        assert_eq!(get(auth).await.status(), StatusCode::UNAUTHORIZED, "{auth}");
    }
}

#[tokio::test]
async fn id_lists_are_bounded() {
    let app = TestApp::seeded().await;
    let too_many: Vec<String> = (0..=quiver_http::handlers::MAX_IDS)
        .map(|i| i.to_string())
        .collect();
    for uri in [
        "/collections/books/points/lookup",
        "/collections/books/points/delete",
    ] {
        let (status, body) = app.post(uri, json!({ "ids": too_many })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
        assert_eq!(body["error"]["code"], "invalid_argument");
        let (status, body) = app.post(uri, json!({ "ids": &too_many[1..] })).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {body}");
    }
}

#[tokio::test]
async fn body_errors_have_consistent_statuses() {
    let app = TestApp::seeded().await;
    let send = |body: &'static str, content_type: Option<&'static str>| {
        let mut req = Request::builder()
            .method(Method::POST)
            .uri("/collections/books/search");
        if let Some(ct) = content_type {
            req = req.header("content-type", ct);
        }
        app.raw(req.body(Body::from(body)).unwrap())
    };

    // Malformed JSON and the wrong shape are invalid input, like any other.
    for body in [
        "{\"vector\": [1.0,",
        "{\"vectr\": [1.0, 1.0]}",
        "{\"vector\": \"x\"}",
    ] {
        let res = send(body, Some("application/json")).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "{body}");
    }
    let res = send("{\"vector\": [1.0, 1.0]}", None).await;
    assert_eq!(res.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn create_collection_rejects_unknown_fields() {
    let app = TestApp::new();
    let (status, body) = app
        .post(
            "/collections",
            json!({"name": "a", "dimension": 2, "metrc": "euclidean"}),
        )
        .await;
    // The same status as every other invalid input.
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_body");

    // Everything accepted before still is, with the same defaults.
    let (status, body) = app
        .post("/collections", json!({"name": "a", "dimension": 2}))
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["metric"], "cosine");
    assert_eq!(body["index"]["type"], "hnsw");
    let (status, body) = app
        .post(
            "/collections",
            json!({"name": "b", "dimension": 2, "metric": "manhattan", "index": {"type": "kd_tree"}}),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["index"]["balance_factor"], 3);
}

/// Holds a collection's read lock for `hold` on its first write, so that a
/// writer queued behind it keeps new readers waiting for that long.
struct SlowWriter(std::time::Duration);

impl std::io::Write for SlowWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        std::thread::sleep(std::mem::take(&mut self.0));
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn point_reads_do_not_block_the_runtime() {
    use std::time::{Duration, Instant};

    let db = Arc::new(Database::in_memory());
    let c = db
        .create_collection("c", quiver_db::CollectionConfig::new(2))
        .unwrap();
    c.upsert([quiver_db::Point::new("a", vec![1.0, 0.0])]).unwrap();
    let app = router(Arc::clone(&db), HttpOptions::default());

    // A reader holding the lock, then a writer queued behind it: the lock is
    // fair, so new readers now wait until both are done (~600ms).
    let hold = Duration::from_millis(600);
    let reader = {
        let c = Arc::clone(&c);
        std::thread::spawn(move || c.export_state(SlowWriter(hold)).unwrap())
    };
    std::thread::sleep(Duration::from_millis(50));
    let writer = {
        let c = Arc::clone(&c);
        std::thread::spawn(move || c.upsert([quiver_db::Point::new("b", vec![0.0, 1.0])]).unwrap())
    };
    std::thread::sleep(Duration::from_millis(50));

    let started = Instant::now();
    let read = tokio::spawn(
        app.clone().oneshot(
            Request::get("/collections/c/points/a")
                .body(Body::empty())
                .unwrap(),
        ),
    );
    tokio::task::yield_now().await;
    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let waited = started.elapsed();
    assert_eq!(health.status(), StatusCode::OK);
    assert!(
        waited < Duration::from_millis(250),
        "the runtime was blocked for {waited:?}"
    );

    assert_eq!(read.await.unwrap().unwrap().status(), StatusCode::OK);
    reader.join().unwrap();
    writer.join().unwrap();
}

fn hit_ids(body: &Value) -> Vec<&str> {
    body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_str().unwrap())
        .collect()
}

#[tokio::test]
async fn metadata_updates_by_ids_and_filter() {
    let app = TestApp::seeded().await;
    let (status, body) = app
        .post(
            "/collections/books/points/metadata",
            json!({"ids": ["a", "nope"], "metadata": {"year": 1979, "genre": null}}),
        )
        .await;
    assert_eq!((status, &body), (StatusCode::OK, &json!({"updated": 1})));
    let (_, body) = app.get("/collections/books/points/a").await;
    assert_eq!(body["metadata"], json!({"year": 1979}));

    let (_, body) = app
        .post(
            "/collections/books/points/metadata",
            json!({"filter": {"genre": "sf"}, "metadata": {"shelf": 2}, "mode": "replace"}),
        )
        .await;
    assert_eq!(body["updated"], 1, "only c is still sf");
    let (_, body) = app.get("/collections/books/points/c").await;
    assert_eq!(body["metadata"], json!({"shelf": 2}));

    for bad in [
        json!({"metadata": {}}),
        json!({"ids": ["a"], "filter": {"x": 1}, "metadata": {}}),
        json!({"ids": ["a"]}),
        json!({"ids": ["a"], "metadata": {}, "mode": "upsert"}),
        json!({"ids": ["a"], "metadata": {}, "extra": 1}),
    ] {
        let (status, body) = app.post("/collections/books/points/metadata", bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad} -> {body}");
    }
}

#[tokio::test]
async fn count_and_delete_by_filter() {
    let app = TestApp::seeded().await;
    let (status, body) = app.post("/collections/books/points/count", json!({})).await;
    assert_eq!((status, body["count"].clone()), (StatusCode::OK, json!(3)));
    let (_, body) = app
        .post(
            "/collections/books/points/count",
            json!({"filter": {"genre": "sf"}}),
        )
        .await;
    assert_eq!(body["count"], 2);

    let (status, body) = app
        .post("/collections/books/points/delete", json!({"filter": {}}))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("drop the collection")
    );
    for bad in [json!({}), json!({"ids": ["a"], "filter": {"genre": "sf"}})] {
        let (status, _) = app.post("/collections/books/points/delete", bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    let (status, body) = app
        .post(
            "/collections/books/points/delete",
            json!({"filter": {"genre": "sf"}}),
        )
        .await;
    assert_eq!((status, body), (StatusCode::OK, json!({"deleted": 2})));
    let (_, body) = app.post("/collections/books/points/count", json!({})).await;
    assert_eq!(body["count"], 1);
}

#[tokio::test]
async fn search_by_id_recommend_and_max_distance() {
    let app = TestApp::seeded().await;
    let (status, body) = app
        .post("/collections/books/search", json!({"id": "a", "k": 5}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(hit_ids(&body), ["b", "c"], "the point itself is left out");

    let (_, body) = app
        .post(
            "/collections/books/search",
            json!({"vector": [0.0, 0.0], "k": 5, "max_distance": 1.5}),
        )
        .await;
    assert_eq!(hit_ids(&body), ["a", "b"]);

    let (status, body) = app.post("/collections/books/search", json!({"id": "nope"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "point_not_found");
    for bad in [json!({}), json!({"id": "a", "vector": [0.0, 0.0]})] {
        let (status, _) = app.post("/collections/books/search", bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    let (_, body) = app
        .post(
            "/collections/books/search/batch",
            json!({"searches": [{"id": "c", "k": 1}, {"vector": [5.0, 5.0], "k": 1}]}),
        )
        .await;
    assert_eq!(body["results"][0][0]["id"], "b");
    assert_eq!(body["results"][1][0]["id"], "c");

    let (status, body) = app
        .post("/collections/books/recommend", json!({"positive": ["a"], "k": 1}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(hit_ids(&body), ["b"]);
    // Both examples are left out, so only c can be recommended.
    let (_, body) = app
        .post(
            "/collections/books/recommend",
            json!({"positive": ["a", "b"], "negative": [], "k": 3, "with_vectors": true}),
        )
        .await;
    assert_eq!(hit_ids(&body), ["c"]);
    assert_eq!(body["results"][0]["vector"], json!([5.0, 5.0]));
    let (status, body) = app
        .post(
            "/collections/books/recommend",
            json!({"positive": ["a"], "negative": ["gone"]}),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "point_not_found");
    assert!(body["error"]["message"].as_str().unwrap().contains("gone"));
    let (status, _) = app
        .post("/collections/books/recommend", json!({"positive": []}))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let too_many = vec!["a"; 10_001];
    let (status, _) = app
        .post(
            "/collections/books/recommend",
            json!({"positive": too_many[..5_001], "negative": too_many[..5_000]}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn read_only_keys_can_count_and_recommend_but_not_change_points() {
    let mut app = TestApp::with_options(HttpOptions {
        keys: keys(),
        ..Default::default()
    });
    app.key = Some("rw-secret");
    app.post("/collections", json!({"name": "x", "dimension": 2}))
        .await;
    app.post(
        "/collections/x/points",
        json!({"points": [{"id": "a", "vector": [1.0, 0.0]}, {"id": "b", "vector": [0.0, 1.0]}]}),
    )
    .await;

    app.key = Some("ro-secret");
    let (status, _) = app.post("/collections/x/points/count", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = app
        .post("/collections/x/recommend", json!({"positive": ["a"]}))
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = app.post("/collections/x/search", json!({"id": "a"})).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = app
        .post(
            "/collections/x/points/metadata",
            json!({"ids": ["a"], "metadata": {"k": 1}}),
        )
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = app
        .post("/collections/x/points/delete", json!({"filter": {"k": 1}}))
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn metrics_count_requests_by_route_template() {
    let app = TestApp::seeded().await;
    app.post("/collections/books/search", json!({"vector": [1.0, 0.0], "k": 1}))
        .await;
    app.get("/collections/books/points/nope").await;
    app.get("/no/such/route").await;

    let res = app
        .raw(Request::get("/metrics").body(Body::empty()).unwrap())
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(
        res.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/plain; version=0.0.4")
    );
    let text = String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
    for line in [
        "quiver_collection_points{collection=\"books\"}",
        "quiver_requests_total{api=\"http\",route=\"/collections/{name}/search\",status=\"200\"} 1",
        "quiver_requests_total{api=\"http\",route=\"/collections/{name}/points/{id}\",status=\"404\"} 1",
        "quiver_requests_total{api=\"http\",route=\"unmatched\",status=\"404\"} 1",
    ] {
        assert!(text.contains(line), "missing `{line}` in:\n{text}");
    }
    assert!(!text.contains("/nope"), "raw paths must not become labels");
}

#[tokio::test]
async fn metrics_need_a_key_when_auth_is_on() {
    let keys = KeyStore::new(vec![ApiKey {
        name: "viewer".into(),
        role: Role::ReadOnly,
        key: "ro".into(),
    }])
    .unwrap();
    let mut app = TestApp::with_options(HttpOptions {
        keys: Arc::new(keys),
        ..Default::default()
    });
    assert_eq!(
        app.call(Method::GET, "/metrics", None).await.0,
        StatusCode::UNAUTHORIZED
    );
    app.key = Some("ro");
    let res = app
        .raw(
            Request::get("/metrics")
                .header("authorization", "Bearer ro")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn field_indexes_over_rest() {
    let app = TestApp::new();
    let (status, body) = app
        .post(
            "/collections",
            json!({"name": "docs", "dimension": 2, "field_indexes": ["genre"]}),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["field_indexes"], json!(["genre"]));

    let (status, body) = app
        .post("/collections/docs/indexes", json!({"field": "stats.views"}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["field_indexes"], json!(["genre", "stats.views"]));
    let (status, _) = app
        .post("/collections/docs/indexes", json!({"field": "$x"}))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, body) = app
        .call(Method::DELETE, "/collections/docs/indexes/stats.views", None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["field_indexes"], json!(["genre"]));
    let (status, _) = app
        .call(Method::DELETE, "/collections/docs/indexes/stats.views", None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn quantized_collections_over_rest() {
    let app = TestApp::new();
    let (status, body) = app
        .post(
            "/collections",
            json!({"name": "q", "dimension": 2, "metric": "euclidean", "quantization": "scalar"}),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["quantization"], "scalar");
    let (status, _) = app
        .post(
            "/collections/q/points",
            json!({"points": [{"id": "a", "vector": [0.25, -1.0]}]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let (_, point) = app.get("/collections/q/points/a").await;
    let v = point["vector"].as_array().unwrap();
    assert!(
        (v[0].as_f64().unwrap() - 0.25).abs() < 0.01 && v[1] == -1.0,
        "{point}"
    );

    // Unquantized collections do not mention it.
    let (_, body) = app
        .post("/collections", json!({"name": "plain", "dimension": 2}))
        .await;
    assert!(body.get("quantization").is_none(), "{body}");

    for bad in [
        json!({"name": "h", "dimension": 2, "metric": "hamming", "quantization": "scalar"}),
        json!({"name": "k", "dimension": 2, "metric": "euclidean", "index": {"type": "kd_tree"}, "quantization": "scalar"}),
        json!({"name": "hb", "dimension": 2, "metric": "hamming", "quantization": "binary"}),
        json!({"name": "x", "dimension": 2, "quantization": "ternary"}),
    ] {
        let (status, body) = app.post("/collections", bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}

#[tokio::test]
async fn binary_quantization_and_expiry_over_rest() {
    let app = TestApp::new();
    let (status, body) = app
        .post(
            "/collections",
            json!({
                "name": "e",
                "dimension": 2,
                "metric": "euclidean",
                "quantization": "binary",
                "expiry": {"field": "exp", "default_ttl": 60},
            }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let (_, body) = app.get("/collections/e").await;
    assert_eq!(body["quantization"], "binary");
    assert_eq!(body["expiry"], json!({"field": "exp", "default_ttl": 60}));
    assert_eq!(body["field_indexes"], json!(["exp"]));

    let (status, body) = app
        .post(
            "/collections/e/points",
            json!({"points": [
                {"id": "gone", "vector": [0.25, -1.0], "metadata": {"exp": 1000}},
                {"id": "kept", "vector": [0.5, 1.0], "metadata": {"exp": 100_000_000_000u64}},
                {"id": "ttl", "vector": [1.0, 1.0]},
            ]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = app.post("/collections/e/points/count", json!({})).await;
    assert_eq!(body["count"], 2, "{body}");
    let (status, _) = app.get("/collections/e/points/gone").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, point) = app.get("/collections/e/points/kept").await;
    assert_eq!(
        point["vector"],
        json!([0.5, 1.0]),
        "binary codes keep the vectors exact"
    );
    let (_, point) = app.get("/collections/e/points/ttl").await;
    assert!(point["metadata"]["exp"].as_u64().is_some(), "{point}");
    let (_, body) = app
        .post("/collections/e/search", json!({"vector": [0.25, -1.0], "k": 10}))
        .await;
    let ids: Vec<&str> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_str().unwrap())
        .collect();
    assert!(!ids.contains(&"gone") && ids.len() == 2, "{body}");

    let (status, body) = app
        .post(
            "/collections/e/points",
            json!({"points": [{"id": "bad", "vector": [1.0, 0.0], "metadata": {"exp": "tomorrow"}}]}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = app
        .call(Method::PATCH, "/collections/e", Some(json!({"expiry": null})))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("expiry").is_none(), "{body}");
    assert_eq!(body["count"], 3, "unpurged points come back");
}

async fn hybrid_app() -> TestApp {
    let app = TestApp::new();
    let (status, body) = app
        .post(
            "/collections",
            json!({
                "name": "docs",
                "dimension": 2,
                "metric": "euclidean",
                "vectors": {"title": {"dimension": 3, "metric": "euclidean", "index": {"type": "flat"}}},
                "sparse_vectors": {"text": {"modifier": "idf"}},
            }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["vectors"]["title"]["dimension"], 3);
    assert_eq!(body["sparse_vectors"]["text"]["modifier"], "idf");
    let (status, body) = app
        .post(
            "/collections/docs/points",
            json!({"points": [
                {"id": "a", "vector": [0.0, 0.0], "vectors": {"title": [1.0, 0.0, 0.0]},
                 "sparse_vectors": {"text": {"indices": [7, 1], "values": [1.0, 2.0]}}},
                {"id": "b", "vector": [1.0, 0.0], "vectors": {"title": [0.0, 1.0, 0.0]},
                 "sparse_vectors": {"text": {"indices": [2], "values": [1.0]}}},
                {"id": "c", "vector": [5.0, 5.0], "vectors": {"title": [0.0, 0.0, 1.0]}}
            ]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    app
}

#[tokio::test]
async fn named_and_sparse_vectors() {
    let app = hybrid_app().await;

    let (status, body) = app
        .post(
            "/collections/docs/search",
            json!({"vector": [0.0, 1.0, 0.0], "using": "title", "k": 1}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"][0]["id"], "b");

    let (status, body) = app
        .post(
            "/collections/docs/search",
            json!({"sparse": {"indices": [1], "values": [1.0]}, "using": "text", "with_vectors": true}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let hits = body["results"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["id"], "a");
    assert!(hits[0]["distance"].as_f64().unwrap() < 0.0, "negated score");
    assert_eq!(
        hits[0]["sparse_vectors"]["text"]["indices"],
        json!([1, 7]),
        "sorted"
    );
    assert_eq!(hits[0]["vectors"]["title"], json!([1.0, 0.0, 0.0]));

    let (status, body) = app.get("/collections/docs/points/a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["vectors"]["title"], json!([1.0, 0.0, 0.0]));
    let (_, body) = app.get("/collections/docs/points/a?with_vectors=false").await;
    assert!(body.get("vectors").is_none() && body.get("sparse_vectors").is_none());

    let (status, body) = app
        .post(
            "/collections/docs/recommend",
            json!({"positive": ["a"], "using": "title", "k": 1}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    for bad in [
        json!({"sparse": {"indices": [1], "values": [1.0]}}),
        json!({"sparse": {"indices": [1], "values": [1.0]}, "vector": [0.0, 0.0], "using": "text"}),
        json!({"vector": [0.0, 0.0], "using": "nope"}),
        json!({"vector": [0.0, 0.0], "using": "title"}),
    ] {
        let (status, body) = app.post("/collections/docs/search", bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
    let (status, body) = app
        .post(
            "/collections/docs/points",
            json!({"points": [{"id": "d", "vector": [0.0, 0.0]}]}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "missing named vector: {body}");
}

#[tokio::test]
async fn hybrid_query() {
    let app = hybrid_app().await;
    let (status, body) = app
        .post(
            "/collections/docs/query",
            json!({
                "prefetch": [
                    {"vector": [5.0, 5.0], "limit": 3},
                    {"sparse": {"indices": [2], "values": [1.0]}, "using": "text", "weight": 2.0},
                ],
                "fusion": {"type": "rrf", "k": 1},
                "k": 2,
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let hits = body["results"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    // b: 2/(1+1) from the keyword search plus 1/(1+2) from the dense one.
    assert_eq!(hits[0]["id"], "b");
    assert!((hits[0]["score"].as_f64().unwrap() - (1.0 + 1.0 / 3.0)).abs() < 1e-6);
    assert_eq!(hits[1]["id"], "c");

    for bad in [
        json!({"prefetch": []}),
        json!({"prefetch": [{"sparse": {"indices": [1], "values": [1.0]}}]}),
        json!({"prefetch": [{"vector": [0.0, 0.0], "sparse": {"indices": [1], "values": [1.0]}}]}),
        json!({"prefetch": [{"vector": [0.0, 0.0]}], "fusion": {"type": "magic"}}),
        json!({"prefetch": [{"vector": [0.0, 0.0], "weight": -1.0}]}),
        json!({"prefetch": [{"vector": [0.0, 0.0], "limit": 0}]}),
    ] {
        let (status, body) = app.post("/collections/docs/query", bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
}

#[tokio::test]
async fn update_rename_and_aliases() {
    let app = TestApp::seeded().await;
    let (status, body) = app
        .call(
            Method::PATCH,
            "/collections/books",
            Some(json!({"index": {"type": "flat"}, "quantization": "scalar"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["index"]["type"], "flat");
    assert_eq!(body["quantization"], "scalar");
    let (status, body) = app
        .call(Method::PATCH, "/collections/books", Some(json!({"dimension": 3})))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_body");
    let (status, _) = app
        .call(
            Method::PATCH,
            "/collections/books",
            Some(json!({"on_disk": true})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "in-memory database");

    let (status, body) = app
        .call(
            Method::PUT,
            "/aliases/library",
            Some(json!({"collection": "books"})),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = app
        .post("/collections/books/rename", json!({"name": "novels"}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["name"], "novels");
    assert_eq!(body["count"], 3);

    let (_, body) = app.get("/aliases").await;
    assert_eq!(
        body["aliases"],
        json!([{"alias": "library", "collection": "novels"}])
    );
    let (status, body) = app.get("/collections/library").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["name"], "novels");
    let (status, body) = app
        .post(
            "/collections/library/search",
            json!({"vector": [0.0, 0.0], "k": 1}),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"][0]["id"], "a");

    let (status, body) = app
        .post("/collections", json!({"name": "library", "dimension": 2}))
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let (status, _) = app
        .call(Method::PUT, "/aliases/x", Some(json!({"collection": "missing"})))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = app.call(Method::DELETE, "/aliases/library", None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, body) = app.call(Method::DELETE, "/aliases/library", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
}

#[tokio::test]
async fn readonly_keys_cannot_change_settings_or_aliases() {
    let keys = KeyStore::new(vec![ApiKey {
        name: "ro".into(),
        role: Role::ReadOnly,
        key: "ro".into(),
    }])
    .unwrap();
    let mut app = TestApp::with_options(HttpOptions {
        keys: Arc::new(keys),
        ..Default::default()
    });
    app.key = Some("ro");
    for (method, uri) in [
        (Method::PATCH, "/collections/x"),
        (Method::POST, "/collections/x/rename"),
        (Method::PUT, "/aliases/a"),
        (Method::DELETE, "/aliases/a"),
    ] {
        let (status, _) = app.call(method.clone(), uri, Some(json!({}))).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri}");
    }
    let (status, _) = app.get("/aliases").await;
    assert_eq!(status, StatusCode::OK);
}
