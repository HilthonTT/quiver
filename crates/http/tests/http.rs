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
    assert_eq!(body["next_offset"], 2);
    let (_, body) = app.get("/collections/books/points?offset=2&limit=2").await;
    assert_eq!(body["points"].as_array().unwrap().len(), 1);
    assert_eq!(body["next_offset"], Value::Null);

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
