//! Async HTTP client for a Quiver server.
//!
//! ```no_run
//! # async fn run() -> Result<(), quiver_client::ClientError> {
//! use quiver_client::{Client, SearchRequest};
//!
//! let client = Client::new("http://localhost:6900")?.with_api_key("secret");
//! let hits = client
//!     .search("movies", &SearchRequest::new(vec![0.1, 0.2, 0.3], 5))
//!     .await?;
//! # Ok(())
//! # }
//! ```

use quiver_types::snapshot::SnapshotInfo;
use quiver_types::{CollectionConfig, CollectionInfo, Point, PointInput, SearchHit};
use reqwest::{Method, RequestBuilder, StatusCode, Url};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

/// Errors returned by [`Client`].
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// The server answered with an error body.
    #[error("{message} ({status}, {code})")]
    Api {
        status: StatusCode,
        code: String,
        message: String,
    },
    /// The request never got a usable answer (connection, timeout, decoding).
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// The base URL could not be parsed.
    #[error("invalid server URL `{0}`")]
    InvalidUrl(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// Overall limit for one request, matching the Python client.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Server-wide statistics from `GET /info`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerInfo {
    pub version: String,
    pub storage: String,
    pub snapshots_enabled: bool,
    pub collections: usize,
    pub points: usize,
}

/// One page of [`Client::scroll`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ScrollPage {
    /// Points in ascending id order.
    pub points: Vec<Point>,
    /// Cursor for the next page; `None` at the end.
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// Result of [`Client::upsert`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct UpsertResult {
    pub upserted: usize,
    /// Ids of the written points, in request order (generated ones included).
    pub ids: Vec<String>,
}

/// A nearest-neighbour query.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    /// Metadata filter, in the server's JSON filter syntax.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    /// HNSW search breadth; ignored by other indexes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef: Option<usize>,
    pub with_vectors: bool,
    pub with_metadata: bool,
}

impl SearchRequest {
    /// Top-`k` search returning metadata but not vectors.
    pub fn new(vector: Vec<f32>, k: usize) -> Self {
        Self {
            vector,
            k,
            filter: None,
            ef: None,
            with_vectors: false,
            with_metadata: true,
        }
    }
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

/// A handle to a Quiver server. Cheap to clone.
#[derive(Debug, Clone)]
pub struct Client {
    base: Url,
    http: reqwest::Client,
    api_key: Option<String>,
}

impl Client {
    /// Creates a client for the server at `base_url`, e.g. `http://localhost:6900`.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self> {
        Self::with_timeout(base_url.as_ref(), REQUEST_TIMEOUT)
    }

    fn with_timeout(raw: &str, timeout: Duration) -> Result<Self> {
        // A trailing slash makes `Url::join` append rather than replace the last segment.
        let base = Url::parse(&format!("{}/", raw.trim_end_matches('/')))
            .map_err(|_| ClientError::InvalidUrl(raw.to_owned()))?;
        if !matches!(base.scheme(), "http" | "https") {
            return Err(ClientError::InvalidUrl(raw.to_owned()));
        }
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(timeout)
            .build()?;
        Ok(Self {
            base,
            http,
            api_key: None,
        })
    }

    /// Sends `key` with every request. An empty key disables authentication.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.api_key = (!key.is_empty()).then_some(key);
        self
    }

    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        let url = self.base.join(path).expect("relative paths always join");
        let req = self.http.request(method, url);
        match &self.api_key {
            Some(key) => req.bearer_auth(key),
            None => req,
        }
    }

    async fn send(req: RequestBuilder) -> Result<reqwest::Response> {
        let res = req.send().await?;
        let status = res.status();
        if status.is_success() {
            return Ok(res);
        }
        let text = res.text().await.unwrap_or_default();
        let (code, message) = match serde_json::from_str::<ErrorBody>(&text) {
            Ok(body) => (body.error.code, body.error.message),
            Err(_) => (status.as_str().to_owned(), text),
        };
        Err(ClientError::Api {
            status,
            code,
            message,
        })
    }

    async fn json<T: DeserializeOwned>(req: RequestBuilder) -> Result<T> {
        Ok(Self::send(req).await?.json().await?)
    }

    /// Like [`Self::json`], but extracts one field of the response object.
    async fn field<T: DeserializeOwned>(req: RequestBuilder, name: &str) -> Result<T> {
        let mut body: Value = Self::json(req).await?;
        let value = body.get_mut(name).map(Value::take).unwrap_or(Value::Null);
        serde_json::from_value(value).map_err(|e| ClientError::Api {
            status: StatusCode::OK,
            code: "invalid_response".into(),
            message: format!("unexpected `{name}` in response: {e}"),
        })
    }

    pub async fn info(&self) -> Result<ServerInfo> {
        Self::json(self.request(Method::GET, "info")).await
    }

    pub async fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        Self::field(self.request(Method::GET, "collections"), "collections").await
    }

    pub async fn get_collection(&self, name: &str) -> Result<CollectionInfo> {
        Self::json(self.request(Method::GET, &format!("collections/{}", seg(name)))).await
    }

    pub async fn create_collection(&self, name: &str, config: &CollectionConfig) -> Result<CollectionInfo> {
        let mut body = serde_json::to_value(config).expect("configs serialize");
        body["name"] = json!(name);
        Self::json(self.request(Method::POST, "collections").json(&body)).await
    }

    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        Self::send(self.request(Method::DELETE, &format!("collections/{}", seg(name)))).await?;
        Ok(())
    }

    pub async fn upsert(&self, collection: &str, points: Vec<PointInput>) -> Result<UpsertResult> {
        let path = format!("collections/{}/points", seg(collection));
        Self::json(
            self.request(Method::POST, &path)
                .json(&json!({ "points": points })),
        )
        .await
    }

    /// Fetches one point with its vector, or `None` if it does not exist.
    pub async fn get_point(&self, collection: &str, id: &str) -> Result<Option<Point>> {
        if id == "." || id == ".." {
            // URLs resolve these as dot-segments, even percent-encoded
            // (`%2E%2E`), so they cannot appear in a path; look them up by body.
            let path = format!("collections/{}/points/lookup", seg(collection));
            let req = self.request(Method::POST, &path).json(&json!({ "ids": [id] }));
            let mut points: Vec<Point> = Self::field(req, "points").await?;
            return Ok(points.pop());
        }
        let path = format!("collections/{}/points/{}", seg(collection), seg(id));
        match Self::json(self.request(Method::GET, &path)).await {
            Ok(point) => Ok(Some(point)),
            // Only a missing point; a missing collection is still an error.
            Err(ClientError::Api { ref code, .. }) if code == "point_not_found" => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Lists up to `limit` points with ids after `cursor` (from the start if
    /// `None`), in ascending id order. Follow `next_cursor` for the next page.
    pub async fn scroll(
        &self,
        collection: &str,
        cursor: Option<&str>,
        limit: usize,
        with_vectors: bool,
    ) -> Result<ScrollPage> {
        let path = format!("collections/{}/points", seg(collection));
        let mut query = vec![
            ("limit", limit.to_string()),
            ("with_vectors", with_vectors.to_string()),
        ];
        if let Some(cursor) = cursor {
            query.push(("cursor", cursor.to_owned()));
        }
        Self::json(self.request(Method::GET, &path).query(&query)).await
    }

    /// Deletes points by id and returns how many existed.
    pub async fn delete_points(&self, collection: &str, ids: &[String]) -> Result<usize> {
        let path = format!("collections/{}/points/delete", seg(collection));
        Self::field(
            self.request(Method::POST, &path).json(&json!({ "ids": ids })),
            "deleted",
        )
        .await
    }

    pub async fn search(&self, collection: &str, req: &SearchRequest) -> Result<Vec<SearchHit>> {
        let path = format!("collections/{}/search", seg(collection));
        Self::field(self.request(Method::POST, &path).json(req), "results").await
    }

    pub async fn create_snapshot(&self, collection: &str) -> Result<SnapshotInfo> {
        let path = format!("collections/{}/snapshots", seg(collection));
        Self::json(self.request(Method::POST, &path)).await
    }

    /// Lists snapshots, optionally only those of one collection.
    pub async fn list_snapshots(&self, collection: Option<&str>) -> Result<Vec<SnapshotInfo>> {
        let mut req = self.request(Method::GET, "snapshots");
        if let Some(c) = collection {
            req = req.query(&[("collection", c)]);
        }
        Self::field(req, "snapshots").await
    }

    /// Restores a snapshot, under `name` if given, else its original name.
    pub async fn restore_snapshot(
        &self,
        id: &str,
        name: Option<&str>,
        replace: bool,
    ) -> Result<CollectionInfo> {
        let path = format!("snapshots/{}/restore", seg(id));
        let body = json!({ "collection": name, "replace": replace });
        Self::json(self.request(Method::POST, &path).json(&body)).await
    }

    pub async fn delete_snapshot(&self, id: &str) -> Result<()> {
        Self::send(self.request(Method::DELETE, &format!("snapshots/{}", seg(id)))).await?;
        Ok(())
    }
}

/// Percent-encodes one path segment.
fn seg(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_urls() {
        assert!(Client::new("not a url").is_err());
        assert!(Client::new("ftp://example.com").is_err());
        assert!(Client::new("http://localhost:6900/").is_ok());
    }

    #[test]
    fn paths_keep_a_base_prefix() {
        let c = Client::new("http://example.com/quiver").unwrap();
        let url = c.base.join("collections").unwrap();
        assert_eq!(url.as_str(), "http://example.com/quiver/collections");
    }

    #[test]
    fn empty_api_key_disables_auth() {
        let c = Client::new("http://localhost").unwrap().with_api_key("");
        assert!(c.api_key.is_none());
    }

    #[test]
    fn segments_are_escaped() {
        assert_eq!(seg("a b/c"), "a%20b%2Fc");
        assert_eq!(seg("plain-id_1.x"), "plain-id_1.x");
    }

    #[tokio::test]
    async fn stalled_requests_time_out() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        // Accept connections but never answer.
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                held.push(socket);
            }
        });
        let c = Client::with_timeout(&url, Duration::from_millis(200)).unwrap();
        let res = tokio::time::timeout(Duration::from_secs(10), c.info())
            .await
            .expect("request should time out on its own");
        assert!(matches!(res, Err(ClientError::Http(e)) if e.is_timeout()));
    }
}
