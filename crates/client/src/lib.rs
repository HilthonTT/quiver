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
pub use quiver_types::{
    CollectionUpdate, ContextPair, Direction, Example, FacetCount, Fusion, HitGroup, MetadataMode, OrderBy,
    Rescore, ScoredHit, SparseVector, VectorUpdate,
};
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
    /// [`TlsOptions`] holds no usable certificate.
    #[error("invalid TLS settings: {0}")]
    Tls(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// Point ids that `GET collections/{name}/points/{id}` cannot address.
const RESERVED_POINT_IDS: [&str; 8] = [
    ".", "..", "lookup", "delete", "count", "metadata", "scroll", "facet",
];

/// Overall limit for one request, matching the Python client.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Limit for snapshot downloads and uploads, which move whole archives.
const TRANSFER_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// TLS settings for [`Client::with_tls`].
#[derive(Clone, Default)]
pub struct TlsOptions {
    /// PEM certificates to trust in addition to the public roots, e.g. the
    /// CA that signed the server's certificate.
    pub ca_pem: Option<Vec<u8>>,
    /// A PEM client certificate chain followed by its private key, for
    /// servers started with `--tls-client-ca`.
    pub identity_pem: Option<Vec<u8>>,
}

impl std::fmt::Debug for TlsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsOptions")
            .field("ca_pem", &self.ca_pem.is_some())
            .field("identity_pem", &self.identity_pem.is_some())
            .finish()
    }
}

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

/// A nearest-neighbour query, by vector or by an existing point's id.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchRequest {
    /// The query vector; leave empty when searching by `id` or `sparse`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vector: Vec<f32>,
    /// Search with this point's stored vector, leaving the point itself out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Search the sparse vector named by `using` with this one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseVector>,
    /// The named vector to search; the main one if `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using: Option<String>,
    pub k: usize,
    /// Metadata filter, in the server's JSON filter syntax.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    /// HNSW search breadth; ignored by other indexes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef: Option<usize>,
    /// Drop hits farther than this; with HNSW it applies to the approximate top `k`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
    #[serde(skip_serializing_if = "is_zero")]
    pub offset: usize,
    pub with_vectors: bool,
    pub with_metadata: bool,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl SearchRequest {
    /// Top-`k` search returning metadata but not vectors.
    pub fn new(vector: Vec<f32>, k: usize) -> Self {
        Self {
            vector,
            id: None,
            sparse: None,
            using: None,
            k,
            filter: None,
            ef: None,
            max_distance: None,
            offset: 0,
            with_vectors: false,
            with_metadata: true,
        }
    }

    /// The `k` points nearest to point `id`, which is left out.
    pub fn by_id(id: impl Into<String>, k: usize) -> Self {
        Self {
            id: Some(id.into()),
            ..Self::new(Vec::new(), k)
        }
    }

    /// The `k` points scoring highest against `vector` in the sparse vector
    /// `using`. Hits' distances are their negated scores.
    pub fn sparse(using: impl Into<String>, vector: SparseVector, k: usize) -> Self {
        Self {
            sparse: Some(vector),
            using: Some(using.into()),
            ..Self::new(Vec::new(), k)
        }
    }

    /// Searches the named dense vector `name` instead of the main one.
    #[must_use]
    pub fn using(mut self, name: impl Into<String>) -> Self {
        self.using = Some(name.into());
        self
    }

    #[must_use]
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Value) -> Self {
        self.filter = Some(filter);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupSearchRequest {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseVector>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using: Option<String>,
    pub group_by: String,
    pub groups: usize,
    pub group_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
    pub with_vectors: bool,
    pub with_metadata: bool,
}

impl GroupSearchRequest {
    pub fn new(vector: Vec<f32>, group_by: impl Into<String>, groups: usize, group_size: usize) -> Self {
        Self {
            vector,
            id: None,
            sparse: None,
            using: None,
            group_by: group_by.into(),
            groups,
            group_size,
            filter: None,
            ef: None,
            max_distance: None,
            with_vectors: false,
            with_metadata: true,
        }
    }

    pub fn by_id(
        id: impl Into<String>,
        group_by: impl Into<String>,
        groups: usize,
        group_size: usize,
    ) -> Self {
        Self {
            id: Some(id.into()),
            ..Self::new(Vec::new(), group_by, groups, group_size)
        }
    }

    pub fn sparse(
        using: impl Into<String>,
        vector: SparseVector,
        group_by: impl Into<String>,
        groups: usize,
        group_size: usize,
    ) -> Self {
        Self {
            sparse: Some(vector),
            using: Some(using.into()),
            ..Self::new(Vec::new(), group_by, groups, group_size)
        }
    }

    #[must_use]
    pub fn using(mut self, name: impl Into<String>) -> Self {
        self.using = Some(name.into());
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Value) -> Self {
        self.filter = Some(filter);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiscoverRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Example>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<ContextPair>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using: Option<String>,
    pub k: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub offset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
    pub with_vectors: bool,
    pub with_metadata: bool,
}

impl DiscoverRequest {
    pub fn new(target: Option<Example>, context: Vec<ContextPair>, k: usize) -> Self {
        Self {
            target,
            context,
            using: None,
            k,
            offset: 0,
            filter: None,
            ef: None,
            max_distance: None,
            with_vectors: false,
            with_metadata: true,
        }
    }

    #[must_use]
    pub fn using(mut self, name: impl Into<String>) -> Self {
        self.using = Some(name.into());
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Value) -> Self {
        self.filter = Some(filter);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScrollRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    pub limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_by: Option<OrderBy>,
    pub with_vectors: bool,
}

impl ScrollRequest {
    pub fn new(limit: usize) -> Self {
        Self {
            cursor: None,
            limit,
            filter: None,
            order_by: None,
            with_vectors: false,
        }
    }

    #[must_use]
    pub fn cursor(mut self, cursor: impl Into<String>) -> Self {
        self.cursor = Some(cursor.into());
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Value) -> Self {
        self.filter = Some(filter);
        self
    }

    #[must_use]
    pub fn order_by(mut self, order_by: OrderBy) -> Self {
        self.order_by = Some(order_by);
        self
    }

    #[must_use]
    pub fn with_vectors(mut self, yes: bool) -> Self {
        self.with_vectors = yes;
        self
    }
}

/// Several searches whose rankings are fused into one (hybrid search).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryRequest {
    pub prefetch: Vec<Prefetch>,
    pub fusion: Fusion,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescore: Option<Rescore>,
    pub k: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub offset: usize,
    /// Metadata filter applied to every search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    pub with_vectors: bool,
    pub with_metadata: bool,
}

impl QueryRequest {
    /// The `k` best points by reciprocal rank fusion of `prefetch`.
    pub fn new(prefetch: Vec<Prefetch>, k: usize) -> Self {
        Self {
            prefetch,
            fusion: Fusion::default(),
            rescore: None,
            k,
            offset: 0,
            filter: None,
            with_vectors: false,
            with_metadata: true,
        }
    }

    #[must_use]
    pub fn fusion(mut self, fusion: Fusion) -> Self {
        self.fusion = fusion;
        self
    }

    #[must_use]
    pub fn rescore(mut self, rescore: Rescore) -> Self {
        self.rescore = Some(rescore);
        self
    }

    #[must_use]
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

/// One search of a [`QueryRequest`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prefetch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseVector>,
    /// The named vector to search; required for `sparse`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using: Option<String>,
    /// How many of its best hits take part (the server's default: `max(k, 50)`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    pub weight: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
}

impl Prefetch {
    /// A search of the main dense vector (see [`using`](Self::using)).
    pub fn dense(vector: Vec<f32>) -> Self {
        Self {
            vector: Some(vector),
            sparse: None,
            using: None,
            limit: None,
            weight: 1.0,
            ef: None,
            max_distance: None,
        }
    }

    /// A search of the sparse vector `using`.
    pub fn sparse(using: impl Into<String>, vector: SparseVector) -> Self {
        Self {
            vector: None,
            sparse: Some(vector),
            using: Some(using.into()),
            ..Self::dense(Vec::new())
        }
    }

    /// Searches the named vector `name`.
    #[must_use]
    pub fn using(mut self, name: impl Into<String>) -> Self {
        self.using = Some(name.into());
        self
    }

    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    #[must_use]
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// An alternative name for a collection.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Alias {
    pub alias: String,
    pub collection: String,
}

/// Points like the `positive` examples and unlike the `negative` ones.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecommendRequest {
    pub positive: Vec<String>,
    /// The named vector to compare; the main one if `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub negative: Vec<String>,
    pub k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f32>,
    #[serde(skip_serializing_if = "is_zero")]
    pub offset: usize,
    pub with_vectors: bool,
    pub with_metadata: bool,
}

impl RecommendRequest {
    /// Top-`k` recommendations from `positive` examples, with metadata.
    pub fn new(positive: Vec<String>, k: usize) -> Self {
        Self {
            positive,
            using: None,
            negative: Vec::new(),
            k,
            filter: None,
            ef: None,
            max_distance: None,
            offset: 0,
            with_vectors: false,
            with_metadata: true,
        }
    }
}

/// Which points [`Client::update_metadata`] changes.
#[derive(Debug, Clone, PartialEq)]
pub enum Selector {
    /// These ids; missing ones are skipped.
    Ids(Vec<String>),
    /// Every point matching this filter (the server's JSON filter syntax).
    Filter(Value),
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

    /// Creates a client for an `https://` server whose certificate is signed
    /// by a private CA, or that wants a client certificate (mutual TLS).
    pub fn with_tls(base_url: impl AsRef<str>, tls: &TlsOptions) -> Result<Self> {
        Self::build(base_url.as_ref(), REQUEST_TIMEOUT, Some(tls))
    }

    fn with_timeout(raw: &str, timeout: Duration) -> Result<Self> {
        Self::build(raw, timeout, None)
    }

    fn build(raw: &str, timeout: Duration, tls: Option<&TlsOptions>) -> Result<Self> {
        // A trailing slash makes `Url::join` append rather than replace the last segment.
        let base = Url::parse(&format!("{}/", raw.trim_end_matches('/')))
            .map_err(|_| ClientError::InvalidUrl(raw.to_owned()))?;
        if !matches!(base.scheme(), "http" | "https") {
            return Err(ClientError::InvalidUrl(raw.to_owned()));
        }
        let mut http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(timeout);
        if let Some(tls) = tls {
            if let Some(pem) = &tls.ca_pem {
                let certs = reqwest::Certificate::from_pem_bundle(pem)?;
                if certs.is_empty() {
                    return Err(ClientError::Tls("no certificate in ca_pem".into()));
                }
                for cert in certs {
                    http = http.add_root_certificate(cert);
                }
            }
            if let Some(pem) = &tls.identity_pem {
                http = http.identity(reqwest::Identity::from_pem(pem)?);
            }
        }
        let http = http.build()?;
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

    /// Changes index, quantization or on-disk settings; may rebuild indexes.
    pub async fn update_collection(&self, name: &str, update: &CollectionUpdate) -> Result<CollectionInfo> {
        let path = format!("collections/{}", seg(name));
        Self::json(self.request(Method::PATCH, &path).json(update)).await
    }

    /// Renames a collection; its aliases follow it.
    pub async fn rename_collection(&self, name: &str, new_name: &str) -> Result<CollectionInfo> {
        let path = format!("collections/{}/rename", seg(name));
        Self::json(
            self.request(Method::POST, &path)
                .json(&json!({ "name": new_name })),
        )
        .await
    }

    pub async fn list_aliases(&self) -> Result<Vec<Alias>> {
        Self::field(self.request(Method::GET, "aliases"), "aliases").await
    }

    /// Creates `alias`, or atomically points it at `collection`.
    pub async fn set_alias(&self, alias: &str, collection: &str) -> Result<()> {
        let path = format!("aliases/{}", seg(alias));
        Self::send(
            self.request(Method::PUT, &path)
                .json(&json!({ "collection": collection })),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_alias(&self, alias: &str) -> Result<()> {
        let path = format!("aliases/{}", seg(alias));
        Self::send(self.request(Method::DELETE, &path)).await?;
        Ok(())
    }

    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        Self::send(self.request(Method::DELETE, &format!("collections/{}", seg(name)))).await?;
        Ok(())
    }

    /// Writes the collection's pending changes to disk.
    pub async fn flush(&self, name: &str) -> Result<()> {
        let path = format!("collections/{}/flush", seg(name));
        Self::send(self.request(Method::POST, &path)).await?;
        Ok(())
    }

    /// Drops deleted points from storage and rebuilds the index.
    pub async fn compact(&self, name: &str) -> Result<CollectionInfo> {
        let path = format!("collections/{}/compact", seg(name));
        Self::json(self.request(Method::POST, &path)).await
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
        if RESERVED_POINT_IDS.contains(&id) {
            // `GET points/{id}` cannot reach these: URLs resolve `.` and `..`
            // as dot-segments (even percent-encoded), and the server's fixed
            // `points/...` routes take the rest. Look them up by body instead.
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

    /// Indexes a metadata field (a dot-separated path, as in filters) so
    /// filters on it only look at matching points.
    pub async fn create_field_index(&self, collection: &str, field: &str) -> Result<CollectionInfo> {
        let path = format!("collections/{}/indexes", seg(collection));
        Self::json(self.request(Method::POST, &path).json(&json!({ "field": field }))).await
    }

    pub async fn drop_field_index(&self, collection: &str, field: &str) -> Result<CollectionInfo> {
        let path = format!("collections/{}/indexes/{}", seg(collection), seg(field));
        Self::json(self.request(Method::DELETE, &path)).await
    }

    /// Fetches several points; missing ids are skipped.
    pub async fn get_points(
        &self,
        collection: &str,
        ids: &[String],
        with_vectors: bool,
    ) -> Result<Vec<Point>> {
        let path = format!("collections/{}/points/lookup", seg(collection));
        let body = json!({ "ids": ids, "with_vectors": with_vectors });
        Self::field(self.request(Method::POST, &path).json(&body), "points").await
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

    pub async fn scroll_with(&self, collection: &str, req: &ScrollRequest) -> Result<ScrollPage> {
        let path = format!("collections/{}/points/scroll", seg(collection));
        Self::json(self.request(Method::POST, &path).json(req)).await
    }

    pub async fn facet(
        &self,
        collection: &str,
        field: &str,
        filter: Option<&Value>,
        limit: usize,
    ) -> Result<Vec<FacetCount>> {
        let path = format!("collections/{}/points/facet", seg(collection));
        let mut body = json!({ "field": field, "limit": limit });
        if let Some(f) = filter {
            body["filter"] = f.clone();
        }
        Self::field(self.request(Method::POST, &path).json(&body), "values").await
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

    /// Deletes every point matching `filter`, and returns how many there were.
    /// The server rejects an empty filter: drop the collection instead.
    pub async fn delete_where(&self, collection: &str, filter: &Value) -> Result<usize> {
        let path = format!("collections/{}/points/delete", seg(collection));
        Self::field(
            self.request(Method::POST, &path)
                .json(&json!({ "filter": filter })),
            "deleted",
        )
        .await
    }

    /// Counts the points, or those matching `filter`.
    pub async fn count(&self, collection: &str, filter: Option<&Value>) -> Result<usize> {
        let path = format!("collections/{}/points/count", seg(collection));
        let body = match filter {
            Some(f) => json!({ "filter": f }),
            None => json!({}),
        };
        Self::field(self.request(Method::POST, &path).json(&body), "count").await
    }

    /// Changes the selected points' metadata without resending vectors, and
    /// returns how many were updated.
    pub async fn update_metadata(
        &self,
        collection: &str,
        selector: &Selector,
        metadata: &Value,
        mode: MetadataMode,
    ) -> Result<usize> {
        let path = format!("collections/{}/points/metadata", seg(collection));
        let mut body = json!({ "metadata": metadata, "mode": mode });
        match selector {
            Selector::Ids(ids) => body["ids"] = json!(ids),
            Selector::Filter(f) => body["filter"] = f.clone(),
        }
        Self::field(self.request(Method::POST, &path).json(&body), "updated").await
    }

    pub async fn search(&self, collection: &str, req: &SearchRequest) -> Result<Vec<SearchHit>> {
        let path = format!("collections/{}/search", seg(collection));
        Self::field(self.request(Method::POST, &path).json(req), "results").await
    }

    /// Runs several searches in one request; results are in request order.
    pub async fn search_batch(
        &self,
        collection: &str,
        reqs: &[SearchRequest],
    ) -> Result<Vec<Vec<SearchHit>>> {
        let path = format!("collections/{}/search/batch", seg(collection));
        Self::field(
            self.request(Method::POST, &path)
                .json(&json!({ "searches": reqs })),
            "results",
        )
        .await
    }

    pub async fn search_groups(&self, collection: &str, req: &GroupSearchRequest) -> Result<Vec<HitGroup>> {
        let path = format!("collections/{}/search/groups", seg(collection));
        Self::field(self.request(Method::POST, &path).json(req), "groups").await
    }

    pub async fn discover(&self, collection: &str, req: &DiscoverRequest) -> Result<Vec<SearchHit>> {
        let path = format!("collections/{}/discover", seg(collection));
        Self::field(self.request(Method::POST, &path).json(req), "results").await
    }

    /// Recommends points from positive (and optionally negative) examples,
    /// which are left out of the results.
    pub async fn recommend(&self, collection: &str, req: &RecommendRequest) -> Result<Vec<SearchHit>> {
        let path = format!("collections/{}/recommend", seg(collection));
        Self::field(self.request(Method::POST, &path).json(req), "results").await
    }

    /// Hybrid search: fuses the rankings of several dense and sparse searches.
    pub async fn query(&self, collection: &str, req: &QueryRequest) -> Result<Vec<ScoredHit>> {
        let path = format!("collections/{}/query", seg(collection));
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

    pub async fn get_snapshot(&self, id: &str) -> Result<SnapshotInfo> {
        Self::json(self.request(Method::GET, &format!("snapshots/{}", seg(id)))).await
    }

    /// Downloads a snapshot archive (`.tar.gz`).
    pub async fn download_snapshot(&self, id: &str) -> Result<Vec<u8>> {
        let path = format!("snapshots/{}/download", seg(id));
        let req = self.request(Method::GET, &path).timeout(TRANSFER_TIMEOUT);
        Ok(Self::send(req).await?.bytes().await?.to_vec())
    }

    /// Uploads an archive from [`Self::download_snapshot`], possibly taken on
    /// another server. Restore it afterwards with [`Self::restore_snapshot`].
    pub async fn upload_snapshot(&self, archive: Vec<u8>) -> Result<SnapshotInfo> {
        let req = self
            .request(Method::POST, "snapshots")
            .header(reqwest::header::CONTENT_TYPE, "application/gzip")
            .timeout(TRANSFER_TIMEOUT)
            .body(archive);
        Self::json(req).await
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
