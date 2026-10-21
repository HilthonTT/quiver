use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, FromRequest, Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

use std::collections::BTreeMap;

use quiver_db::{
    Collection, CollectionConfig, CollectionInfo, CollectionUpdate, Database, Filter, Fusion, HybridQuery,
    IndexConfig, MetadataMode, Metric, Point, PointInput, PointSelector, Prefetch, Quantization, QueryVector,
    RestoreOptions, SearchHit, SearchOptions, SnapshotInfo, SparseVector, SparseVectorConfig, VectorConfig,
};

use crate::error::ApiError;

/// Upper bound on `k` for a single search.
pub const MAX_K: usize = 10_000;
/// Upper bound on searches in one batch request.
pub const MAX_BATCH: usize = 256;
/// Upper bound on `limit` when scrolling.
pub const MAX_SCROLL: usize = 10_000;
/// Upper bound on ids in one lookup or delete request.
pub const MAX_IDS: usize = 10_000;
/// Upper bound on the searches a hybrid query fuses.
pub const MAX_PREFETCH: usize = 16;

/// Maximum snapshot upload size in bytes, attached to the upload route.
#[derive(Debug, Clone, Copy)]
pub struct UploadLimit(pub usize);

pub type AppState = Arc<Database>;
type ApiResult<T> = Result<T, ApiError>;

#[derive(Debug, FromRequest)]
#[from_request(via(axum::Json), rejection(ApiError))]
pub struct Json<T>(pub T);

impl<T: Serialize> IntoResponse for Json<T> {
    fn into_response(self) -> Response {
        axum::Json(self.0).into_response()
    }
}

async fn blocking<T, F>(f: F) -> ApiResult<T>
where
    F: FnOnce() -> Result<T, quiver_db::Error> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::internal(format!("task failed: {e}")))?
        .map_err(ApiError::from)
}

fn collection(db: &Database, name: &str) -> ApiResult<Arc<Collection>> {
    Ok(db.collection(name)?)
}

pub async fn root() -> Json<Value> {
    Json(json!({
        "name": "Quiver",
        "version": env!("CARGO_PKG_VERSION"),
        "docs": "https://github.com/HilthonTT/quiver#readme",
    }))
}

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

/// Server-wide statistics.
pub async fn info(State(db): State<AppState>) -> ApiResult<Json<Value>> {
    let collections = {
        let db = Arc::clone(&db);
        blocking(move || Ok(db.list_collections())).await?
    };
    let points: usize = collections.iter().map(|c| c.count).sum();
    Ok(Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "storage": db.options().storage,
        "snapshots_enabled": db.snapshots().is_ok(),
        "collections": collections.len(),
        "points": points,
    })))
}

/// Prometheus text-format metrics.
pub async fn metrics(State(db): State<AppState>) -> ApiResult<Response> {
    let text = {
        let db = Arc::clone(&db);
        blocking(move || Ok(db.metrics().render(&db))).await?
    };
    Ok((
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        text,
    )
        .into_response())
}

/// The fields of [`CollectionConfig`] spelled out (rather than flattened)
/// so that unknown or misspelt fields are rejected.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateCollection {
    name: String,
    dimension: usize,
    #[serde(default)]
    metric: Metric,
    #[serde(default)]
    index: IndexConfig,
    #[serde(default)]
    field_indexes: Vec<String>,
    #[serde(default)]
    quantization: Quantization,
    #[serde(default)]
    on_disk: bool,
    #[serde(default)]
    vectors: BTreeMap<String, VectorConfig>,
    #[serde(default)]
    sparse_vectors: BTreeMap<String, SparseVectorConfig>,
}

pub async fn list_collections(State(db): State<AppState>) -> ApiResult<Json<Value>> {
    let collections = blocking(move || Ok(db.list_collections())).await?;
    Ok(Json(json!({ "collections": collections })))
}

pub async fn create_collection(
    State(db): State<AppState>,
    Json(req): Json<CreateCollection>,
) -> ApiResult<(StatusCode, Json<CollectionInfo>)> {
    let config = CollectionConfig {
        dimension: req.dimension,
        metric: req.metric,
        index: req.index,
        quantization: req.quantization,
        on_disk: req.on_disk,
        vectors: req.vectors,
        sparse_vectors: req.sparse_vectors,
    };
    let info = blocking(move || {
        Ok(db
            .create_collection_with_fields(&req.name, config, &req.field_indexes)?
            .info())
    })
    .await?;
    tracing::info!(collection = %info.name, dimension = info.config.dimension, "created collection");
    Ok((StatusCode::CREATED, Json(info)))
}

pub async fn get_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<CollectionInfo>> {
    let c = collection(&db, &name)?;
    Ok(Json(blocking(move || Ok(c.info())).await?))
}

/// Changes index, quantization or on-disk settings; may rebuild indexes.
pub async fn update_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(update): Json<CollectionUpdate>,
) -> ApiResult<Json<CollectionInfo>> {
    let info = blocking(move || db.update_collection(&name, &update)).await?;
    tracing::info!(collection = %info.name, "updated collection settings");
    Ok(Json(info))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenameRequest {
    /// The new name.
    name: String,
}

pub async fn rename_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<RenameRequest>,
) -> ApiResult<Json<CollectionInfo>> {
    let info = blocking(move || Ok(db.rename_collection(&name, &req.name)?.info())).await?;
    Ok(Json(info))
}

pub async fn list_aliases(State(db): State<AppState>) -> Json<Value> {
    let aliases: Vec<Value> = db
        .aliases()
        .into_iter()
        .map(|(alias, collection)| json!({ "alias": alias, "collection": collection }))
        .collect();
    Json(json!({ "aliases": aliases }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasRequest {
    collection: String,
}

/// Creates the alias, or atomically points it at another collection.
pub async fn set_alias(
    State(db): State<AppState>,
    Path(alias): Path<String>,
    Json(req): Json<AliasRequest>,
) -> ApiResult<Json<Value>> {
    let response = json!({ "alias": alias, "collection": req.collection });
    blocking(move || db.set_alias(&alias, &req.collection)).await?;
    Ok(Json(response))
}

pub async fn delete_alias(State(db): State<AppState>, Path(alias): Path<String>) -> ApiResult<StatusCode> {
    blocking(move || db.delete_alias(&alias)).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn drop_collection(State(db): State<AppState>, Path(name): Path<String>) -> ApiResult<StatusCode> {
    blocking(move || db.drop_collection(&name)).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn flush_collection(State(db): State<AppState>, Path(name): Path<String>) -> ApiResult<StatusCode> {
    let c = collection(&db, &name)?;
    blocking(move || c.flush()).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn compact_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<CollectionInfo>> {
    let c = collection(&db, &name)?;
    let info = blocking(move || c.compact().map(|()| c.info())).await?;
    Ok(Json(info))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldIndexRequest {
    field: String,
}

/// Indexes a metadata field; indexing one that already has an index is a no-op.
pub async fn create_field_index(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<FieldIndexRequest>,
) -> ApiResult<Json<CollectionInfo>> {
    let c = collection(&db, &name)?;
    let info = blocking(move || c.create_field_index(&req.field).map(|()| c.info())).await?;
    Ok(Json(info))
}

pub async fn drop_field_index(
    State(db): State<AppState>,
    Path((name, field)): Path<(String, String)>,
) -> ApiResult<Json<CollectionInfo>> {
    let c = collection(&db, &name)?;
    let (dropped, info) = blocking(move || c.drop_field_index(&field).map(|d| (d, c.info()))).await?;
    if !dropped {
        return Err(ApiError::not_found("this field has no index"));
    }
    Ok(Json(info))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpsertRequest {
    points: Vec<PointInput>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdsRequest {
    ids: Vec<String>,
    #[serde(default = "yes")]
    with_vectors: bool,
}

#[derive(Debug, Deserialize)]
pub struct PointQuery {
    #[serde(default = "yes")]
    with_vectors: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrollQuery {
    /// Exclusive: the page starts after this id.
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    with_vectors: bool,
}

fn check_ids(ids: &[String]) -> ApiResult<()> {
    if ids.len() > MAX_IDS {
        return Err(ApiError::bad_request(format!(
            "at most {MAX_IDS} ids per request"
        )));
    }
    Ok(())
}

/// A filter from a request body; JSON `null` means none.
fn parse_filter(filter: Option<&Value>) -> ApiResult<Option<Filter>> {
    match filter {
        Some(Value::Null) | None => Ok(None),
        Some(f) => Ok(Some(Filter::parse(f)?)),
    }
}

/// Exactly one of `ids` and `filter`.
fn selector(ids: Option<Vec<String>>, filter: Option<&Value>) -> ApiResult<PointSelector> {
    match (ids, parse_filter(filter)?) {
        (Some(ids), None) => {
            check_ids(&ids)?;
            Ok(PointSelector::Ids(ids))
        }
        (None, Some(filter)) => Ok(PointSelector::Filter(filter)),
        _ => Err(ApiError::bad_request("give exactly one of `ids` and `filter`")),
    }
}

fn yes() -> bool {
    true
}

fn default_limit() -> usize {
    100
}

fn present(mut p: Point, with_vectors: bool) -> Value {
    if !with_vectors {
        p.vector.clear();
        p.vectors.clear();
        p.sparse_vectors.clear();
    }
    let mut v = serde_json::to_value(p).expect("points serialize");
    if !with_vectors {
        v.as_object_mut().expect("object").remove("vector");
    }
    v
}

pub async fn upsert_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    let points: Vec<Point> = req.points.into_iter().map(Point::from).collect();
    let ids: Vec<String> = points.iter().map(|p| p.id.clone()).collect();
    let upserted = blocking(move || c.upsert(points)).await?;
    Ok(Json(json!({ "upserted": upserted, "ids": ids })))
}

pub async fn get_point(
    State(db): State<AppState>,
    Path((name, id)): Path<(String, String)>,
    query: Result<Query<PointQuery>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let Query(q) = query?;
    let c = collection(&db, &name)?;
    let point = {
        let id = id.clone();
        blocking(move || Ok(c.get(&id))).await?
    };
    let point = point.ok_or_else(|| ApiError::point_not_found(&id))?;
    Ok(Json(present(point, q.with_vectors)))
}

pub async fn lookup_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<IdsRequest>,
) -> ApiResult<Json<Value>> {
    check_ids(&req.ids)?;
    let c = collection(&db, &name)?;
    let with_vectors = req.with_vectors;
    let points = blocking(move || Ok(c.get_many(&req.ids))).await?;
    let points: Vec<Value> = points.into_iter().map(|p| present(p, with_vectors)).collect();
    Ok(Json(json!({ "points": points })))
}

pub async fn scroll_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    query: Result<Query<ScrollQuery>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let Query(q) = query?;
    if q.limit > MAX_SCROLL {
        return Err(ApiError::bad_request(format!("limit must be <= {MAX_SCROLL}")));
    }
    let c = collection(&db, &name)?;
    let (cursor, limit) = (q.cursor, q.limit);
    let page = blocking(move || Ok(c.scroll(cursor.as_deref(), limit))).await?;
    let points: Vec<Value> = page
        .points
        .into_iter()
        .map(|p| present(p, q.with_vectors))
        .collect();
    Ok(Json(json!({ "points": points, "next_cursor": page.next_cursor })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeleteRequest {
    #[serde(default)]
    ids: Option<Vec<String>>,
    #[serde(default)]
    filter: Option<Value>,
}

pub async fn delete_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<DeleteRequest>,
) -> ApiResult<Json<Value>> {
    if req
        .filter
        .as_ref()
        .and_then(Value::as_object)
        .is_some_and(|f| f.is_empty())
    {
        return Err(ApiError::bad_request(
            "an empty filter would delete every point; drop the collection to delete everything",
        ));
    }
    let selector = selector(req.ids, req.filter.as_ref())?;
    let c = collection(&db, &name)?;
    let deleted = blocking(move || match selector {
        PointSelector::Ids(ids) => c.delete(&ids),
        PointSelector::Filter(filter) => c.delete_where(&filter),
    })
    .await?;
    Ok(Json(json!({ "deleted": deleted })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountRequest {
    #[serde(default)]
    filter: Option<Value>,
}

pub async fn count_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<CountRequest>,
) -> ApiResult<Json<Value>> {
    let filter = parse_filter(req.filter.as_ref())?;
    let c = collection(&db, &name)?;
    let count = blocking(move || Ok(c.count(filter.as_ref()))).await?;
    Ok(Json(json!({ "count": count })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataRequest {
    #[serde(default)]
    ids: Option<Vec<String>>,
    #[serde(default)]
    filter: Option<Value>,
    metadata: Value,
    #[serde(default)]
    mode: MetadataMode,
}

pub async fn update_metadata(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<MetadataRequest>,
) -> ApiResult<Json<Value>> {
    let selector = selector(req.ids, req.filter.as_ref())?;
    let c = collection(&db, &name)?;
    let updated = blocking(move || c.update_metadata(&selector, req.metadata, req.mode)).await?;
    Ok(Json(json!({ "updated": updated })))
}

pub async fn delete_point(
    State(db): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let c = collection(&db, &name)?;
    let ids = [id.clone()];
    match blocking(move || c.delete(&ids)).await? {
        0 => Err(ApiError::point_not_found(&id)),
        _ => Ok(StatusCode::NO_CONTENT),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    /// Exactly one of `vector`, `id` and `sparse`.
    #[serde(default)]
    vector: Option<Vec<f32>>,
    /// Search with this point's vector, leaving the point out of the results.
    #[serde(default)]
    id: Option<String>,
    /// Search a sparse vector (named by `using`) with this one.
    #[serde(default)]
    sparse: Option<SparseVector>,
    /// The named vector to search; the main one if absent.
    #[serde(default)]
    using: Option<String>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    ef: Option<usize>,
    /// Drop hits farther than this distance.
    #[serde(default)]
    max_distance: Option<f32>,
    #[serde(default)]
    with_vectors: bool,
    #[serde(default = "yes")]
    with_metadata: bool,
}

fn default_k() -> usize {
    10
}

/// The search settings shared by `search` and `recommend`.
#[derive(Debug, Clone, Copy)]
struct Settings<'a> {
    using: Option<&'a str>,
    k: usize,
    filter: Option<&'a Value>,
    ef: Option<usize>,
    max_distance: Option<f32>,
    with_vectors: bool,
    with_metadata: bool,
}

impl Settings<'_> {
    fn options(self) -> ApiResult<SearchOptions> {
        if self.k == 0 || self.k > MAX_K {
            return Err(ApiError::bad_request(format!("k must be between 1 and {MAX_K}")));
        }
        Ok(SearchOptions {
            using: self.using.map(str::to_owned),
            filter: parse_filter(self.filter)?,
            ef: self.ef,
            include_vectors: self.with_vectors,
            include_metadata: self.with_metadata,
            max_distance: self.max_distance,
        })
    }
}

/// What a validated search runs against.
enum SearchBy {
    Vector(Vec<f32>),
    Id(String),
    /// A sparse vector, against the sparse vector named.
    Sparse(SparseVector, String),
}

impl SearchRequest {
    fn prepare(self) -> ApiResult<(SearchBy, usize, SearchOptions)> {
        let mut opts = Settings {
            using: self.using.as_deref(),
            k: self.k,
            filter: self.filter.as_ref(),
            ef: self.ef,
            max_distance: self.max_distance,
            with_vectors: self.with_vectors,
            with_metadata: self.with_metadata,
        }
        .options()?;
        let by = match (self.vector, self.id, self.sparse) {
            (Some(v), None, None) => SearchBy::Vector(v),
            (None, Some(id), None) => SearchBy::Id(id),
            (None, None, Some(sparse)) => {
                let name = opts
                    .using
                    .take()
                    .ok_or_else(|| ApiError::bad_request("name the sparse vector to search in `using`"))?;
                SearchBy::Sparse(sparse, name)
            }
            _ => {
                return Err(ApiError::bad_request(
                    "give exactly one of `vector`, `id` and `sparse`",
                ));
            }
        };
        Ok((by, self.k, opts))
    }
}

fn run_search(
    c: &Collection,
    by: &SearchBy,
    k: usize,
    opts: &SearchOptions,
) -> quiver_db::Result<Vec<SearchHit>> {
    match by {
        SearchBy::Vector(v) => c.search_with(v, k, opts),
        SearchBy::Id(id) => c.search_by_id(id, k, opts),
        SearchBy::Sparse(v, name) => c.search_sparse(name, v, k, opts),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSearchRequest {
    searches: Vec<SearchRequest>,
}

pub async fn search(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    let (by, k, opts) = req.prepare()?;
    let results = blocking(move || run_search(&c, &by, k, &opts)).await?;
    Ok(Json(json!({ "results": results })))
}

pub async fn search_batch(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<BatchSearchRequest>,
) -> ApiResult<Json<Value>> {
    if req.searches.len() > MAX_BATCH {
        return Err(ApiError::bad_request(format!(
            "at most {MAX_BATCH} searches per batch"
        )));
    }
    let c = collection(&db, &name)?;
    let prepared = req
        .searches
        .into_iter()
        .map(SearchRequest::prepare)
        .collect::<ApiResult<Vec<_>>>()?;
    let results = blocking(move || {
        prepared
            .iter()
            .map(|(by, k, opts)| run_search(&c, by, *k, opts))
            .collect::<Result<Vec<_>, _>>()
    })
    .await?;
    Ok(Json(json!({ "results": results })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecommendRequest {
    positive: Vec<String>,
    /// The named vector to compare; the main one if absent.
    #[serde(default)]
    using: Option<String>,
    #[serde(default)]
    negative: Vec<String>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    ef: Option<usize>,
    /// Drop hits farther than this distance.
    #[serde(default)]
    max_distance: Option<f32>,
    #[serde(default)]
    with_vectors: bool,
    #[serde(default = "yes")]
    with_metadata: bool,
}

pub async fn recommend(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<RecommendRequest>,
) -> ApiResult<Json<Value>> {
    if req.positive.len().saturating_add(req.negative.len()) > MAX_IDS {
        return Err(ApiError::bad_request(format!(
            "at most {MAX_IDS} example ids per request"
        )));
    }
    let opts = Settings {
        using: req.using.as_deref(),
        k: req.k,
        filter: req.filter.as_ref(),
        ef: req.ef,
        max_distance: req.max_distance,
        with_vectors: req.with_vectors,
        with_metadata: req.with_metadata,
    }
    .options()?;
    let c = collection(&db, &name)?;
    let results = blocking(move || c.recommend(&req.positive, &req.negative, req.k, &opts)).await?;
    Ok(Json(json!({ "results": results })))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    /// The searches whose rankings are fused.
    prefetch: Vec<PrefetchRequest>,
    #[serde(default)]
    fusion: Fusion,
    #[serde(default = "default_k")]
    k: usize,
    /// Applies to every search.
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    with_vectors: bool,
    #[serde(default = "yes")]
    with_metadata: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefetchRequest {
    /// Exactly one of `vector` and `sparse`.
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default)]
    sparse: Option<SparseVector>,
    /// The named vector to search: the main one if absent for `vector`;
    /// required for `sparse`.
    #[serde(default)]
    using: Option<String>,
    /// How many of this search's best hits take part (default: `max(k, 50)`).
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default = "default_weight")]
    weight: f32,
    #[serde(default)]
    ef: Option<usize>,
    #[serde(default)]
    max_distance: Option<f32>,
}

fn default_weight() -> f32 {
    1.0
}

impl PrefetchRequest {
    fn prepare(self, k: usize) -> ApiResult<Prefetch> {
        let limit = self.limit.unwrap_or(k.max(50));
        if limit == 0 || limit > MAX_K {
            return Err(ApiError::bad_request(format!(
                "prefetch limit must be between 1 and {MAX_K}"
            )));
        }
        let query = match (self.vector, self.sparse) {
            (Some(vector), None) => QueryVector::Dense {
                vector,
                using: self.using,
            },
            (None, Some(vector)) => QueryVector::Sparse {
                vector,
                using: self
                    .using
                    .ok_or_else(|| ApiError::bad_request("name the sparse vector to search in `using`"))?,
            },
            _ => {
                return Err(ApiError::bad_request(
                    "give each prefetch exactly one of `vector` and `sparse`",
                ));
            }
        };
        Ok(Prefetch {
            query,
            limit,
            weight: self.weight,
            ef: self.ef,
            max_distance: self.max_distance,
        })
    }
}

/// Hybrid search: several searches, dense and sparse, fused into one ranking.
pub async fn query(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<QueryRequest>,
) -> ApiResult<Json<Value>> {
    if req.k == 0 || req.k > MAX_K {
        return Err(ApiError::bad_request(format!("k must be between 1 and {MAX_K}")));
    }
    if req.prefetch.is_empty() || req.prefetch.len() > MAX_PREFETCH {
        return Err(ApiError::bad_request(format!(
            "give between 1 and {MAX_PREFETCH} prefetch searches"
        )));
    }
    let k = req.k;
    let query = HybridQuery {
        prefetch: req
            .prefetch
            .into_iter()
            .map(|p| p.prepare(k))
            .collect::<ApiResult<_>>()?,
        fusion: req.fusion,
        filter: parse_filter(req.filter.as_ref())?,
        include_vectors: req.with_vectors,
        include_metadata: req.with_metadata,
    };
    let c = collection(&db, &name)?;
    let results = blocking(move || c.query(&query, k)).await?;
    Ok(Json(json!({ "results": results })))
}

#[derive(Debug, Deserialize)]
pub struct SnapshotQuery {
    #[serde(default)]
    collection: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestoreRequest {
    /// Restore under this name instead of the original.
    #[serde(default)]
    collection: Option<String>,
    /// Replace an existing collection of the same name.
    #[serde(default)]
    replace: bool,
}

pub async fn create_snapshot(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<(StatusCode, Json<SnapshotInfo>)> {
    let c = collection(&db, &name)?;
    let info = blocking(move || c.create_snapshot()).await?;
    Ok((StatusCode::CREATED, Json(info)))
}

pub async fn list_collection_snapshots(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<Value>> {
    let snapshots = blocking(move || db.list_snapshots(Some(&name))).await?;
    Ok(Json(json!({ "snapshots": snapshots })))
}

pub async fn list_snapshots(
    State(db): State<AppState>,
    query: Result<Query<SnapshotQuery>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let Query(q) = query?;
    // `?collection=` (empty) means no filter, as in the gRPC API.
    let collection = q.collection.filter(|c| !c.is_empty());
    let snapshots = blocking(move || db.list_snapshots(collection.as_deref())).await?;
    Ok(Json(json!({ "snapshots": snapshots })))
}

pub async fn get_snapshot(
    State(db): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<SnapshotInfo>> {
    let info = blocking(move || db.snapshots()?.get(&id)).await?;
    Ok(Json(info))
}

pub async fn delete_snapshot(State(db): State<AppState>, Path(id): Path<String>) -> ApiResult<StatusCode> {
    blocking(move || db.delete_snapshot(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn restore_snapshot(
    State(db): State<AppState>,
    Path(id): Path<String>,
    body: axum::body::Bytes,
) -> ApiResult<Json<CollectionInfo>> {
    // The body is optional: an empty request restores under the original name.
    let req: RestoreRequest = if body.iter().all(u8::is_ascii_whitespace) {
        RestoreRequest::default()
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "invalid_body", e.to_string()))?
    };
    let info = blocking(move || {
        let opts = RestoreOptions {
            name: req.collection,
            replace: req.replace,
        };
        Ok(db.restore_snapshot(&id, opts)?.info())
    })
    .await?;
    Ok(Json(info))
}

/// Streams the archive file (`application/gzip`).
pub async fn download_snapshot(State(db): State<AppState>, Path(id): Path<String>) -> ApiResult<Response> {
    let (path, info) = {
        let db = Arc::clone(&db);
        let id = id.clone();
        blocking(move || {
            let reg = db.snapshots()?;
            Ok((reg.path(&id)?, reg.get(&id)?))
        })
        .await?
    };
    let file = tokio::fs::File::open(&path)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let body = Body::from_stream(tokio_util::io::ReaderStream::new(file));
    let filename = format!("{}-{}.tar.gz", info.manifest.collection, info.manifest.id);
    Ok((
        [
            (header::CONTENT_TYPE, "application/gzip".to_owned()),
            (header::CONTENT_LENGTH, info.archive_bytes.to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        body,
    )
        .into_response())
}

/// Accepts a raw archive body (as produced by `download`) and adds it to the registry.
pub async fn upload_snapshot(
    State(db): State<AppState>,
    Extension(UploadLimit(limit)): Extension<UploadLimit>,
    headers: HeaderMap,
    body: Body,
) -> ApiResult<(StatusCode, Json<SnapshotInfo>)> {
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|len| len > limit as u64) {
        return Err(ApiError::payload_too_large(limit));
    }

    // Removed on every early return, and also if the client goes away and
    // this future is dropped mid-upload.
    let path = db.snapshots()?.staging_path();
    let mut staging = StagingFile(Some(path.clone()));
    let written: Result<(), ApiError> = async {
        let mut file = tokio::fs::File::create(&path)
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut stream = body.into_data_stream();
        let mut received = 0usize;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ApiError::bad_request(format!("upload failed: {e}")))?;
            received = received.saturating_add(chunk.len());
            if received > limit {
                return Err(ApiError::payload_too_large(limit));
            }
            file.write_all(&chunk)
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
        }
        file.sync_all()
            .await
            .map_err(|e| ApiError::internal(e.to_string()))
    }
    .await;
    written?;
    // `import_staged` removes the staging file itself from here on.
    staging.0 = None;
    let imported = tokio::task::spawn_blocking(move || db.snapshots()?.import_staged(&path))
        .await
        .map_err(|e| ApiError::internal(format!("task failed: {e}")))?;
    // The archive came from the client: an unreadable one is its error.
    let info = imported.map_err(|e| match e {
        quiver_db::Error::Corrupt(_) | quiver_db::Error::Serde(_) | quiver_db::Error::Io(_) => {
            ApiError::invalid_snapshot(e.to_string())
        }
        other => ApiError::from(other),
    })?;
    Ok((StatusCode::CREATED, Json(info)))
}

/// A staging file that is deleted when dropped, unless taken.
struct StagingFile(Option<std::path::PathBuf>);

impl Drop for StagingFile {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}
