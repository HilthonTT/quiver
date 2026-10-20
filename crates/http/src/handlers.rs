use std::sync::Arc;

use axum::body::Body;
use axum::extract::{FromRequest, Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

use vectordb_db::{
    Collection, CollectionConfig, CollectionInfo, Database, Filter, Point, PointInput,
    RestoreOptions, SearchOptions, SnapshotInfo,
};

use crate::error::ApiError;

/// Upper bound on `k` for a single search.
pub const MAX_K: usize = 10_000;
/// Upper bound on searches in one batch request.
pub const MAX_BATCH: usize = 256;
/// Upper bound on `limit` when scrolling.
pub const MAX_SCROLL: usize = 10_000;

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
    F: FnOnce() -> Result<T, vectordb_db::Error> + Send + 'static,
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
        "name": "SagittaDB",
        "version": env!("CARGO_PKG_VERSION"),
        "docs": "https://github.com/HilthonTT/sagittadb/blob/main/docs/API.md",
    }))
}

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

/// Server-wide statistics.
pub async fn info(State(db): State<AppState>) -> Json<Value> {
    let collections = db.list_collections();
    let points: usize = collections.iter().map(|c| c.count).sum();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "storage": db.options().storage,
        "snapshots_enabled": db.snapshots().is_ok(),
        "collections": collections.len(),
        "points": points,
    }))
}

#[derive(Debug, Deserialize)]
pub struct CreateCollection {
    name: String,
    #[serde(flatten)]
    config: CollectionConfig,
}

pub async fn list_collections(State(db): State<AppState>) -> Json<Value> {
    Json(json!({ "collections": db.list_collections() }))
}

pub async fn create_collection(
    State(db): State<AppState>,
    Json(req): Json<CreateCollection>,
) -> ApiResult<(StatusCode, Json<CollectionInfo>)> {
    let info = blocking(move || Ok(db.create_collection(&req.name, req.config)?.info())).await?;
    tracing::info!(collection = %info.name, dimension = info.config.dimension, "created collection");
    Ok((StatusCode::CREATED, Json(info)))
}

pub async fn get_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<CollectionInfo>> {
    Ok(Json(collection(&db, &name)?.info()))
}

pub async fn drop_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    blocking(move || db.drop_collection(&name)).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn flush_collection(
    State(db): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
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
pub struct ScrollQuery {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    with_vectors: bool,
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
    let point = collection(&db, &name)?
        .get(&id)
        .ok_or_else(|| ApiError::not_found(format!("point `{id}` not found")))?;
    Ok(Json(present(point, q.with_vectors)))
}

pub async fn lookup_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<IdsRequest>,
) -> ApiResult<Json<Value>> {
    let points: Vec<Value> = collection(&db, &name)?
        .get_many(&req.ids)
        .into_iter()
        .map(|p| present(p, req.with_vectors))
        .collect();
    Ok(Json(json!({ "points": points })))
}

pub async fn scroll_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    query: Result<Query<ScrollQuery>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let Query(q) = query?;
    if q.limit > MAX_SCROLL {
        return Err(ApiError::bad_request(format!(
            "limit must be <= {MAX_SCROLL}"
        )));
    }
    let c = collection(&db, &name)?;
    let page = blocking(move || Ok(c.scroll(q.offset, q.limit))).await?;
    // An empty page is the end even for `limit=0`, which would otherwise
    // hand back the same cursor forever.
    let next_offset =
        (!page.is_empty() && page.len() == q.limit).then(|| q.offset.saturating_add(q.limit));
    let points: Vec<Value> = page
        .into_iter()
        .map(|p| present(p, q.with_vectors))
        .collect();
    Ok(Json(
        json!({ "points": points, "next_offset": next_offset }),
    ))
}

pub async fn delete_points(
    State(db): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<IdsRequest>,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    let deleted = blocking(move || c.delete(&req.ids)).await?;
    Ok(Json(json!({ "deleted": deleted })))
}

pub async fn delete_point(
    State(db): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let c = collection(&db, &name)?;
    match blocking(move || c.delete(&[id])).await? {
        0 => Err(ApiError::not_found("point not found")),
        _ => Ok(StatusCode::NO_CONTENT),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    ef: Option<usize>,
    #[serde(default)]
    with_vectors: bool,
    #[serde(default = "yes")]
    with_metadata: bool,
}

fn default_k() -> usize {
    10
}

impl SearchRequest {
    fn options(&self) -> ApiResult<SearchOptions> {
        if self.k == 0 || self.k > MAX_K {
            return Err(ApiError::bad_request(format!(
                "k must be between 1 and {MAX_K}"
            )));
        }
        let filter = match &self.filter {
            Some(Value::Null) | None => None,
            Some(f) => Some(Filter::parse(f)?),
        };
        Ok(SearchOptions {
            filter,
            ef: self.ef,
            include_vectors: self.with_vectors,
            include_metadata: self.with_metadata,
        })
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
    let opts = req.options()?;
    let results = blocking(move || c.search_with(&req.vector, req.k, &opts)).await?;
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
        .map(|s| s.options().map(|o| (s, o)))
        .collect::<ApiResult<Vec<_>>>()?;
    let results = blocking(move || {
        prepared
            .iter()
            .map(|(s, o)| c.search_with(&s.vector, s.k, o))
            .collect::<Result<Vec<_>, _>>()
    })
    .await?;
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
    let snapshots = blocking(move || db.list_snapshots(q.collection.as_deref())).await?;
    Ok(Json(json!({ "snapshots": snapshots })))
}

pub async fn get_snapshot(
    State(db): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<SnapshotInfo>> {
    let info = blocking(move || db.snapshots()?.get(&id)).await?;
    Ok(Json(info))
}

pub async fn delete_snapshot(
    State(db): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
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
pub async fn download_snapshot(
    State(db): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
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
    body: Body,
) -> ApiResult<(StatusCode, Json<SnapshotInfo>)> {
    let staging = db.snapshots()?.staging_path();
    let written: Result<(), ApiError> = async {
        let mut file = tokio::fs::File::create(&staging)
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut stream = body.into_data_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ApiError::bad_request(format!("upload failed: {e}")))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
        }
        file.sync_all()
            .await
            .map_err(|e| ApiError::internal(e.to_string()))
    }
    .await;
    if let Err(e) = written {
        let _ = tokio::fs::remove_file(&staging).await;
        return Err(e);
    }
    let info = blocking(move || db.snapshots()?.import_staged(&staging)).await?;
    Ok((StatusCode::CREATED, Json(info)))
}
