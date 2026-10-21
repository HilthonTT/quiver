//! The `quiver.v1.Quiver` service implementation.

use std::pin::Pin;
use std::sync::Arc;

use quiver_db::auth::Role;
use quiver_db::{
    Collection, Database, Filter, MetadataMode, Point, PointSelector, RestoreOptions, SearchHit,
    SearchOptions,
};
use tokio::io::AsyncWriteExt;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::auth::require;
use crate::convert::*;
use crate::pb;

/// Upper bound on `k` for a single search.
const MAX_K: usize = 10_000;
/// Upper bound on searches in one batch.
const MAX_BATCH: usize = 256;
/// Upper bound on a scroll page.
const MAX_SCROLL: usize = 10_000;
/// Upper bound on ids in one GetPoints or DeletePoints request.
const MAX_IDS: usize = 10_000;
/// Largest snapshot upload accepted, as in the REST API.
const MAX_UPLOAD_BYTES: u64 = 16 << 30;
/// Size of the chunks a download is sent in.
const DOWNLOAD_CHUNK_BYTES: usize = 1 << 20;

type ChunkStream = Pin<Box<dyn Stream<Item = Result<pb::SnapshotChunk, Status>> + Send>>;

type Rpc<T> = Result<Response<T>, Status>;

/// Implements the gRPC service over a [`Database`].
#[derive(Debug, Clone)]
pub struct QuiverService {
    db: Arc<Database>,
}

impl QuiverService {
    /// A service backed by `db`.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Runs blocking database work off the async runtime.
    async fn blocking<T, F>(&self, f: F) -> Result<T, Status>
    where
        F: FnOnce(&Database) -> quiver_db::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db = Arc::clone(&self.db);
        tokio::task::spawn_blocking(move || f(&db))
            .await
            .map_err(|e| Status::internal(format!("task failed: {e}")))?
            .map_err(status)
    }
}

fn check_ids(ids: &[String]) -> Result<(), Status> {
    if ids.len() > MAX_IDS {
        return Err(Status::invalid_argument(format!(
            "at most {MAX_IDS} ids per request"
        )));
    }
    Ok(())
}

/// A JSON filter; empty or JSON `null` means none, as in the REST API.
fn filter_from_pb(json: &str) -> Result<Option<Filter>, Status> {
    json_from_pb(json, "filter_json")?
        .filter(|f| !f.is_null())
        .map(|f| Filter::parse(&f))
        .transpose()
        .map_err(status)
}

/// Either `ids` or a filter, not both. No filter selects the (maybe empty) ids.
fn selector(ids: Vec<String>, filter_json: &str) -> Result<PointSelector, Status> {
    check_ids(&ids)?;
    match filter_from_pb(filter_json)? {
        None => Ok(PointSelector::Ids(ids)),
        Some(_) if !ids.is_empty() => Err(Status::invalid_argument(
            "give either ids or filter_json, not both",
        )),
        Some(filter) => Ok(PointSelector::Filter(filter)),
    }
}

/// The search settings shared by `Search` and `Recommend`.
struct Settings<'a> {
    k: u32,
    filter_json: &'a str,
    ef: Option<u32>,
    max_distance: Option<f32>,
    with_vectors: bool,
    with_metadata: Option<bool>,
}

impl Settings<'_> {
    fn options(&self) -> Result<(usize, SearchOptions), Status> {
        let k = if self.k == 0 { 10 } else { self.k as usize };
        if k > MAX_K {
            return Err(Status::invalid_argument(format!(
                "k must be between 1 and {MAX_K}"
            )));
        }
        Ok((
            k,
            SearchOptions {
                filter: filter_from_pb(self.filter_json)?,
                ef: self.ef.map(|e| e as usize),
                include_vectors: self.with_vectors,
                include_metadata: self.with_metadata.unwrap_or(true),
                max_distance: self.max_distance,
            },
        ))
    }
}

fn search_options(req: &pb::SearchRequest) -> Result<(usize, SearchOptions), Status> {
    if req.vector.is_empty() == req.id.is_empty() {
        return Err(Status::invalid_argument("give either vector or id, not both"));
    }
    Settings {
        k: req.k,
        filter_json: &req.filter_json,
        ef: req.ef,
        max_distance: req.max_distance,
        with_vectors: req.with_vectors,
        with_metadata: req.with_metadata,
    }
    .options()
}

fn run_search(
    c: &Collection,
    req: &pb::SearchRequest,
    k: usize,
    opts: &SearchOptions,
) -> quiver_db::Result<Vec<SearchHit>> {
    if req.id.is_empty() {
        c.search_with(&req.vector, k, opts)
    } else {
        c.search_by_id(&req.id, k, opts)
    }
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

#[tonic::async_trait]
impl pb::quiver_server::Quiver for QuiverService {
    type DownloadSnapshotStream = ChunkStream;
    async fn health(&self, _req: Request<pb::HealthRequest>) -> Rpc<pb::HealthResponse> {
        Ok(Response::new(pb::HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }))
    }

    async fn info(&self, req: Request<pb::InfoRequest>) -> Rpc<pb::InfoResponse> {
        require(&req, Role::ReadOnly)?;
        let (collections, storage, snapshots_enabled) = self
            .blocking(|db| {
                Ok((
                    db.list_collections(),
                    db.options().storage.to_string(),
                    db.snapshots().is_ok(),
                ))
            })
            .await?;
        Ok(Response::new(pb::InfoResponse {
            version: env!("CARGO_PKG_VERSION").into(),
            storage,
            snapshots_enabled,
            collections: collections.len() as u64,
            points: collections.iter().map(|c| c.count as u64).sum(),
        }))
    }

    async fn list_collections(
        &self,
        req: Request<pb::ListCollectionsRequest>,
    ) -> Rpc<pb::ListCollectionsResponse> {
        require(&req, Role::ReadOnly)?;
        let collections = self
            .blocking(|db| Ok(db.list_collections()))
            .await?
            .into_iter()
            .map(info_to_pb)
            .collect();
        Ok(Response::new(pb::ListCollectionsResponse { collections }))
    }

    async fn create_collection(&self, req: Request<pb::CreateCollectionRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let config = config_from_pb(&req)?;
        let info = self
            .blocking(move |db| {
                Ok(db
                    .create_collection_with_fields(&req.name, config, &req.field_indexes)?
                    .info())
            })
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn get_collection(&self, req: Request<pb::GetCollectionRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadOnly)?;
        let name = req.into_inner().name;
        let info = self.blocking(move |db| Ok(db.collection(&name)?.info())).await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn drop_collection(
        &self,
        req: Request<pb::DropCollectionRequest>,
    ) -> Rpc<pb::DropCollectionResponse> {
        require(&req, Role::ReadWrite)?;
        let name = req.into_inner().name;
        self.blocking(move |db| db.drop_collection(&name)).await?;
        Ok(Response::new(pb::DropCollectionResponse {}))
    }

    async fn flush(&self, req: Request<pb::FlushRequest>) -> Rpc<pb::FlushResponse> {
        require(&req, Role::ReadWrite)?;
        let name = req.into_inner().collection;
        self.blocking(move |db| db.collection(&name)?.flush()).await?;
        Ok(Response::new(pb::FlushResponse {}))
    }

    async fn compact(&self, req: Request<pb::CompactRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let name = req.into_inner().collection;
        let info = self
            .blocking(move |db| {
                let c = db.collection(&name)?;
                c.compact()?;
                Ok(c.info())
            })
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn create_field_index(&self, req: Request<pb::FieldIndexRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let info = self
            .blocking(move |db| {
                let c = db.collection(&req.collection)?;
                c.create_field_index(&req.field)?;
                Ok(c.info())
            })
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn drop_field_index(&self, req: Request<pb::FieldIndexRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let (dropped, info) = self
            .blocking(move |db| {
                let c = db.collection(&req.collection)?;
                Ok((c.drop_field_index(&req.field)?, c.info()))
            })
            .await?;
        if !dropped {
            return Err(Status::not_found("this field has no index"));
        }
        Ok(Response::new(info_to_pb(info)))
    }

    async fn upsert(&self, req: Request<pb::UpsertRequest>) -> Rpc<pb::UpsertResponse> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let points: Vec<Point> = req
            .points
            .into_iter()
            .map(point_from_pb)
            .collect::<Result<_, _>>()?;
        let ids: Vec<String> = points.iter().map(|p| p.id.clone()).collect();
        let upserted = self
            .blocking(move |db| db.collection(&req.collection)?.upsert(points))
            .await?;
        Ok(Response::new(pb::UpsertResponse {
            upserted: upserted as u64,
            ids,
        }))
    }

    async fn get_points(&self, req: Request<pb::GetPointsRequest>) -> Rpc<pb::GetPointsResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        check_ids(&req.ids)?;
        let with_vectors = req.with_vectors;
        let points = self
            .blocking(move |db| Ok(db.collection(&req.collection)?.get_many(&req.ids)))
            .await?
            .into_iter()
            .map(|p| point_to_pb(p, with_vectors))
            .collect();
        Ok(Response::new(pb::GetPointsResponse { points }))
    }

    async fn delete_points(&self, req: Request<pb::DeletePointsRequest>) -> Rpc<pb::DeletePointsResponse> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        if json_from_pb(&req.filter_json, "filter_json")?
            .and_then(|f| f.as_object().map(serde_json::Map::is_empty))
            .unwrap_or(false)
        {
            return Err(Status::invalid_argument(
                "an empty filter would delete every point; drop the collection to delete everything",
            ));
        }
        let selector = selector(req.ids, &req.filter_json)?;
        let deleted = self
            .blocking(move |db| {
                let c = db.collection(&req.collection)?;
                match selector {
                    PointSelector::Ids(ids) => c.delete(&ids),
                    PointSelector::Filter(filter) => c.delete_where(&filter),
                }
            })
            .await?;
        Ok(Response::new(pb::DeletePointsResponse {
            deleted: deleted as u64,
        }))
    }

    async fn scroll(&self, req: Request<pb::ScrollRequest>) -> Rpc<pb::ScrollResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let limit = if req.limit == 0 { 100 } else { req.limit as usize };
        if limit > MAX_SCROLL {
            return Err(Status::invalid_argument(format!("limit must be <= {MAX_SCROLL}")));
        }
        let cursor = (!req.cursor.is_empty()).then(|| req.cursor.clone());
        let collection = req.collection.clone();
        let page = self
            .blocking(move |db| Ok(db.collection(&collection)?.scroll(cursor.as_deref(), limit)))
            .await?;
        let points = page
            .points
            .into_iter()
            .map(|p| point_to_pb(p, req.with_vectors))
            .collect();
        Ok(Response::new(pb::ScrollResponse {
            points,
            next_cursor: page.next_cursor,
        }))
    }

    async fn count(&self, req: Request<pb::CountRequest>) -> Rpc<pb::CountResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let filter = filter_from_pb(&req.filter_json)?;
        let count = self
            .blocking(move |db| Ok(db.collection(&req.collection)?.count(filter.as_ref())))
            .await?;
        Ok(Response::new(pb::CountResponse { count: count as u64 }))
    }

    async fn update_metadata(
        &self,
        req: Request<pb::UpdateMetadataRequest>,
    ) -> Rpc<pb::UpdateMetadataResponse> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let mode = match pb::MetadataMode::try_from(req.mode) {
            Ok(pb::MetadataMode::Merge) => MetadataMode::Merge,
            Ok(pb::MetadataMode::Replace) => MetadataMode::Replace,
            Err(_) => {
                return Err(Status::invalid_argument(format!(
                    "unknown metadata mode {}",
                    req.mode
                )));
            }
        };
        let metadata = json_from_pb(&req.metadata_json, "metadata_json")?
            .ok_or_else(|| Status::invalid_argument("metadata_json is required (use `null` to remove)"))?;
        let selector = selector(req.ids, &req.filter_json)?;
        let updated = self
            .blocking(move |db| {
                db.collection(&req.collection)?
                    .update_metadata(&selector, metadata, mode)
            })
            .await?;
        Ok(Response::new(pb::UpdateMetadataResponse {
            updated: updated as u64,
        }))
    }

    async fn recommend(&self, req: Request<pb::RecommendRequest>) -> Rpc<pb::SearchResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        if req.positive.len().saturating_add(req.negative.len()) > MAX_IDS {
            return Err(Status::invalid_argument(format!(
                "at most {MAX_IDS} example ids per request"
            )));
        }
        let (k, opts) = Settings {
            k: req.k,
            filter_json: &req.filter_json,
            ef: req.ef,
            max_distance: req.max_distance,
            with_vectors: req.with_vectors,
            with_metadata: req.with_metadata,
        }
        .options()?;
        let hits = self
            .blocking(move |db| {
                db.collection(&req.collection)?
                    .recommend(&req.positive, &req.negative, k, &opts)
            })
            .await?;
        Ok(Response::new(pb::SearchResponse {
            hits: hits.into_iter().map(hit_to_pb).collect(),
        }))
    }

    async fn search(&self, req: Request<pb::SearchRequest>) -> Rpc<pb::SearchResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let (k, opts) = search_options(&req)?;
        let hits = self
            .blocking(move |db| run_search(&*db.collection(&req.collection)?, &req, k, &opts))
            .await?;
        Ok(Response::new(pb::SearchResponse {
            hits: hits.into_iter().map(hit_to_pb).collect(),
        }))
    }

    async fn search_batch(&self, req: Request<pb::SearchBatchRequest>) -> Rpc<pb::SearchBatchResponse> {
        require(&req, Role::ReadOnly)?;
        let searches = req.into_inner().searches;
        if searches.len() > MAX_BATCH {
            return Err(Status::invalid_argument(format!(
                "at most {MAX_BATCH} searches per batch"
            )));
        }
        let prepared = searches
            .into_iter()
            .map(|s| search_options(&s).map(|o| (s, o)))
            .collect::<Result<Vec<_>, _>>()?;
        let results = self
            .blocking(move |db| {
                prepared
                    .iter()
                    .map(|(s, (k, opts))| {
                        let hits = run_search(&*db.collection(&s.collection)?, s, *k, opts)?;
                        Ok(pb::SearchResponse {
                            hits: hits.into_iter().map(hit_to_pb).collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, quiver_db::Error>>()
            })
            .await?;
        Ok(Response::new(pb::SearchBatchResponse { results }))
    }

    async fn create_snapshot(&self, req: Request<pb::CreateSnapshotRequest>) -> Rpc<pb::SnapshotInfo> {
        require(&req, Role::ReadWrite)?;
        let name = req.into_inner().collection;
        let info = self.blocking(move |db| db.create_snapshot(&name)).await?;
        Ok(Response::new(snapshot_to_pb(info)))
    }

    async fn list_snapshots(&self, req: Request<pb::ListSnapshotsRequest>) -> Rpc<pb::ListSnapshotsResponse> {
        require(&req, Role::ReadOnly)?;
        let name = req.into_inner().collection;
        let list = self
            .blocking(move |db| db.list_snapshots(Some(name.as_str()).filter(|n| !n.is_empty())))
            .await?;
        Ok(Response::new(pb::ListSnapshotsResponse {
            snapshots: list.into_iter().map(snapshot_to_pb).collect(),
        }))
    }

    async fn get_snapshot(&self, req: Request<pb::GetSnapshotRequest>) -> Rpc<pb::SnapshotInfo> {
        require(&req, Role::ReadOnly)?;
        let id = req.into_inner().id;
        let info = self.blocking(move |db| db.snapshots()?.get(&id)).await?;
        Ok(Response::new(snapshot_to_pb(info)))
    }

    async fn delete_snapshot(
        &self,
        req: Request<pb::DeleteSnapshotRequest>,
    ) -> Rpc<pb::DeleteSnapshotResponse> {
        require(&req, Role::ReadWrite)?;
        let id = req.into_inner().id;
        self.blocking(move |db| db.delete_snapshot(&id)).await?;
        Ok(Response::new(pb::DeleteSnapshotResponse {}))
    }

    async fn restore_snapshot(&self, req: Request<pb::RestoreSnapshotRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let info = self
            .blocking(move |db| {
                let opts = RestoreOptions {
                    name: Some(req.collection).filter(|n| !n.is_empty()),
                    replace: req.replace,
                };
                Ok(db.restore_snapshot(&req.id, opts)?.info())
            })
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn download_snapshot(
        &self,
        req: Request<pb::DownloadSnapshotRequest>,
    ) -> Rpc<Self::DownloadSnapshotStream> {
        require(&req, Role::ReadOnly)?;
        let id = req.into_inner().id;
        let path = self
            .blocking(move |db| {
                let reg = db.snapshots()?;
                // `get` checks that the archive exists and is readable.
                reg.get(&id)?;
                reg.path(&id)
            })
            .await?;
        let file = tokio::fs::File::open(&path)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let chunks = tokio_util::io::ReaderStream::with_capacity(file, DOWNLOAD_CHUNK_BYTES).map(|chunk| {
            chunk
                .map(|data| pb::SnapshotChunk { data: data.to_vec() })
                .map_err(|e| Status::internal(e.to_string()))
        });
        Ok(Response::new(Box::pin(chunks)))
    }

    async fn upload_snapshot(&self, req: Request<Streaming<pb::SnapshotChunk>>) -> Rpc<pb::SnapshotInfo> {
        require(&req, Role::ReadWrite)?;
        let mut chunks = req.into_inner();
        let path = self.db.snapshots().map_err(status)?.staging_path();
        // Removed on every early return, and if the client goes away mid-upload.
        let mut staging = StagingFile(Some(path.clone()));
        let mut file = tokio::fs::File::create(&path)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let mut received = 0u64;
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            received = received.saturating_add(chunk.data.len() as u64);
            if received > MAX_UPLOAD_BYTES {
                return Err(Status::resource_exhausted(format!(
                    "snapshot uploads are limited to {MAX_UPLOAD_BYTES} bytes"
                )));
            }
            file.write_all(&chunk.data)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
        }
        file.sync_all()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        drop(file);
        // `import_staged` removes the staging file itself from here on.
        staging.0 = None;
        let db = Arc::clone(&self.db);
        let imported = tokio::task::spawn_blocking(move || db.snapshots()?.import_staged(&path))
            .await
            .map_err(|e| Status::internal(format!("task failed: {e}")))?;
        // The archive came from the client: an unreadable one is its error.
        let info = imported.map_err(|e| match e {
            quiver_db::Error::Corrupt(_) | quiver_db::Error::Serde(_) | quiver_db::Error::Io(_) => {
                Status::invalid_argument(format!("invalid snapshot: {e}"))
            }
            other => status(other),
        })?;
        Ok(Response::new(snapshot_to_pb(info)))
    }
}
