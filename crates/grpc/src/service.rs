//! The `quiver.v1.Quiver` service implementation.

use std::sync::Arc;

use quiver_db::auth::Role;
use quiver_db::{Database, Filter, Point, RestoreOptions, SearchOptions};
use tonic::{Request, Response, Status};

use crate::auth::require;
use crate::convert::*;
use crate::pb;

/// Upper bound on `k` for a single search.
const MAX_K: usize = 10_000;
/// Upper bound on searches in one batch.
const MAX_BATCH: usize = 256;
/// Upper bound on a scroll page.
const MAX_SCROLL: usize = 10_000;

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

fn search_options(req: &pb::SearchRequest) -> Result<(usize, SearchOptions), Status> {
    let k = if req.k == 0 { 10 } else { req.k as usize };
    if k > MAX_K {
        return Err(Status::invalid_argument(format!(
            "k must be between 1 and {MAX_K}"
        )));
    }
    let filter = json_from_pb(&req.filter_json, "filter_json")?
        .map(|f| Filter::parse(&f))
        .transpose()
        .map_err(status)?;
    Ok((
        k,
        SearchOptions {
            filter,
            ef: req.ef.map(|e| e as usize),
            include_vectors: req.with_vectors,
            include_metadata: req.with_metadata.unwrap_or(true),
        },
    ))
}

#[tonic::async_trait]
impl pb::quiver_server::Quiver for QuiverService {
    async fn health(&self, _req: Request<pb::HealthRequest>) -> Rpc<pb::HealthResponse> {
        Ok(Response::new(pb::HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }))
    }

    async fn list_collections(
        &self,
        req: Request<pb::ListCollectionsRequest>,
    ) -> Rpc<pb::ListCollectionsResponse> {
        require(&req, Role::ReadOnly)?;
        let collections = self.db.list_collections().into_iter().map(info_to_pb).collect();
        Ok(Response::new(pb::ListCollectionsResponse { collections }))
    }

    async fn create_collection(&self, req: Request<pb::CreateCollectionRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let config = config_from_pb(&req)?;
        let info = self
            .blocking(move |db| Ok(db.create_collection(&req.name, config)?.info()))
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn get_collection(&self, req: Request<pb::GetCollectionRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadOnly)?;
        let c = self.db.collection(&req.get_ref().name).map_err(status)?;
        Ok(Response::new(info_to_pb(c.info())))
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
        let c = self.db.collection(&req.collection).map_err(status)?;
        let points = c
            .get_many(&req.ids)
            .into_iter()
            .map(|p| point_to_pb(p, req.with_vectors))
            .collect();
        Ok(Response::new(pb::GetPointsResponse { points }))
    }

    async fn delete_points(&self, req: Request<pb::DeletePointsRequest>) -> Rpc<pb::DeletePointsResponse> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let deleted = self
            .blocking(move |db| db.collection(&req.collection)?.delete(&req.ids))
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
        let offset = req.offset as usize;
        let page = self
            .blocking(move |db| Ok(db.collection(&req.collection)?.scroll(offset, limit)))
            .await?;
        let next_offset = (page.len() == limit).then(|| offset.saturating_add(limit) as u64);
        let points = page
            .into_iter()
            .map(|p| point_to_pb(p, req.with_vectors))
            .collect();
        Ok(Response::new(pb::ScrollResponse { points, next_offset }))
    }

    async fn search(&self, req: Request<pb::SearchRequest>) -> Rpc<pb::SearchResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let (k, opts) = search_options(&req)?;
        let hits = self
            .blocking(move |db| db.collection(&req.collection)?.search_with(&req.vector, k, &opts))
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
                        let hits = db.collection(&s.collection)?.search_with(&s.vector, *k, opts)?;
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
}
