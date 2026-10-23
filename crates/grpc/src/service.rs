//! The `quiver.v1.Quiver` service implementation.

use std::pin::Pin;
use std::sync::Arc;

use quiver_db::auth::Role;
use quiver_db::{
    Collection, CollectionUpdate, ContextPair, Database, Direction, Example, Filter, Fusion, GroupBy,
    HybridQuery, MetadataMode, OrderBy, Point, PointSelector, Prefetch, QueryVector, Rescore, RestoreOptions,
    ScrollOptions, SearchHit, SearchOptions, SearchTarget,
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
/// Upper bound on the searches a query fuses.
const MAX_PREFETCH: usize = 16;
const MAX_FACET: usize = 1_000;
const MAX_CONTEXT: usize = 64;
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
    using: &'a str,
    k: u32,
    filter_json: &'a str,
    ef: Option<u32>,
    max_distance: Option<f32>,
    with_vectors: bool,
    with_metadata: Option<bool>,
    offset: u32,
}

impl Settings<'_> {
    fn options(&self) -> Result<(usize, SearchOptions), Status> {
        let k = if self.k == 0 { 10 } else { self.k as usize };
        if k > MAX_K {
            return Err(Status::invalid_argument(format!(
                "k must be between 1 and {MAX_K}"
            )));
        }
        if k + self.offset as usize > MAX_K {
            return Err(Status::invalid_argument(format!(
                "k + offset must be at most {MAX_K}"
            )));
        }
        Ok((
            k,
            SearchOptions {
                using: (!self.using.is_empty()).then(|| self.using.to_owned()),
                filter: filter_from_pb(self.filter_json)?,
                ef: self.ef.map(|e| e as usize),
                include_vectors: self.with_vectors,
                include_metadata: self.with_metadata.unwrap_or(true),
                max_distance: self.max_distance,
                offset: self.offset as usize,
            },
        ))
    }
}

fn search_options(req: &pb::SearchRequest) -> Result<(usize, SearchOptions), Status> {
    let given = [!req.vector.is_empty(), !req.id.is_empty(), req.sparse.is_some()];
    if given.iter().filter(|&&g| g).count() != 1 {
        return Err(Status::invalid_argument(
            "give exactly one of vector, id and sparse",
        ));
    }
    if req.sparse.is_some() && req.using.is_empty() {
        return Err(Status::invalid_argument(
            "name the sparse vector to search in `using`",
        ));
    }
    Settings {
        using: &req.using,
        k: req.k,
        filter_json: &req.filter_json,
        ef: req.ef,
        max_distance: req.max_distance,
        with_vectors: req.with_vectors,
        with_metadata: req.with_metadata,
        offset: req.offset,
    }
    .options()
}

fn target_from_pb(
    vector: Vec<f32>,
    id: String,
    sparse: Option<pb::SparseVector>,
    using: &str,
) -> Result<SearchTarget, Status> {
    match (vector.is_empty(), id.is_empty(), sparse) {
        (false, true, None) => Ok(SearchTarget::Vector(vector)),
        (true, false, None) => Ok(SearchTarget::Id(id)),
        (true, true, Some(sparse)) => {
            if using.is_empty() {
                return Err(Status::invalid_argument(
                    "name the sparse vector to search in `using`",
                ));
            }
            Ok(SearchTarget::Sparse {
                vector: sparse_from_pb(sparse),
                using: using.to_owned(),
            })
        }
        _ => Err(Status::invalid_argument(
            "give exactly one of vector, id and sparse",
        )),
    }
}

fn example_from_pb(e: Option<pb::Example>, what: &str) -> Result<Example, Status> {
    match e.and_then(|e| e.example) {
        Some(pb::example::Example::Id(id)) if !id.is_empty() => Ok(Example::Id(id)),
        Some(pb::example::Example::Vector(v)) if !v.values.is_empty() => Ok(Example::Vector(v.values)),
        _ => Err(Status::invalid_argument(format!(
            "{what} needs an id or a vector"
        ))),
    }
}

fn fusion_from_pb(fusion: i32, rrf_k: Option<u32>) -> Result<Fusion, Status> {
    match pb::Fusion::try_from(fusion) {
        Ok(pb::Fusion::Rrf) => Ok(rrf_k.map_or_else(Fusion::default, |k| Fusion::Rrf { k })),
        Ok(_) if rrf_k.is_some() => Err(Status::invalid_argument(
            "rrf_k only applies to reciprocal rank fusion",
        )),
        Ok(pb::Fusion::RelativeScore) => Ok(Fusion::RelativeScore),
        Ok(pb::Fusion::Dbsf) => Ok(Fusion::Dbsf),
        Err(_) => Err(Status::invalid_argument(format!("unknown fusion {fusion}"))),
    }
}

fn run_search(
    c: &Collection,
    req: &pb::SearchRequest,
    k: usize,
    opts: &SearchOptions,
) -> quiver_db::Result<Vec<SearchHit>> {
    if let Some(sparse) = &req.sparse {
        let opts = SearchOptions {
            using: None,
            ..opts.clone()
        };
        c.search_sparse(&req.using, &sparse_from_pb(sparse.clone()), k, &opts)
    } else if req.id.is_empty() {
        c.search_with(&req.vector, k, opts)
    } else {
        c.search_by_id(&req.id, k, opts)
    }
}

fn prefetch_from_pb(p: pb::Prefetch, k: usize) -> Result<Prefetch, Status> {
    let limit = if p.limit == 0 { k.max(50) } else { p.limit as usize };
    if limit > MAX_K {
        return Err(Status::invalid_argument(format!(
            "prefetch limit must be between 1 and {MAX_K}"
        )));
    }
    let query = match (p.vector.is_empty(), p.sparse) {
        (false, None) => QueryVector::Dense {
            vector: p.vector,
            using: (!p.using.is_empty()).then_some(p.using),
        },
        (true, Some(sparse)) => {
            if p.using.is_empty() {
                return Err(Status::invalid_argument(
                    "name the sparse vector to search in `using`",
                ));
            }
            QueryVector::Sparse {
                vector: sparse_from_pb(sparse),
                using: p.using,
            }
        }
        _ => {
            return Err(Status::invalid_argument(
                "give each prefetch exactly one of vector and sparse",
            ));
        }
    };
    Ok(Prefetch {
        query,
        limit,
        weight: p.weight.unwrap_or(1.0),
        ef: p.ef.map(|e| e as usize),
        max_distance: p.max_distance,
    })
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

    async fn update_collection(&self, req: Request<pb::UpdateCollectionRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let update: CollectionUpdate = serde_json::from_str(&req.update_json)
            .map_err(|e| Status::invalid_argument(format!("update_json is invalid: {e}")))?;
        let info = self
            .blocking(move |db| db.update_collection(&req.name, &update))
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn rename_collection(&self, req: Request<pb::RenameCollectionRequest>) -> Rpc<pb::CollectionInfo> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        let info = self
            .blocking(move |db| Ok(db.rename_collection(&req.name, &req.new_name)?.info()))
            .await?;
        Ok(Response::new(info_to_pb(info)))
    }

    async fn list_aliases(&self, req: Request<pb::ListAliasesRequest>) -> Rpc<pb::ListAliasesResponse> {
        require(&req, Role::ReadOnly)?;
        let aliases = self
            .db
            .aliases()
            .into_iter()
            .map(|(alias, collection)| pb::Alias { alias, collection })
            .collect();
        Ok(Response::new(pb::ListAliasesResponse { aliases }))
    }

    async fn set_alias(&self, req: Request<pb::SetAliasRequest>) -> Rpc<pb::SetAliasResponse> {
        require(&req, Role::ReadWrite)?;
        let req = req.into_inner();
        self.blocking(move |db| db.set_alias(&req.alias, &req.collection))
            .await?;
        Ok(Response::new(pb::SetAliasResponse {}))
    }

    async fn delete_alias(&self, req: Request<pb::DeleteAliasRequest>) -> Rpc<pb::DeleteAliasResponse> {
        require(&req, Role::ReadWrite)?;
        let alias = req.into_inner().alias;
        self.blocking(move |db| db.delete_alias(&alias)).await?;
        Ok(Response::new(pb::DeleteAliasResponse {}))
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
        let mut opts = ScrollOptions::new(limit);
        opts.cursor = (!req.cursor.is_empty()).then(|| req.cursor.clone());
        opts.filter = filter_from_pb(&req.filter_json)?;
        opts.order_by = match &req.order_by {
            None => None,
            Some(o) => Some(OrderBy {
                field: o.field.clone(),
                direction: match pb::Direction::try_from(o.direction) {
                    Ok(pb::Direction::Asc) => Direction::Asc,
                    Ok(pb::Direction::Desc) => Direction::Desc,
                    Err(_) => {
                        return Err(Status::invalid_argument(format!(
                            "unknown direction {}",
                            o.direction
                        )));
                    }
                },
            }),
        };
        let collection = req.collection.clone();
        let page = self
            .blocking(move |db| db.collection(&collection)?.scroll_with(&opts))
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
            using: &req.using,
            k: req.k,
            filter_json: &req.filter_json,
            ef: req.ef,
            max_distance: req.max_distance,
            with_vectors: req.with_vectors,
            with_metadata: req.with_metadata,
            offset: req.offset,
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

    async fn query(&self, req: Request<pb::QueryRequest>) -> Rpc<pb::QueryResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let k = if req.k == 0 { 10 } else { req.k as usize };
        if k > MAX_K {
            return Err(Status::invalid_argument(format!(
                "k must be between 1 and {MAX_K}"
            )));
        }
        if k + req.offset as usize > MAX_K {
            return Err(Status::invalid_argument(format!(
                "k + offset must be at most {MAX_K}"
            )));
        }
        if req.prefetch.is_empty() || req.prefetch.len() > MAX_PREFETCH {
            return Err(Status::invalid_argument(format!(
                "give between 1 and {MAX_PREFETCH} prefetch searches"
            )));
        }
        let query = HybridQuery {
            prefetch: req
                .prefetch
                .into_iter()
                .map(|p| prefetch_from_pb(p, k + req.offset as usize))
                .collect::<Result<_, _>>()?,
            fusion: fusion_from_pb(req.fusion, req.rrf_k)?,
            filter: filter_from_pb(&req.filter_json)?,
            include_vectors: req.with_vectors,
            include_metadata: req.with_metadata.unwrap_or(true),
            rescore: req.rescore.map(|r| Rescore {
                vector: r.vector,
                using: (!r.using.is_empty()).then_some(r.using),
            }),
            offset: req.offset as usize,
        };
        let collection = req.collection;
        let hits = self
            .blocking(move |db| db.collection(&collection)?.query(&query, k))
            .await?;
        Ok(Response::new(pb::QueryResponse {
            hits: hits.into_iter().map(scored_to_pb).collect(),
        }))
    }

    async fn search_groups(&self, req: Request<pb::SearchGroupsRequest>) -> Rpc<pb::SearchGroupsResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let groups = if req.groups == 0 { 10 } else { req.groups as usize };
        let group_size = if req.group_size == 0 {
            3
        } else {
            req.group_size as usize
        };
        if groups * group_size > MAX_K {
            return Err(Status::invalid_argument(format!(
                "groups * group_size must be at most {MAX_K}"
            )));
        }
        if req.group_by.is_empty() {
            return Err(Status::invalid_argument("group_by is required"));
        }
        let target = target_from_pb(req.vector, req.id, req.sparse, &req.using)?;
        let (_, mut opts) = Settings {
            using: &req.using,
            k: 1,
            filter_json: &req.filter_json,
            ef: req.ef,
            max_distance: req.max_distance,
            with_vectors: req.with_vectors,
            with_metadata: req.with_metadata,
            offset: 0,
        }
        .options()?;
        if matches!(target, SearchTarget::Sparse { .. }) {
            opts.using = None;
        }
        let group = GroupBy::new(req.group_by, groups, group_size);
        let collection = req.collection;
        let found = self
            .blocking(move |db| db.collection(&collection)?.search_groups(&target, &group, &opts))
            .await?;
        Ok(Response::new(pb::SearchGroupsResponse {
            groups: found
                .into_iter()
                .map(|g| pb::HitGroup {
                    key_json: g.key.to_string(),
                    hits: g.hits.into_iter().map(hit_to_pb).collect(),
                })
                .collect(),
        }))
    }

    async fn discover(&self, req: Request<pb::DiscoverRequest>) -> Rpc<pb::SearchResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        if req.context.len() > MAX_CONTEXT {
            return Err(Status::invalid_argument(format!(
                "at most {MAX_CONTEXT} context pairs per request"
            )));
        }
        let target = req
            .target
            .map(|t| example_from_pb(Some(t), "target"))
            .transpose()?;
        let context = req
            .context
            .into_iter()
            .map(|p| {
                Ok(ContextPair {
                    positive: example_from_pb(p.positive, "a context pair's positive")?,
                    negative: example_from_pb(p.negative, "a context pair's negative")?,
                })
            })
            .collect::<Result<Vec<_>, Status>>()?;
        let (k, opts) = Settings {
            using: &req.using,
            k: req.k,
            filter_json: &req.filter_json,
            ef: req.ef,
            max_distance: req.max_distance,
            with_vectors: req.with_vectors,
            with_metadata: req.with_metadata,
            offset: req.offset,
        }
        .options()?;
        let collection = req.collection;
        let hits = self
            .blocking(move |db| {
                db.collection(&collection)?
                    .discover(target.as_ref(), &context, k, &opts)
            })
            .await?;
        Ok(Response::new(pb::SearchResponse {
            hits: hits.into_iter().map(hit_to_pb).collect(),
        }))
    }

    async fn facet(&self, req: Request<pb::FacetRequest>) -> Rpc<pb::FacetResponse> {
        require(&req, Role::ReadOnly)?;
        let req = req.into_inner();
        let limit = if req.limit == 0 { 10 } else { req.limit as usize };
        if limit > MAX_FACET {
            return Err(Status::invalid_argument(format!("limit must be <= {MAX_FACET}")));
        }
        let filter = filter_from_pb(&req.filter_json)?;
        let values = self
            .blocking(move |db| {
                db.collection(&req.collection)?
                    .facet(&req.field, filter.as_ref(), limit)
            })
            .await?;
        Ok(Response::new(pb::FacetResponse {
            values: values
                .into_iter()
                .map(|v| pb::FacetValue {
                    value_json: v.value.to_string(),
                    count: v.count as u64,
                })
                .collect(),
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
