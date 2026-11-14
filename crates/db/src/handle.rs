use std::path::Path;

use quiver_snapshot::SnapshotInfo;
use quiver_types::{
    CollectionConfig, CollectionInfo, CollectionUpdate, ContextPair, Error, Example, FacetCount, Filter,
    GroupBy, HitGroup, HybridQuery, MetadataMode, Point, PointSelector, Result, ScoredHit, ScrollOptions,
    SearchHit, SearchOptions, SearchTarget, SparseVector,
};
use serde_json::Value;

use crate::collection::LocalCollection;
use crate::remote::RemoteCollection;
use crate::replication::{Outcome, Write};
use crate::{LogPosition, ScrollPage, StorageKind};

/// A collection of points: held by this process, or sharded across the
/// nodes of a cluster. Both offer the same operations.
#[derive(Debug)]
pub struct Collection {
    repr: Repr,
}

// Collections are handed out behind an `Arc`, so the size difference costs
// one allocation per collection, not per use.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum Repr {
    Local(LocalCollection),
    Remote(RemoteCollection),
}

macro_rules! dispatch {
    ($self:ident, $c:ident => $e:expr) => {
        match &$self.repr {
            Repr::Local($c) => $e,
            Repr::Remote($c) => $e,
        }
    };
}

fn local_only(what: &str) -> Error {
    Error::Unsupported(format!(
        "{what} is not available for collections in a cluster yet"
    ))
}

impl Collection {
    pub(crate) fn local(c: LocalCollection) -> Self {
        Self { repr: Repr::Local(c) }
    }

    pub(crate) fn remote(c: RemoteCollection) -> Self {
        Self {
            repr: Repr::Remote(c),
        }
    }

    pub(crate) fn as_local(&self) -> Option<&LocalCollection> {
        match &self.repr {
            Repr::Local(c) => Some(c),
            Repr::Remote(_) => None,
        }
    }

    /// Whether this collection's points live on a cluster's nodes rather
    /// than in this process.
    pub fn is_distributed(&self) -> bool {
        matches!(self.repr, Repr::Remote(_))
    }

    pub fn name(&self) -> &str {
        dispatch!(self, c => c.name())
    }

    /// The collection's current configuration.
    pub fn config(&self) -> CollectionConfig {
        dispatch!(self, c => c.config())
    }

    pub fn storage(&self) -> StorageKind {
        dispatch!(self, c => c.storage())
    }

    /// Changes whenever the collection is written to (always 0 for
    /// distributed collections).
    pub fn generation(&self) -> u64 {
        dispatch!(self, c => c.generation())
    }

    /// Number of points, not counting expired ones.
    pub fn len(&self) -> usize {
        dispatch!(self, c => c.len())
    }

    pub fn is_empty(&self) -> bool {
        dispatch!(self, c => c.is_empty())
    }

    /// Bytes of memory the vectors take, tombstones included.
    pub fn vector_bytes(&self) -> usize {
        dispatch!(self, c => c.vector_bytes())
    }

    /// The collection's statistics. For a distributed collection whose
    /// shards cannot all be reached, the counts are 0; see [`Self::try_info`].
    pub fn info(&self) -> CollectionInfo {
        dispatch!(self, c => c.info())
    }

    /// [`Self::info`], failing when a distributed collection's shards cannot
    /// all be reached.
    pub fn try_info(&self) -> Result<CollectionInfo> {
        match &self.repr {
            Repr::Local(c) => Ok(c.info()),
            Repr::Remote(c) => c.try_info(),
        }
    }

    /// Changes the collection's index, quantization or on-disk settings.
    ///
    /// Indexes whose settings changed (other than `ef_search`) are rebuilt,
    /// and vectors are moved to or from disk. That can take a while; reads
    /// carry on meanwhile, writes wait.
    pub fn update(&self, update: &CollectionUpdate) -> Result<CollectionInfo> {
        dispatch!(self, c => c.update(update))
    }

    /// Indexes a metadata field (a dot-separated path, as in filters) so
    /// filters on it look at matching points only. Indexing a field that
    /// already has an index does nothing.
    pub fn create_field_index(&self, field: &str) -> Result<()> {
        dispatch!(self, c => c.create_field_index(field))
    }

    /// Drops a field index; returns whether there was one.
    pub fn drop_field_index(&self, field: &str) -> Result<bool> {
        dispatch!(self, c => c.drop_field_index(field))
    }

    pub fn insert(&self, vector: Vec<f32>, metadata: Option<Value>) -> Result<String> {
        dispatch!(self, c => c.insert(vector, metadata))
    }

    pub fn upsert(&self, points: impl IntoIterator<Item = Point>) -> Result<usize> {
        dispatch!(self, c => c.upsert(points))
    }

    pub fn delete<S: AsRef<str>>(&self, ids: &[S]) -> Result<usize> {
        dispatch!(self, c => c.delete(ids))
    }

    /// Deletes every point whose metadata matches `filter`; returns how many.
    pub fn delete_where(&self, filter: &Filter) -> Result<usize> {
        dispatch!(self, c => c.delete_where(filter))
    }

    /// Number of points, or of points whose metadata matches `filter`. For a
    /// distributed collection whose shards cannot all be reached this is 0;
    /// see [`Self::try_count`].
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        dispatch!(self, c => c.count(filter))
    }

    /// [`Self::count`], failing when a distributed collection's shards
    /// cannot all be reached.
    pub fn try_count(&self, filter: Option<&Filter>) -> Result<usize> {
        match &self.repr {
            Repr::Local(c) => Ok(c.count(filter)),
            Repr::Remote(c) => c.try_count(filter),
        }
    }

    /// Changes the metadata of the selected points without touching their
    /// vectors, and returns how many were updated (missing ids are skipped).
    ///
    /// Every new value is validated before anything changes, so a batch is
    /// applied completely or not at all (on each shard, for a distributed
    /// collection).
    pub fn update_metadata(
        &self,
        selector: &PointSelector,
        metadata: Value,
        mode: MetadataMode,
    ) -> Result<usize> {
        dispatch!(self, c => c.update_metadata(selector, metadata, mode))
    }

    /// The point `id`. `None` also when a distributed collection's shard
    /// cannot be reached; see [`Self::try_get`].
    pub fn get(&self, id: &str) -> Option<Point> {
        dispatch!(self, c => c.get(id))
    }

    /// [`Self::get`], failing when the point's shard cannot be reached.
    pub fn try_get(&self, id: &str) -> Result<Option<Point>> {
        match &self.repr {
            Repr::Local(c) => Ok(c.get(id)),
            Repr::Remote(c) => c.try_get(id),
        }
    }

    pub fn get_many<S: AsRef<str>>(&self, ids: &[S]) -> Vec<Point> {
        dispatch!(self, c => c.get_many(ids))
    }

    /// [`Self::get_many`], failing when a shard holding some of the points
    /// cannot be reached.
    pub fn try_get_many<S: AsRef<str>>(&self, ids: &[S]) -> Result<Vec<Point>> {
        match &self.repr {
            Repr::Local(c) => Ok(c.get_many(ids)),
            Repr::Remote(c) => c.try_get_many(ids),
        }
    }

    /// Up to `limit` points with an id greater than `cursor` (from the start
    /// if `None`), in ascending byte-wise id order. Paging by id rather than
    /// position means deletes and updates between pages never cause points
    /// to be skipped or repeated.
    pub fn scroll(&self, cursor: Option<&str>, limit: usize) -> Result<ScrollPage> {
        dispatch!(self, c => c.scroll(cursor, limit))
    }

    pub fn scroll_with(&self, opts: &ScrollOptions) -> Result<ScrollPage> {
        dispatch!(self, c => c.scroll_with(opts))
    }

    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        dispatch!(self, c => c.search(vector, k))
    }

    /// The `k` points nearest to `vector` in the dense vector `opts.using`.
    pub fn search_with(&self, vector: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        dispatch!(self, c => c.search_with(vector, k, opts))
    }

    /// Searches with the stored vector of point `id`, leaving that point out
    /// of the results.
    pub fn search_by_id(&self, id: &str, k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        dispatch!(self, c => c.search_by_id(id, k, opts))
    }

    /// The `k` points scoring highest against `vector` in the sparse vector
    /// `name`. Hits' distances are their negated scores, so lower is better.
    pub fn search_sparse(
        &self,
        name: &str,
        vector: &SparseVector,
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        dispatch!(self, c => c.search_sparse(name, vector, k, opts))
    }

    pub fn search_groups(
        &self,
        target: &SearchTarget,
        group: &GroupBy,
        opts: &SearchOptions,
    ) -> Result<Vec<HitGroup>> {
        dispatch!(self, c => c.search_groups(target, group, opts))
    }

    pub fn facet(&self, field: &str, filter: Option<&Filter>, limit: usize) -> Result<Vec<FacetCount>> {
        dispatch!(self, c => c.facet(field, filter, limit))
    }

    pub fn discover(
        &self,
        target: Option<&Example>,
        context: &[ContextPair],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        dispatch!(self, c => c.discover(target, context, k, opts))
    }

    /// Runs several searches, dense and sparse, and fuses their rankings.
    /// Returns the `k` best points by fused score.
    pub fn query(&self, query: &HybridQuery, k: usize) -> Result<Vec<ScoredHit>> {
        dispatch!(self, c => c.query(query, k))
    }

    /// Finds points like the `positive` examples and unlike the `negative`
    /// ones, in the dense vector `opts.using`. The query is the mean of the
    /// positives, pushed away from the mean of the negatives
    /// (`2·mean(pos) − mean(neg)`) when there are any. The examples
    /// themselves are left out of the results.
    pub fn recommend<S: AsRef<str>>(
        &self,
        positive: &[S],
        negative: &[S],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        dispatch!(self, c => c.recommend(positive, negative, k, opts))
    }

    pub fn flush(&self) -> Result<()> {
        dispatch!(self, c => c.flush())
    }

    pub fn compact(&self) -> Result<()> {
        dispatch!(self, c => c.compact())
    }

    /// Deletes the points that have expired; returns how many. Reads never
    /// see expired points anyway; this frees the space they take.
    pub fn purge_expired(&self) -> Result<usize> {
        dispatch!(self, c => c.purge_expired())
    }

    pub fn create_snapshot(&self) -> Result<SnapshotInfo> {
        match &self.repr {
            Repr::Local(c) => c.create_snapshot(),
            Repr::Remote(_) => Err(local_only("snapshots")),
        }
    }

    /// Writes the collection's points and indexes, every vector included.
    pub fn export_state(&self, w: impl std::io::Write) -> Result<()> {
        match &self.repr {
            Repr::Local(c) => c.export_state(w),
            Repr::Remote(_) => Err(local_only("exporting state")),
        }
    }

    /// The last replicated log entry this collection applied, when it is
    /// the replica of a cluster's shard.
    pub fn applied(&self) -> Option<LogPosition> {
        self.as_local().and_then(LocalCollection::applied)
    }

    /// Applies one entry of a shard's replicated log; see
    /// [`crate::replication`]. Only for a shard's local replica.
    pub fn apply(&self, at: LogPosition, write: &Write) -> Result<Outcome> {
        match &self.repr {
            Repr::Local(c) => c.apply(at, write),
            Repr::Remote(_) => Err(local_only("applying log entries")),
        }
    }

    /// Records that log entries up to the one at `at` are applied, for
    /// entries that carry no command.
    pub fn mark_applied(&self, at: LogPosition) -> Result<()> {
        match &self.repr {
            Repr::Local(c) => c.mark_applied(at),
            Repr::Remote(_) => Err(local_only("applying log entries")),
        }
    }

    /// Writes this replica to a snapshot archive at `path`.
    pub fn export_archive(&self, path: &Path) -> Result<()> {
        match &self.repr {
            Repr::Local(c) => c.export_archive(path),
            Repr::Remote(_) => Err(local_only("exporting archives")),
        }
    }
}
