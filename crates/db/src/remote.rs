//! Collections sharded across a cluster, as seen from any of its nodes.
//!
//! A [`ClusterBackend`] moves commands and reads to the shards' replicas;
//! [`RemoteCollection`] splits each operation into per-shard parts and
//! merges what the shards answer, so callers see one collection.

use std::collections::{BTreeMap, HashMap, HashSet};

use quiver_index::Neighbor;
use quiver_types::{
    CollectionConfig, CollectionInfo, CollectionUpdate, ContextPair, Direction, Error, Example, FacetCount,
    Filter, GroupBy, HitGroup, HybridQuery, MetadataMode, Point, PointSelector, QueryVector, Result,
    ScoredHit, ScrollOptions, SearchHit, SearchOptions, SearchTarget, SparseVector, validate_field_path,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{fuse, parse_order_cursor, validate_point, validate_vector};
use crate::grouping::{self, Key};
use crate::replication::{Command, Outcome, ShardInfo, ShardRead, ShardReply, Write, shard_of};
use crate::{ScrollPage, StorageKind};

/// A collection as a cluster's catalog records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardedCollection {
    pub config: CollectionConfig,
    /// How many shards its points are spread over; fixed at creation.
    pub shards: u32,
}

/// The collections and aliases of a cluster.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterCatalog {
    pub collections: BTreeMap<String, ShardedCollection>,
    /// Alias to collection name.
    pub aliases: BTreeMap<String, String>,
}

impl ClusterCatalog {
    /// The collection `name` names, directly or as an alias.
    pub fn resolve(&self, name: &str) -> Option<(&str, &ShardedCollection)> {
        let (canonical, c) = self.collections.get_key_value(name).or_else(|| {
            self.aliases
                .get(name)
                .and_then(|target| self.collections.get_key_value(target))
        })?;
        Some((canonical.as_str(), c))
    }
}

/// Maintenance every replica of a collection runs on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Maintenance {
    Flush,
    Compact,
}

/// How a node reaches the rest of its cluster. Implemented by the cluster
/// crate; the database calls it from blocking threads.
pub trait ClusterBackend: Send + Sync + std::fmt::Debug {
    /// The storage engine shard replicas use.
    fn storage(&self) -> StorageKind;

    /// The catalog as this node knows it.
    fn catalog(&self) -> Result<ClusterCatalog>;

    /// The collection `name` names, directly or as an alias. Unlike
    /// [`Self::catalog`], asks the cluster's leader before answering `None`,
    /// so a collection just created elsewhere is found.
    fn resolve(&self, name: &str) -> Result<Option<(String, ShardedCollection)>>;

    fn create_collection(
        &self,
        name: &str,
        config: &CollectionConfig,
        field_indexes: &[String],
    ) -> Result<()>;
    fn drop_collection(&self, name: &str) -> Result<()>;
    fn rename_collection(&self, from: &str, to: &str) -> Result<()>;
    fn set_alias(&self, alias: &str, collection: &str) -> Result<()>;
    fn delete_alias(&self, alias: &str) -> Result<()>;
    /// Records a collection's new settings in the catalog, once its shards
    /// have applied them.
    fn set_config(&self, collection: &str, config: &CollectionConfig) -> Result<()>;

    /// Applies each write to its shard of `collection`, all at once.
    /// Succeeds only if every one does.
    fn write(&self, collection: &str, writes: Vec<(u32, Write)>) -> Result<Vec<Outcome>>;

    /// Sends each read to its shard of `collection`, all at once.
    fn read(&self, collection: &str, reads: Vec<(u32, ShardRead)>) -> Result<Vec<ShardReply>>;

    /// Runs maintenance on every replica of `collection`.
    fn maintain(&self, collection: &str, op: Maintenance) -> Result<()>;
}

/// A collection whose shards live on a cluster's nodes.
#[derive(Debug, Clone)]
pub(crate) struct RemoteCollection {
    name: String,
    shards: u32,
    config: CollectionConfig,
    backend: std::sync::Arc<dyn ClusterBackend>,
}

fn unexpected(reply: &ShardReply) -> Error {
    Error::Storage(format!("unexpected reply from a shard: {reply:?}"))
}

/// Hits of several shards in one ranking: by distance, then id.
fn merge_hits(lists: Vec<Vec<SearchHit>>, skip: &HashSet<&str>, offset: usize, k: usize) -> Vec<SearchHit> {
    let mut all: Vec<SearchHit> = lists
        .into_iter()
        .flatten()
        .filter(|h| !skip.contains(h.id.as_str()))
        .collect();
    all.sort_by(|a, b| a.distance.total_cmp(&b.distance).then_with(|| a.id.cmp(&b.id)));
    all.into_iter().skip(offset).take(k).collect()
}

/// The vector `using` names in `p`.
fn vector_of(p: &Point, using: Option<&str>) -> Result<Vec<f32>> {
    match using {
        None => Ok(p.vector.clone()),
        Some(name) => p
            .vectors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::invalid(format!("the collection has no vector named `{name}`"))),
    }
}

impl RemoteCollection {
    pub(crate) fn new(
        name: String,
        sharded: ShardedCollection,
        backend: std::sync::Arc<dyn ClusterBackend>,
    ) -> Self {
        Self {
            name,
            shards: sharded.shards.max(1),
            config: sharded.config,
            backend,
        }
    }

    fn all_shards(&self) -> impl Iterator<Item = u32> + use<> {
        0..self.shards
    }

    fn read_all(&self, read: ShardRead) -> Result<Vec<ShardReply>> {
        let reads = self.all_shards().map(|s| (s, read.clone())).collect();
        self.backend.read(&self.name, reads)
    }

    fn write_all(&self, command: Command) -> Result<Vec<Outcome>> {
        let write = Write::now(command);
        let writes = self.all_shards().map(|s| (s, write.clone())).collect();
        self.backend.write(&self.name, writes)
    }

    fn hits_of(replies: Vec<ShardReply>) -> Result<Vec<Vec<SearchHit>>> {
        replies
            .into_iter()
            .map(|r| match r {
                ShardReply::Hits(h) => Ok(h),
                other => Err(unexpected(&other)),
            })
            .collect()
    }

    /// Ids grouped by the shard holding them, each group in input order.
    fn by_shard<S: AsRef<str>>(&self, ids: &[S]) -> BTreeMap<u32, Vec<String>> {
        let mut out: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for id in ids {
            let id = id.as_ref();
            out.entry(shard_of(id, self.shards))
                .or_default()
                .push(id.to_owned());
        }
        out
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> CollectionConfig {
        match self.backend.resolve(&self.name) {
            Ok(Some((_, c))) => c.config,
            _ => self.config.clone(),
        }
    }

    pub fn storage(&self) -> StorageKind {
        self.backend.storage()
    }

    pub fn generation(&self) -> u64 {
        0
    }

    pub fn len(&self) -> usize {
        self.count(None)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn vector_bytes(&self) -> usize {
        self.shard_infos()
            .map(|infos| infos.iter().map(|i| i.vector_bytes).sum())
            .unwrap_or_default()
    }

    fn shard_infos(&self) -> Result<Vec<ShardInfo>> {
        self.read_all(ShardRead::Info)?
            .into_iter()
            .map(|r| match r {
                ShardReply::Info(i) => Ok(i),
                other => Err(unexpected(&other)),
            })
            .collect()
    }

    pub fn try_info(&self) -> Result<CollectionInfo> {
        let infos = self.shard_infos()?;
        Ok(CollectionInfo {
            name: self.name.clone(),
            config: self.config(),
            count: infos.iter().map(|i| i.count).sum(),
            tombstones: infos.iter().map(|i| i.tombstones).sum(),
            field_indexes: infos.first().map(|i| i.field_indexes.clone()).unwrap_or_default(),
        })
    }

    pub fn info(&self) -> CollectionInfo {
        self.try_info().unwrap_or_else(|e| {
            tracing::warn!(collection = %self.name, error = %e, "cannot reach every shard");
            CollectionInfo {
                name: self.name.clone(),
                config: self.config(),
                count: 0,
                tombstones: 0,
                field_indexes: Vec::new(),
            }
        })
    }

    pub fn update(&self, update: &CollectionUpdate) -> Result<CollectionInfo> {
        let old = self.config();
        let new = update.apply(&old)?;
        if new.uses_disk() && !self.storage().is_durable() {
            return Err(Error::invalid(
                "vectors can only be kept on disk with a durable storage engine",
            ));
        }
        if new != old {
            self.write_all(Command::Update {
                update: update.clone(),
            })?;
            self.backend.set_config(&self.name, &new)?;
        }
        self.try_info()
    }

    pub fn create_field_index(&self, field: &str) -> Result<()> {
        validate_field_path(field)?;
        self.write_all(Command::CreateFieldIndex {
            field: field.to_owned(),
        })?;
        Ok(())
    }

    pub fn drop_field_index(&self, field: &str) -> Result<bool> {
        let outcomes = self.write_all(Command::DropFieldIndex {
            field: field.to_owned(),
        })?;
        Ok(outcomes.contains(&Outcome::Changed(true)))
    }

    pub fn insert(&self, vector: Vec<f32>, metadata: Option<Value>) -> Result<String> {
        let mut point = Point::with_random_id(vector);
        point.metadata = metadata;
        let id = point.id.clone();
        self.upsert([point])?;
        Ok(id)
    }

    pub fn upsert(&self, points: impl IntoIterator<Item = Point>) -> Result<usize> {
        let config = self.config();
        let mut by_shard: BTreeMap<u32, Vec<Point>> = BTreeMap::new();
        let mut n = 0;
        for mut p in points {
            // Checked here too, so a bad point fails the batch before any
            // shard has written any of it.
            validate_point(&config, &mut p)?;
            by_shard.entry(shard_of(&p.id, self.shards)).or_default().push(p);
            n += 1;
        }
        if n == 0 {
            return Ok(0);
        }
        let now = crate::expiry::now();
        let writes = by_shard
            .into_iter()
            .map(|(s, points)| {
                (
                    s,
                    Write {
                        now,
                        command: Command::Upsert { points },
                    },
                )
            })
            .collect();
        self.backend.write(&self.name, writes)?;
        Ok(n)
    }

    pub fn delete<S: AsRef<str>>(&self, ids: &[S]) -> Result<usize> {
        let now = crate::expiry::now();
        let writes = self
            .by_shard(ids)
            .into_iter()
            .map(|(s, ids)| {
                (
                    s,
                    Write {
                        now,
                        command: Command::Delete { ids },
                    },
                )
            })
            .collect();
        Ok(self
            .backend
            .write(&self.name, writes)?
            .iter()
            .map(Outcome::count)
            .sum())
    }

    pub fn delete_where(&self, filter: &Filter) -> Result<usize> {
        let outcomes = self.write_all(Command::DeleteWhere {
            filter: filter.clone(),
        })?;
        Ok(outcomes.iter().map(Outcome::count).sum())
    }

    pub fn try_count(&self, filter: Option<&Filter>) -> Result<usize> {
        self.read_all(ShardRead::Count {
            filter: filter.cloned(),
        })?
        .into_iter()
        .map(|r| match r {
            ShardReply::Count(n) => Ok(n),
            other => Err(unexpected(&other)),
        })
        .sum()
    }

    pub fn count(&self, filter: Option<&Filter>) -> usize {
        self.try_count(filter).unwrap_or_else(|e| {
            tracing::warn!(collection = %self.name, error = %e, "cannot count every shard");
            0
        })
    }

    pub fn update_metadata(
        &self,
        selector: &PointSelector,
        metadata: Value,
        mode: MetadataMode,
    ) -> Result<usize> {
        let now = crate::expiry::now();
        let outcomes = match selector {
            PointSelector::Ids(ids) => {
                let writes = self
                    .by_shard(ids)
                    .into_iter()
                    .map(|(s, ids)| {
                        (
                            s,
                            Write {
                                now,
                                command: Command::UpdateMetadata {
                                    selector: PointSelector::Ids(ids),
                                    metadata: metadata.clone(),
                                    mode,
                                },
                            },
                        )
                    })
                    .collect();
                self.backend.write(&self.name, writes)?
            }
            PointSelector::Filter(_) => self.write_all(Command::UpdateMetadata {
                selector: selector.clone(),
                metadata,
                mode,
            })?,
        };
        Ok(outcomes.iter().map(Outcome::count).sum())
    }

    pub fn try_get_many<S: AsRef<str>>(&self, ids: &[S]) -> Result<Vec<Point>> {
        let reads = self
            .by_shard(ids)
            .into_iter()
            .map(|(s, ids)| (s, ShardRead::Get { ids }))
            .collect();
        let mut found: HashMap<String, Point> = HashMap::new();
        for reply in self.backend.read(&self.name, reads)? {
            match reply {
                ShardReply::Points(points) => found.extend(points.into_iter().map(|p| (p.id.clone(), p))),
                other => return Err(unexpected(&other)),
            }
        }
        Ok(ids
            .iter()
            .filter_map(|id| found.get(id.as_ref()).cloned())
            .collect())
    }

    pub fn try_get(&self, id: &str) -> Result<Option<Point>> {
        Ok(self.try_get_many(&[id])?.pop())
    }

    pub fn get(&self, id: &str) -> Option<Point> {
        self.try_get(id).unwrap_or_else(|e| {
            tracing::warn!(collection = %self.name, error = %e, "cannot read a shard");
            None
        })
    }

    pub fn get_many<S: AsRef<str>>(&self, ids: &[S]) -> Vec<Point> {
        self.try_get_many(ids).unwrap_or_else(|e| {
            tracing::warn!(collection = %self.name, error = %e, "cannot read every shard");
            Vec::new()
        })
    }

    /// The point `id`, or [`Error::PointNotFound`].
    fn point(&self, id: &str) -> Result<Point> {
        self.try_get(id)?
            .ok_or_else(|| Error::PointNotFound(id.to_owned()))
    }

    pub fn scroll(&self, cursor: Option<&str>, limit: usize) -> Result<ScrollPage> {
        let mut opts = ScrollOptions::new(limit);
        opts.cursor = cursor.map(str::to_owned);
        self.scroll_with(&opts)
    }

    pub fn scroll_with(&self, opts: &ScrollOptions) -> Result<ScrollPage> {
        if let Some(order) = &opts.order_by {
            validate_field_path(&order.field)?;
            if let Some(cursor) = &opts.cursor {
                parse_order_cursor(cursor)?;
            }
        }
        let mut more = false;
        let mut points = Vec::new();
        for reply in self.read_all(ShardRead::Scroll { opts: opts.clone() })? {
            match reply {
                ShardReply::Page(page) => {
                    more |= page.next_cursor.is_some();
                    points.extend(page.points);
                }
                other => return Err(unexpected(&other)),
            }
        }
        let Some(order) = &opts.order_by else {
            points.sort_by(|a, b| a.id.cmp(&b.id));
            more |= points.len() > opts.limit;
            points.truncate(opts.limit);
            let next_cursor = points.last().filter(|_| more).map(|p| p.id.clone());
            return Ok(ScrollPage { points, next_cursor });
        };
        let path = grouping::split_path(&order.field);
        let key = |p: &Point| {
            grouping::lookup(p.metadata.as_ref(), &path)
                .cloned()
                .unwrap_or(Value::Null)
        };
        let mut rows: Vec<(Value, Point)> = points.into_iter().map(|p| (key(&p), p)).collect();
        rows.sort_by(|a, b| {
            let by_value = grouping::value_order(&a.0, &b.0);
            let by_value = match order.direction {
                Direction::Asc => by_value,
                Direction::Desc => by_value.reverse(),
            };
            by_value.then_with(|| a.1.id.cmp(&b.1.id))
        });
        more |= rows.len() > opts.limit;
        rows.truncate(opts.limit);
        let next_cursor = rows
            .last()
            .filter(|_| more)
            .map(|(v, p)| serde_json::json!([v, p.id]).to_string());
        Ok(ScrollPage {
            points: rows.into_iter().map(|(_, p)| p).collect(),
            next_cursor,
        })
    }

    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        self.search_with(vector, k, &SearchOptions::default())
    }

    /// Asks every shard for its best `k + offset + skip.len()` and keeps
    /// the overall best `k` after `offset`, leaving out `skip`.
    fn dense(
        &self,
        vector: &[f32],
        k: usize,
        opts: &SearchOptions,
        skip: &HashSet<&str>,
    ) -> Result<Vec<SearchHit>> {
        let mut query = vector.to_vec();
        validate_vector(&self.config().dense(opts.using.as_deref())?, &mut query)?;
        let per_shard = k.saturating_add(opts.offset).saturating_add(skip.len());
        let shard_opts = SearchOptions {
            offset: 0,
            ..opts.clone()
        };
        let replies = self.read_all(ShardRead::Search {
            vector: query,
            k: per_shard,
            opts: shard_opts,
        })?;
        Ok(merge_hits(Self::hits_of(replies)?, skip, opts.offset, k))
    }

    pub fn search_with(&self, vector: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        self.dense(vector, k, opts, &HashSet::new())
    }

    pub fn search_by_id(&self, id: &str, k: usize, opts: &SearchOptions) -> Result<Vec<SearchHit>> {
        let config = self.config();
        config.dense(opts.using.as_deref())?;
        let vector = vector_of(&self.point(id)?, opts.using.as_deref())?;
        self.dense(&vector, k, opts, &HashSet::from([id]))
    }

    pub fn search_sparse(
        &self,
        name: &str,
        vector: &SparseVector,
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        let mut query = vector.clone();
        query.normalize()?;
        self.config().sparse(name)?;
        let replies = self.read_all(ShardRead::SearchSparse {
            name: name.to_owned(),
            vector: query,
            k: k.saturating_add(opts.offset),
            opts: SearchOptions {
                offset: 0,
                ..opts.clone()
            },
        })?;
        Ok(merge_hits(
            Self::hits_of(replies)?,
            &HashSet::new(),
            opts.offset,
            k,
        ))
    }

    pub fn search_groups(
        &self,
        target: &SearchTarget,
        group: &GroupBy,
        opts: &SearchOptions,
    ) -> Result<Vec<HitGroup>> {
        validate_field_path(&group.field)?;
        if group.groups == 0 || group.group_size == 0 {
            return Err(Error::invalid("`groups` and `group_size` must be at least 1"));
        }
        let using = opts.using.as_deref();
        let (target, skip) = match target {
            SearchTarget::Id(id) => {
                self.config().dense(using)?;
                let vector = vector_of(&self.point(id)?, using)?;
                (SearchTarget::Vector(vector), Some(id.as_str()))
            }
            other => (other.clone(), None),
        };
        // One more per group where the target point itself may take a place.
        let shard_group = GroupBy {
            group_size: group.group_size + usize::from(skip.is_some()),
            ..group.clone()
        };
        let replies = self.read_all(ShardRead::Groups {
            target,
            group: shard_group,
            opts: SearchOptions {
                offset: 0,
                ..opts.clone()
            },
        })?;
        let mut index: HashMap<Key, usize> = HashMap::new();
        let mut groups: Vec<HitGroup> = Vec::new();
        for reply in replies {
            let ShardReply::Groups(found) = reply else {
                return Err(unexpected(&reply));
            };
            for g in found {
                let Some(key) = Key::of(&g.key) else { continue };
                let i = *index.entry(key).or_insert_with(|| {
                    groups.push(HitGroup {
                        key: g.key.clone(),
                        hits: Vec::new(),
                    });
                    groups.len() - 1
                });
                groups[i]
                    .hits
                    .extend(g.hits.into_iter().filter(|h| Some(h.id.as_str()) != skip));
            }
        }
        for g in &mut groups {
            g.hits
                .sort_by(|a, b| a.distance.total_cmp(&b.distance).then_with(|| a.id.cmp(&b.id)));
            g.hits.truncate(group.group_size);
        }
        groups.retain(|g| !g.hits.is_empty());
        groups.sort_by(|a, b| {
            let best = |g: &HitGroup| (g.hits[0].distance, g.hits[0].id.clone());
            let (da, ia) = best(a);
            let (db, ib) = best(b);
            da.total_cmp(&db).then(ia.cmp(&ib))
        });
        groups.truncate(group.groups);
        Ok(groups)
    }

    pub fn facet(&self, field: &str, filter: Option<&Filter>, limit: usize) -> Result<Vec<FacetCount>> {
        validate_field_path(field)?;
        let replies = self.read_all(ShardRead::Facet {
            field: field.to_owned(),
            filter: filter.cloned(),
            // Every value: one shard's tail can be another's head.
            limit: usize::MAX,
        })?;
        let mut counts: HashMap<Key, FacetCount> = HashMap::new();
        for reply in replies {
            let ShardReply::Facets(found) = reply else {
                return Err(unexpected(&reply));
            };
            for f in found {
                let Some(key) = Key::of(&f.value) else { continue };
                counts
                    .entry(key)
                    .or_insert_with(|| FacetCount {
                        value: f.value.clone(),
                        count: 0,
                    })
                    .count += f.count;
            }
        }
        let mut values: Vec<FacetCount> = counts.into_values().collect();
        values.sort_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| grouping::value_order(&a.value, &b.value))
        });
        values.truncate(limit);
        Ok(values)
    }

    pub fn discover(
        &self,
        target: Option<&Example>,
        context: &[ContextPair],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        if target.is_none() && context.is_empty() {
            return Err(Error::invalid("discover needs a target, context pairs or both"));
        }
        let using = opts.using.as_deref();
        let config = self.config().dense(using)?;
        // Resolve examples given by id to vectors: they may live on any shard.
        let mut ids: Vec<&str> = Vec::new();
        let examples = target
            .into_iter()
            .chain(context.iter().flat_map(|p| [&p.positive, &p.negative]));
        for e in examples {
            if let Example::Id(id) = e {
                ids.push(id);
            }
        }
        let points: HashMap<String, Point> = self
            .try_get_many(&ids)?
            .into_iter()
            .map(|p| (p.id.clone(), p))
            .collect();
        let resolve = |e: &Example| -> Result<Vec<f32>> {
            let mut v = match e {
                Example::Id(id) => vector_of(
                    points.get(id).ok_or_else(|| Error::PointNotFound(id.clone()))?,
                    using,
                )?,
                Example::Vector(v) => v.clone(),
            };
            validate_vector(&config, &mut v)?;
            Ok(v)
        };
        let target_vector = target.map(resolve).transpose()?;
        let pairs = context
            .iter()
            .map(|p| Ok((resolve(&p.positive)?, resolve(&p.negative)?)))
            .collect::<Result<Vec<_>>>()?;
        let skip: HashSet<&str> = ids.iter().copied().collect();
        let wanted = k.saturating_add(opts.offset);
        let replies = self.read_all(ShardRead::Discover {
            target: target_vector.clone().map(Example::Vector),
            context: pairs
                .iter()
                .map(|(p, n)| ContextPair {
                    positive: Example::Vector(p.clone()),
                    negative: Example::Vector(n.clone()),
                })
                .collect(),
            k: wanted.saturating_add(skip.len()),
            opts: SearchOptions {
                offset: 0,
                // A target ranks by the context pairs, which needs the vectors.
                include_vectors: opts.include_vectors || target.is_some(),
                ..opts.clone()
            },
        })?;
        let mut hits: Vec<SearchHit> = Self::hits_of(replies)?
            .into_iter()
            .flatten()
            .filter(|h| !skip.contains(h.id.as_str()))
            .collect();
        match &target_vector {
            None => hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then_with(|| a.id.cmp(&b.id))),
            Some(_) => {
                let metric = config.metric;
                let rank = |h: &SearchHit| -> usize {
                    let v = match using {
                        None => h.vector.as_deref(),
                        Some(name) => h.vectors.get(name).map(Vec::as_slice),
                    };
                    v.map_or(0, |v| {
                        pairs
                            .iter()
                            .filter(|(p, n)| metric.distance(p, v) < metric.distance(n, v))
                            .count()
                    })
                };
                let mut ranked: Vec<(usize, SearchHit)> = hits.into_iter().map(|h| (rank(&h), h)).collect();
                ranked.sort_by(|a, b| {
                    b.0.cmp(&a.0)
                        .then(a.1.distance.total_cmp(&b.1.distance))
                        .then_with(|| a.1.id.cmp(&b.1.id))
                });
                hits = ranked.into_iter().map(|(_, h)| h).collect();
                if !opts.include_vectors {
                    for h in &mut hits {
                        h.vector = None;
                        h.vectors.clear();
                        h.sparse_vectors.clear();
                    }
                }
            }
        }
        Ok(hits.into_iter().skip(opts.offset).take(k).collect())
    }

    pub fn query(&self, query: &HybridQuery, k: usize) -> Result<Vec<ScoredHit>> {
        if query.prefetch.is_empty() {
            return Err(Error::invalid("a query needs at least one search to fuse"));
        }
        let config = self.config();
        // Each search runs over every shard and is merged first, so that
        // ranks and scores are fused across the whole collection.
        let mut slots: HashMap<String, u32> = HashMap::new();
        let mut ids: Vec<String> = Vec::new();
        let mut ranked: Vec<(f32, Vec<Neighbor>)> = Vec::new();
        for p in &query.prefetch {
            if !p.weight.is_finite() || p.weight < 0.0 {
                return Err(Error::invalid("search weights must be finite and not negative"));
            }
            if p.weight == 0.0 && query.rescore.is_none() {
                continue;
            }
            let opts = SearchOptions {
                using: None,
                filter: query.filter.clone(),
                ef: p.ef,
                include_vectors: false,
                include_metadata: false,
                max_distance: p.max_distance,
                offset: 0,
            };
            let found = match &p.query {
                QueryVector::Dense { vector, using } => {
                    let opts = SearchOptions {
                        using: using.clone(),
                        ..opts
                    };
                    self.dense(vector, p.limit, &opts, &HashSet::new())?
                }
                QueryVector::Sparse { vector, using } => {
                    config.sparse(using)?;
                    self.search_sparse(using, vector, p.limit, &opts)?
                }
            };
            let neighbours = found
                .into_iter()
                .map(|h| {
                    let slot = *slots.entry(h.id.clone()).or_insert_with(|| {
                        ids.push(h.id.clone());
                        (ids.len() - 1) as u32
                    });
                    Neighbor {
                        slot,
                        dist: h.distance,
                    }
                })
                .collect();
            ranked.push((p.weight, neighbours));
        }

        let mut fused: Vec<(u32, f64)> = match &query.rescore {
            Some(r) => {
                let using = r.using.as_deref();
                let vconfig = config.dense(using)?;
                let mut v = r.vector.clone();
                validate_vector(&vconfig, &mut v)?;
                let points = self.try_get_many(&ids)?;
                points
                    .iter()
                    .map(|p| {
                        let stored = vector_of(p, using)?;
                        Ok((slots[&p.id], -f64::from(vconfig.metric.distance(&v, &stored))))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            None => fuse(query.fusion, &ranked),
        };
        fused.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| ids[a.0 as usize].cmp(&ids[b.0 as usize]))
        });
        fused.truncate(k.saturating_add(query.offset));
        fused.drain(..query.offset.min(fused.len()));

        let wanted: Vec<&String> = fused.iter().map(|(slot, _)| &ids[*slot as usize]).collect();
        let points: HashMap<String, Point> = if query.include_metadata || query.include_vectors {
            self.try_get_many(&wanted)?
                .into_iter()
                .map(|p| (p.id.clone(), p))
                .collect()
        } else {
            HashMap::new()
        };
        Ok(fused
            .into_iter()
            .map(|(slot, score)| {
                let id = ids[slot as usize].clone();
                let p = points.get(&id);
                ScoredHit {
                    score: score as f32,
                    metadata: p
                        .filter(|_| query.include_metadata)
                        .and_then(|p| p.metadata.clone()),
                    vector: p.filter(|_| query.include_vectors).map(|p| p.vector.clone()),
                    vectors: p
                        .filter(|_| query.include_vectors)
                        .map(|p| p.vectors.clone())
                        .unwrap_or_default(),
                    sparse_vectors: p
                        .filter(|_| query.include_vectors)
                        .map(|p| p.sparse_vectors.clone())
                        .unwrap_or_default(),
                    id,
                }
            })
            .collect())
    }

    pub fn recommend<S: AsRef<str>>(
        &self,
        positive: &[S],
        negative: &[S],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<Vec<SearchHit>> {
        if positive.is_empty() {
            return Err(Error::invalid("recommend needs at least one positive example"));
        }
        let using = opts.using.as_deref();
        let config = self.config().dense(using)?;
        let all: Vec<&str> = positive.iter().chain(negative).map(AsRef::as_ref).collect();
        let points: HashMap<String, Point> = self
            .try_get_many(&all)?
            .into_iter()
            .map(|p| (p.id.clone(), p))
            .collect();
        let vectors = |ids: &[S]| {
            ids.iter()
                .map(|id| {
                    let id = id.as_ref();
                    let p = points
                        .get(id)
                        .ok_or_else(|| Error::PointNotFound(id.to_owned()))?;
                    vector_of(p, using)
                })
                .collect::<Result<Vec<_>>>()
        };
        let (pos, neg) = (vectors(positive)?, vectors(negative)?);
        let mean = |vs: &[Vec<f32>]| {
            let mut sum = vec![0.0f64; config.dimension];
            for v in vs {
                for (s, &x) in sum.iter_mut().zip(v) {
                    *s += f64::from(x);
                }
            }
            sum.into_iter().map(|s| s / vs.len() as f64).collect::<Vec<f64>>()
        };
        let pos_mean = mean(&pos);
        let query: Vec<f32> = if neg.is_empty() {
            pos_mean.iter().map(|&x| x as f32).collect()
        } else {
            let neg_mean = mean(&neg);
            pos_mean
                .iter()
                .zip(&neg_mean)
                .map(|(&p, &n)| (2.0 * p - n) as f32)
                .collect()
        };
        let skip: HashSet<&str> = all.into_iter().collect();
        self.dense(&query, k, opts, &skip)
    }

    pub fn flush(&self) -> Result<()> {
        self.backend.maintain(&self.name, Maintenance::Flush)
    }

    pub fn compact(&self) -> Result<()> {
        self.backend.maintain(&self.name, Maintenance::Compact)
    }

    pub fn purge_expired(&self) -> Result<usize> {
        Ok(self
            .write_all(Command::PurgeExpired)?
            .iter()
            .map(Outcome::count)
            .sum())
    }
}
