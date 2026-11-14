//! What the replicas of a cluster's shards exchange: the commands their
//! replicated logs hold, and the reads a coordinating node sends them.
//!
//! These types travel between Quiver nodes of the same version only; they
//! are not a stable format.

use quiver_types::{
    CollectionConfig, CollectionUpdate, ContextPair, Error, Example, FacetCount, Filter, GroupBy, HitGroup,
    MetadataMode, Point, PointSelector, Result, ScrollOptions, SearchHit, SearchOptions, SearchTarget,
    SparseVector,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ScrollPage;

/// A change to one shard, as its replicated log holds it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    Upsert {
        points: Vec<Point>,
    },
    Delete {
        ids: Vec<String>,
    },
    DeleteWhere {
        filter: Filter,
    },
    UpdateMetadata {
        selector: PointSelector,
        metadata: Value,
        mode: MetadataMode,
    },
    PurgeExpired,
    Update {
        update: CollectionUpdate,
    },
    CreateFieldIndex {
        field: String,
    },
    DropFieldIndex {
        field: String,
    },
    /// Changes nothing; marks a log entry that carries no command.
    Noop,
}

/// A [`Command`] with the time it is applied as of, so that every replica
/// judges expiry the same way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Write {
    /// Seconds since the Unix epoch, as the node that proposed it saw them.
    pub now: f64,
    pub command: Command,
}

impl Write {
    /// `command` as of now.
    pub fn now(command: Command) -> Self {
        Self {
            now: crate::expiry::now(),
            command,
        }
    }
}

/// What applying a [`Command`] returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// How many points were written or removed.
    Count(usize),
    /// Whether something changed (dropping a field index).
    Changed(bool),
    Done,
}

impl Outcome {
    /// The count, or 0 for outcomes without one.
    pub fn count(&self) -> usize {
        match self {
            Outcome::Count(n) => *n,
            _ => 0,
        }
    }
}

/// An [`Error`] as it crosses the network.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireError {
    pub code: String,
    /// The error's own argument (a name, a message).
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<(usize, usize)>,
}

impl From<&Error> for WireError {
    fn from(e: &Error) -> Self {
        let (code, detail) = match e {
            Error::DimensionMismatch { expected, actual } => {
                return Self {
                    code: "dimension_mismatch".into(),
                    detail: String::new(),
                    dims: Some((*expected, *actual)),
                };
            }
            Error::CollectionNotFound(s) => ("collection_not_found", s.clone()),
            Error::CollectionExists(s) => ("collection_exists", s.clone()),
            Error::AliasNotFound(s) => ("alias_not_found", s.clone()),
            Error::AliasExists(s) => ("alias_exists", s.clone()),
            Error::SnapshotNotFound(s) => ("snapshot_not_found", s.clone()),
            Error::PointNotFound(s) => ("point_not_found", s.clone()),
            Error::Forbidden(s) => ("forbidden", s.clone()),
            Error::InvalidArgument(s) => ("invalid_argument", s.clone()),
            Error::InvalidFilter(s) => ("invalid_filter", s.clone()),
            Error::Unsupported(s) => ("unsupported", s.clone()),
            Error::Unavailable(s) => ("unavailable", s.clone()),
            Error::Corrupt(s) => ("corrupt", s.clone()),
            Error::Storage(s) => ("storage", s.clone()),
            other => ("storage", other.to_string()),
        };
        Self {
            code: code.into(),
            detail,
            dims: None,
        }
    }
}

impl From<WireError> for Error {
    fn from(w: WireError) -> Self {
        let d = w.detail;
        match w.code.as_str() {
            "dimension_mismatch" => {
                let (expected, actual) = w.dims.unwrap_or_default();
                Error::DimensionMismatch { expected, actual }
            }
            "collection_not_found" => Error::CollectionNotFound(d),
            "collection_exists" => Error::CollectionExists(d),
            "alias_not_found" => Error::AliasNotFound(d),
            "alias_exists" => Error::AliasExists(d),
            "snapshot_not_found" => Error::SnapshotNotFound(d),
            "point_not_found" => Error::PointNotFound(d),
            "forbidden" => Error::Forbidden(d),
            "invalid_argument" => Error::InvalidArgument(d),
            "invalid_filter" => Error::InvalidFilter(d),
            "unsupported" => Error::Unsupported(d),
            "unavailable" => Error::Unavailable(d),
            "corrupt" => Error::Corrupt(d),
            _ => Error::Storage(d),
        }
    }
}

/// A read of one shard, which a coordinating node merges with the same
/// read of the collection's other shards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ShardRead {
    Info,
    Get {
        ids: Vec<String>,
    },
    Count {
        filter: Option<Filter>,
    },
    Scroll {
        opts: ScrollOptions,
    },
    Search {
        vector: Vec<f32>,
        k: usize,
        opts: SearchOptions,
    },
    SearchSparse {
        name: String,
        vector: SparseVector,
        k: usize,
        opts: SearchOptions,
    },
    Groups {
        target: SearchTarget,
        group: GroupBy,
        opts: SearchOptions,
    },
    Facet {
        field: String,
        filter: Option<Filter>,
        limit: usize,
    },
    Discover {
        target: Option<Example>,
        context: Vec<ContextPair>,
        k: usize,
        opts: SearchOptions,
    },
}

/// A shard's answer to a [`ShardRead`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardReply {
    Info(ShardInfo),
    Points(Vec<Point>),
    Count(usize),
    Page(ScrollPage),
    Hits(Vec<SearchHit>),
    Groups(Vec<HitGroup>),
    Facets(Vec<FacetCount>),
}

/// One shard's statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardInfo {
    pub config: CollectionConfig,
    pub count: usize,
    pub tombstones: usize,
    pub vector_bytes: usize,
    pub field_indexes: Vec<String>,
}

impl crate::Collection {
    /// Answers one shard's part of a read spread over a cluster.
    pub fn read_shard(&self, read: &ShardRead) -> Result<ShardReply> {
        Ok(match read {
            ShardRead::Info => {
                let info = self.info();
                ShardReply::Info(ShardInfo {
                    config: info.config,
                    count: info.count,
                    tombstones: info.tombstones,
                    vector_bytes: self.vector_bytes(),
                    field_indexes: info.field_indexes,
                })
            }
            ShardRead::Get { ids } => ShardReply::Points(self.get_many(ids)),
            ShardRead::Count { filter } => ShardReply::Count(self.count(filter.as_ref())),
            ShardRead::Scroll { opts } => ShardReply::Page(self.scroll_with(opts)?),
            ShardRead::Search { vector, k, opts } => ShardReply::Hits(self.search_with(vector, *k, opts)?),
            ShardRead::SearchSparse {
                name,
                vector,
                k,
                opts,
            } => ShardReply::Hits(self.search_sparse(name, vector, *k, opts)?),
            ShardRead::Groups { target, group, opts } => {
                ShardReply::Groups(self.search_groups(target, group, opts)?)
            }
            ShardRead::Facet { field, filter, limit } => {
                ShardReply::Facets(self.facet(field, filter.as_ref(), *limit)?)
            }
            ShardRead::Discover {
                target,
                context,
                k,
                opts,
            } => ShardReply::Hits(self.discover(target.as_ref(), context, *k, opts)?),
        })
    }
}

/// The shard of `shards` that holds the point `id`.
///
/// This decides where every point of a sharded collection lives, so it must
/// never change: FNV-1a over the id's bytes.
pub fn shard_of(id: &str, shards: u32) -> u32 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    (hash % u64::from(shards.max(1))) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_of_is_stable() {
        // Fixed forever: changing these moves every point of every cluster.
        assert_eq!(shard_of("", 7), (0xcbf2_9ce4_8422_2325u64 % 7) as u32);
        assert_eq!(shard_of("a", 1), 0);
        let spread: std::collections::BTreeSet<u32> =
            (0..1000).map(|i| shard_of(&format!("p{i}"), 4)).collect();
        assert_eq!(spread.len(), 4);
    }

    #[test]
    fn wire_errors_round_trip() {
        for e in [
            Error::PointNotFound("x".into()),
            Error::DimensionMismatch {
                expected: 3,
                actual: 2,
            },
            Error::Unavailable("no leader".into()),
        ] {
            let back: Error = WireError::from(&e).into();
            assert_eq!(back.to_string(), e.to_string());
        }
    }
}
