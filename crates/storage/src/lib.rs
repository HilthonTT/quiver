use serde::{Deserialize, Serialize};
use vectordb_types::{Error, Point, Result};

pub mod codec;
mod store;
mod wal;

/// A mutation, as handed to [`StorageEngine::log`].
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op<'a> {
    Upsert { points: &'a [Point] },
    Delete { ids: &'a [String] },
}

/// An owned mutation, as read back from a log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Mutation {
    Upsert { points: Vec<Point> },
    Delete { ids: Vec<String> },
}
