use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::CollectionConfig;
use crate::filter::Filter;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Unique (per collection) identifier.
    pub id: String,
    /// The embedding. Empty when fetched without vectors.
    #[serde(default)]
    pub vector: Vec<f32>,
    /// Arbitrary JSON metadata ("payload"), usable in filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Point {
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: None,
        }
    }

    pub fn with_random_id(vector: Vec<f32>) -> Self {
        Self::new(new_id(), vector)
    }

    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointInput {
    /// Optional id; a UUID v4 is generated if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The embedding.
    pub vector: Vec<f32>,
    /// Arbitrary JSON metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl From<PointInput> for Point {
    fn from(p: PointInput) -> Self {
        Point {
            id: p.id.unwrap_or_else(new_id),
            vector: p.vector,
            metadata: p.metadata,
        }
    }
}

pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Only return points whose metadata matches this filter.
    pub filter: Option<Filter>,
    /// HNSW beam width for this query (defaults to the collection's `ef_search`).
    /// Higher = better recall, slower. Ignored by exact indexes.
    pub ef: Option<usize>,
    /// Include each hit's vector in the results.
    pub include_vectors: bool,
    /// Include each hit's metadata in the results.
    pub include_metadata: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            filter: None,
            ef: None,
            include_vectors: false,
            include_metadata: true,
        }
    }
}

impl SearchOptions {
    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    #[must_use]
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    #[must_use]
    pub fn include_vectors(mut self, yes: bool) -> Self {
        self.include_vectors = yes;
        self
    }

    #[must_use]
    pub fn include_metadata(mut self, yes: bool) -> Self {
        self.include_metadata = yes;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    /// The point's id.
    pub id: String,
    /// Distance to the query (lower is closer). See [`crate::Metric`].
    pub distance: f32,
    /// Metadata, if requested and present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Vector, if requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// The collection's name.
    pub name: String,
    /// The collection's configuration.
    #[serde(flatten)]
    pub config: CollectionConfig,
    /// Number of live points.
    pub count: usize,
    /// Number of deleted slots awaiting compaction.
    pub tombstones: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_input_generates_ids() {
        let a: Point = PointInput {
            id: None,
            vector: vec![1.0],
            metadata: None,
        }
        .into();
        let b: Point = PointInput {
            id: Some("x".into()),
            vector: vec![1.0],
            metadata: None,
        }
        .into();
        assert_eq!(a.id.len(), 36);
        assert_eq!(b.id, "x");
    }
}
