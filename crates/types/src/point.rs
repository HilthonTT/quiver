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
    /// Drop hits farther than this distance. Results are still capped at
    /// `k`; with HNSW the threshold applies to the approximate top `k`.
    pub max_distance: Option<f32>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            filter: None,
            ef: None,
            include_vectors: false,
            include_metadata: true,
            max_distance: None,
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

    #[must_use]
    pub fn max_distance(mut self, max: f32) -> Self {
        self.max_distance = Some(max);
        self
    }
}

/// Which points an operation applies to.
#[derive(Debug, Clone, PartialEq)]
pub enum PointSelector {
    /// These ids; ids that do not exist are skipped.
    Ids(Vec<String>),
    /// Every point whose metadata matches.
    Filter(Filter),
}

/// How a metadata update combines new metadata
/// with what a point already has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    /// RFC 7386 JSON Merge Patch: objects merge recursively and `null`
    /// removes a key.
    #[default]
    Merge,
    /// Replace the whole value; `null` removes the metadata.
    Replace,
}

/// Applies `patch` to `target` as an RFC 7386 JSON Merge Patch.
pub fn merge_patch(target: &mut Value, patch: &Value) {
    let Value::Object(patch) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = Value::Object(serde_json::Map::new());
    }
    let target = target.as_object_mut().expect("just made an object");
    for (key, value) in patch {
        if value.is_null() {
            target.remove(key);
        } else {
            merge_patch(target.entry(key.clone()).or_insert(Value::Null), value);
        }
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
    /// Metadata fields with an index, in the order they were added.
    #[serde(default)]
    pub field_indexes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_patch_follows_rfc_7386() {
        use serde_json::json;
        let cases = [
            (json!({"a": "b"}), json!({"a": "c"}), json!({"a": "c"})),
            (json!({"a": "b"}), json!({"b": "c"}), json!({"a": "b", "b": "c"})),
            (json!({"a": "b"}), json!({"a": null}), json!({})),
            (
                json!({"a": {"b": "c", "d": 1}}),
                json!({"a": {"b": null, "e": [1]}}),
                json!({"a": {"d": 1, "e": [1]}}),
            ),
            (json!({"a": [1]}), json!({"a": {"b": 1}}), json!({"a": {"b": 1}})),
            (json!(["a"]), json!({"a": "b"}), json!({"a": "b"})),
            (json!({"a": "b"}), json!(["c"]), json!(["c"])),
            (json!({"a": "foo"}), json!(null), json!(null)),
            (
                json!({}),
                json!({"a": {"bb": {"ccc": null}}}),
                json!({"a": {"bb": {}}}),
            ),
        ];
        for (mut target, patch, expected) in cases {
            merge_patch(&mut target, &patch);
            assert_eq!(target, expected, "patch {patch}");
        }
    }

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
