use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::CollectionConfig;
use crate::error::{Error, Result};
use crate::filter::Filter;

/// Maximum number of non-zero components in a sparse vector.
pub const MAX_SPARSE_NNZ: usize = 65_536;

/// Named dense vectors, by name.
pub type NamedVectors = BTreeMap<String, Vec<f32>>;
/// Sparse vectors, by name.
pub type SparseVectors = BTreeMap<String, SparseVector>;

/// A vector given by its non-zero components: `values[i]` at `indices[i]`.
/// Typically term weights, with one index per vocabulary term.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self { indices, values }
    }

    /// Number of non-zero components.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Checks the vector and sorts it by index. Zero values are dropped.
    pub fn normalize(&mut self) -> Result<()> {
        if self.indices.len() != self.values.len() {
            return Err(Error::invalid("sparse vectors need as many values as indices"));
        }
        if self.indices.len() > MAX_SPARSE_NNZ {
            return Err(Error::invalid(format!(
                "sparse vectors can have at most {MAX_SPARSE_NNZ} non-zero values"
            )));
        }
        if self.values.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("sparse vectors must not contain NaN or infinity"));
        }
        if !self.indices.is_sorted_by(|a, b| a < b) {
            let mut pairs: Vec<(u32, f32)> = self
                .indices
                .iter()
                .copied()
                .zip(self.values.iter().copied())
                .collect();
            pairs.sort_unstable_by_key(|&(i, _)| i);
            if pairs.windows(2).any(|w| w[0].0 == w[1].0) {
                return Err(Error::invalid("sparse vector indices must be unique"));
            }
            (self.indices, self.values) = pairs.into_iter().unzip();
        }
        if self.values.contains(&0.0) {
            let (indices, values) = self
                .indices
                .iter()
                .zip(&self.values)
                .filter(|&(_, &v)| v != 0.0)
                .map(|(&i, &v)| (i, v))
                .unzip();
            (self.indices, self.values) = (indices, values);
        }
        Ok(())
    }

    /// The dot product with `other`; both must be sorted by index.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let (mut i, mut j, mut sum) = (0, 0, 0.0f32);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Unique (per collection) identifier.
    pub id: String,
    /// The main embedding. Empty when fetched without vectors.
    #[serde(default)]
    pub vector: Vec<f32>,
    /// The collection's named dense vectors; every point has all of them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: NamedVectors,
    /// Sparse vectors, by name; any of them may be missing.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: SparseVectors,
    /// Arbitrary JSON metadata ("payload"), usable in filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Point {
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            vectors: BTreeMap::new(),
            sparse_vectors: BTreeMap::new(),
            metadata: None,
        }
    }

    /// Adds a named dense vector.
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, vector: Vec<f32>) -> Self {
        self.vectors.insert(name.into(), vector);
        self
    }

    /// Adds a sparse vector.
    #[must_use]
    pub fn with_sparse_vector(mut self, name: impl Into<String>, vector: SparseVector) -> Self {
        self.sparse_vectors.insert(name.into(), vector);
        self
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
    /// The main embedding.
    pub vector: Vec<f32>,
    /// Named dense vectors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: NamedVectors,
    /// Sparse vectors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: SparseVectors,
    /// Arbitrary JSON metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl From<PointInput> for Point {
    fn from(p: PointInput) -> Self {
        Point {
            id: p.id.unwrap_or_else(new_id),
            vector: p.vector,
            vectors: p.vectors,
            sparse_vectors: p.sparse_vectors,
            metadata: p.metadata,
        }
    }
}

pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Which named vector to search (the main one if `None`).
    pub using: Option<String>,
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
    pub offset: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            using: None,
            filter: None,
            ef: None,
            include_vectors: false,
            include_metadata: true,
            max_distance: None,
            offset: 0,
        }
    }
}

impl SearchOptions {
    /// Searches the named vector `name`.
    #[must_use]
    pub fn using(mut self, name: impl Into<String>) -> Self {
        self.using = Some(name.into());
        self
    }

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

    #[must_use]
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchTarget {
    Vector(Vec<f32>),
    Id(String),
    Sparse { vector: SparseVector, using: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupBy {
    pub field: String,
    pub groups: usize,
    pub group_size: usize,
}

impl GroupBy {
    pub fn new(field: impl Into<String>, groups: usize, group_size: usize) -> Self {
        Self {
            field: field.into(),
            groups,
            group_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HitGroup {
    pub key: Value,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetCount {
    pub value: Value,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    #[default]
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderBy {
    pub field: String,
    #[serde(default)]
    pub direction: Direction,
}

impl OrderBy {
    pub fn asc(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            direction: Direction::Asc,
        }
    }

    pub fn desc(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            direction: Direction::Desc,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScrollOptions {
    pub cursor: Option<String>,
    pub limit: usize,
    pub filter: Option<Filter>,
    pub order_by: Option<OrderBy>,
}

impl ScrollOptions {
    pub fn new(limit: usize) -> Self {
        Self {
            cursor: None,
            limit,
            filter: None,
            order_by: None,
        }
    }

    #[must_use]
    pub fn cursor(mut self, cursor: impl Into<String>) -> Self {
        self.cursor = Some(cursor.into());
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    #[must_use]
    pub fn order_by(mut self, order_by: OrderBy) -> Self {
        self.order_by = Some(order_by);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Example {
    Id(String),
    Vector(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPair {
    pub positive: Example,
    pub negative: Example,
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
    /// The main vector, if requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Named dense vectors, if vectors were requested.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: NamedVectors,
    /// Sparse vectors, if vectors were requested.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: SparseVectors,
}

/// What one part of a [`HybridQuery`] searches with.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryVector {
    /// A dense vector, against the main vector or a named one.
    Dense { vector: Vec<f32>, using: Option<String> },
    /// A sparse vector, against the named sparse vector `using`.
    Sparse { vector: SparseVector, using: String },
}

/// One search whose ranking a [`HybridQuery`] fuses with the others.
#[derive(Debug, Clone, PartialEq)]
pub struct Prefetch {
    pub query: QueryVector,
    /// How many of this search's best hits take part in the fusion.
    pub limit: usize,
    /// Multiplies this search's contribution to the fused score.
    pub weight: f32,
    /// HNSW beam width for this search.
    pub ef: Option<usize>,
    /// Hits farther than this take no part (see [`SearchOptions::max_distance`]).
    pub max_distance: Option<f32>,
}

impl Prefetch {
    pub fn new(query: QueryVector, limit: usize) -> Self {
        Self {
            query,
            limit,
            weight: 1.0,
            ef: None,
            max_distance: None,
        }
    }
}

/// How the rankings of a [`HybridQuery`] are combined.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Fusion {
    /// Reciprocal rank fusion: a point at rank `r` (from 1) of a search
    /// scores `weight / (k + r)` there; scores add up across searches.
    /// Needs no score calibration between dense and sparse searches.
    Rrf {
        #[serde(default = "default_rrf_k")]
        k: u32,
    },
    RelativeScore,
    Dbsf,
}

fn default_rrf_k() -> u32 {
    60
}

impl Default for Fusion {
    fn default() -> Self {
        Fusion::Rrf { k: default_rrf_k() }
    }
}

/// Several searches, e.g. one dense and one sparse, fused into one ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridQuery {
    pub prefetch: Vec<Prefetch>,
    pub fusion: Fusion,
    /// Applies to every search.
    pub filter: Option<Filter>,
    pub include_vectors: bool,
    pub include_metadata: bool,
    pub rescore: Option<Rescore>,
    pub offset: usize,
}

impl HybridQuery {
    pub fn new(prefetch: Vec<Prefetch>) -> Self {
        Self {
            prefetch,
            fusion: Fusion::default(),
            filter: None,
            include_vectors: false,
            include_metadata: true,
            rescore: None,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rescore {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub using: Option<String>,
}

/// A result of a [`HybridQuery`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredHit {
    pub id: String,
    /// The fused score (higher is better).
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: NamedVectors,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: SparseVectors,
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
    fn sparse_vectors_are_checked_and_sorted() {
        let mut v = SparseVector::new(vec![5, 1, 3], vec![0.5, 1.0, 0.0]);
        v.normalize().unwrap();
        assert_eq!(v, SparseVector::new(vec![1, 5], vec![1.0, 0.5]));
        let w = SparseVector::new(vec![0, 5, 9], vec![2.0, 4.0, 1.0]);
        assert_eq!(v.dot(&w), 2.0);
        for bad in [
            SparseVector::new(vec![1, 1], vec![1.0, 2.0]),
            SparseVector::new(vec![1], vec![]),
            SparseVector::new(vec![1], vec![f32::NAN]),
        ] {
            assert!(bad.clone().normalize().is_err(), "{bad:?}");
        }
        assert!(serde_json::from_str::<SparseVector>(r#"{"indices":[1],"values":[1],"x":1}"#).is_err());
    }

    #[test]
    fn point_input_generates_ids() {
        let input = |id: Option<&str>| PointInput {
            id: id.map(Into::into),
            vector: vec![1.0],
            vectors: BTreeMap::new(),
            sparse_vectors: BTreeMap::new(),
            metadata: None,
        };
        let a: Point = input(None).into();
        let b: Point = input(Some("x")).into();
        assert_eq!(a.id.len(), 36);
        assert_eq!(b.id, "x");
    }
}
