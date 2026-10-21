use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Error, Result};

/// Maximum vector dimension accepted.
pub const MAX_DIMENSION: usize = 65_536;
/// Maximum length of a point id, in bytes.
pub const MAX_ID_LEN: usize = 512;
/// Maximum collection-name length.
pub const MAX_NAME_LEN: usize = 64;
/// Maximum number of metadata fields a collection can index.
pub const MAX_FIELD_INDEXES: usize = 32;
/// Maximum length of an indexed field path, in bytes.
pub const MAX_FIELD_PATH_LEN: usize = 256;
/// Maximum nesting depth of point metadata. Stored data is wrapped in a few
/// more levels and must stay under serde_json's read limit of 128.
pub const MAX_METADATA_DEPTH: usize = 64;

/// Maximum number of named dense vectors a collection can have.
pub const MAX_NAMED_VECTORS: usize = 16;
/// Maximum number of sparse vectors a collection can have.
pub const MAX_SPARSE_VECTORS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Dimension of the main vector.
    pub dimension: usize,
    #[serde(default)]
    pub metric: Metric,
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default, skip_serializing_if = "Quantization::is_none")]
    pub quantization: Quantization,
    /// Keep the main vectors in a file instead of memory.
    #[serde(default, skip_serializing_if = "is_false")]
    pub on_disk: bool,
    /// Further dense vectors every point carries, by name, each with its
    /// own dimension, metric and index.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: BTreeMap<String, VectorConfig>,
    /// Sparse vectors points may carry, by name, for keyword-style search.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: BTreeMap<String, SparseVectorConfig>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl CollectionConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: Metric::default(),
            index: IndexConfig::default(),
            quantization: Quantization::None,
            on_disk: false,
            vectors: BTreeMap::new(),
            sparse_vectors: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }

    #[must_use]
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    #[must_use]
    pub fn index(mut self, index: IndexConfig) -> Self {
        self.index = index;
        self
    }

    #[must_use]
    pub fn on_disk(mut self, on_disk: bool) -> Self {
        self.on_disk = on_disk;
        self
    }

    /// Adds a named dense vector.
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, config: VectorConfig) -> Self {
        self.vectors.insert(name.into(), config);
        self
    }

    /// Adds a named sparse vector.
    #[must_use]
    pub fn with_sparse_vector(mut self, name: impl Into<String>, config: SparseVectorConfig) -> Self {
        self.sparse_vectors.insert(name.into(), config);
        self
    }

    /// The main vector's settings.
    pub fn main_vector(&self) -> VectorConfig {
        VectorConfig {
            dimension: self.dimension,
            metric: self.metric,
            index: self.index,
            quantization: self.quantization,
            on_disk: self.on_disk,
        }
    }

    /// Settings of the dense vector `using` names (the main one for `None`).
    pub fn dense(&self, using: Option<&str>) -> Result<VectorConfig> {
        match using {
            None => Ok(self.main_vector()),
            Some(name) => self.vectors.get(name).copied().ok_or_else(|| {
                if self.sparse_vectors.contains_key(name) {
                    Error::invalid(format!("`{name}` is a sparse vector; query it with `sparse`"))
                } else {
                    Error::invalid(format!("this collection has no vector named `{name}`"))
                }
            }),
        }
    }

    /// Settings of the sparse vector `name`.
    pub fn sparse(&self, name: &str) -> Result<SparseVectorConfig> {
        self.sparse_vectors.get(name).copied().ok_or_else(|| {
            if self.vectors.contains_key(name) {
                Error::invalid(format!("`{name}` is a dense vector; query it with `vector`"))
            } else {
                Error::invalid(format!("this collection has no sparse vector named `{name}`"))
            }
        })
    }

    /// Whether any vector lives on disk.
    pub fn uses_disk(&self) -> bool {
        self.on_disk || self.vectors.values().any(|v| v.on_disk)
    }

    pub fn validate(&self) -> Result<()> {
        self.main_vector().validate()?;
        if self.vectors.len() > MAX_NAMED_VECTORS {
            return Err(Error::invalid(format!(
                "a collection can have at most {MAX_NAMED_VECTORS} named vectors"
            )));
        }
        if self.sparse_vectors.len() > MAX_SPARSE_VECTORS {
            return Err(Error::invalid(format!(
                "a collection can have at most {MAX_SPARSE_VECTORS} sparse vectors"
            )));
        }
        for (name, config) in &self.vectors {
            validate_vector_name(name)?;
            config
                .validate()
                .map_err(|e| Error::invalid(format!("vector `{name}`: {}", message(&e))))?;
        }
        for name in self.sparse_vectors.keys() {
            validate_vector_name(name)?;
            if self.vectors.contains_key(name) {
                return Err(Error::invalid(format!(
                    "`{name}` cannot be both a dense and a sparse vector"
                )));
            }
        }
        Ok(())
    }
}

/// The message of an `InvalidArgument`, for wrapping it with context.
fn message(e: &Error) -> String {
    match e {
        Error::InvalidArgument(msg) => msg.clone(),
        other => other.to_string(),
    }
}

/// Vector names follow the rules for collection names.
fn validate_vector_name(name: &str) -> Result<()> {
    validate_name(name).map_err(|_| {
        Error::invalid(format!(
            "invalid vector name `{name}`: use 1-{MAX_NAME_LEN} characters from [A-Za-z0-9_-]"
        ))
    })
}

/// Settings of one dense vector: the main one or a named one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorConfig {
    pub dimension: usize,
    #[serde(default)]
    pub metric: Metric,
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default, skip_serializing_if = "Quantization::is_none")]
    pub quantization: Quantization,
    /// Keep these vectors in a file instead of memory. Searches then read
    /// them from disk (the OS page cache keeps hot ones in memory); with
    /// scalar quantization, the compact codes stay in memory for the search
    /// and the exact vectors on disk re-rank the results.
    #[serde(default, skip_serializing_if = "is_false")]
    pub on_disk: bool,
}

impl VectorConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: Metric::default(),
            index: IndexConfig::default(),
            quantization: Quantization::None,
            on_disk: false,
        }
    }

    #[must_use]
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    #[must_use]
    pub fn index(mut self, index: IndexConfig) -> Self {
        self.index = index;
        self
    }

    #[must_use]
    pub fn quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }

    #[must_use]
    pub fn on_disk(mut self, on_disk: bool) -> Self {
        self.on_disk = on_disk;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 || self.dimension > MAX_DIMENSION {
            return Err(Error::invalid(format!(
                "dimension must be between 1 and {MAX_DIMENSION}"
            )));
        }

        match &self.index {
            IndexConfig::Flat => {}
            IndexConfig::Hnsw(p) => p.validate()?,
            IndexConfig::KdTree(p) => {
                p.validate()?;
                if self.metric == Metric::DotProduct {
                    return Err(Error::invalid(
                        "the kd_tree index does not support dot_product: \
                         inner products give no distance bound to prune the tree with",
                    ));
                }
            }
        }

        if self.quantization == Quantization::Scalar {
            if self.metric == Metric::Hamming {
                return Err(Error::invalid(
                    "scalar quantization does not support hamming: \
                     it compares exact values, which quantization rounds",
                ));
            }
            if matches!(self.index, IndexConfig::KdTree(_)) {
                return Err(Error::invalid("scalar quantization needs an hnsw or flat index"));
            }
        }

        Ok(())
    }
}

/// Settings of one sparse vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseVectorConfig {
    /// How query weights are adjusted before scoring.
    #[serde(default, skip_serializing_if = "SparseModifier::is_none")]
    pub modifier: SparseModifier,
}

/// Adjustment applied to sparse query weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseModifier {
    /// Scores are plain dot products.
    #[default]
    None,
    /// Each query weight is multiplied by its index's inverse document
    /// frequency over the collection, `ln(1 + (N - n + 0.5) / (n + 0.5))`
    /// for `n` of `N` points having it. With term frequencies as values
    /// (saturated as BM25 does), scores are BM25 scores.
    Idf,
}

impl SparseModifier {
    pub fn is_none(&self) -> bool {
        *self == SparseModifier::None
    }
}

/// How vectors are held in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    /// Full 32-bit floats.
    #[default]
    None,
    /// One byte per component plus 8 bytes per vector (its range), about a
    /// quarter of the memory. Lossy: searches and returned vectors use the
    /// decoded values, which differ from the originals by at most
    /// `(max - min) / 510` per component, where `min`/`max` are taken over
    /// that vector's components.
    Scalar,
}

impl Quantization {
    pub fn is_none(&self) -> bool {
        *self == Quantization::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexConfig {
    /// Exact brute-force search. Best for small collections (< ~50k vectors)
    /// or when perfect recall is required.
    Flat,
    /// Approximate nearest-neighbour search with an HNSW graph. The default;
    /// best for large, high-dimensional collections.
    Hnsw(HnswParams),
    /// Exact search with a k-d tree. Fast for low-dimensional data
    /// (roughly < 20 dimensions); degrades towards brute force above that.
    KdTree(KdTreeParams),
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig::Hnsw(HnswParams::default())
    }
}

impl IndexConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            IndexConfig::Flat => "flat",
            IndexConfig::Hnsw(_) => "hnsw",
            IndexConfig::KdTree(_) => "kd_tree",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HnswParams {
    /// Max neighbours per node on upper layers (layer 0 uses `2 * m`).
    /// Higher = better recall, more memory. Typical: 8–48.
    pub m: usize,
    /// Candidate list size while building. Higher = better graph, slower inserts.
    pub ef_construction: usize,
    /// Default candidate list size while searching. Can be overridden per query.
    pub ef_search: usize,
    /// Seed for level assignment, for reproducible graphs.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 128,
            ef_search: 64,
            seed: 0x5A61_7474_6100_0001,
        }
    }
}

impl HnswParams {
    fn validate(&self) -> Result<()> {
        if !(2..=256).contains(&self.m) {
            return Err(Error::invalid("hnsw.m must be between 2 and 256"));
        }
        if self.ef_construction == 0 || self.ef_search == 0 {
            return Err(Error::invalid(
                "hnsw.ef_construction and hnsw.ef_search must be > 0",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KdTreeParams {
    /// Incremental inserts can unbalance the tree. Once its depth exceeds
    /// `balance_factor × log2(n)`, it is rebuilt around per-axis medians.
    pub balance_factor: u32,
}

impl Default for KdTreeParams {
    fn default() -> Self {
        Self { balance_factor: 3 }
    }
}

impl KdTreeParams {
    fn validate(&self) -> Result<()> {
        if !(2..=64).contains(&self.balance_factor) {
            return Err(Error::invalid("kd_tree.balance_factor must be between 2 and 64"));
        }
        Ok(())
    }
}

/// Collection names: 1–64 characters of `[A-Za-z0-9_-]`, not starting with `-`.
pub fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && !name.starts_with('-')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');

    if ok {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "invalid name `{name}`: use 1-{MAX_NAME_LEN} characters from [A-Za-z0-9_-]"
        )))
    }
}

/// Indexed metadata fields: a dot-separated path, as in filters, of
/// non-empty segments, not starting with `$` (that marks filter operators).
pub fn validate_field_path(path: &str) -> Result<()> {
    let ok = !path.is_empty()
        && path.len() <= MAX_FIELD_PATH_LEN
        && !path.starts_with('$')
        && path.split('.').all(|seg| !seg.is_empty());
    if ok {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "invalid field `{path}`: use a dot-separated path of 1-{MAX_FIELD_PATH_LEN} bytes \
             with no empty segments, not starting with `$`"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_path_validation() {
        for good in ["genre", "stats.views", "a-b_c.d1"] {
            assert!(validate_field_path(good).is_ok(), "{good}");
        }
        for bad in ["", "$and", "a..b", ".a", "a.", &"x".repeat(257)] {
            assert!(validate_field_path(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn name_validation() {
        for good in ["a", "my_docs", "Docs-2", &"x".repeat(64)] {
            assert!(validate_name(good).is_ok(), "{good}");
        }
        for bad in ["", "-a", "a/b", "..", "a b", "é", &"x".repeat(65)] {
            assert!(validate_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn index_config_serde() {
        let c: IndexConfig = serde_json::from_str(r#"{"type":"hnsw","m":32}"#).unwrap();
        assert_eq!(
            c,
            IndexConfig::Hnsw(HnswParams {
                m: 32,
                ..Default::default()
            })
        );
        let c: IndexConfig = serde_json::from_str(r#"{"type":"flat"}"#).unwrap();
        assert_eq!(c, IndexConfig::Flat);
        let c: IndexConfig = serde_json::from_str(r#"{"type":"kd_tree"}"#).unwrap();
        assert_eq!(c, IndexConfig::KdTree(KdTreeParams::default()));
    }

    #[test]
    fn quantization_serde_and_validation() {
        let c: CollectionConfig = serde_json::from_str(r#"{"dimension":3}"#).unwrap();
        assert_eq!(c.quantization, Quantization::None);
        assert!(!serde_json::to_string(&c).unwrap().contains("quantization"));
        let c: CollectionConfig = serde_json::from_str(r#"{"dimension":3,"quantization":"scalar"}"#).unwrap();
        assert_eq!(c.quantization, Quantization::Scalar);
        assert!(c.validate().is_ok());
        assert!(c.clone().metric(Metric::Hamming).validate().is_err());
        assert!(
            c.clone()
                .metric(Metric::Euclidean)
                .index(IndexConfig::KdTree(KdTreeParams::default()))
                .validate()
                .is_err()
        );
        assert!(c.index(IndexConfig::Flat).validate().is_ok());
    }

    #[test]
    fn named_and_sparse_vectors() {
        let c: CollectionConfig = serde_json::from_value(serde_json::json!({
            "dimension": 3,
            "vectors": {"title": {"dimension": 2, "metric": "dot", "index": {"type": "flat"}}},
            "sparse_vectors": {"text": {"modifier": "idf"}},
        }))
        .unwrap();
        assert!(c.validate().is_ok());
        assert_eq!(c.dense(Some("title")).unwrap().dimension, 2);
        assert_eq!(c.dense(None).unwrap().dimension, 3);
        assert!(c.dense(Some("text")).is_err());
        assert!(c.dense(Some("nope")).is_err());
        assert_eq!(c.sparse("text").unwrap().modifier, SparseModifier::Idf);
        assert!(c.sparse("title").is_err());
        assert!(
            !serde_json::to_string(&CollectionConfig::new(3))
                .unwrap()
                .contains("vectors")
        );

        let both = c.clone().with_vector("text", VectorConfig::new(2));
        assert!(both.validate().is_err(), "a name can't be dense and sparse");
        let bad_name = c.clone().with_vector("a b", VectorConfig::new(2));
        assert!(bad_name.validate().is_err());
        let bad_dim = c.with_vector("x", VectorConfig::new(0));
        assert!(bad_dim.validate().is_err());
        assert!(
            serde_json::from_str::<VectorConfig>(r#"{"dimension": 2, "dimensions": 3}"#).is_err(),
            "unknown fields"
        );
    }

    #[test]
    fn kd_tree_rejects_dot_product() {
        let config = CollectionConfig::new(3)
            .metric(Metric::DotProduct)
            .index(IndexConfig::KdTree(KdTreeParams::default()));
        assert!(config.validate().is_err());
        assert!(config.metric(Metric::Manhattan).validate().is_ok());
    }
}
