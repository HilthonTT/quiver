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
/// Longest default time to live: 100 years, in seconds.
pub const MAX_TTL_SECS: u64 = 100 * 365 * 24 * 60 * 60;

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
    /// How the main vectors' components are stored; fixed at creation.
    #[serde(default, skip_serializing_if = "Datatype::is_float32")]
    pub datatype: Datatype,
    /// Further dense vectors every point carries, by name, each with its
    /// own dimension, metric and index.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: BTreeMap<String, VectorConfig>,
    /// Sparse vectors points may carry, by name, for keyword-style search.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: BTreeMap<String, SparseVectorConfig>,
    /// Lets points expire: see [`ExpiryConfig`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiry: Option<ExpiryConfig>,
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
            datatype: Datatype::Float32,
            vectors: BTreeMap::new(),
            sparse_vectors: BTreeMap::new(),
            expiry: None,
        }
    }

    #[must_use]
    pub fn expiry(mut self, expiry: ExpiryConfig) -> Self {
        self.expiry = Some(expiry);
        self
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

    #[must_use]
    pub fn datatype(mut self, datatype: Datatype) -> Self {
        self.datatype = datatype;
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
            datatype: self.datatype,
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
        if let Some(expiry) = &self.expiry {
            expiry.validate()?;
        }
        Ok(())
    }
}

/// Lets points expire. A point expires once the number in its metadata
/// field `field`, a Unix time in seconds, is not after the current time.
/// Expired points disappear from every read at once and are deleted by a
/// periodic sweep. The field is indexed while expiry is on, and holding
/// anything but a number in it is refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpiryConfig {
    /// The metadata field (a dot-separated path, as in filters) holding
    /// each point's expiry time.
    pub field: String,
    /// Points written without `field` get it set to the time they were
    /// written plus this many seconds. Without it, such points never expire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_ttl: Option<u64>,
}

impl ExpiryConfig {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            default_ttl: None,
        }
    }

    #[must_use]
    pub fn default_ttl(mut self, seconds: u64) -> Self {
        self.default_ttl = Some(seconds);
        self
    }

    pub fn validate(&self) -> Result<()> {
        validate_field_path(&self.field).map_err(|e| Error::invalid(format!("expiry: {}", message(&e))))?;
        if self.default_ttl.is_some_and(|ttl| ttl == 0 || ttl > MAX_TTL_SECS) {
            return Err(Error::invalid(format!(
                "expiry.default_ttl must be between 1 and {MAX_TTL_SECS} seconds"
            )));
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
    /// How the components are stored; fixed at creation.
    #[serde(default, skip_serializing_if = "Datatype::is_float32")]
    pub datatype: Datatype,
}

impl VectorConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: Metric::default(),
            index: IndexConfig::default(),
            quantization: Quantization::None,
            on_disk: false,
            datatype: Datatype::Float32,
        }
    }

    #[must_use]
    pub fn datatype(mut self, datatype: Datatype) -> Self {
        self.datatype = datatype;
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

        if let Some(kind) = self.quantization.name() {
            if self.metric == Metric::Hamming {
                return Err(Error::invalid(format!(
                    "{kind} quantization does not support hamming: \
                     it compares exact values, which quantization rounds"
                )));
            }
            if matches!(self.index, IndexConfig::KdTree(_)) {
                return Err(Error::invalid(format!(
                    "{kind} quantization needs an hnsw or flat index"
                )));
            }
        }

        match self.datatype {
            Datatype::Float32 => {}
            Datatype::Float16 | Datatype::Uint8 if self.quantization == Quantization::Scalar => {
                return Err(Error::invalid(format!(
                    "scalar quantization needs the float32 datatype ({} vectors are already compact; \
                     binary quantization works with them)",
                    self.datatype.as_str()
                )));
            }
            Datatype::Uint8 if self.metric == Metric::Cosine => {
                return Err(Error::invalid(
                    "the uint8 datatype does not support cosine: cosine vectors are normalised, \
                     which leaves fractions; use dot_product, euclidean, manhattan or hamming",
                ));
            }
            Datatype::Float16 | Datatype::Uint8 => {}
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

/// Changes to a collection's settings, for `Collection::update`.
///
/// Dimensions and metrics are fixed once a collection exists, as are the
/// sets of named and sparse vectors.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionUpdate {
    /// Changes to the main vector's index; see [`VectorUpdate::index`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_disk: Option<bool>,
    /// Changes to named vectors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vectors: BTreeMap<String, VectorUpdate>,
    /// Changes to sparse vectors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_vectors: BTreeMap<String, SparseVectorConfig>,
    /// New expiry settings, replacing the current ones; `Some(None)`
    /// (`null` in JSON) turns expiry off. Points already stored keep their
    /// expiry times; a `default_ttl` applies to points written from then on.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub expiry: Option<Option<ExpiryConfig>>,
}

/// Deserializes a field that is present, even as `null`, to `Some`, so
/// `Option<Option<T>>` tells an absent field from a `null` one.
fn present<'de, D, T>(d: D) -> std::result::Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// Changes to one dense vector's settings. Absent fields stay as they are.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorUpdate {
    /// The new index settings, as a JSON Merge Patch (RFC 7386) over the
    /// current ones: `{"ef_search": 128}` changes just that parameter, and a
    /// different `type` replaces the index (its other parameters take their
    /// defaults). Anything but an `ef_search` change rebuilds the index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_disk: Option<bool>,
}

impl VectorUpdate {
    pub fn is_empty(&self) -> bool {
        self.index.is_none() && self.quantization.is_none() && self.on_disk.is_none()
    }

    /// `config` with these changes made.
    pub fn apply(&self, config: &VectorConfig) -> Result<VectorConfig> {
        let mut out = *config;
        if let Some(patch) = &self.index {
            let patch_type = patch.get("type").and_then(|t| t.as_str());
            let mut index = if patch_type.is_none_or(|t| t == config.index.kind()) {
                serde_json::to_value(config.index)?
            } else {
                serde_json::Value::Object(serde_json::Map::new())
            };
            crate::merge_patch(&mut index, patch);
            out.index = serde_json::from_value(index)
                .map_err(|e| Error::invalid(format!("invalid index settings: {e}")))?;
        }
        if let Some(q) = self.quantization {
            out.quantization = q;
        }
        if let Some(on_disk) = self.on_disk {
            out.on_disk = on_disk;
        }
        out.validate()?;
        Ok(out)
    }
}

impl CollectionUpdate {
    /// `config` with these changes made.
    pub fn apply(&self, config: &CollectionConfig) -> Result<CollectionConfig> {
        let mut out = config.clone();
        let main = VectorUpdate {
            index: self.index.clone(),
            quantization: self.quantization,
            on_disk: self.on_disk,
        }
        .apply(&config.main_vector())?;
        out.index = main.index;
        out.quantization = main.quantization;
        out.on_disk = main.on_disk;
        for (name, update) in &self.vectors {
            let current = config.vectors.get(name).ok_or_else(|| {
                Error::invalid(format!(
                    "this collection has no vector named `{name}` (vectors cannot be added later)"
                ))
            })?;
            let updated = update
                .apply(current)
                .map_err(|e| Error::invalid(format!("vector `{name}`: {}", message(&e))))?;
            out.vectors.insert(name.clone(), updated);
        }
        for (name, sparse) in &self.sparse_vectors {
            if !config.sparse_vectors.contains_key(name) {
                return Err(Error::invalid(format!(
                    "this collection has no sparse vector named `{name}`"
                )));
            }
            out.sparse_vectors.insert(name.clone(), *sparse);
        }
        if let Some(expiry) = &self.expiry {
            out.expiry = expiry.clone();
        }
        out.validate()?;
        Ok(out)
    }
}

/// How a vector's components are stored, in memory and in on-disk vector
/// files. Queries are always 32-bit floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Datatype {
    /// 32-bit floats, exactly as written.
    #[default]
    #[serde(alias = "f32")]
    Float32,
    /// 16-bit floats (IEEE 754 half precision): half the memory. Written
    /// components are rounded to the nearest half (about three significant
    /// digits, at most 65504 in magnitude); stored and returned vectors are
    /// the rounded values.
    #[serde(alias = "f16")]
    Float16,
    /// Unsigned bytes: a quarter of the memory. Written components must be
    /// whole numbers from 0 to 255, and are stored exactly. Not with cosine.
    #[serde(alias = "u8")]
    Uint8,
}

/// Largest finite `f16`.
const F16_MAX: f32 = 65504.0;

impl Datatype {
    pub fn is_float32(&self) -> bool {
        *self == Datatype::Float32
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Datatype::Float32 => "float32",
            Datatype::Float16 => "float16",
            Datatype::Uint8 => "uint8",
        }
    }

    /// Bytes per component.
    pub fn width(self) -> usize {
        match self {
            Datatype::Float32 => 4,
            Datatype::Float16 => 2,
            Datatype::Uint8 => 1,
        }
    }

    /// Makes `v` hold exactly what this datatype stores: rounds to halves,
    /// or checks for whole bytes. Fails for components out of range.
    pub fn store(self, v: &mut [f32]) -> Result<()> {
        match self {
            Datatype::Float32 => {}
            Datatype::Float16 => {
                for x in v.iter_mut() {
                    if x.abs() > F16_MAX {
                        return Err(Error::invalid(format!(
                            "float16 vectors cannot hold {x}: components must be between -{F16_MAX} and {F16_MAX}"
                        )));
                    }
                    *x = half::f16::from_f32(*x).to_f32();
                }
            }
            Datatype::Uint8 => {
                if let Some(x) = v.iter().find(|x| !Self::is_byte(**x)) {
                    return Err(Error::invalid(format!(
                        "uint8 vectors cannot hold {x}: components must be whole numbers from 0 to 255"
                    )));
                }
            }
        }
        Ok(())
    }

    fn is_byte(x: f32) -> bool {
        (0.0..=255.0).contains(&x) && x.fract() == 0.0
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
    /// One bit per component (its sign) plus 4 bytes per vector (its mean
    /// magnitude), searched instead of the vectors, which are kept exactly
    /// and re-rank the results. The codes take about a thirty-second of the
    /// memory of the vectors; keep those on disk (`on_disk`) to save it.
    /// Best for high-dimensional embeddings (several hundred dimensions or
    /// more) whose components are centred around zero.
    Binary,
}

impl Quantization {
    pub fn is_none(&self) -> bool {
        *self == Quantization::None
    }

    /// The name used in messages, for quantizations other than `None`.
    pub fn name(&self) -> Option<&'static str> {
        match self {
            Quantization::None => None,
            Quantization::Scalar => Some("scalar"),
            Quantization::Binary => Some("binary"),
        }
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

        let c: CollectionConfig = serde_json::from_str(r#"{"dimension":3,"quantization":"binary"}"#).unwrap();
        assert_eq!(c.quantization, Quantization::Binary);
        assert!(c.validate().is_ok());
        assert!(c.clone().metric(Metric::Hamming).validate().is_err());
        assert!(
            c.metric(Metric::Euclidean)
                .index(IndexConfig::KdTree(KdTreeParams::default()))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn expiry_serde_validation_and_updates() {
        let c: CollectionConfig = serde_json::from_value(serde_json::json!({
            "dimension": 3,
            "expiry": {"field": "meta.expires_at", "default_ttl": 60},
        }))
        .unwrap();
        assert_eq!(
            c.expiry,
            Some(ExpiryConfig::new("meta.expires_at").default_ttl(60))
        );
        assert!(c.validate().is_ok());
        assert!(
            !serde_json::to_string(&CollectionConfig::new(3))
                .unwrap()
                .contains("expiry")
        );
        for bad in [
            ExpiryConfig::new(""),
            ExpiryConfig::new("$and"),
            ExpiryConfig::new("t").default_ttl(0),
            ExpiryConfig::new("t").default_ttl(MAX_TTL_SECS + 1),
        ] {
            assert!(
                CollectionConfig::new(3).expiry(bad.clone()).validate().is_err(),
                "{bad:?}"
            );
        }

        let unchanged: CollectionUpdate = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(unchanged.expiry, None);
        assert_eq!(unchanged.apply(&c).unwrap().expiry, c.expiry);
        let off: CollectionUpdate = serde_json::from_value(serde_json::json!({"expiry": null})).unwrap();
        assert_eq!(off.expiry, Some(None));
        assert_eq!(off.apply(&c).unwrap().expiry, None);
        assert_eq!(
            serde_json::to_value(&off).unwrap(),
            serde_json::json!({"expiry": null})
        );
        let on: CollectionUpdate =
            serde_json::from_value(serde_json::json!({"expiry": {"field": "ttl"}})).unwrap();
        assert_eq!(on.apply(&c).unwrap().expiry, Some(ExpiryConfig::new("ttl")));
        let bad: CollectionUpdate =
            serde_json::from_value(serde_json::json!({"expiry": {"field": "a..b"}})).unwrap();
        assert!(bad.apply(&c).is_err());
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
    fn updates_patch_the_index() {
        let config = CollectionConfig::new(3)
            .index(IndexConfig::Hnsw(HnswParams {
                m: 32,
                ..Default::default()
            }))
            .with_vector("title", VectorConfig::new(2));
        let update: CollectionUpdate = serde_json::from_value(serde_json::json!({
            "index": {"ef_search": 200},
            "quantization": "scalar",
            "vectors": {"title": {"index": {"type": "flat"}, "on_disk": true}},
        }))
        .unwrap();
        let out = update.apply(&config).unwrap();
        let IndexConfig::Hnsw(p) = out.index else { panic!() };
        assert_eq!((p.m, p.ef_search), (32, 200), "other parameters are kept");
        assert_eq!(out.quantization, Quantization::Scalar);
        assert_eq!(out.vectors["title"].index, IndexConfig::Flat);
        assert!(out.vectors["title"].on_disk);

        let switch: CollectionUpdate =
            serde_json::from_value(serde_json::json!({"index": {"type": "flat"}})).unwrap();
        assert_eq!(switch.apply(&config).unwrap().index, IndexConfig::Flat);
        for bad in [
            serde_json::json!({"index": {"m": 1}}),
            serde_json::json!({"index": {"type": "kd_tree"}, "quantization": "scalar"}),
            serde_json::json!({"vectors": {"body": {"on_disk": true}}}),
            serde_json::json!({"sparse_vectors": {"text": {}}}),
        ] {
            let update: CollectionUpdate = serde_json::from_value(bad.clone()).unwrap();
            assert!(update.apply(&config).is_err(), "{bad}");
        }
        assert!(serde_json::from_value::<CollectionUpdate>(serde_json::json!({"dimension": 4})).is_err());
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
