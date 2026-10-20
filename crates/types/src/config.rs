use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Error, Result};

/// Maximum vector dimension accepted.
pub const MAX_DIMENSION: usize = 65_536;
/// Maximum length of a point id, in bytes.
pub const MAX_ID_LEN: usize = 512;
/// Maximum collection-name length.
pub const MAX_NAME_LEN: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub dimension: usize,
    #[serde(default)]
    pub metric: Metric,
    #[serde(default)]
    pub index: IndexConfig,
}

impl CollectionConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: Metric::default(),
            index: IndexConfig::default(),
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

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn kd_tree_rejects_dot_product() {
        let config = CollectionConfig::new(3)
            .metric(Metric::DotProduct)
            .index(IndexConfig::KdTree(KdTreeParams::default()));
        assert!(config.validate().is_err());
        assert!(config.metric(Metric::Manhattan).validate().is_ok());
    }
}
