use std::collections::{BTreeMap, HashMap};

use quiver_db::{
    CollectionConfig, CollectionInfo, Error, ExpiryConfig, HnswParams, IndexConfig, KdTreeParams, Metric,
    NamedVectors, Point, Quantization, ScoredHit, SearchHit, SnapshotInfo, SparseModifier, SparseVector,
    SparseVectorConfig, SparseVectors, VectorConfig, new_id,
};
use serde_json::Value;
use tonic::Status;

use crate::pb;

pub(crate) fn status(err: Error) -> Status {
    let msg = err.to_string();
    match err {
        Error::CollectionNotFound(_)
        | Error::SnapshotNotFound(_)
        | Error::PointNotFound(_)
        | Error::AliasNotFound(_) => Status::not_found(msg),
        Error::CollectionExists(_) | Error::AliasExists(_) => Status::already_exists(msg),
        Error::DimensionMismatch { .. } | Error::InvalidArgument(_) | Error::InvalidFilter(_) => {
            Status::invalid_argument(msg)
        }
        Error::Unsupported(_) => Status::unimplemented(msg),
        _ => {
            tracing::error!(error = %msg, "internal error");
            Status::internal(msg)
        }
    }
}

pub(crate) fn metric_from_pb(m: i32) -> Result<Metric, Status> {
    Ok(
        match pb::Metric::try_from(m).map_err(|_| Status::invalid_argument("unknown metric"))? {
            pb::Metric::Unspecified | pb::Metric::Cosine => Metric::Cosine,
            pb::Metric::Euclidean => Metric::Euclidean,
            pb::Metric::DotProduct => Metric::DotProduct,
            pb::Metric::Manhattan => Metric::Manhattan,
            pb::Metric::Hamming => Metric::Hamming,
        },
    )
}

pub(crate) fn metric_to_pb(m: Metric) -> i32 {
    (match m {
        Metric::Cosine => pb::Metric::Cosine,
        Metric::Euclidean => pb::Metric::Euclidean,
        Metric::DotProduct => pb::Metric::DotProduct,
        Metric::Manhattan => pb::Metric::Manhattan,
        Metric::Hamming => pb::Metric::Hamming,
    }) as i32
}

pub(crate) fn index_from_pb(index: Option<pb::IndexConfig>) -> IndexConfig {
    use pb::index_config::Kind;
    match index.and_then(|i| i.kind) {
        None => IndexConfig::default(),
        Some(Kind::Flat(_)) => IndexConfig::Flat,
        Some(Kind::Hnsw(h)) => {
            let d = HnswParams::default();
            IndexConfig::Hnsw(HnswParams {
                m: h.m.map_or(d.m, |v| v as usize),
                ef_construction: h.ef_construction.map_or(d.ef_construction, |v| v as usize),
                ef_search: h.ef_search.map_or(d.ef_search, |v| v as usize),
                seed: h.seed.unwrap_or(d.seed),
            })
        }
        Some(Kind::KdTree(k)) => IndexConfig::KdTree(KdTreeParams {
            balance_factor: k.balance_factor.unwrap_or(KdTreeParams::default().balance_factor),
        }),
    }
}

pub(crate) fn index_to_pb(index: &IndexConfig) -> pb::IndexConfig {
    use pb::index_config::Kind;
    let kind = match index {
        IndexConfig::Flat => Kind::Flat(pb::FlatIndex {}),
        IndexConfig::Hnsw(h) => Kind::Hnsw(pb::HnswIndex {
            m: Some(h.m as u32),
            ef_construction: Some(h.ef_construction as u32),
            ef_search: Some(h.ef_search as u32),
            seed: Some(h.seed),
        }),
        IndexConfig::KdTree(k) => Kind::KdTree(pb::KdTreeIndex {
            balance_factor: Some(k.balance_factor),
        }),
    };
    pb::IndexConfig { kind: Some(kind) }
}

pub(crate) fn config_from_pb(req: &pb::CreateCollectionRequest) -> Result<CollectionConfig, Status> {
    Ok(CollectionConfig {
        dimension: req.dimension as usize,
        metric: metric_from_pb(req.metric)?,
        index: index_from_pb(req.index),
        quantization: quantization_from_pb(req.quantization)?,
        on_disk: req.on_disk,
        vectors: req
            .vectors
            .iter()
            .map(|(name, v)| Ok((name.clone(), vector_params_from_pb(v)?)))
            .collect::<Result<_, Status>>()?,
        sparse_vectors: req
            .sparse_vectors
            .iter()
            .map(|(name, v)| Ok((name.clone(), sparse_params_from_pb(v)?)))
            .collect::<Result<_, Status>>()?,
        expiry: req.expiry.as_ref().map(expiry_from_pb),
    })
}

fn expiry_from_pb(e: &pb::ExpiryConfig) -> ExpiryConfig {
    ExpiryConfig {
        field: e.field.clone(),
        default_ttl: e.default_ttl,
    }
}

fn expiry_to_pb(e: &ExpiryConfig) -> pb::ExpiryConfig {
    pb::ExpiryConfig {
        field: e.field.clone(),
        default_ttl: e.default_ttl,
    }
}

fn vector_params_from_pb(v: &pb::VectorParams) -> Result<VectorConfig, Status> {
    Ok(VectorConfig {
        dimension: v.dimension as usize,
        metric: metric_from_pb(v.metric)?,
        index: index_from_pb(v.index),
        quantization: quantization_from_pb(v.quantization)?,
        on_disk: v.on_disk,
    })
}

fn vector_params_to_pb(v: &VectorConfig) -> pb::VectorParams {
    pb::VectorParams {
        dimension: v.dimension as u32,
        metric: metric_to_pb(v.metric),
        index: Some(index_to_pb(&v.index)),
        quantization: quantization_to_pb(v.quantization),
        on_disk: v.on_disk,
    }
}

fn sparse_params_from_pb(v: &pb::SparseVectorParams) -> Result<SparseVectorConfig, Status> {
    let modifier = match pb::SparseModifier::try_from(v.modifier)
        .map_err(|_| Status::invalid_argument("unknown sparse modifier"))?
    {
        pb::SparseModifier::None => SparseModifier::None,
        pb::SparseModifier::Idf => SparseModifier::Idf,
    };
    Ok(SparseVectorConfig { modifier })
}

fn sparse_params_to_pb(v: &SparseVectorConfig) -> pb::SparseVectorParams {
    let modifier = match v.modifier {
        SparseModifier::None => pb::SparseModifier::None,
        SparseModifier::Idf => pb::SparseModifier::Idf,
    };
    pb::SparseVectorParams {
        modifier: modifier.into(),
    }
}

type PbVectors = (
    HashMap<String, pb::VectorParams>,
    HashMap<String, pb::SparseVectorParams>,
);

fn vector_maps_to_pb(config: &CollectionConfig) -> PbVectors {
    (
        config
            .vectors
            .iter()
            .map(|(n, v)| (n.clone(), vector_params_to_pb(v)))
            .collect(),
        config
            .sparse_vectors
            .iter()
            .map(|(n, v)| (n.clone(), sparse_params_to_pb(v)))
            .collect(),
    )
}

pub(crate) fn sparse_from_pb(v: pb::SparseVector) -> SparseVector {
    SparseVector::new(v.indices, v.values)
}

fn sparse_to_pb(v: SparseVector) -> pb::SparseVector {
    pb::SparseVector {
        indices: v.indices,
        values: v.values,
    }
}

fn named_to_pb(v: NamedVectors) -> HashMap<String, pb::DenseVector> {
    v.into_iter()
        .map(|(n, values)| (n, pb::DenseVector { values }))
        .collect()
}

fn sparse_map_to_pb(v: SparseVectors) -> HashMap<String, pb::SparseVector> {
    v.into_iter().map(|(n, s)| (n, sparse_to_pb(s))).collect()
}

pub(crate) fn quantization_from_pb(q: i32) -> Result<Quantization, Status> {
    match pb::Quantization::try_from(q).map_err(|_| Status::invalid_argument("unknown quantization"))? {
        pb::Quantization::None => Ok(Quantization::None),
        pb::Quantization::Scalar => Ok(Quantization::Scalar),
        pb::Quantization::Binary => Ok(Quantization::Binary),
    }
}

pub(crate) fn quantization_to_pb(q: Quantization) -> i32 {
    match q {
        Quantization::None => pb::Quantization::None,
        Quantization::Scalar => pb::Quantization::Scalar,
        Quantization::Binary => pb::Quantization::Binary,
    }
    .into()
}

pub(crate) fn info_to_pb(info: CollectionInfo) -> pb::CollectionInfo {
    let (vectors, sparse_vectors) = vector_maps_to_pb(&info.config);
    pb::CollectionInfo {
        on_disk: info.config.on_disk,
        expiry: info.config.expiry.as_ref().map(expiry_to_pb),
        vectors,
        sparse_vectors,
        name: info.name,
        dimension: info.config.dimension as u32,
        metric: metric_to_pb(info.config.metric),
        index: Some(index_to_pb(&info.config.index)),
        count: info.count as u64,
        tombstones: info.tombstones as u64,
        field_indexes: info.field_indexes,
        quantization: quantization_to_pb(info.config.quantization),
    }
}

pub(crate) fn json_from_pb(s: &str, what: &str) -> Result<Option<Value>, Status> {
    if s.trim().is_empty() {
        return Ok(None);
    }
    serde_json::from_str(s)
        .map(Some)
        .map_err(|e| Status::invalid_argument(format!("{what} is not valid JSON: {e}")))
}

fn json_to_pb(v: Option<&Value>) -> String {
    v.map(Value::to_string).unwrap_or_default()
}

pub(crate) fn point_from_pb(p: pb::Point) -> Result<Point, Status> {
    Ok(Point {
        id: if p.id.is_empty() { new_id() } else { p.id },
        vector: p.vector,
        vectors: p
            .vectors
            .into_iter()
            .map(|(n, v)| (n, v.values))
            .collect::<BTreeMap<_, _>>(),
        sparse_vectors: p
            .sparse_vectors
            .into_iter()
            .map(|(n, v)| (n, sparse_from_pb(v)))
            .collect(),
        metadata: json_from_pb(&p.metadata_json, "metadata_json")?,
    })
}

pub(crate) fn point_to_pb(p: Point, with_vectors: bool) -> pb::Point {
    let (vector, vectors, sparse_vectors) = if with_vectors {
        (
            p.vector,
            named_to_pb(p.vectors),
            sparse_map_to_pb(p.sparse_vectors),
        )
    } else {
        Default::default()
    };
    pb::Point {
        metadata_json: json_to_pb(p.metadata.as_ref()),
        vector,
        vectors,
        sparse_vectors,
        id: p.id,
    }
}

pub(crate) fn hit_to_pb(h: SearchHit) -> pb::SearchHit {
    pb::SearchHit {
        metadata_json: json_to_pb(h.metadata.as_ref()),
        vector: h.vector.unwrap_or_default(),
        vectors: named_to_pb(h.vectors),
        sparse_vectors: sparse_map_to_pb(h.sparse_vectors),
        id: h.id,
        distance: h.distance,
    }
}

pub(crate) fn scored_to_pb(h: ScoredHit) -> pb::ScoredHit {
    pb::ScoredHit {
        metadata_json: json_to_pb(h.metadata.as_ref()),
        vector: h.vector.unwrap_or_default(),
        vectors: named_to_pb(h.vectors),
        sparse_vectors: sparse_map_to_pb(h.sparse_vectors),
        id: h.id,
        score: h.score,
    }
}

pub(crate) fn snapshot_to_pb(s: SnapshotInfo) -> pb::SnapshotInfo {
    let m = s.manifest;
    let (vectors, sparse_vectors) = vector_maps_to_pb(&m.config);
    pb::SnapshotInfo {
        on_disk: m.config.on_disk,
        expiry: m.config.expiry.as_ref().map(expiry_to_pb),
        vectors,
        sparse_vectors,
        dimension: m.config.dimension as u32,
        metric: metric_to_pb(m.config.metric),
        index: Some(index_to_pb(&m.config.index)),
        id: m.id,
        collection: m.collection,
        created_at_ms: m.created_at_ms,
        quiver_version: m.quiver_version,
        points: m.points as u64,
        data_bytes: m.data_bytes,
        data_sha256: m.data_sha256,
        archive_bytes: s.archive_bytes,
        field_indexes: m.field_indexes,
        quantization: quantization_to_pb(m.config.quantization),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip() {
        for index in [
            IndexConfig::Flat,
            IndexConfig::default(),
            IndexConfig::KdTree(KdTreeParams { balance_factor: 7 }),
        ] {
            assert_eq!(index_from_pb(Some(index_to_pb(&index))), index);
        }
        assert_eq!(index_from_pb(None), IndexConfig::default());
    }

    #[test]
    fn config_roundtrip() {
        let config = CollectionConfig::new(3)
            .on_disk(true)
            .quantization(Quantization::Binary)
            .expiry(ExpiryConfig::new("exp").default_ttl(60))
            .with_vector(
                "t",
                VectorConfig::new(2)
                    .metric(Metric::DotProduct)
                    .index(IndexConfig::Flat),
            )
            .with_sparse_vector(
                "s",
                SparseVectorConfig {
                    modifier: SparseModifier::Idf,
                },
            );
        let info = info_to_pb(CollectionInfo {
            name: "c".into(),
            config: config.clone(),
            count: 0,
            tombstones: 0,
            field_indexes: Vec::new(),
        });
        let req = pb::CreateCollectionRequest {
            dimension: info.dimension,
            metric: info.metric,
            index: info.index,
            quantization: info.quantization,
            on_disk: info.on_disk,
            vectors: info.vectors,
            sparse_vectors: info.sparse_vectors,
            expiry: info.expiry,
            ..Default::default()
        };
        assert_eq!(config_from_pb(&req).unwrap(), config);
    }

    #[test]
    fn metric_roundtrip() {
        for m in Metric::ALL {
            assert_eq!(metric_from_pb(metric_to_pb(m)).unwrap(), m);
        }
        assert_eq!(metric_from_pb(0).unwrap(), Metric::Cosine);
        assert!(metric_from_pb(99).is_err());
    }

    #[test]
    fn points_get_ids_and_validate_json() {
        let p = point_from_pb(pb::Point {
            id: String::new(),
            vector: vec![1.0],
            metadata_json: r#"{"a":1}"#.into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(p.id.len(), 36);
        assert_eq!(p.metadata.unwrap()["a"], 1);
        assert!(
            point_from_pb(pb::Point {
                id: "x".into(),
                metadata_json: "{nope".into(),
                ..Default::default()
            })
            .is_err()
        );
    }
}
