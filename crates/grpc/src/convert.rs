use quiver_db::{
    CollectionConfig, CollectionInfo, Error, HnswParams, IndexConfig, KdTreeParams, Metric, Point, SearchHit,
    SnapshotInfo, new_id,
};
use serde_json::Value;
use tonic::Status;

use crate::pb;

pub(crate) fn status(err: Error) -> Status {
    let msg = err.to_string();
    match err {
        Error::CollectionNotFound(_) | Error::SnapshotNotFound(_) | Error::PointNotFound(_) => {
            Status::not_found(msg)
        }
        Error::CollectionExists(_) => Status::already_exists(msg),
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
    })
}

pub(crate) fn info_to_pb(info: CollectionInfo) -> pb::CollectionInfo {
    pb::CollectionInfo {
        name: info.name,
        dimension: info.config.dimension as u32,
        metric: metric_to_pb(info.config.metric),
        index: Some(index_to_pb(&info.config.index)),
        count: info.count as u64,
        tombstones: info.tombstones as u64,
        field_indexes: info.field_indexes,
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
        metadata: json_from_pb(&p.metadata_json, "metadata_json")?,
    })
}

pub(crate) fn point_to_pb(p: Point, with_vectors: bool) -> pb::Point {
    pb::Point {
        metadata_json: json_to_pb(p.metadata.as_ref()),
        vector: if with_vectors { p.vector } else { Vec::new() },
        id: p.id,
    }
}

pub(crate) fn hit_to_pb(h: SearchHit) -> pb::SearchHit {
    pb::SearchHit {
        metadata_json: json_to_pb(h.metadata.as_ref()),
        vector: h.vector.unwrap_or_default(),
        id: h.id,
        distance: h.distance,
    }
}

pub(crate) fn snapshot_to_pb(s: SnapshotInfo) -> pb::SnapshotInfo {
    let m = s.manifest;
    pb::SnapshotInfo {
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
        })
        .unwrap();
        assert_eq!(p.id.len(), 36);
        assert_eq!(p.metadata.unwrap()["a"], 1);
        assert!(
            point_from_pb(pb::Point {
                id: "x".into(),
                vector: vec![],
                metadata_json: "{nope".into(),
            })
            .is_err()
        );
    }
}
