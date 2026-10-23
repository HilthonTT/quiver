use std::sync::Arc;

use quiver_db::{
    Collection, CollectionConfig, ContextPair, Database, Error, Example, Filter, Fusion, GroupBy,
    HybridQuery, IndexConfig, Metric, OrderBy, Point, Prefetch, QueryVector, Rescore, ScrollOptions,
    SearchOptions, SearchTarget, SparseVector, SparseVectorConfig, VectorConfig,
};
use serde_json::{Value, json};

fn line(n: usize, index: IndexConfig) -> Arc<Collection> {
    let c = Database::in_memory()
        .create_collection(
            "c",
            CollectionConfig::new(1).metric(Metric::Euclidean).index(index),
        )
        .unwrap();
    c.upsert((0..n).map(|i| {
        Point::new(format!("p{i:03}"), vec![i as f32]).with_metadata(json!({
            "doc": format!("d{}", i / 4),
            "n": i,
            "parity": if i % 2 == 0 { "even" } else { "odd" },
        }))
    }))
    .unwrap();
    c
}

fn ids<T>(hits: &[T], id: impl Fn(&T) -> &str) -> Vec<String> {
    hits.iter().map(|h| id(h).to_owned()).collect()
}

fn filter(v: Value) -> Filter {
    Filter::parse(&v).unwrap()
}

#[test]
fn offset_pages_through_search_results() {
    for index in [IndexConfig::Flat, IndexConfig::Hnsw(Default::default())] {
        let c = line(50, index);
        let all = c.search(&[0.0], 20).unwrap();
        let page = c
            .search_with(&[0.0], 5, &SearchOptions::default().offset(10))
            .unwrap();
        assert_eq!(page, all[10..15].to_vec(), "{index:?}");
        let past = c
            .search_with(&[0.0], 5, &SearchOptions::default().offset(100))
            .unwrap();
        assert!(past.is_empty());
        let by_id = c
            .search_by_id("p000", 2, &SearchOptions::default().offset(1))
            .unwrap();
        assert_eq!(ids(&by_id, |h| &h.id), ["p002", "p003"]);
        let rec = c
            .recommend(&["p000"], &[], 2, &SearchOptions::default().offset(2))
            .unwrap();
        assert_eq!(ids(&rec, |h| &h.id), ["p003", "p004"]);
    }
}

#[test]
fn groups_collect_the_best_hits_per_key() {
    let c = line(40, IndexConfig::Flat);
    let groups = c
        .search_groups(
            &SearchTarget::Vector(vec![9.0]),
            &GroupBy::new("doc", 3, 2),
            &SearchOptions::default(),
        )
        .unwrap();
    let keys: Vec<Value> = groups.iter().map(|g| g.key.clone()).collect();
    assert_eq!(keys, [json!("d2"), json!("d1"), json!("d3")]);
    assert_eq!(ids(&groups[0].hits, |h| &h.id), ["p009", "p008"]);
    assert_eq!(ids(&groups[1].hits, |h| &h.id), ["p007", "p006"]);
    assert_eq!(ids(&groups[2].hits, |h| &h.id), ["p012", "p013"]);

    let wide = c
        .search_groups(
            &SearchTarget::Vector(vec![0.0]),
            &GroupBy::new("parity", 5, 30),
            &SearchOptions::default().filter(filter(json!({"n": {"$lt": 10}}))),
        )
        .unwrap();
    assert_eq!(wide.len(), 2);
    assert_eq!(wide[0].key, json!("even"));
    assert_eq!(wide[0].hits.len(), 5);
    assert_eq!(wide[1].hits.len(), 5);

    let by_id = c
        .search_groups(
            &SearchTarget::Id("p000".into()),
            &GroupBy::new("doc", 1, 4),
            &SearchOptions::default(),
        )
        .unwrap();
    assert_eq!(ids(&by_id[0].hits, |h| &h.id), ["p001", "p002", "p003"]);

    assert!(matches!(
        c.search_groups(
            &SearchTarget::Vector(vec![0.0]),
            &GroupBy::new("doc", 0, 1),
            &SearchOptions::default()
        ),
        Err(Error::InvalidArgument(_))
    ));
    assert!(
        c.search_groups(
            &SearchTarget::Vector(vec![0.0]),
            &GroupBy::new("missing", 3, 3),
            &SearchOptions::default()
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn groups_search_sparse_vectors_and_split_arrays() {
    let c = Database::in_memory()
        .create_collection(
            "s",
            CollectionConfig::new(1).with_sparse_vector("text", SparseVectorConfig::default()),
        )
        .unwrap();
    c.upsert([
        Point::new("a", vec![1.0])
            .with_sparse_vector("text", SparseVector::new(vec![1], vec![3.0]))
            .with_metadata(json!({"tags": ["x", "y"]})),
        Point::new("b", vec![1.0])
            .with_sparse_vector("text", SparseVector::new(vec![1], vec![2.0]))
            .with_metadata(json!({"tags": "y"})),
        Point::new("c", vec![1.0])
            .with_sparse_vector("text", SparseVector::new(vec![1], vec![1.0]))
            .with_metadata(json!({"tags": ["z"]})),
    ])
    .unwrap();
    let groups = c
        .search_groups(
            &SearchTarget::Sparse {
                vector: SparseVector::new(vec![1], vec![1.0]),
                using: "text".into(),
            },
            &GroupBy::new("tags", 10, 2),
            &SearchOptions::default(),
        )
        .unwrap();
    let summary: Vec<(Value, Vec<String>)> = groups
        .iter()
        .map(|g| (g.key.clone(), ids(&g.hits, |h| &h.id)))
        .collect();
    assert_eq!(
        summary,
        [
            (json!("x"), vec!["a".to_owned()]),
            (json!("y"), vec!["a".to_owned(), "b".to_owned()]),
            (json!("z"), vec!["c".to_owned()]),
        ]
    );
}

#[test]
fn facets_count_values_of_matching_points() {
    let c = line(10, IndexConfig::Flat);
    let all = c.facet("parity", None, 10).unwrap();
    assert_eq!(
        all.iter().map(|f| (f.value.clone(), f.count)).collect::<Vec<_>>(),
        [(json!("even"), 5), (json!("odd"), 5)]
    );
    let some = c
        .facet("doc", Some(&filter(json!({"n": {"$gte": 3}}))), 2)
        .unwrap();
    assert_eq!(
        some.iter()
            .map(|f| (f.value.clone(), f.count))
            .collect::<Vec<_>>(),
        [(json!("d1"), 4), (json!("d2"), 2)]
    );
    assert!(c.facet("$bad", None, 10).is_err());
}

#[test]
fn ordered_scrolls_page_by_value_then_id() {
    let c = Database::in_memory()
        .create_collection("o", CollectionConfig::new(1))
        .unwrap();
    let years = [1965, 1937, 1965, 2001, 1990];
    c.upsert(years.iter().enumerate().map(|(i, y)| {
        Point::new(format!("b{i}"), vec![1.0]).with_metadata(json!({"year": y, "even": i % 2 == 0}))
    }))
    .unwrap();
    c.upsert([
        Point::new("none", vec![1.0]),
        Point::new("text", vec![1.0]).with_metadata(json!({"year": "unknown"})),
    ])
    .unwrap();

    let collect = |opts: ScrollOptions| {
        let mut out = Vec::new();
        let mut opts = opts;
        loop {
            let page = c.scroll_with(&opts).unwrap();
            out.extend(page.points.into_iter().map(|p| p.id));
            match page.next_cursor {
                Some(cursor) => opts.cursor = Some(cursor),
                None => return out,
            }
        }
    };
    assert_eq!(
        collect(ScrollOptions::new(2).order_by(OrderBy::asc("year"))),
        ["b1", "b0", "b2", "b4", "b3", "text"]
    );
    assert_eq!(
        collect(ScrollOptions::new(3).order_by(OrderBy::desc("year"))),
        ["text", "b3", "b4", "b0", "b2", "b1"]
    );
    assert_eq!(
        collect(
            ScrollOptions::new(1)
                .order_by(OrderBy::desc("year"))
                .filter(filter(json!({"even": true})))
        ),
        ["b4", "b0", "b2"]
    );
    assert_eq!(
        collect(ScrollOptions::new(2).filter(filter(json!({"year": {"$gt": 1950}})))),
        ["b0", "b2", "b3", "b4"]
    );
    let bad = ScrollOptions::new(2).order_by(OrderBy::asc("year")).cursor("b1");
    assert!(matches!(c.scroll_with(&bad), Err(Error::InvalidArgument(_))));
}

fn hybrid() -> Arc<Collection> {
    let c = Database::in_memory()
        .create_collection(
            "h",
            CollectionConfig::new(2)
                .metric(Metric::Euclidean)
                .index(IndexConfig::Flat)
                .with_vector(
                    "fine",
                    VectorConfig::new(2)
                        .metric(Metric::Euclidean)
                        .index(IndexConfig::Flat),
                )
                .with_sparse_vector("text", SparseVectorConfig::default()),
        )
        .unwrap();
    c.upsert((0..6).map(|i| {
        let x = i as f32;
        Point::new(format!("p{i}"), vec![x, 0.0])
            .with_vector("fine", vec![5.0 - x, 0.0])
            .with_sparse_vector("text", SparseVector::new(vec![1], vec![6.0 - x]))
    }))
    .unwrap();
    c
}

fn prefetch_dense(v: [f32; 2], limit: usize) -> Prefetch {
    Prefetch::new(
        QueryVector::Dense {
            vector: v.to_vec(),
            using: None,
        },
        limit,
    )
}

#[test]
fn score_fusions_normalise_each_search() {
    let c = hybrid();
    let sparse = Prefetch::new(
        QueryVector::Sparse {
            vector: SparseVector::new(vec![1], vec![1.0]),
            using: "text".into(),
        },
        6,
    );
    for fusion in [Fusion::RelativeScore, Fusion::Dbsf] {
        let mut q = HybridQuery::new(vec![prefetch_dense([5.0, 0.0], 6), sparse.clone()]);
        q.fusion = fusion;
        let hits = c.query(&q, 6).unwrap();
        assert_eq!(hits.len(), 6, "{fusion:?}");
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));

        let mut solo = HybridQuery::new(vec![prefetch_dense([0.0, 0.0], 6)]);
        solo.fusion = fusion;
        let hits = c.query(&solo, 6).unwrap();
        assert_eq!(hits[0].id, "p0", "{fusion:?}");
    }
    let mut minmax = HybridQuery::new(vec![prefetch_dense([0.0, 0.0], 6)]);
    minmax.fusion = Fusion::RelativeScore;
    let hits = c.query(&minmax, 6).unwrap();
    assert_eq!(hits[0].score, 1.0);
    assert_eq!(hits[5].score, 0.0);
}

#[test]
fn rescoring_reorders_prefetched_candidates() {
    let c = hybrid();
    let mut q = HybridQuery::new(vec![prefetch_dense([0.0, 0.0], 3)]);
    q.rescore = Some(Rescore {
        vector: vec![5.0, 0.0],
        using: Some("fine".into()),
    });
    let hits = c.query(&q, 10).unwrap();
    assert_eq!(ids(&hits, |h| &h.id), ["p0", "p1", "p2"]);
    assert_eq!(hits[0].score, 0.0);
    assert_eq!(hits[1].score, -1.0);

    let mut weightless = HybridQuery::new(vec![prefetch_dense([0.0, 0.0], 2)]);
    weightless.prefetch[0].weight = 0.0;
    weightless.rescore = q.rescore.clone();
    assert_eq!(c.query(&weightless, 10).unwrap().len(), 2);

    q.offset = 1;
    let hits = c.query(&q, 1).unwrap();
    assert_eq!(ids(&hits, |h| &h.id), ["p1"]);

    q.rescore = Some(Rescore {
        vector: vec![1.0],
        using: None,
    });
    assert!(matches!(c.query(&q, 1), Err(Error::DimensionMismatch { .. })));
}

#[test]
fn discovery_ranks_by_context_then_target() {
    let c = line(20, IndexConfig::Flat);
    let pair = |p: &str, n: &str| ContextPair {
        positive: Example::Id(p.into()),
        negative: Example::Id(n.into()),
    };
    let hits = c
        .discover(
            Some(&Example::Vector(vec![10.0])),
            &[pair("p015", "p005")],
            3,
            &SearchOptions::default(),
        )
        .unwrap();
    assert_eq!(ids(&hits, |h| &h.id), ["p011", "p012", "p013"]);
    assert_eq!(hits[0].distance, 1.0);

    let context_only = c
        .discover(
            None,
            &[pair("p018", "p002"), pair("p017", "p009")],
            4,
            &SearchOptions::default(),
        )
        .unwrap();
    assert!(context_only.iter().all(|h| h.distance == 0.0));
    assert!(context_only.iter().all(|h| h.id.as_str() >= "p013"));
    assert!(!ids(&context_only, |h| &h.id).contains(&"p018".to_owned()));

    let vectors = c
        .discover(
            None,
            &[ContextPair {
                positive: Example::Vector(vec![0.0]),
                negative: Example::Vector(vec![19.0]),
            }],
            2,
            &SearchOptions::default().filter(filter(json!({"parity": "odd"}))),
        )
        .unwrap();
    assert_eq!(vectors.len(), 2);
    assert!(vectors.iter().all(|h| h.distance == 0.0));

    assert!(matches!(
        c.discover(None, &[], 3, &SearchOptions::default()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        c.discover(
            Some(&Example::Id("nope".into())),
            &[],
            3,
            &SearchOptions::default()
        ),
        Err(Error::PointNotFound(_))
    ));
}

#[test]
fn new_filter_operators_work_in_searches() {
    let c = Database::in_memory()
        .create_collection("f", CollectionConfig::new(1))
        .unwrap();
    c.upsert([
        Point::new("bxl", vec![1.0]).with_metadata(json!({
            "name": "Brussels Central",
            "loc": {"lat": 50.845, "lon": 4.357},
            "lines": ["1", "5"],
        })),
        Point::new("ant", vec![1.0]).with_metadata(json!({
            "name": "Antwerp Central",
            "loc": {"lat": 51.217, "lon": 4.421},
            "lines": ["1"],
        })),
    ])
    .unwrap();
    c.create_field_index("name").unwrap();
    let count = |f: Value| c.count(Some(&filter(f)));
    assert_eq!(count(json!({"name": {"$text": "central"}})), 2);
    assert_eq!(count(json!({"name": {"$text": "brussels central"}})), 1);
    assert_eq!(count(json!({"name": {"$prefix": "Ant"}})), 1);
    assert_eq!(count(json!({"name": {"$regex": "^[AB]\\w+ C"}})), 2);
    assert_eq!(count(json!({"lines": {"$all": ["1", "5"]}})), 1);
    assert_eq!(count(json!({"lines": {"$size": 1}})), 1);
    assert_eq!(
        count(json!({"loc": {"$geoRadius": {"center": {"lat": 50.85, "lon": 4.35}, "meters": 5_000}}})),
        1
    );
}
