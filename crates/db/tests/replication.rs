//! Shard replicas applying replicated commands, and collections sharded
//! over several local "shards" through an in-process cluster backend.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use quiver_db::replication::{Command, Outcome, ShardRead, ShardReply, Write};
use quiver_db::{
    ClusterBackend, ClusterCatalog, Collection, CollectionConfig, ContextPair, Database, DbOptions, Error,
    Example, Filter, Fusion, GroupBy, HybridQuery, IndexConfig, LogPosition, Maintenance, MetadataMode,
    Metric, OrderBy, Point, PointSelector, Prefetch, QueryVector, Rescore, Result, ScrollOptions,
    SearchOptions, SearchTarget, ShardedCollection, SparseVector, SparseVectorConfig, StorageKind,
};
use serde_json::{Value, json};

/// A fixed time near the clock's, so stamped expiry times stay ahead of it.
fn now() -> f64 {
    static NOW: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *NOW.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            .floor()
    })
}

fn at(index: u64) -> LogPosition {
    LogPosition {
        index,
        term: 1,
        node: 1,
    }
}

fn write(command: Command) -> Write {
    Write { now: now(), command }
}

#[test]
fn applied_index_survives_restarts_and_failed_commands() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.create_collection("s", CollectionConfig::new(2)).unwrap();
        assert_eq!(c.applied(), None);
        let points = vec![Point::new("a", vec![1.0, 0.0]), Point::new("b", vec![0.0, 1.0])];
        assert_eq!(
            c.apply(at(1), &write(Command::Upsert { points })).unwrap(),
            Outcome::Count(2)
        );
        let deleted = c
            .apply(
                at(2),
                &write(Command::Delete {
                    ids: vec!["a".into()],
                }),
            )
            .unwrap();
        assert_eq!(deleted, Outcome::Count(1));
        // Refused the same way on every replica, so it still counts as applied.
        let bad = Command::Upsert {
            points: vec![Point::new("c", vec![1.0])],
        };
        assert!(matches!(
            c.apply(at(3), &write(bad)),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(c.applied(), Some(at(3)));
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("s").unwrap();
    assert_eq!(c.applied(), Some(at(3)));
    assert_eq!(c.len(), 1);
    c.mark_applied(at(4)).unwrap();
    c.flush().unwrap();
    drop(c);
    drop(db);
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("s").unwrap().applied(), Some(at(4)));
}

#[test]
fn applying_a_command_twice_changes_nothing_more() {
    let db = Database::in_memory();
    let config = CollectionConfig::new(1).expiry(quiver_db::ExpiryConfig::new("exp").default_ttl(3600));
    let c = db.create_collection("s", config).unwrap();
    let commands = [
        Command::Upsert {
            points: (0..4)
                .map(|i| Point::new(format!("p{i}"), vec![i as f32 + 1.0]).with_metadata(json!({"n": i})))
                .collect(),
        },
        Command::UpdateMetadata {
            selector: PointSelector::Filter(Filter::parse(&json!({"n": {"$gte": 2}})).unwrap()),
            metadata: json!({"big": true}),
            mode: MetadataMode::Merge,
        },
        Command::DeleteWhere {
            filter: Filter::parse(&json!({"n": 0})).unwrap(),
        },
    ];
    for (i, command) in commands.into_iter().enumerate() {
        let w = write(command);
        let index = i as u64 + 1;
        c.apply(at(index), &w).unwrap();
        let once = c.scroll(None, 100).unwrap();
        c.apply(at(index), &w).unwrap();
        assert_eq!(c.scroll(None, 100).unwrap(), once, "command {index}");
    }
    let p = c.get("p3").unwrap();
    // Stamped as of the command's time, not the clock's.
    assert_eq!(p.metadata.unwrap()["exp"].as_f64(), Some(now() + 3600.0));
}

#[test]
fn archives_install_with_their_applied_index() {
    let src = Database::in_memory();
    let c = src.create_collection("s", CollectionConfig::new(2)).unwrap();
    c.create_field_index("tag").unwrap();
    c.upsert([Point::new("a", vec![1.0, 2.0]).with_metadata(json!({"tag": "x"}))])
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("snap.tar.gz");
    c.export_archive(&archive).unwrap();

    let dst_dir = tempfile::tempdir().unwrap();
    {
        let dst = Database::open(dst_dir.path()).unwrap();
        // Replaces whatever the replica held.
        dst.create_collection("s", CollectionConfig::new(2)).unwrap();
        let installed = dst.install_archive("s", &archive, Some(at(42))).unwrap();
        assert_eq!(installed.applied(), Some(at(42)));
        assert_eq!(installed.info().field_indexes, ["tag"]);
    }
    let dst = Database::open(dst_dir.path()).unwrap();
    let c = dst.collection("s").unwrap();
    assert_eq!(c.applied(), Some(at(42)));
    assert_eq!(c.get("a").unwrap().metadata, Some(json!({"tag": "x"})));
}

/// A cluster in one process: each shard is a local collection, and every
/// command is applied straight away.
#[derive(Debug)]
struct LocalShards {
    shards: Database,
    catalog: Mutex<ClusterCatalog>,
    index: AtomicU64,
}

impl LocalShards {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            shards: Database::in_memory(),
            catalog: Mutex::default(),
            index: AtomicU64::new(0),
        })
    }

    fn shard(&self, collection: &str, shard: u32) -> Result<Arc<Collection>> {
        self.shards.collection(&format!("{collection}-{shard}"))
    }
}

impl ClusterBackend for LocalShards {
    fn storage(&self) -> StorageKind {
        StorageKind::Memory
    }

    fn catalog(&self) -> Result<ClusterCatalog> {
        Ok(self.catalog.lock().clone())
    }

    fn resolve(&self, name: &str) -> Result<Option<(String, ShardedCollection)>> {
        Ok(self
            .catalog
            .lock()
            .resolve(name)
            .map(|(n, c)| (n.to_owned(), c.clone())))
    }

    fn create_collection(
        &self,
        name: &str,
        config: &CollectionConfig,
        field_indexes: &[String],
    ) -> Result<()> {
        let mut catalog = self.catalog.lock();
        if catalog.collections.contains_key(name) {
            return Err(Error::CollectionExists(name.into()));
        }
        for s in 0..3 {
            self.shards.create_collection_with_fields(
                &format!("{name}-{s}"),
                config.clone(),
                field_indexes,
            )?;
        }
        catalog.collections.insert(
            name.into(),
            ShardedCollection {
                config: config.clone(),
                shards: 3,
            },
        );
        Ok(())
    }

    fn drop_collection(&self, name: &str) -> Result<()> {
        self.catalog.lock().collections.remove(name);
        Ok(())
    }

    fn rename_collection(&self, _from: &str, _to: &str) -> Result<()> {
        Err(Error::Unsupported("rename".into()))
    }

    fn set_alias(&self, alias: &str, collection: &str) -> Result<()> {
        self.catalog
            .lock()
            .aliases
            .insert(alias.into(), collection.into());
        Ok(())
    }

    fn delete_alias(&self, alias: &str) -> Result<()> {
        self.catalog.lock().aliases.remove(alias);
        Ok(())
    }

    fn set_config(&self, collection: &str, config: &CollectionConfig) -> Result<()> {
        if let Some(c) = self.catalog.lock().collections.get_mut(collection) {
            c.config = config.clone();
        }
        Ok(())
    }

    fn write(&self, collection: &str, writes: Vec<(u32, Write)>) -> Result<Vec<Outcome>> {
        writes
            .into_iter()
            .map(|(s, w)| {
                let index = self.index.fetch_add(1, Ordering::Relaxed) + 1;
                self.shard(collection, s)?.apply(at(index), &w)
            })
            .collect()
    }

    fn read(&self, collection: &str, reads: Vec<(u32, ShardRead)>) -> Result<Vec<ShardReply>> {
        reads
            .into_iter()
            .map(|(s, r)| self.shard(collection, s)?.read_shard(&r))
            .collect()
    }

    fn maintain(&self, _collection: &str, _op: Maintenance) -> Result<()> {
        Ok(())
    }
}

fn config() -> CollectionConfig {
    CollectionConfig::new(2)
        .metric(Metric::Euclidean)
        .index(IndexConfig::Flat)
        .with_sparse_vector("text", SparseVectorConfig::default())
}

fn points() -> Vec<Point> {
    (0..60)
        .map(|i| {
            let x = i as f32;
            let kind = ["a", "b", "c"][i as usize % 3];
            Point::new(format!("p{i:02}"), vec![x.cos() * x, x.sin() * x])
                .with_sparse_vector("text", SparseVector::new(vec![i % 7, 10 + i % 3], vec![1.0, 0.5]))
                .with_metadata(json!({
                    "n": i,
                    "kind": kind,
                    "tags": [i % 2, i % 5],
                }))
        })
        .collect()
}

/// The same data, in one local collection and sharded three ways.
fn pair() -> (Arc<Collection>, Arc<Collection>) {
    let single = Database::in_memory().create_collection("c", config()).unwrap();
    single.upsert(points()).unwrap();
    let cluster = Database::distributed(LocalShards::new());
    let sharded = cluster.create_collection("c", config()).unwrap();
    assert!(sharded.is_distributed());
    sharded.upsert(points()).unwrap();
    (single, sharded)
}

fn filter(v: Value) -> Filter {
    Filter::parse(&v).unwrap()
}

#[test]
fn sharded_points_spread_and_read_back() {
    let (single, sharded) = pair();
    assert_eq!(sharded.len(), 60);
    let info = sharded.try_info().unwrap();
    assert_eq!(info.count, 60);
    assert_eq!(sharded.get("p07"), single.get("p07"));
    let ids = ["p09", "nope", "p01", "p09"];
    assert_eq!(sharded.get_many(&ids), single.get_many(&ids));
    let f = filter(json!({"kind": "b"}));
    assert_eq!(sharded.count(Some(&f)), single.count(Some(&f)));
}

#[test]
fn sharded_scrolls_page_like_one_collection() {
    let (single, sharded) = pair();
    for opts in [
        ScrollOptions::new(7),
        ScrollOptions {
            filter: Some(filter(json!({"kind": {"$ne": "a"}}))),
            ..ScrollOptions::new(9)
        },
        ScrollOptions {
            order_by: Some(OrderBy::desc("n")),
            ..ScrollOptions::new(8)
        },
    ] {
        let (mut a, mut b) = (opts.clone(), opts.clone());
        loop {
            let (pa, pb) = (single.scroll_with(&a).unwrap(), sharded.scroll_with(&b).unwrap());
            assert_eq!(pa, pb);
            match pa.next_cursor {
                Some(c) => {
                    a.cursor = Some(c.clone());
                    b.cursor = Some(c);
                }
                None => break,
            }
        }
    }
}

#[test]
fn sharded_searches_match_one_collection() {
    let (single, sharded) = pair();
    let opts = SearchOptions {
        include_metadata: true,
        offset: 3,
        filter: Some(filter(json!({"n": {"$lt": 50}}))),
        ..SearchOptions::default()
    };
    assert_eq!(
        sharded.search_with(&[3.0, 1.0], 10, &opts).unwrap(),
        single.search_with(&[3.0, 1.0], 10, &opts).unwrap()
    );
    assert_eq!(
        sharded.search_by_id("p10", 5, &opts).unwrap(),
        single.search_by_id("p10", 5, &opts).unwrap()
    );
    let sparse = SparseVector::new(vec![3, 11], vec![1.0, 1.0]);
    let a = single
        .search_sparse("text", &sparse, 8, &SearchOptions::default())
        .unwrap();
    let b = sharded
        .search_sparse("text", &sparse, 8, &SearchOptions::default())
        .unwrap();
    let dists = |h: &[quiver_db::SearchHit]| h.iter().map(|h| h.distance).collect::<Vec<_>>();
    assert_eq!(dists(&a), dists(&b));
    assert_eq!(
        sharded.recommend(&["p05", "p06"], &["p30"], 6, &opts).unwrap(),
        single.recommend(&["p05", "p06"], &["p30"], 6, &opts).unwrap()
    );
    assert!(matches!(
        sharded.search_by_id("missing", 3, &opts),
        Err(Error::PointNotFound(_))
    ));
}

#[test]
fn sharded_groups_facets_and_discovery_match_one_collection() {
    let (single, sharded) = pair();
    let group = GroupBy::new("kind", 2, 3);
    for target in [
        SearchTarget::Vector(vec![0.0, 0.0]),
        SearchTarget::Id("p12".into()),
    ] {
        assert_eq!(
            sharded
                .search_groups(&target, &group, &SearchOptions::default())
                .unwrap(),
            single
                .search_groups(&target, &group, &SearchOptions::default())
                .unwrap()
        );
    }
    assert_eq!(
        sharded.facet("tags", None, 10).unwrap(),
        single.facet("tags", None, 10).unwrap()
    );
    let context = [ContextPair {
        positive: Example::Id("p03".into()),
        negative: Example::Vector(vec![-20.0, 5.0]),
    }];
    for target in [None, Some(Example::Id("p20".into()))] {
        assert_eq!(
            sharded
                .discover(target.as_ref(), &context, 5, &SearchOptions::default())
                .unwrap(),
            single
                .discover(target.as_ref(), &context, 5, &SearchOptions::default())
                .unwrap()
        );
    }
}

#[test]
fn sharded_hybrid_queries_fuse_over_every_shard() {
    let (single, sharded) = pair();
    let prefetch = vec![
        Prefetch::new(
            QueryVector::Dense {
                vector: vec![5.0, -2.0],
                using: None,
            },
            20,
        ),
        Prefetch::new(
            QueryVector::Sparse {
                vector: SparseVector::new(vec![2], vec![1.0]),
                using: "text".into(),
            },
            20,
        ),
    ];
    for fusion in [Fusion::Rrf { k: 60 }, Fusion::RelativeScore, Fusion::Dbsf] {
        let mut q = HybridQuery::new(prefetch.clone());
        q.fusion = fusion;
        q.include_metadata = true;
        q.offset = 1;
        let ids =
            |c: &Collection| -> Vec<String> { c.query(&q, 6).unwrap().into_iter().map(|h| h.id).collect() };
        assert_eq!(ids(&sharded), ids(&single), "{fusion:?}");
    }
    let mut q = HybridQuery::new(prefetch);
    q.rescore = Some(Rescore {
        vector: vec![1.0, 1.0],
        using: None,
    });
    assert_eq!(sharded.query(&q, 5).unwrap(), single.query(&q, 5).unwrap());
}

#[test]
fn sharded_writes_route_by_id() {
    let (single, sharded) = pair();
    let f = filter(json!({"n": {"$gte": 40}}));
    assert_eq!(
        sharded.delete_where(&f).unwrap(),
        single.delete_where(&f).unwrap()
    );
    let sel = PointSelector::Ids(vec!["p01".into(), "p02".into(), "zzz".into()]);
    assert_eq!(
        sharded
            .update_metadata(&sel, json!({"x": 1}), MetadataMode::Merge)
            .unwrap(),
        2
    );
    assert_eq!(sharded.get("p02").unwrap().metadata.unwrap()["x"], json!(1));
    assert_eq!(sharded.delete(&["p01", "p02", "nope"]).unwrap(), 2);
    assert_eq!(sharded.len(), 38);
    sharded.create_field_index("kind").unwrap();
    assert_eq!(sharded.try_info().unwrap().field_indexes, ["kind"]);
    assert!(sharded.drop_field_index("kind").unwrap());
    // Bad points fail before any shard writes.
    let bad = vec![Point::new("ok", vec![1.0, 1.0]), Point::new("bad", vec![1.0])];
    assert!(sharded.upsert(bad).is_err());
    assert!(sharded.get("ok").is_none());
}

#[test]
fn distributed_catalog_goes_through_the_backend() {
    let cluster = Database::distributed(LocalShards::new());
    assert!(cluster.is_distributed());
    cluster.create_collection("c", config()).unwrap();
    cluster.set_alias("live", "c").unwrap();
    assert_eq!(cluster.collection("live").unwrap().name(), "c");
    assert_eq!(cluster.aliases(), BTreeMap::from([("live".into(), "c".into())]));
    assert_eq!(cluster.list_collections().len(), 1);
    assert!(matches!(cluster.snapshots(), Err(Error::Unsupported(_))));
    assert!(matches!(
        cluster.collection("nope"),
        Err(Error::CollectionNotFound(_))
    ));
    let _ = DbOptions::default();
}
