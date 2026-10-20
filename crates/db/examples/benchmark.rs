//! Measures HNSW build time, query throughput and recall@k against exact search.
//!
//! ```sh
//! cargo run --release --example benchmark -- [points] [dim] [queries] [clustered|uniform]
//! cargo run --release --example benchmark -- 100000 128 1000
//! ```
//!
//! `clustered` (the default) draws points around 64 random centroids, which
//! resembles real embedding distributions. `uniform` draws from the unit
//! hypercube: a worst case for every graph index, because distances in high
//! dimensions concentrate and neighbourhoods become nearly meaningless.

use std::collections::HashSet;
use std::time::Instant;

use quiver_db::{CollectionConfig, Database, IndexConfig, Metric, Point, SearchOptions};

const K: usize = 10;

fn main() -> quiver_db::Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<usize> = raw.iter().filter_map(|a| a.parse().ok()).collect();
    let uniform = raw.iter().any(|a| a == "uniform");
    let n = args.first().copied().unwrap_or(20_000);
    let dim = args.get(1).copied().unwrap_or(128);
    let nq = args.get(2).copied().unwrap_or(500);

    let dist = if uniform { "uniform" } else { "clustered" };
    println!("points={n} dim={dim} queries={nq} k={K} data={dist}\n");
    let mut rng = Rng(42);
    let centroids: Vec<Vec<f32>> = (0..64).map(|_| rng.vector(dim)).collect();
    let sample = |rng: &mut Rng| {
        if uniform {
            return rng.vector(dim);
        }
        let c = &centroids[(rng.next() * centroids.len() as f32) as usize % centroids.len()];
        c.iter().map(|x| x + (rng.next() - 0.5) * 0.4).collect()
    };
    let data: Vec<Vec<f32>> = (0..n).map(|_| sample(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..nq).map(|_| sample(&mut rng)).collect();

    let db = Database::in_memory();
    let config = CollectionConfig::new(dim).metric(Metric::Euclidean);
    let flat = db.create_collection("flat", config.clone().index(IndexConfig::Flat))?;
    let hnsw = db.create_collection("hnsw", config.index(IndexConfig::default()))?;
    let points = || {
        data.iter()
            .enumerate()
            .map(|(i, v)| Point::new(i.to_string(), v.clone()))
    };

    flat.upsert(points())?;
    let t = Instant::now();
    hnsw.upsert(points())?;
    let build = t.elapsed();
    println!(
        "HNSW build: {:.2?} ({:.0} inserts/s)\n",
        build,
        n as f64 / build.as_secs_f64()
    );

    let t = Instant::now();
    let truth: Vec<HashSet<String>> = queries
        .iter()
        .map(|q| {
            flat.search(q, K)
                .map(|h| h.into_iter().map(|h| h.id).collect())
        })
        .collect::<Result<_, _>>()?;
    let flat_qps = nq as f64 / t.elapsed().as_secs_f64();
    println!("{:<14} {:>10} {:>10}", "index", "QPS", "recall@10");
    println!("{:<14} {:>10.0} {:>10.4}", "flat (exact)", flat_qps, 1.0);

    // k-d trees only pay off in low dimensions; above that they degrade
    // towards brute force, so skip them rather than waste minutes.
    if dim <= 32 {
        let kd = db.create_collection(
            "kd",
            CollectionConfig::new(dim)
                .metric(Metric::Euclidean)
                .index(IndexConfig::KdTree(Default::default())),
        )?;
        kd.upsert(points())?;
        let opts = SearchOptions::default().include_metadata(false);
        let t = Instant::now();
        let mut hits = 0;
        for (q, truth) in queries.iter().zip(&truth) {
            hits += kd
                .search_with(q, K, &opts)?
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
        }
        let qps = nq as f64 / t.elapsed().as_secs_f64();
        let recall = hits as f64 / (nq * K) as f64;
        println!("{:<14} {:>10.0} {:>10.4}", "kd_tree", qps, recall);
    }

    for ef in [16, 32, 64, 128, 256] {
        let opts = SearchOptions::default().ef(ef).include_metadata(false);
        let t = Instant::now();
        let mut hits = 0;
        for (q, truth) in queries.iter().zip(&truth) {
            hits += hnsw
                .search_with(q, K, &opts)?
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
        }
        let qps = nq as f64 / t.elapsed().as_secs_f64();
        let recall = hits as f64 / (nq * K) as f64;
        println!(
            "{:<14} {:>10.0} {:>10.4}",
            format!("hnsw ef={ef}"),
            qps,
            recall
        );
    }
    Ok(())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next() * 2.0 - 1.0).collect()
    }
}
