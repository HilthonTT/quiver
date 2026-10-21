//! Request counters and a Prometheus text-format exposition of them, plus
//! gauges describing each collection. Served by the REST API at `/metrics`.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::time::Duration;

use parking_lot::Mutex;

use crate::Database;

/// Upper bounds of the latency histogram buckets, in seconds.
const BUCKETS: [f64; 12] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// A collection, as the per-collection gauges see it.
struct Row {
    info: quiver_types::CollectionInfo,
    vector_bytes: usize,
}

/// Identifies one request series.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    /// `http` or `grpc`.
    api: &'static str,
    /// The route template (`/collections/{name}/search`) or gRPC method path.
    route: String,
    /// HTTP status code or gRPC status code.
    status: String,
}

#[derive(Debug, Default, Clone)]
struct Series {
    count: u64,
    sum_secs: f64,
    /// Non-cumulative counts per bucket; the last slot is `+Inf`.
    buckets: [u64; BUCKETS.len() + 1],
}

/// Request counts and latencies, shared by the REST and gRPC layers.
#[derive(Debug, Default)]
pub struct Metrics {
    requests: Mutex<BTreeMap<Key, Series>>,
}

impl Metrics {
    /// Records one finished request. `route` must be a template, never a raw
    /// path, or every distinct id would become a series of its own.
    pub fn record(&self, api: &'static str, route: &str, status: &str, elapsed: Duration) {
        let secs = elapsed.as_secs_f64();
        let bucket = BUCKETS.iter().position(|&b| secs <= b).unwrap_or(BUCKETS.len());
        let key = Key {
            api,
            route: route.to_owned(),
            status: status.to_owned(),
        };
        let mut requests = self.requests.lock();
        let series = requests.entry(key).or_default();
        series.count += 1;
        series.sum_secs += secs;
        series.buckets[bucket] += 1;
    }

    /// Everything in the Prometheus text exposition format (version 0.0.4).
    pub fn render(&self, db: &Database) -> String {
        let mut out = String::new();
        let w = &mut out;

        gauge_header(w, "quiver_info", "Build and storage information; always 1.");
        let _ = writeln!(
            w,
            "quiver_info{{version=\"{}\",storage=\"{}\",snapshots_enabled=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION"),
            db.options().storage,
            db.snapshots().is_ok()
        );

        let collections: Vec<Row> = db
            .collections()
            .iter()
            .map(|c| Row {
                info: c.info(),
                vector_bytes: c.vector_bytes(),
            })
            .collect();
        gauge_header(w, "quiver_collections", "Number of collections.");
        let _ = writeln!(w, "quiver_collections {}", collections.len());

        type Field = fn(&Row) -> u64;
        let per_collection: [(&str, &str, Field); 4] = [
            (
                "quiver_collection_points",
                "Live points in the collection.",
                |c| c.info.count as u64,
            ),
            (
                "quiver_collection_tombstones",
                "Deleted slots awaiting compaction.",
                |c| c.info.tombstones as u64,
            ),
            ("quiver_collection_dimension", "Main vector dimension.", |c| {
                c.info.config.dimension as u64
            }),
            (
                "quiver_collection_vector_bytes",
                "Memory held by stored vectors, tombstones included (not vectors kept on disk).",
                |c| c.vector_bytes as u64,
            ),
        ];
        for (name, help, field) in per_collection {
            gauge_header(w, name, help);
            for c in &collections {
                // Collection names are [A-Za-z0-9_-], so they need no escaping.
                let _ = writeln!(w, "{name}{{collection=\"{}\"}} {}", c.info.name, field(c));
            }
        }

        let requests = self.requests.lock().clone();
        let _ = writeln!(
            w,
            "# HELP quiver_requests_total Requests served, by API, route and status."
        );
        let _ = writeln!(w, "# TYPE quiver_requests_total counter");
        for (key, series) in &requests {
            let _ = writeln!(w, "quiver_requests_total{{{}}} {}", labels(key), series.count);
        }

        let name = "quiver_request_duration_seconds";
        let _ = writeln!(
            w,
            "# HELP {name} Time to produce a response, by API, route and status."
        );
        let _ = writeln!(w, "# TYPE {name} histogram");
        for (key, series) in &requests {
            let labels = labels(key);
            let mut cumulative = 0;
            for (i, count) in series.buckets.iter().enumerate() {
                cumulative += count;
                let le = BUCKETS
                    .get(i)
                    .map_or_else(|| "+Inf".to_owned(), |b| b.to_string());
                let _ = writeln!(w, "{name}_bucket{{{labels},le=\"{le}\"}} {cumulative}");
            }
            let _ = writeln!(w, "{name}_sum{{{labels}}} {}", series.sum_secs);
            let _ = writeln!(w, "{name}_count{{{labels}}} {}", series.count);
        }
        out
    }
}

fn gauge_header(w: &mut String, name: &str, help: &str) {
    let _ = writeln!(w, "# HELP {name} {help}");
    let _ = writeln!(w, "# TYPE {name} gauge");
}

fn labels(key: &Key) -> String {
    format!(
        "api=\"{}\",route=\"{}\",status=\"{}\"",
        key.api,
        escape(&key.route),
        escape(&key.status)
    )
}

/// Escapes a label value as the text format requires.
fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use quiver_types::{CollectionConfig, Point};

    #[test]
    fn renders_collections_and_cumulative_histograms() {
        let db = Database::in_memory();
        db.create_collection("docs", CollectionConfig::new(4))
            .unwrap()
            .upsert([
                Point::new("a", vec![1.0, 0.0, 0.0, 0.0]),
                Point::new("b", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .unwrap();
        let m = Metrics::default();
        m.record(
            "http",
            "/collections/{name}/search",
            "200",
            Duration::from_millis(3),
        );
        m.record(
            "http",
            "/collections/{name}/search",
            "200",
            Duration::from_secs(20),
        );
        m.record("grpc", "/quiver.v1.Quiver/Search", "0", Duration::from_micros(10));

        let text = m.render(&db);
        for line in [
            "quiver_collections 1",
            "quiver_collection_points{collection=\"docs\"} 2",
            "quiver_collection_vector_bytes{collection=\"docs\"} 32",
            "quiver_requests_total{api=\"http\",route=\"/collections/{name}/search\",status=\"200\"} 2",
            "quiver_request_duration_seconds_bucket{api=\"http\",route=\"/collections/{name}/search\",status=\"200\",le=\"0.001\"} 0",
            "quiver_request_duration_seconds_bucket{api=\"http\",route=\"/collections/{name}/search\",status=\"200\",le=\"0.005\"} 1",
            "quiver_request_duration_seconds_bucket{api=\"http\",route=\"/collections/{name}/search\",status=\"200\",le=\"10\"} 1",
            "quiver_request_duration_seconds_bucket{api=\"http\",route=\"/collections/{name}/search\",status=\"200\",le=\"+Inf\"} 2",
            "quiver_request_duration_seconds_count{api=\"grpc\",route=\"/quiver.v1.Quiver/Search\",status=\"0\"} 1",
        ] {
            assert!(text.lines().any(|l| l == line), "missing `{line}` in:\n{text}");
        }
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
