<p align="center">
  <img src="assets/wordmark.svg" alt="Quiver" width="520">
</p>

<p align="center">
  <b>A fast, embeddable vector database written from scratch in Rust.</b><br>
  HNSW &amp; k-d tree indexes · metadata filtering · pluggable storage · snapshots · REST + gRPC · TUI · Python client
</p>

<p align="center">
  <a href="https://github.com/HilthonTT/vector-db/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/HilthonTT/vector-db/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#license"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue"></a>
  <img alt="MSRV 1.89" src="https://img.shields.io/badge/rust-1.89%2B-orange">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
</p>

---

*Quiver* is Latin for **arrow**. It's also the Arrow constellation, a few stars joined by lines, which is roughly what a vector index is too.

## Features

- **Three index types**, all implemented from scratch with no ANN library: exact `flat` search (parallelised with rayon), approximate **HNSW**, and an exact **k-d tree** for low-dimensional data.
- **Five metrics**: cosine, Euclidean, dot product, Manhattan and Hamming. The kernels are auto-vectorised and use no `unsafe`.
- **Filtered search**: MongoDB-style metadata filters (`$gte`, `$in`, `$or`, nested fields, …) are evaluated *inside* the index traversal, so a selective filter still returns `k` results.
- **Pluggable storage engines**:
  - `file` (default): a checksummed write-ahead log plus atomic checkpoints that include the HNSW graph, so restarts don't rebuild it.
  - `rocksdb`: behind a cargo feature.
  - `memory`.
- **Snapshots**: portable, SHA-256-verified `.tar.gz` backups that can be restored on another machine, under a new name, or into another storage engine. Includes retention (keep the last *k*), download/upload, and periodic background snapshots.
- **Two APIs, one server**: REST/JSON (axum) and gRPC (tonic). Both use **API keys with read-only / read-write roles**.
- **Clients**:
  - a typed Python client (sync and async, numpy-aware)
  - an async Rust client
  - a **terminal UI** (`quiver-tui`) with a dashboard, search, inserts, snapshots, and optional text embedding via any OpenAI-compatible endpoint (Ollama, OpenAI, …)
- **Embeddable**: use the `quiver` crate in-process, like SQLite, with no server at all.

## Quick start

### Run the server

```bash
docker compose up -d       # REST on :6900, gRPC on :6901, data in a volume

# or from source
cargo run --release -p quiver-server -- --data-dir ./quiver-data
```

### Use it from Python

```bash
pip install ./clients/python        # PyPI release: pip install quiver
```

```python
from quiver import Client, HNSW

client = Client("http://localhost:6900")  # api_key="..." when auth is on
books = client.get_or_create_collection("books", dimension=3, metric="cosine", index=HNSW(m=16))

books.upsert([
    ("dune",       [0.9, 0.1, 0.0], {"genre": "sci-fi",  "year": 1965}),
    ("foundation", [0.8, 0.2, 0.1], {"genre": "sci-fi",  "year": 1951}),
    ("hobbit",     [0.1, 0.9, 0.2], {"genre": "fantasy", "year": 1937}),
])
new_id = books.insert([0.0, 0.2, 0.9], {"genre": "romance"})  # server-generated UUID

for hit in books.search([1.0, 0.0, 0.0], k=2, filter={"year": {"$lt": 1960}}):
    print(hit.id, round(hit.distance, 4), hit.metadata)

snap = books.create_snapshot()                       # backup
client.download_snapshot(snap.id, "books.tar.gz")    # ...and take it with you
```

The full client API, including async, is in the [Python client README](clients/python/README.md).

### Use the terminal UI

```bash
cargo run --release -p quiver-tui -- --url http://localhost:6900
# optional: search and insert by *text* through a local embedding model
QUIVER_EMBEDDING_URL=http://localhost:11434/v1/embeddings QUIVER_EMBEDDING_MODEL=nomic-embed-text quiver-tui
```

The UI has a collections dashboard, point browsing, insert, search and filter forms, and snapshot management. Press `?` for key bindings.

### Use curl or gRPC

```bash
curl -X POST localhost:6900/collections -H 'content-type: application/json' \
  -d '{"name": "places", "dimension": 2, "metric": "euclidean", "index": {"type": "kd_tree"}}'

curl -X POST localhost:6900/collections/places/points -H 'content-type: application/json' \
  -d '{"points": [{"vector": [4.35, 50.85], "metadata": {"city": "Brussels"}}]}'

curl -X POST localhost:6900/collections/places/search -H 'content-type: application/json' \
  -d '{"vector": [5.5, 50.5], "k": 3}'

grpcurl -plaintext -import-path crates/grpc/proto -proto quiver.proto \
  -d '{"collection": "places", "vector": [5.5, 50.5], "k": 3}' localhost:6901 quiver.v1.Quiver/Search
```

All REST and gRPC endpoints are listed in the [API reference](docs/API.md).

### Embed it in Rust

```toml
[dependencies]
quiver = { git = "https://github.com/HilthonTT/vector-db" }   # features = ["rocksdb"] for RocksDB
```

```rust
use quiver_db::{CollectionConfig, Database, Filter, Metric, Point, SearchOptions};
use serde_json::json;

let db = Database::open("./data")?;
let books = db.get_or_create_collection("books", CollectionConfig::new(3).metric(Metric::Cosine))?;

books.upsert([
    Point::new("dune", vec![0.9, 0.1, 0.0]).with_metadata(json!({"year": 1965})),
    Point::new("hobbit", vec![0.1, 0.9, 0.2]).with_metadata(json!({"year": 1937})),
])?;

let filter = Filter::parse(&json!({"year": {"$gt": 1950}}))?;
let hits = books.search_with(&[1.0, 0.0, 0.0], 5, &SearchOptions::default().filter(filter))?;

let backup = books.create_snapshot()?; // stored in ./data/snapshots
```

`cargo run --example quickstart` runs a complete example.

## Performance

Measured with `cargo run --release --example benchmark -- <points> <dim> 1000`: clustered vectors, 1,000 queries, k = 10, single-threaded queries, on an Intel Core Ultra 7 255HX.

**100k × 128-d**

| Index | Queries/s | Recall@10 |
|---|---:|---:|
| flat (exact) | 654 | 1.000 |
| HNSW `ef=32` | 8,669 | 0.851 |
| HNSW `ef=64` (default) | 5,191 | 0.959 |
| HNSW `ef=128` | 3,278 | 0.993 |

**100k × 8-d**

| Index | Queries/s | Recall@10 |
|---|---:|---:|
| flat (exact) | 2,288 | 1.000 |
| k-d tree (exact) | 34,312 | 1.000 |
| HNSW `ef=64` | 38,140 | 0.999 |

Pick **HNSW** for high-dimensional embeddings and **k-d tree** for exact answers on low-dimensional data (geo, tabular features). **Flat** is fine for small collections. To trade recall for speed, set `ef` per query or `ef_search` per collection.

## Configuration

`quiver` reads command-line flags, then environment variables, then a `.env` file (see [`.env.example`](.env.example)).

| Flag | Env | Default | |
|---|---|---|---|
| `--host` | `QUIVER_HOST` | `127.0.0.1` | Bind address for both APIs (`0.0.0.0` in Docker) |
| `--http-port` | `QUIVER_HTTP_PORT` | `6900` | REST |
| `--grpc-port` | `QUIVER_GRPC_PORT` | `6901` | gRPC |
| `--disable-http` / `--disable-grpc` | `QUIVER_DISABLE_HTTP` / `_GRPC` | off | Run only one API |
| `--data-dir` | `QUIVER_DATA_DIR` | `./quiver-data` | |
| `--storage` | `QUIVER_STORAGE` | `file` | `file`, `rocksdb` (build with `--features rocksdb`) or `memory` |
| `--in-memory` | `QUIVER_IN_MEMORY` | off | Touch no disk at all |
| `--fsync` | `QUIVER_FSYNC` | off | fsync every write |
| `--keys-file` | `QUIVER_KEYS_FILE` | none | JSON list of `{name, role, key}`; see [`keys.example.json`](keys.example.json) |
| `--api-key` / `--readonly-api-key` | `QUIVER_API_KEY` / `QUIVER_READONLY_API_KEY` | none | Single keys, combinable with the file |
| `--snapshot-dir` | `QUIVER_SNAPSHOT_DIR` | `<data-dir>/snapshots` | |
| `--snapshot-interval` | `QUIVER_SNAPSHOT_INTERVAL` | `0` (off) | Seconds between automatic snapshots of changed collections |
| `--snapshot-keep` | `QUIVER_SNAPSHOT_KEEP` | `5` | Snapshots kept per collection (`0` = all) |
| `--max-body-mb` | `QUIVER_MAX_BODY_MB` | `64` | |
| `--cors` | `QUIVER_CORS` | off | Allow any origin |
| | `RUST_LOG` | `info` | e.g. `quiver=debug,tower_http=debug` |

## Architecture

The workspace is split into crates layered from definitions up to binaries:

| Crate | |
|---|---|
| [`quiver-types`](crates/types) | Points, metrics, configs, filters, errors, API-key roles. No I/O |
| [`quiver-storage`](crates/storage) | Columnar vector store and the `StorageEngine` trait (memory / file / RocksDB) |
| [`quiver-index`](crates/index) | Flat, HNSW and k-d tree indexes |
| [`quiver-snapshot`](crates/snapshot) | Backup archives, registry, retention |
| [`quiver`](crates/db) | **The embeddable database API**: `Database`, `Collection`, snapshot scheduler |
| [`quiver-http`](crates/http) | REST API (axum) |
| [`quiver-grpc`](crates/grpc) | gRPC API (tonic; proto compiled without `protoc`) |
| [`quiver-server`](crates/server) | The `quiver` binary |
| [`quiver-client`](crates/client) | Async Rust client |
| [`quiver-tui`](crates/tui) | The `quiver-tui` terminal UI |

The [Architecture](docs/ARCHITECTURE.md) doc covers how the HNSW and k-d tree work, filtered traversal, storage engines, the recovery protocol and the snapshot format.

## Development

```bash
make check          # fmt + clippy + Rust tests + Python lint/type-check/tests (what CI runs)
make test-rocksdb   # also run the storage tests against RocksDB (needs clang)
make bench          # recall/throughput benchmark
make run            # start a local server;  make tui  to open the terminal UI
```

The Python tooling uses [uv](https://docs.astral.sh/uv/). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

These projects shaped Quiver's design:

- [VortexDB](https://github.com/sdslabs/VortexDB): the crate layering, pluggable storage engines including RocksDB, flat/k-d tree/HNSW indexes, snapshots with retention, REST + gRPC servers, role-based API keys, and the TUI
- [nano-quiver-rs](https://github.com/VectorInstitute/nano-quiver-rs): a small embeddable API that's easy to hack on
- [vecdb](https://github.com/zaydmulani09/vecdb): "SQLite for vectors", with WAL-based crash recovery and payload filtering
- Malkov & Yashunin, [*Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*](https://arxiv.org/abs/1603.09320)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
