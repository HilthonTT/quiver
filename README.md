<p align="center">
  <img src="assets/wordmark.svg" alt="Quiver" width="480">
</p>

<p align="center">
  <a href="https://github.com/HilthonTT/quiver/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/HilthonTT/quiver/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="MSRV 1.89" src="https://img.shields.io/badge/rust-1.89%2B-orange">
</p>

A fast, embeddable vector database written from scratch in Rust.

Documentation: **[hilthontt.github.io/quiver](https://hilthontt.github.io/quiver/)**

- **Indexes:** HNSW, exact flat search and a k-d tree, with cosine, Euclidean, dot, Manhattan and Hamming metrics.
- **Filtered search:** MongoDB-style metadata filters (`$gte`, `$in`, `$or`, …) applied during the index traversal.
- **Data operations:** update metadata in place (merge or replace), count and delete by filter, search by an existing point, recommend from positive and negative examples, and distance thresholds.
- **Storage:** a checksummed WAL with checkpoints by default, RocksDB behind a feature flag, or in-memory.
- **Snapshots:** SHA-256-verified `.tar.gz` backups with retention and periodic scheduling.
- **Access:** REST and gRPC with read-only and read-write API keys, plus a Python client, a Rust client, a terminal UI, or the `quiver-db` crate embedded in-process.

## Quick start

```bash
docker compose up -d                       # REST on :6900, gRPC on :6901
# or: cargo run --release -p quiver-server
```

```python
# pip install ./clients/python
from quiver import Client

client = Client("http://localhost:6900")
books = client.get_or_create_collection("books", dimension=3)
books.upsert([
    ("dune", [0.9, 0.1, 0.0], {"year": 1965}),
    ("hobbit", [0.1, 0.9, 0.2], {"year": 1937}),
])
for hit in books.search([1.0, 0.0, 0.0], k=1, filter={"year": {"$gt": 1950}}):
    print(hit.id, hit.distance)
```

Or with curl:

```bash
curl -X POST localhost:6900/collections -H 'content-type: application/json' \
  -d '{"name": "places", "dimension": 2, "metric": "euclidean"}'
curl -X POST localhost:6900/collections/places/points -H 'content-type: application/json' \
  -d '{"points": [{"id": "brussels", "vector": [4.35, 50.85]}]}'
curl -X POST localhost:6900/collections/places/search -H 'content-type: application/json' \
  -d '{"vector": [5.5, 50.5], "k": 1}'
```

Run `quiver --help` for every option; each one can also be set with a `QUIVER_*` environment variable (see [`.env.example`](.env.example)). Set an API key before exposing the server; see [SECURITY.md](SECURITY.md).

## More

- [Python client](clients/python/README.md)
- Terminal UI: `cargo run --release -p quiver-tui`
- Embedding in Rust: `cargo run -p quiver-db --example quickstart`
- gRPC API: [`crates/grpc/proto/quiver.proto`](crates/grpc/proto/quiver.proto)
- Development: `make check` runs what CI runs; see [CONTRIBUTING.md](CONTRIBUTING.md)

## License

[MIT](LICENSE)
