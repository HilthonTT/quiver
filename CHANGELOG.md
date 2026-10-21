# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Metadata updates** without resending vectors: merge (RFC 7386 JSON Merge Patch) or replace, by ids or by filter.
  REST `POST /collections/{name}/points/metadata`, gRPC `UpdateMetadata`, `Collection::update_metadata`, Rust and Python `update_metadata`.
- **Count and delete by filter**: `POST /collections/{name}/points/count`, `points/delete` with `{"filter": ...}`
  (an empty filter is refused), gRPC `Count` and `DeletePoints.filter_json`, and `count`/`delete_where` in every client.
- **Search by point id** (`{"id": ...}` instead of a vector; the point itself is left out) and **recommendations** from
  positive and negative example ids (`POST /collections/{name}/recommend`, gRPC `Recommend`). Missing ids give `point_not_found`.
- **`max_distance`** on searches and recommendations drops hits farther than a threshold.
- **`quiver --healthcheck`** probes a running server and exits 0 or 1.
- gRPC `Info`, `GetSnapshot`, `DownloadSnapshot` (server stream) and `UploadSnapshot` (client stream), so gRPC covers the REST API.
- Rust client: `get_points`, `search_batch`, `flush`, `compact`, `get_snapshot`, `download_snapshot` and `upload_snapshot`.

### Changed

- **Breaking:** listing points pages by id cursor instead of offset, so changes between pages never skip or repeat points.
  Points come back in ascending id order; REST `offset`/`next_offset` are now `cursor`/`next_cursor` (an `offset` parameter is rejected),
  likewise in gRPC `Scroll`, `Collection::scroll` and the Rust client. In Python, `scroll(cursor=, limit=)` returns one `ScrollPage`
  and `iter_points()` iterates over everything.
- **Breaking:** Python `AsyncClient.collection()` is now a plain method (no `await`), like `Client.collection()`.
- Malformed or wrongly shaped JSON bodies (including unknown fields) now return 400 `invalid_body` instead of 422.
- Uploaded snapshots are verified by streaming (manifest first, size and SHA-256 checked) instead of being decoded in memory,
  so an upload that decompresses to far more than its size cannot exhaust memory.
- Stale `*.partial` files in the snapshot directory are removed at startup once they are an hour old.

### Fixed

- The Docker `HEALTHCHECK` always failed: it used bash's `/dev/tcp` under dash. It now runs `quiver --healthcheck`.

## [0.2.0] - 2026-10-16

### Changed

- **Split into a layered workspace of ten crates**, modelled on VortexDB:
  `quiver-types`, `-storage`, `-index`, `-snapshot`, `quiver`, `-http`, `-grpc`, `-server`, `-client` and `-tui`.
- The server binary now serves REST on `--http-port` (default 6900; `--port` still works) and gRPC on `--grpc-port` (6901).
- File engine checkpoints are now `state.bin` (format v2, with an opaque index blob). Data written by 0.1.0 must be re-imported.
- Upsert responses include the written `ids`.

### Added

- **k-d tree index** (`{"type": "kd_tree"}`): exact search, fast in low dimensions.
- **Manhattan and Hamming metrics**.
- **Pluggable storage engines** behind a `StorageEngine` trait: `file` (default), `memory`, and `rocksdb` (cargo feature `rocksdb`). The engine is recorded per collection.
- **Snapshots**: SHA-256-verified `.tar.gz` backups with retention, download/upload, restore under a new name or over an existing collection, and a periodic `SnapshotScheduler` (`--snapshot-interval`, `--snapshot-keep`).
- **gRPC API** (`quiver.v1.Quiver`, tonic), with the proto compiled by protox (no `protoc` needed).
- **Role-based API keys** (`readonly` / `readwrite`) from `--keys-file`, `--api-key` and `--readonly-api-key`, on both APIs.
- Optional point ids: a UUID v4 is generated when missing (`Collection::insert`, REST, gRPC, Python `insert()`).
- `GET /` and `GET /info` endpoints.
- `.env` file support. Precedence is CLI > env > `.env` > defaults.
- **`quiver-tui`** terminal UI with a dashboard, collection browsing, insert/search/filter forms, snapshot management and optional OpenAI-compatible text embeddings.
- **`quiver-client`**, an async Rust client.
- Python client: `KDTree`, the new metrics, `insert()`, `info()`, and the snapshot API (`create_snapshot`, `list_snapshots`, `restore_snapshot`, `download_snapshot`, `upload_snapshot`, `delete_snapshot`).

## [0.1.0] - 2026-10-16

### Added

- `quiver` library: collections with flat and HNSW indexes; cosine, Euclidean and dot-product metrics.
- MongoDB-style metadata filters, evaluated during HNSW traversal.
- Durability: CRC-checked write-ahead log, atomic snapshots (including the HNSW graph), torn-write recovery, automatic checkpointing and compaction, single-process file lock.
- `quiver` HTTP server: collection, point, search and batch-search endpoints; API-key auth; graceful shutdown with flush; Docker image.
- `quiver` Python client: sync and async, typed, numpy support, automatic upsert batching.

[Unreleased]: https://github.com/HilthonTT/quiver/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HilthonTT/quiver/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HilthonTT/quiver/releases/tag/v0.1.0
