# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Per-collection API keys**: a key in the keys file can carry `"collections": ["books", "tenant-a-*"]` (exact names, or
  prefixes ending in `*`) to limit it to those collections, over REST and gRPC alike. Aliases are judged by the collection
  they point to; new names (create, rename, restore under a name, aliases) must be in scope; listings, `/info` and snapshot
  listings show only what is in scope; snapshots are judged by the collection they were taken of; `/metrics` refuses scoped
  keys. Out-of-scope requests answer `403 forbidden` / `PERMISSION_DENIED`, also for names that do not exist.
  `ApiKey::new`/`with_collections`/`access`, `Access` and `Scope` in `quiver_db::auth`; `Database::authorize` and friends.
- **Vector datatypes**: `"datatype": "float16"` (half the memory; components rounded to half precision) or `"uint8"`
  (a quarter; whole numbers 0-255, stored exactly; not with cosine), per vector, at creation. On-disk vector files use the
  datatype's width. Queries stay 32-bit floats. Works with binary quantization, not scalar. gRPC `Datatype`; Python
  `datatype=` and `VectorParams(datatype=...)`; `Datatype` in the embedded crate.
- **SIMD distance kernels**: AVX2 + FMA (and F16C for `float16`) on x86-64, chosen at run time, with the portable kernels as
  the fallback. About 25-35% more HNSW queries per second and 20% faster builds at 768 dimensions.

### Changed

- The keys file refuses unknown fields, so that a misspelt `collections` cannot grant a key every collection.

- **Grouped search**: `POST /collections/{name}/search/groups` (gRPC `SearchGroups`, `Collection::search_groups`) returns the
  best `groups` values of a metadata field `group_by`, each with up to `group_size` hits, for a vector, an id or a sparse
  query. Points with an array there join each element's group.
- **Discovery**: `POST /collections/{name}/discover` (gRPC `Discover`, `Collection::discover`) ranks a target's neighbours by
  how many context pairs (`positive`, `negative`; ids or vectors) they are on the right side of, or without a target scores
  every point by its context penalty.
- **Facets**: `POST /collections/{name}/points/facet` (gRPC `Facet`, `Collection::facet`) counts the values of a field,
  optionally over a filter.
- **Filtered and ordered scrolls**: `POST /collections/{name}/points/scroll` (gRPC `Scroll.filter_json`/`order_by`,
  `Collection::scroll_with`) takes a `filter` and `order_by: {"field", "direction"}`; ordered pages use an opaque cursor.
- **`offset`** on searches, batch entries, recommendations, discovery and hybrid queries (`k + offset` at most 10,000).
- **Score fusion and rescoring** in `POST /query`: `fusion` `{"type": "relative_score"}` (min-max) and `{"type": "dbsf"}`
  (distribution-based), and `rescore: {"vector", "using"?}` to re-rank the prefetched candidates by exact distance. gRPC
  `Fusion` and `Rescore`.
- **Filter operators**: `$text` (all words, case-insensitive), `$prefix`, `$regex`, `$all`, `$size`, `$elemMatch`,
  `$geoRadius` and `$geoBox`. Field indexes answer `$text` (they now keep each string's words), `$prefix` and `$all`.
- The Rust and Python clients cover all of the above. Point ids `scroll` and `facet` are read through `points/lookup`, as
  the new routes shadow `GET points/{id}`.

- **Binary quantization** (`"quantization": "binary"`, per vector, at creation or through `PATCH`): searches score one bit per
  component plus a 4-byte scale per vector, fetch 20×k candidates and re-rank them by the exact vectors, which are kept, so
  results carry exact distances and vectors. With `on_disk`, only the bits stay in memory, about 1/32 of the vectors. HNSW
  graphs are built from the exact vectors. HNSW and flat indexes only, not with `hamming`. gRPC `QUANTIZATION_BINARY`.
- **Expiring points**: `"expiry": {"field": "expires_at", "default_ttl": 3600}` at creation or through `PATCH` (`null` turns
  it off). A point expires once the number in its metadata field `field` (Unix seconds) has passed: it is hidden from every
  read at once, and a background sweep deletes it (`--expiry-interval`, `QUIVER_EXPIRY_INTERVAL`, default 60 s; `0` disables
  it). `default_ttl` gives points written without the field an expiry time. The field is indexed while expiry is on, and
  non-numeric values are refused. `Collection::purge_expired` and `ExpirySweeper` in the embedded crate; `Expiry` in the
  Python client; gRPC `ExpiryConfig`.
- **TLS** for both APIs: `--tls-cert`/`--tls-key` (PEM; `QUIVER_TLS_CERT`/`QUIVER_TLS_KEY`), and `--tls-client-ca` to require
  client certificates (mutual TLS). REST negotiates HTTP/2 or HTTP/1.1 over ALPN; handshakes run concurrently and time out
  after 10 s. `quiver --healthcheck` probes over TLS. The server warns when API keys are configured on a non-loopback
  address without TLS. Clients: Rust `Client::with_tls` with `TlsOptions`, Python `ca_cert=`/`client_cert=`/`client_key=`,
  TUI `--ca-cert`/`--client-cert`.
- **Named vectors**: `vectors` at creation adds dense vectors every point carries, each with its own dimension, metric,
  index, quantization and on-disk setting. Search, search by id, batch search and recommend take `using` to pick one.
  Points and hits carry them in `vectors`.
- **Sparse vectors and hybrid search**: `sparse_vectors` at creation (optionally with `"modifier": "idf"` for BM25-style
  weighting) adds sparse vectors points may carry, searched exactly through an inverted index with `{"sparse": ..., "using": ...}`
  (distances are negated dot-product scores). `POST /collections/{name}/query` (gRPC `Query`) fuses several dense and sparse
  searches with weighted reciprocal rank fusion.
- **On-disk vectors** (`"on_disk": true`, per vector): vectors live in a file next to the collection instead of memory and are
  read with positional reads. Combined with scalar quantization, the codes stay in memory for the search and the exact vectors
  on disk re-rank 4×k candidates, so results carry exact distances. Durable storage engines only.
- **Collection settings can change**: `PATCH /collections/{name}` (gRPC `UpdateCollection`, `Collection::update`) changes the
  index (as a JSON Merge Patch over the current settings), quantization, on-disk storage and sparse modifiers. Changed indexes
  are rebuilt while reads continue; an `ef_search`-only change rebuilds nothing.
- **Renaming collections**: `POST /collections/{name}/rename` (gRPC `RenameCollection`, `Database::rename_collection`).
- **Aliases**: `GET /aliases`, `PUT /aliases/{alias}` to create or atomically re-point one, `DELETE /aliases/{alias}` (gRPC
  `ListAliases`/`SetAlias`/`DeleteAlias`). Aliases work wherever a collection is named except dropping and renaming, share a
  namespace with collections, follow renames, go away with their collection, and persist in `aliases.json`.
- The Rust and Python clients cover all of the above; the TUI lists each collection's other vectors.

- **Metadata updates** without resending vectors: merge (RFC 7386 JSON Merge Patch) or replace, by ids or by filter.
  REST `POST /collections/{name}/points/metadata`, gRPC `UpdateMetadata`, `Collection::update_metadata`, Rust and Python `update_metadata`.
- **Count and delete by filter**: `POST /collections/{name}/points/count`, `points/delete` with `{"filter": ...}`
  (an empty filter is refused), gRPC `Count` and `DeletePoints.filter_json`, and `count`/`delete_where` in every client.
- **Search by point id** (`{"id": ...}` instead of a vector; the point itself is left out) and **recommendations** from
  positive and negative example ids (`POST /collections/{name}/recommend`, gRPC `Recommend`). Missing ids give `point_not_found`.
- **`max_distance`** on searches and recommendations drops hits farther than a threshold.
- **Field indexes** on metadata fields: `POST /collections/{name}/indexes`, `DELETE /collections/{name}/indexes/{field}`,
  `field_indexes` when creating a collection, gRPC `CreateFieldIndex`/`DropFieldIndex`, and `create_field_index`/`drop_field_index`
  in every client. Equality, `$in`, ranges and `$exists: true` on indexed fields only look at matching points; selective
  filtered searches score the matches exactly instead of walking the HNSW graph, which could miss them. Results never change.
  Indexed fields are saved with the collection and carried by snapshots; the indexes are rebuilt on open.
- **Scalar quantization** (`"quantization": "scalar"` at creation): one byte per component plus 8 bytes per vector, about a
  quarter of the memory. Lossy; HNSW and flat indexes only, not with `hamming`. On-disk formats are unchanged.
- **Prometheus metrics** at `GET /metrics` (read role): per-collection gauges and request counts and latencies for REST and gRPC.
- **`quiver --healthcheck`** probes a running server and exits 0 or 1.
- gRPC `Info`, `GetSnapshot`, `DownloadSnapshot` (server stream) and `UploadSnapshot` (client stream), so gRPC covers the REST API.
- Rust client: `get_points`, `search_batch`, `flush`, `compact`, `get_snapshot`, `download_snapshot` and `upload_snapshot`.

### Changed

- Checkpoints are written in state format v3 (columnar, with references to on-disk vector files); v2 checkpoints and
  snapshots still load. Older versions cannot read data written by this one.
- Snapshot data is staged in a temporary file instead of memory while the archive is written.
- `quiver_collection_vector_bytes` reports the memory vectors actually take, across all vectors, excluding those on disk.
- `Collection::config()` returns an owned `CollectionConfig`, since settings can now change.
- `quiver_grpc::serve` takes an optional `ServerTlsConfig`.
- **Breaking:** listing points pages by id cursor instead of offset, so changes between pages never skip or repeat points.
  Points come back in ascending id order; REST `offset`/`next_offset` are now `cursor`/`next_cursor` (an `offset` parameter is rejected),
  likewise in gRPC `Scroll`, `Collection::scroll` and the Rust client. In Python, `scroll(cursor=, limit=)` returns one `ScrollPage`
  and `iter_points()` iterates over everything.
- **Breaking:** Python `AsyncClient.collection()` is now a plain method (no `await`), like `Client.collection()`.
- Malformed or wrongly shaped JSON bodies (including unknown fields) now return 400 `invalid_body` instead of 422.
- Uploaded snapshots are verified by streaming (manifest first, size and SHA-256 checked) instead of being decoded in memory,
  so an upload that decompresses to far more than its size cannot exhaust memory.
- Stale `*.partial` files in the snapshot directory are removed at startup once they are an hour old.
- `docker-compose.yml` publishes the ports on `127.0.0.1` only; set an API key before exposing them.
- `CollectionInfo` gains `field_indexes`; `VectorStore::vector` returns a `Cow<[f32]>`; `write_archive` takes the indexed fields.

### Fixed

- The Docker `HEALTHCHECK` always failed: it used bash's `/dev/tcp` under dash. It now runs `quiver --healthcheck`.
- Restoring a snapshot now validates its points (finite vectors within the magnitude limit, no zero vectors under cosine,
  id lengths, metadata depth) before any existing collection is replaced. A forged upload could otherwise load invalid data.
- The website build job no longer holds Pages and OIDC write permissions while running `npm ci` on pull requests.

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
