# quiver — Python client

Typed sync and async Python client for [Quiver](https://github.com/HilthonTT/vector-db), a vector database written in Rust.

```bash
pip install quiver            # or: uv add quiver
pip install "quiver[numpy]"   # optional: numpy arrays are accepted anywhere a vector is
```

Requires Python 3.10+ and a running Quiver server (`docker compose up` or `cargo run -p quiver-server`). The client uses the REST API on port 6900.

## Quick start

```python
from quiver import Client, HNSW

client = Client("http://localhost:6900")  # api_key="..." if the server requires one

movies = client.get_or_create_collection(
    "movies",
    dimension=4,
    metric="cosine",  # "cosine" | "euclidean" | "dot_product" | "manhattan" | "hamming"
    index=HNSW(m=16),  # or Flat() / KDTree() for exact search
)

movies.upsert(
    [
        ("alien", [0.9, 0.1, 0.0, 0.2], {"genre": "sci-fi", "year": 1979}),
        ("arrival", [0.8, 0.3, 0.1, 0.1], {"genre": "sci-fi", "year": 2016}),
        ("amelie", [0.0, 0.2, 0.9, 0.4], {"genre": "romance", "year": 2001}),
    ]
)

for hit in movies.search([1.0, 0.0, 0.0, 0.1], k=2, filter={"year": {"$gte": 2000}}):
    print(hit.id, round(hit.distance, 3), hit.metadata)
```

### Points

`upsert` accepts any mix of:

```python
Point("id", vector, {"meta": 1})  # dataclass
{"id": "id", "vector": vector, "metadata": {...}}  # dict (id optional: generated)
("id", vector) / ("id", vector, {"meta": 1})  # tuple
```

Vectors can be lists, tuples or numpy arrays. Large iterables (including generators) are streamed in batches of `batch_size` (default 1000).

### Collection API

| Method | Description |
|---|---|
| `upsert(points, batch_size=1000)` | Insert or replace points; returns count written |
| `insert(vector, metadata=None, id=None)` | Insert one point; returns its (generated) id |
| `search(vector, k=10, filter=None, ef=None, with_vectors=False, with_metadata=True)` | k-nearest-neighbour search |
| `search_batch(vectors, k=10, ...)` | Many searches in one request |
| `get(id)` / `get_many(ids)` | Fetch points (`get` returns `None` if missing) |
| `scroll(page_size=500)` | Iterate over every point |
| `delete(ids)` | Delete points; returns how many existed |
| `info()` / `len(col)` | Configuration and point count |
| `flush()` / `compact()` | Force a checkpoint / reclaim deleted space |
| `create_snapshot()` / `snapshots()` | Back up this collection / list its backups |

Distances are always **lower = closer**: `1 - cos` for cosine, L2 for Euclidean, `-(a·b)` for dot product, L1 for Manhattan, and the count of differing components for Hamming.

### Snapshots (backups)

```python
snap = movies.create_snapshot()  # SnapshotInfo(id=..., points=..., created_at=...)
client.list_snapshots(collection="movies")
client.download_snapshot(snap.id, "movies.tar.gz")  # portable, checksummed archive
other = Client("http://other-host:6900")
other.upload_snapshot("movies.tar.gz")
other.restore_snapshot(snap.id, collection="movies_copy", replace=False)
client.delete_snapshot(snap.id)
```

### Filters

MongoDB-style metadata filters:

```python
{"genre": "sci-fi"}  # equality (matches any element of an array)
{"year": {"$gte": 1990, "$lt": 2000}}  # $gt $gte $lt $lte $eq $ne
{"tags": {"$in": ["classic", "cult"]}}  # $in $nin
{"award": {"$exists": True}}
{"$or": [{"genre": "drama"}, {"rating": {"$gt": 8}}]}  # $and $or $not
{"author.country": "BE"}  # nested fields
```

### Async

```python
import asyncio
from quiver import AsyncClient


async def main():
    async with AsyncClient("http://localhost:6900") as client:
        col = await client.get_or_create_collection("docs", dimension=3)
        await col.upsert([("a", [1, 0, 0])])
        print(await col.search([1, 0, 0], k=1))


asyncio.run(main())
```

### Errors

All exceptions derive from `QuiverError` and carry `.status_code` and `.code`:
`InvalidRequestError` (400), `AuthenticationError` (401), `PermissionDeniedError` (403, a
read-only key used for a write), `NotFoundError` (404), `ConflictError` (409), `ServerError` (5xx)
and `QuiverConnectionError` (network).

## Development

```bash
uv sync                      # create .venv with dev dependencies
uv run pytest                # unit tests (no server needed)
uv run ruff check . && uv run ruff format --check . && uv run mypy src

# integration tests against a live server
cargo run -p quiver-server -- --in-memory &
QUIVER_URL=http://localhost:6900 uv run pytest
```
