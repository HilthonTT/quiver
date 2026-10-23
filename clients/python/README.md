# quiver — Python client

Typed sync and async Python client for [Quiver](https://github.com/HilthonTT/quiver), a vector database written in Rust.

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
| `search(vector, k=10, filter=None, ef=None, max_distance=None, with_vectors=False, with_metadata=True)` | k-nearest-neighbour search |
| `search_by_id(id, k=10, ...)` | Neighbours of an existing point (which is left out) |
| `recommend(positive, negative=(), k=10, ...)` | Points like the positive examples and unlike the negative ones |
| `search_batch(vectors, k=10, ...)` | Many searches in one request |
| `search_sparse(vector, k=10, using=...)` | Sparse (keyword) search; `distance` is the negated score |
| `query(prefetch, k=10, filter=None, fusion="rrf", rrf_k=60, rescore=None)` | Hybrid search: fuse several dense and sparse searches, or re-rank them |
| `search_groups(vector, group_by=..., groups=10, group_size=3)` | Best hits per value of a metadata field (also `id=` or `sparse=`) |
| `discover(target=None, context=(), k=10, ...)` | Points on the positive side of each `(positive, negative)` pair |
| `facet(field, filter=None, limit=10)` | Point counts per value of a field, most common first |
| `update(index=, quantization=, on_disk=, vectors=, sparse_vectors=, expiry=)` | Change settings (may rebuild indexes) |
| `rename(new_name)` | Rename the collection; the handle follows it |
| `get(id)` / `get_many(ids)` | Fetch points (`get` returns `None` if missing) |
| `scroll(cursor=None, limit=100, filter=None, order_by=None)` | One page of points (in id order unless `order_by`), with `next_cursor` |
| `iter_points(page_size=500, filter=None, order_by=None)` | Iterate over every point, following the cursors |
| `delete(ids)` / `delete_where(filter)` | Delete points by id or by filter; returns how many existed |
| `update_metadata(metadata, ids=None, filter=None, mode="merge")` | Change metadata without resending vectors |
| `count(filter=None)` | Number of points, optionally matching a filter |
| `info()` / `len(col)` | Configuration and point count |
| `flush()` / `compact()` | Force a checkpoint / reclaim deleted space |
| `create_snapshot()` / `snapshots()` | Back up this collection / list its backups |

Every search method takes `using="name"` to search a named vector instead of the main one,
and `offset=` to skip that many of the best hits (for paging; `k + offset` is at most 10 000).

Distances are always **lower = closer**: `1 - cos` for cosine, L2 for Euclidean, `-(a·b)` for dot product, L1 for Manhattan, and the count of differing components for Hamming.

### Paging

Points are paged in ascending id order. Each page carries a `next_cursor`
(`None` after the last page); points inserted, updated or deleted between pages
are never skipped or returned twice.

```python
page = movies.scroll(limit=100)
while True:
    for point in page.points:
        print(point.id, point.metadata)
    if page.next_cursor is None:
        break
    page = movies.scroll(page.next_cursor, limit=100)

ids = [p.id for p in movies.iter_points()]  # or let the client follow the cursors
```

`filter=` keeps only matching points, and `order_by="year"` (or
`OrderBy("year", "desc")`) pages by a numeric or string metadata field instead,
then by id; points without it are left out. Cursors are opaque strings either way.

```python
from quiver import OrderBy

newest = movies.scroll(limit=20, order_by=OrderBy("year", "desc"), filter={"genre": "sci-fi"})
```

### Data operations

```python
movies.count({"genre": "sci-fi"})  # 2

# Merge (JSON Merge Patch: nested objects merge, None removes a key)...
movies.update_metadata({"seen": True, "draft": None}, ids=["alien", "arrival"])
# ...or replace the whole value, here for every match of a filter.
movies.update_metadata({"genre": "classic"}, filter={"year": {"$lt": 1990}}, mode="replace")

movies.delete_where({"genre": "romance"})  # an empty filter is refused

movies.search_by_id("alien", k=5)  # neighbours of a stored point, without itself
movies.recommend(["alien", "arrival"], negative=["amelie"], k=5)
movies.search([1.0, 0.0, 0.0, 0.1], k=10, max_distance=0.3)  # only close hits
```

`max_distance` drops hits farther than the threshold; results are still capped
at `k`, and with HNSW the threshold applies to the approximate top `k`.

### Groups, facets and discovery

```python
for group in chunks.search_groups(embedding, group_by="doc_id", groups=5, group_size=2):
    print(group.key, [hit.id for hit in group.hits])  # the 2 best chunks of each of 5 documents

movies.facet("genre", filter={"year": {"$gte": 2000}})  # [FacetValue(value="drama", count=41), ...]

movies.discover("alien", context=[("arrival", "amelie")], k=5)  # ids or vectors
movies.discover(context=[("arrival", "amelie"), ([0.1, 0.9, 0.0, 0.0], "titanic")])
```

With a `target`, discovery ranks hits first by how many context pairs they
satisfy (closer to the positive than to the negative), then by distance to the
target. Without one, it scores every point by how far it is from satisfying all
pairs (`distance` 0 means it satisfies them all).

### Named vectors, sparse vectors and hybrid search

```python
from quiver import Flat, Point, Prefetch, SparseVectorParams, VectorParams

papers = client.create_collection(
    "papers",
    dimension=4,
    vectors={"title": VectorParams(2, metric="euclidean", index=Flat())},
    sparse_vectors={"text": SparseVectorParams("idf")},  # BM25-style weighting
)
papers.upsert(
    [
        Point(
            "p1",
            [0.1, 0.2, 0.3, 0.4],
            vectors={"title": [0.5, 0.5]},
            sparse_vectors={"text": {17: 1.0, 4: 0.5}},
        )
    ]
)
papers.search([0.5, 0.5], k=5, using="title")
papers.search_sparse({17: 1.0}, k=5, using="text")
papers.query(
    [Prefetch(vector=[0.1, 0.2, 0.3, 0.4]), Prefetch(sparse={17: 1.0}, using="text", weight=2.0)],
    k=10,
)
```

Every point must carry every named vector; sparse vectors (a `SparseVector(indices, values)`
or an `{index: value}` dict) are optional. `query` returns `ScoredResult`s, highest `score` first.

`fusion="relative_score"` (min-max) and `fusion="dbsf"` (mean ± 3σ) fuse normalised
scores instead of ranks. `rescore=Rescore(vector, using="name")` re-ranks every
prefetched point by its exact distance to `vector` (its score is the negated distance):

```python
from quiver import Rescore

papers.query([Prefetch(vector=small, using="coarse", limit=100)], k=10, rescore=Rescore(full))
```

### Binary quantization and expiring points

```python
from quiver import Expiry

sessions = client.create_collection(
    "sessions",
    dimension=1536,
    quantization="binary",  # 1 bit per component to search; exact vectors re-rank
    on_disk=True,  # ...and live on disk, so memory holds only the bits
    expiry=Expiry("expires_at", default_ttl=3600),
)
sessions.upsert([("s1", embedding)])  # expires in an hour
sessions.upsert([("s2", embedding, {"expires_at": 1_900_000_000})])  # at this Unix time
sessions.update(expiry=False)  # stop expiring points
```

Expired points vanish from every read immediately and are deleted by a periodic sweep.

### Settings, renames and aliases

```python
papers.update(index={"ef_search": 128}, on_disk=True)  # dict: merged into the index settings
papers.rename("papers-v1")
client.set_alias("papers", "papers-v1")  # create, or atomically re-point
client.list_aliases()  # {"papers": "papers-v1"}
client.delete_alias("papers")
```

### TLS

```python
Client("https://quiver.internal:6900", api_key="...", ca_cert="ca.pem")
Client("https://quiver.internal:6900", client_cert="client.pem", client_key="client-key.pem")
```

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
{"title": {"$text": "left hand darkness"}}  # every word, any order, case-insensitive
{"sku": {"$prefix": "BK-"}}  # also {"$regex": "^BK-\\d+$"}
{"tags": {"$all": ["a", "b"]}}  # also {"$size": 2}
{"reviews": {"$elemMatch": {"stars": {"$gte": 4}, "verified": True}}}
{"loc": {"$geoRadius": {"center": {"lat": 50.85, "lon": 4.35}, "meters": 5000}}}
{"loc": {"$geoBox": {"top_left": {"lat": 51, "lon": 4}, "bottom_right": {"lat": 50, "lon": 5}}}}
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
        docs = client.collection("docs")  # plain method: no await
        print([p.id async for p in docs.iter_points()])


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
