"""End-to-end tests against a running server.

Start one with ``cargo run -p quiver-server -- --in-memory`` and run
``QUIVER_URL=http://localhost:6900 pytest``. Skipped when QUIVER_URL is unset.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from quiver import (
    HNSW,
    AsyncClient,
    Client,
    ConflictError,
    Flat,
    InvalidRequestError,
    KDTree,
    NotFoundError,
    Point,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

URL = os.environ.get("QUIVER_URL")
pytestmark = pytest.mark.skipif(not URL, reason="QUIVER_URL not set")


@pytest.fixture
def client() -> Iterator[Client]:
    assert URL
    with Client(URL, api_key=os.environ.get("QUIVER_API_KEY")) as c:
        yield c


@pytest.fixture
def name(client: Client) -> Iterator[str]:
    n = f"test_{uuid.uuid4().hex[:12]}"
    yield n
    for info in client.list_collections():
        if info.name == n:
            client.delete_collection(n)


def test_health(client: Client) -> None:
    assert client.health()["status"] == "ok"


@pytest.mark.parametrize("index", [HNSW(m=8), Flat(), KDTree()])
def test_round_trip(client: Client, name: str, index: HNSW | Flat | KDTree) -> None:
    col = client.create_collection(name, 8, metric="euclidean", index=index)
    rng = np.random.default_rng(0)
    vectors = rng.random((500, 8), dtype=np.float32)
    col.upsert((f"p{i}", v, {"bucket": i % 5, "even": i % 2 == 0}) for i, v in enumerate(vectors))
    assert len(col) == 500

    hits = col.search(vectors[42], k=5)
    assert hits[0].id == "p42"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-5)

    filtered = col.search(vectors[42], k=10, filter={"bucket": {"$in": [1, 3]}, "even": False})
    assert filtered
    assert all(h.metadata and h.metadata["bucket"] in (1, 3) for h in filtered)

    point = col.get("p7")
    assert point is not None
    assert point.vector == pytest.approx(vectors[7].tolist(), rel=1e-6)

    assert col.delete([f"p{i}" for i in range(100)]) == 100
    assert len(col) == 400
    assert sum(1 for _ in col.iter_points(page_size=64)) == 400
    col.flush()


def test_scroll_pages_by_cursor_while_points_change(client: Client, name: str) -> None:
    col = client.create_collection(name, 2, index=Flat())
    col.upsert((f"p{i:02}", [1.0, float(i)]) for i in range(30))
    seen: list[str] = []
    cursor = None
    while True:
        page = col.scroll(cursor, limit=7)
        seen += [p.id for p in page.points]
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
        # Delete one point behind the cursor and one ahead; update another.
        last = int(cursor[1:])
        col.delete([f"p{last - 2:02}", f"p{min(last + 3, 29):02}"])
        col.upsert([(f"p{min(last + 1, 29):02}", [2.0, 1.0])])
    assert seen == sorted(set(seen)), "no point repeated, in id order"
    assert set(seen) >= {p.id for p in col.iter_points()}, "no surviving point skipped"
    assert col.scroll(seen[-1]).points == []


def test_data_operations(client: Client, name: str) -> None:
    col = client.create_collection(name, 2, metric="euclidean", index=Flat())
    col.upsert((f"x{i}", [float(i), 0.0], {"i": i, "even": i % 2 == 0}) for i in range(10))

    assert col.count() == 10
    assert col.count({"even": True}) == 5

    assert col.update_metadata({"tag": "low", "even": None}, filter={"i": {"$lt": 3}}) == 3
    x1 = col.get("x1")
    assert x1 is not None and x1.metadata == {"i": 1, "tag": "low"}
    assert col.update_metadata(None, ids=["x1", "missing"], mode="replace") == 1
    x1 = col.get("x1")
    assert x1 is not None and x1.metadata is None and x1.vector == [1.0, 0.0]

    hits = col.search_by_id("x4", k=2)
    assert sorted(h.id for h in hits) == ["x3", "x5"]
    assert [h.id for h in col.search([0, 0], k=10, max_distance=1.5)] == ["x0", "x1"]
    assert col.recommend("x4", ["x2"], k=1)[0].id == "x6"
    hits = col.recommend(["x4", "x5"], k=8)
    assert len(hits) == 8 and not {"x4", "x5"} & {h.id for h in hits}
    with pytest.raises(NotFoundError) as info:
        col.search_by_id("nope")
    assert info.value.code == "point_not_found"

    with pytest.raises(InvalidRequestError):
        col.delete_where({})
    assert col.delete_where({"even": True}) == 3  # x0 and x2 lost "even" above
    assert col.count() == 7


def test_async_data_operations(name: str) -> None:
    async def main() -> None:
        assert URL
        async with AsyncClient(URL, api_key=os.environ.get("QUIVER_API_KEY")) as client:
            col = await client.create_collection(name, 2, metric="euclidean", index=Flat())
            await col.upsert([("a", [0, 0], {"k": 1}), ("b", [1, 0]), ("c", [5, 5])])
            assert await col.count({"k": 1}) == 1
            assert await col.update_metadata({"k": 2}, ids="b") == 1
            assert await col.count({"k": {"$exists": True}}) == 2
            assert [h.id for h in await col.search_by_id("a", k=5)] == ["b", "c"]
            assert [h.id for h in await col.recommend(["a"], k=1)] == ["b"]
            assert await col.delete_where({"k": 2}) == 1
            await client.delete_collection(name)

    asyncio.run(main())


def test_errors(client: Client, name: str) -> None:
    client.create_collection(name, 3)
    with pytest.raises(ConflictError):
        client.create_collection(name, 3)
    with pytest.raises(InvalidRequestError) as info:
        client.collection(name).search([1.0, 2.0], k=1)
    assert info.value.code == "dimension_mismatch"


def test_async(name: str) -> None:
    async def main() -> None:
        assert URL
        async with AsyncClient(URL, api_key=os.environ.get("QUIVER_API_KEY")) as client:
            col = await client.create_collection(name, 2, metric="dot_product")
            await col.upsert([("a", [1, 0]), ("b", [0, 1])])
            hits = await col.search([1, 0.1], k=1)
            assert hits[0].id == "a"
            assert [p.id async for p in col.iter_points()] == ["a", "b"]
            await client.delete_collection(name)

    asyncio.run(main())


def test_snapshots(client: Client, name: str, tmp_path: Path) -> None:
    if not client.info()["snapshots_enabled"]:
        pytest.skip("server has no snapshot directory")
    col = client.create_collection(name, 2, metric="manhattan")
    generated = col.insert([1.0, 2.0], {"k": "v"})
    col.upsert([("b", [3.0, 4.0])])
    snap = col.create_snapshot()
    assert snap.points == 2
    assert snap.id in [s.id for s in col.snapshots()]

    archive = client.download_snapshot(snap.id, tmp_path / "s.tar.gz")
    client.delete_snapshot(snap.id)
    assert client.upload_snapshot(archive).id == snap.id

    copy = client.restore_snapshot(snap.id, collection=f"{name}_copy")
    try:
        point = copy.get(generated)
        assert point is not None and point.metadata == {"k": "v"}
    finally:
        client.delete_collection(copy.name)
        client.delete_snapshot(snap.id)


def test_named_sparse_and_hybrid(client: Client, name: str) -> None:
    col = client.create_collection(
        name,
        2,
        metric="euclidean",
        vectors={"title": VectorParams(2, metric="euclidean", index=Flat())},
        sparse_vectors={"text": SparseVectorParams("idf")},
    )
    info = col.info()
    assert info.vectors["title"]["dimension"] == 2
    assert info.sparse_vectors["text"]["modifier"] == "idf"
    col.upsert(
        [
            Point("a", [0, 0], vectors={"title": [1, 1]}, sparse_vectors={"text": {1: 1.0}}),
            Point("b", [1, 1], vectors={"title": [0, 0]}, sparse_vectors={"text": {2: 1.0}}),
            {"id": "c", "vector": [5, 5], "vectors": {"title": [5, 5]}},
        ]
    )
    assert col.search([0, 0], k=1, using="title")[0].id == "b"
    hits = col.search_sparse({2: 1.0}, using="text", with_vectors=True)
    assert [h.id for h in hits] == ["b"]
    assert hits[0].distance < 0
    assert hits[0].sparse_vectors["text"] == SparseVector([2], [1.0])
    assert hits[0].vectors["title"] == [0.0, 0.0]
    point = col.get("a")
    assert point is not None and point.vectors["title"] == [1.0, 1.0]

    fused = col.query(
        [Prefetch(vector=[1, 1]), Prefetch(sparse={2: 1.0}, using="text", weight=2.0)], k=2
    )
    assert fused[0].id == "b"
    assert fused[0].score > fused[1].score
    with pytest.raises(InvalidRequestError):
        col.upsert([("d", [0, 0])])


def test_update_rename_and_aliases(client: Client, name: str) -> None:
    col = client.create_collection(name, 2, index=HNSW(m=32))
    col.upsert([("a", [1, 0]), ("b", [0, 1])])
    info = col.update(index={"ef_search": 200}, quantization="scalar")
    assert info.index["m"] == 32 and info.index["ef_search"] == 200
    assert info.quantization == "scalar"
    with pytest.raises(InvalidRequestError):
        col.update(index={"m": 1})

    alias = f"{name}_alias"
    client.set_alias(alias, name)
    assert client.get_collection(alias).info().name == name
    new_name = f"{name}_renamed"
    col.rename(new_name)
    try:
        assert col.name == new_name
        assert len(col) == 2
        assert client.list_aliases()[alias] == new_name
        client.delete_alias(alias)
        with pytest.raises(NotFoundError):
            client.delete_alias(alias)
    finally:
        client.delete_collection(new_name)


def test_async_named_vectors(name: str) -> None:
    assert URL

    async def run() -> None:
        async with AsyncClient(URL, api_key=os.environ.get("QUIVER_API_KEY")) as client:
            col = await client.create_collection(
                name, 2, metric="euclidean", sparse_vectors=["text"]
            )
            await col.upsert([Point("a", [0, 0], sparse_vectors={"text": {3: 2.0}})])
            hits = await col.search_sparse({3: 1.0}, using="text")
            assert hits[0].id == "a" and hits[0].distance == -2.0
            fused = await col.query([Prefetch(sparse={3: 1.0}, using="text")], k=1, rrf_k=1)
            assert fused[0].score == 0.5
            await col.rename(f"{name}_x")
            await client.rename_collection(f"{name}_x", name)

    asyncio.run(run())
