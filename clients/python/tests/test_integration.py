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
    assert sum(1 for _ in col.scroll(page_size=64)) == 400
    col.flush()


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
            assert [p.id async for p in col.scroll()] == ["a", "b"]
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
