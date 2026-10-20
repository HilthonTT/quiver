from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from conftest import Recorder

from quiver import (
    HNSW,
    AsyncClient,
    AuthenticationError,
    Client,
    ConflictError,
    Flat,
    InvalidRequestError,
    KDTree,
    NotFoundError,
    PermissionDeniedError,
    Point,
    QuiverConnectionError,
    SearchResult,
    ServerError,
)

INFO = {
    "name": "docs",
    "dimension": 3,
    "metric": "cosine",
    "index": {"type": "hnsw", "m": 16, "ef_construction": 128, "ef_search": 64},
    "count": 2,
    "tombstones": 0,
}


def test_create_collection_sends_config(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, INFO)
    col = client.create_collection("docs", 3, metric="euclidean", index=HNSW(m=32, seed=7))
    assert col.name == "docs"
    assert mock.last_json() == {
        "name": "docs",
        "dimension": 3,
        "metric": "euclidean",
        "index": {"type": "hnsw", "m": 32, "ef_construction": 128, "ef_search": 64, "seed": 7},
    }


def test_flat_index_payload(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, INFO)
    client.create_collection("docs", 3, index=Flat())
    assert mock.last_json()["index"] == {"type": "flat"}


def test_info_and_len(client: Client, mock: Recorder) -> None:
    mock.on("GET", "/collections/docs", 200, INFO)
    col = client.get_collection("docs")
    info = col.info()
    assert info.dimension == 3
    assert info.index["m"] == 16
    assert len(col) == 2


def test_get_or_create_creates_when_missing(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, INFO)
    client.get_or_create_collection("docs", 3)
    assert [r.method for r in mock.requests] == ["GET", "POST"]


def test_upsert_accepts_many_point_shapes_and_batches(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections/docs/points", 200, {"upserted": 2})
    col = client.collection("docs")
    written = col.upsert(
        [
            Point("a", [1, 0, 0], {"k": 1}),
            {"id": "b", "vector": (0, 1, 0)},
            ("c", np.array([0, 0, 1], dtype=np.float32)),
            ("d", [1, 1, 1], {"k": 2}),
        ],
        batch_size=2,
    )
    assert written == 4
    assert len(mock.requests) == 2
    first = mock.requests[0]
    import json

    points = json.loads(first.content)["points"]
    assert points[0] == {"id": "a", "vector": [1.0, 0.0, 0.0], "metadata": {"k": 1}}
    assert points[1] == {"id": "b", "vector": [0.0, 1.0, 0.0]}
    assert mock.last_json()["points"][0]["vector"] == [0.0, 0.0, 1.0]


def test_upsert_rejects_garbage(client: Client) -> None:
    with pytest.raises(TypeError):
        client.collection("docs").upsert([42])  # type: ignore[list-item]


def test_search_builds_request_and_parses_hits(client: Client, mock: Recorder) -> None:
    mock.on(
        "POST",
        "/collections/docs/search",
        200,
        {"results": [{"id": "a", "distance": 0.1, "metadata": {"lang": "en"}}]},
    )
    hits = client.collection("docs").search(
        np.array([1.0, 0.0, 0.0]), k=3, filter={"lang": "en"}, ef=128
    )
    assert hits == [SearchResult(id="a", distance=0.1, metadata={"lang": "en"})]
    assert mock.last_json() == {
        "vector": [1.0, 0.0, 0.0],
        "k": 3,
        "with_vectors": False,
        "with_metadata": True,
        "filter": {"lang": "en"},
        "ef": 128,
    }


def test_search_batch(client: Client, mock: Recorder) -> None:
    mock.on(
        "POST",
        "/collections/docs/search/batch",
        200,
        {"results": [[{"id": "a", "distance": 0}], []]},
    )
    out = client.collection("docs").search_batch([[1, 0, 0], [0, 1, 0]], k=1)
    assert [len(r) for r in out] == [1, 0]
    assert len(mock.last_json()["searches"]) == 2


def test_get_returns_none_when_missing(client: Client, mock: Recorder) -> None:
    body = {"error": {"code": "point_not_found", "message": "point `nope` not found"}}
    mock.on("GET", "/collections/docs/points/nope", 404, body)
    assert client.collection("docs").get("nope") is None


def test_get_on_missing_collection_raises(
    client: Client, async_client: AsyncClient, mock: Recorder
) -> None:
    # The mock answers unknown routes with a plain `not_found`, like a missing collection.
    with pytest.raises(NotFoundError):
        client.collection("typo").get("a")

    async def main() -> None:
        async with async_client as c:
            with pytest.raises(NotFoundError):
                await (await c.collection("typo")).get("a")

    asyncio.run(main())


def test_ids_are_url_encoded(client: Client, mock: Recorder) -> None:
    mock.on("GET", "/collections/docs/points/a%2Fb%20c", 200, {"id": "a/b c", "vector": [1]})
    point = client.collection("docs").get("a/b c")
    assert point is not None and point.id == "a/b c"


@pytest.mark.parametrize(("point_id", "path"), [(".", "%2E"), ("..", "%2E%2E")])
def test_dot_ids_are_not_dropped_from_the_path(
    client: Client, async_client: AsyncClient, mock: Recorder, point_id: str, path: str
) -> None:
    mock.on("GET", f"/collections/docs/points/{path}", 200, {"id": point_id, "vector": [1]})
    point = client.collection("docs").get(point_id)
    assert point is not None and point.id == point_id

    async def main() -> None:
        async with async_client as c:
            p = await (await c.collection("docs")).get(point_id)
            assert p is not None and p.id == point_id

    asyncio.run(main())


def test_scroll_follows_pages(client: Client, mock: Recorder) -> None:
    pages = iter(
        [
            {"points": [{"id": "a"}, {"id": "b"}], "next_offset": 2},
            {"points": [{"id": "c"}], "next_offset": None},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(pages))

    c = Client(
        http_client=httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler))
    )
    assert [p.id for p in c.collection("docs").scroll(page_size=2)] == ["a", "b", "c"]


def test_delete(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections/docs/points/delete", 200, {"deleted": 1})
    assert client.collection("docs").delete(["a", "zz"]) == 1
    assert mock.last_json() == {"ids": ["a", "zz"]}


@pytest.mark.parametrize(
    ("status", "code", "exc"),
    [
        (400, "dimension_mismatch", InvalidRequestError),
        (401, "unauthorized", AuthenticationError),
        (403, "forbidden", PermissionDeniedError),
        (404, "not_found", NotFoundError),
        (409, "already_exists", ConflictError),
        (500, "internal", ServerError),
    ],
)
def test_errors_are_mapped(
    client: Client, mock: Recorder, status: int, code: str, exc: type[Exception]
) -> None:
    mock.on("GET", "/collections", status, {"error": {"code": code, "message": "boom"}})
    with pytest.raises(exc) as info:
        client.list_collections()
    assert info.value.code == code  # type: ignore[attr-defined]
    assert info.value.status_code == status  # type: ignore[attr-defined]
    assert str(info.value) == "boom"


def test_connection_errors_are_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    c = Client(
        http_client=httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler))
    )
    with pytest.raises(QuiverConnectionError):
        c.health()


def test_api_key_header() -> None:
    c = Client("http://localhost:1/", api_key="k")
    assert c._http.headers["authorization"] == "Bearer k"
    assert str(c._http.base_url) == "http://localhost:1"
    c.close()


def test_async_client(async_client: AsyncClient, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, INFO)
    mock.on("POST", "/collections/docs/points", 200, {"upserted": 1})
    mock.on("POST", "/collections/docs/search", 200, {"results": [{"id": "a", "distance": 0}]})
    mock.on("GET", "/collections/docs", 200, INFO)

    async def main() -> None:
        async with async_client as client:
            col = await client.get_or_create_collection("docs", 3)
            assert await col.upsert([("a", [1, 0, 0])]) == 1
            hits = await col.search([1, 0, 0], k=1)
            assert hits[0].id == "a"
            assert await col.count() == 2

    asyncio.run(main())


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    return Client(
        http_client=httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler))
    )


SNAP: dict[str, Any] = {
    "format_version": 1,
    "id": "4c8ded18-d753-468b-bdbf-5b78c45fbdb3",
    "collection": "docs",
    "created_at_ms": 1_760_000_000_000,
    "quiver_version": "0.2.0",
    "config": {"dimension": 3, "metric": "cosine", "index": {"type": "flat"}},
    "points": 2,
    "data_bytes": 10,
    "data_sha256": "ab",
    "archive_bytes": 99,
}


def test_kd_tree_and_new_metrics(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, INFO)
    client.create_collection("docs", 3, metric="manhattan", index=KDTree(balance_factor=4))
    assert mock.last_json()["metric"] == "manhattan"
    assert mock.last_json()["index"] == {"type": "kd_tree", "balance_factor": 4}


def test_insert_returns_generated_id(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections/docs/points", 200, {"upserted": 1, "ids": ["gen-1"]})
    assert client.collection("docs").insert([1, 0, 0], {"k": 1}) == "gen-1"
    assert mock.last_json() == {"points": [{"vector": [1.0, 0.0, 0.0], "metadata": {"k": 1}}]}
    client.collection("docs").insert([1, 0, 0], id="mine")
    assert mock.last_json()["points"][0]["id"] == "mine"


def test_dicts_without_ids(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections/docs/points", 200, {"upserted": 1, "ids": ["x"]})
    client.collection("docs").upsert([{"vector": [1, 2, 3]}])
    assert mock.last_json() == {"points": [{"vector": [1.0, 2.0, 3.0]}]}


def test_snapshot_api(client: Client, mock: Recorder, tmp_path: Path) -> None:
    sid = SNAP["id"]
    mock.on("POST", "/collections/docs/snapshots", 201, SNAP)
    mock.on("GET", "/snapshots", 200, {"snapshots": [SNAP]})
    mock.on("POST", f"/snapshots/{sid}/restore", 200, {**INFO, "name": "copy"})
    mock.on("DELETE", f"/snapshots/{sid}", 204)

    snap = client.collection("docs").create_snapshot()
    assert snap.id == sid and snap.points == 2 and snap.dimension == 3
    assert snap.created_at.year == 2025
    assert [s.id for s in client.list_snapshots(collection="docs")] == [sid]
    assert mock.requests[-1].url.params["collection"] == "docs"

    restored = client.restore_snapshot(sid, collection="copy", replace=True)
    assert restored.name == "copy"
    assert mock.last_json() == {"collection": "copy", "replace": True}
    client.delete_snapshot(sid)


def test_snapshot_download_and_upload(tmp_path: Path) -> None:
    sid = SNAP["id"]
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"\x1f\x8barchive")
        seen["body"] = request.content
        seen["type"] = request.headers["content-type"].encode()
        return httpx.Response(201, json=SNAP)

    c = mock_client(handler)
    path = c.download_snapshot(sid, tmp_path / "backup.tar.gz")
    assert path.read_bytes() == b"\x1f\x8barchive"
    assert c.upload_snapshot(path).id == sid
    assert seen == {"body": b"\x1f\x8barchive", "type": b"application/gzip"}


def test_download_errors_are_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "gone"}})

    c = mock_client(handler)
    with pytest.raises(NotFoundError):
        c.download_snapshot("x", "unused.tar.gz")
