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
    Expiry,
    Flat,
    InvalidRequestError,
    KDTree,
    NotFoundError,
    OrderBy,
    PermissionDeniedError,
    Point,
    Prefetch,
    QuiverConnectionError,
    Rescore,
    ScrollPage,
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


@pytest.mark.parametrize("point_id", ["lookup", "delete", "count", "metadata", "scroll", "facet"])
def test_ids_taken_by_fixed_routes_are_fetched_by_lookup(
    client: Client, async_client: AsyncClient, mock: Recorder, point_id: str
) -> None:
    mock.on("POST", "/collections/docs/points/lookup", 200, {"points": [{"id": point_id}]})
    point = client.collection("docs").get(point_id)
    assert point is not None and point.id == point_id
    assert mock.last_json()["ids"] == [point_id]

    async def main() -> None:
        async with async_client as c:
            p = await c.collection("docs").get(point_id)
            assert p is not None and p.id == point_id

    asyncio.run(main())

    mock.on("POST", "/collections/docs/points/lookup", 200, {"points": []})
    assert client.collection("docs").get(point_id) is None


def test_get_on_missing_collection_raises(
    client: Client, async_client: AsyncClient, mock: Recorder
) -> None:
    # The mock answers unknown routes with a plain `not_found`, like a missing collection.
    with pytest.raises(NotFoundError):
        client.collection("typo").get("a")

    async def main() -> None:
        async with async_client as c:
            with pytest.raises(NotFoundError):
                await c.collection("typo").get("a")

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
            p = await c.collection("docs").get(point_id)
            assert p is not None and p.id == point_id

    asyncio.run(main())


def test_scroll_returns_one_page_with_its_cursor(client: Client, mock: Recorder) -> None:
    mock.on("GET", "/collections/docs/points", 200, {"points": [{"id": "a"}], "next_cursor": "a"})
    page = client.collection("docs").scroll(limit=1)
    assert page == ScrollPage([Point("a", [])], "a")
    assert dict(mock.requests[-1].url.params) == {"limit": "1", "with_vectors": "false"}

    mock.on("GET", "/collections/docs/points", 200, {"points": []})
    page = client.collection("docs").scroll("a b", limit=5, with_vectors=True)
    assert page.next_cursor is None
    params = mock.requests[-1].url.params
    assert (params["cursor"], params["with_vectors"]) == ("a b", "true")
    assert "offset" not in params


def test_iter_points_follows_cursors(mock: Recorder) -> None:
    pages = {
        None: {"points": [{"id": "a"}, {"id": "b"}], "next_cursor": "b"},
        "b": {"points": [{"id": "c"}], "next_cursor": None},
    }
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        seen.append(cursor)
        return httpx.Response(200, json=pages[cursor])

    transport = httpx.MockTransport(handler)
    c = Client(http_client=httpx.Client(base_url="http://t", transport=transport))
    assert [p.id for p in c.collection("docs").iter_points(page_size=2)] == ["a", "b", "c"]
    assert seen == [None, "b"]

    async def main() -> None:
        http = httpx.AsyncClient(base_url="http://t", transport=transport)
        async with AsyncClient(http_client=http) as ac:
            col = ac.collection("docs")
            assert [p.id async for p in col.iter_points(page_size=2)] == ["a", "b", "c"]
            assert (await col.scroll("b")).next_cursor is None

    asyncio.run(main())


def test_delete(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections/docs/points/delete", 200, {"deleted": 1})
    assert client.collection("docs").delete(["a", "zz"]) == 1
    assert mock.last_json() == {"ids": ["a", "zz"]}


def test_a_lone_string_is_one_id_or_name(client: Client, mock: Recorder) -> None:
    col = client.collection("docs")
    mock.on("POST", "/collections/docs/points/delete", 200, {"deleted": 1})
    col.delete("ab")
    assert mock.last_json() == {"ids": ["ab"]}
    mock.on("POST", "/collections/docs/points/lookup", 200, {"points": []})
    col.get_many("ab")
    assert mock.last_json()["ids"] == ["ab"]
    mock.on("POST", "/collections", 201, INFO)
    client.create_collection("docs", 3, field_indexes="lang", sparse_vectors="text")
    body = mock.last_json()
    assert body["field_indexes"] == ["lang"]
    assert body["sparse_vectors"] == {"text": {}}


def _data_ops_routes(mock: Recorder) -> None:
    hit = {"results": [{"id": "b", "distance": 0.5}]}
    mock.on("POST", "/collections/docs/points/count", 200, {"count": 3})
    mock.on("POST", "/collections/docs/points/delete", 200, {"deleted": 2})
    mock.on("POST", "/collections/docs/points/metadata", 200, {"updated": 1})
    mock.on("POST", "/collections/docs/search", 200, hit)
    mock.on("POST", "/collections/docs/recommend", 200, hit)
    mock.on("POST", "/collections/docs/discover", 200, hit)
    mock.on("POST", "/collections/docs/search/batch", 200, {"results": [hit["results"]]})
    mock.on(
        "POST",
        "/collections/docs/query",
        200,
        {"results": [{"id": "b", "score": 0.5}]},
    )
    mock.on(
        "POST",
        "/collections/docs/search/groups",
        200,
        {"groups": [{"key": "sf", "hits": hit["results"]}]},
    )
    mock.on(
        "POST",
        "/collections/docs/points/facet",
        200,
        {"values": [{"value": "sf", "count": 2}]},
    )
    mock.on(
        "POST",
        "/collections/docs/points/scroll",
        200,
        {"points": [{"id": "b"}], "next_cursor": '[1965,"b"]'},
    )


# Each data operation, its result from the routes above, and the body it must send.
DATA_OPS: list[tuple[str, tuple[Any, ...], dict[str, Any], Any, dict[str, Any]]] = [
    ("count", (), {}, 3, {}),
    ("count", ({"genre": "sf"},), {}, 3, {"filter": {"genre": "sf"}}),
    ("delete_where", ({"year": {"$lt": 1990}},), {}, 2, {"filter": {"year": {"$lt": 1990}}}),
    (
        "update_metadata",
        ({"seen": True, "old": None},),
        {"ids": ["a"]},
        1,
        {"metadata": {"seen": True, "old": None}, "mode": "merge", "ids": ["a"]},
    ),
    (
        "update_metadata",
        (None,),
        {"filter": {"genre": "sf"}, "mode": "replace"},
        1,
        {"metadata": None, "mode": "replace", "filter": {"genre": "sf"}},
    ),
    (
        "search",
        (np.array([1, 0], dtype=np.float64),),
        {"k": 3, "max_distance": 0.25},
        "b",
        {
            "vector": [1.0, 0.0],
            "k": 3,
            "with_vectors": False,
            "with_metadata": True,
            "max_distance": 0.25,
        },
    ),
    (
        "search_by_id",
        ("a",),
        {"k": 2, "filter": {"x": 1}},
        "b",
        {"id": "a", "k": 2, "with_vectors": False, "with_metadata": True, "filter": {"x": 1}},
    ),
    (
        "recommend",
        ("a",),
        {"k": 5},
        "b",
        {"positive": ["a"], "negative": [], "k": 5, "with_vectors": False, "with_metadata": True},
    ),
    (
        "recommend",
        (["a", "c"], ["d"]),
        {"max_distance": 1.0, "with_vectors": True},
        "b",
        {
            "positive": ["a", "c"],
            "negative": ["d"],
            "k": 10,
            "with_vectors": True,
            "with_metadata": True,
            "max_distance": 1.0,
        },
    ),
    (
        "search",
        ([1, 0],),
        {"k": 3, "offset": 6},
        "b",
        {"vector": [1.0, 0.0], "k": 3, "offset": 6, "with_vectors": False, "with_metadata": True},
    ),
    (
        "search_batch",
        ([[1, 0]],),
        {"offset": 2},
        "b",
        {
            "searches": [
                {
                    "vector": [1.0, 0.0],
                    "k": 10,
                    "offset": 2,
                    "with_vectors": False,
                    "with_metadata": True,
                }
            ]
        },
    ),
    (
        "search_sparse",
        ({3: 1.0},),
        {"using": "text", "offset": 1},
        "b",
        {
            "sparse": {"indices": [3], "values": [1.0]},
            "k": 10,
            "offset": 1,
            "using": "text",
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "recommend",
        ("a",),
        {"offset": 4},
        "b",
        {
            "positive": ["a"],
            "negative": [],
            "k": 10,
            "offset": 4,
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "query",
        ([Prefetch(vector=[1, 0], limit=5)],),
        {"fusion": "dbsf", "offset": 1},
        "b",
        {
            "prefetch": [{"vector": [1.0, 0.0], "weight": 1.0, "limit": 5}],
            "fusion": {"type": "dbsf"},
            "k": 10,
            "offset": 1,
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "query",
        ([Prefetch(vector=[1, 0])],),
        {"rescore": Rescore([0, 1, 0], using="fine"), "rrf_k": 10},
        "b",
        {
            "prefetch": [{"vector": [1.0, 0.0], "weight": 1.0}],
            "fusion": {"type": "rrf", "k": 10},
            "rescore": {"vector": [0.0, 1.0, 0.0], "using": "fine"},
            "k": 10,
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "search_groups",
        (np.array([1, 0]),),
        {"group_by": "genre", "groups": 2, "group_size": 1, "filter": {"year": {"$gt": 1}}},
        "sf",
        {
            "vector": [1.0, 0.0],
            "group_by": "genre",
            "groups": 2,
            "group_size": 1,
            "with_vectors": False,
            "with_metadata": True,
            "filter": {"year": {"$gt": 1}},
        },
    ),
    (
        "search_groups",
        (),
        {"id": "a", "group_by": "genre"},
        "sf",
        {
            "id": "a",
            "group_by": "genre",
            "groups": 10,
            "group_size": 3,
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "search_groups",
        (),
        {"sparse": {1: 2.0}, "using": "text", "group_by": "genre"},
        "sf",
        {
            "sparse": {"indices": [1], "values": [2.0]},
            "group_by": "genre",
            "groups": 10,
            "group_size": 3,
            "using": "text",
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "facet",
        ("genre",),
        {"filter": {"title": {"$text": "dune"}}, "limit": 5},
        "sf",
        {"field": "genre", "limit": 5, "filter": {"title": {"$text": "dune"}}},
    ),
    (
        "discover",
        ("a", [("b", np.array([0.5, 0.5]))]),
        {"k": 2, "offset": 1},
        "b",
        {
            "target": "a",
            "context": [{"positive": "b", "negative": [0.5, 0.5]}],
            "k": 2,
            "offset": 1,
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "discover",
        (None, [([1, 0], "c")]),
        {},
        "b",
        {
            "context": [{"positive": [1.0, 0.0], "negative": "c"}],
            "k": 10,
            "with_vectors": False,
            "with_metadata": True,
        },
    ),
    (
        "scroll",
        (),
        {"order_by": "year", "limit": 1},
        (["b"], '[1965,"b"]'),
        {"limit": 1, "with_vectors": False, "order_by": {"field": "year", "direction": "asc"}},
    ),
    (
        "scroll",
        ('[1965,"b"]',),
        {"order_by": OrderBy("year", "desc"), "filter": {"tags": {"$all": ["a"]}}},
        (["b"], '[1965,"b"]'),
        {
            "limit": 100,
            "with_vectors": False,
            "cursor": '[1965,"b"]',
            "filter": {"tags": {"$all": ["a"]}},
            "order_by": {"field": "year", "direction": "desc"},
        },
    ),
]


def _summary(result: Any) -> Any:
    if isinstance(result, ScrollPage):
        return [p.id for p in result.points], result.next_cursor
    if isinstance(result, list) and result:
        first = result[0][0] if isinstance(result[0], list) else result[0]
        for attr in ("id", "key", "value"):
            if hasattr(first, attr):
                return getattr(first, attr)
    return result


@pytest.mark.parametrize(("method", "args", "kwargs", "expected", "body"), DATA_OPS)
def test_data_operations_send_the_same_requests_sync_and_async(
    client: Client,
    async_client: AsyncClient,
    mock: Recorder,
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: Any,
    body: dict[str, Any],
) -> None:
    _data_ops_routes(mock)
    result = getattr(client.collection("docs"), method)(*args, **kwargs)
    assert _summary(result) == expected
    assert mock.last_json() == body

    async def main() -> Any:
        async with async_client as c:
            return await getattr(c.collection("docs"), method)(*args, **kwargs)

    assert _summary(asyncio.run(main())) == expected
    assert mock.last_json() == body


def test_update_metadata_needs_exactly_one_selector(client: Client) -> None:
    col = client.collection("docs")
    with pytest.raises(ValueError):
        col.update_metadata({"a": 1})
    with pytest.raises(ValueError):
        col.update_metadata({"a": 1}, ids=["a"], filter={"x": 1})


def test_missing_example_points_raise_not_found(client: Client, mock: Recorder) -> None:
    error = {"error": {"code": "point_not_found", "message": "point `x` not found"}}
    mock.on("POST", "/collections/docs/recommend", 404, error)
    with pytest.raises(NotFoundError) as info:
        client.collection("docs").recommend(["x"])
    assert info.value.code == "point_not_found"


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
    mock.on("POST", "/collections/docs/points/count", 200, {"count": 2})

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


def test_field_indexes(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, {**INFO, "field_indexes": ["genre"]})
    col = client.create_collection("docs", 3, field_indexes=["genre"])
    assert mock.last_json()["field_indexes"] == ["genre"]

    mock.on("POST", "/collections/docs/indexes", 200, {**INFO, "field_indexes": ["genre", "a.b"]})
    info = col.create_field_index("a.b")
    assert mock.last_json() == {"field": "a.b"}
    assert info.field_indexes == ["genre", "a.b"]

    mock.on("DELETE", "/collections/docs/indexes/a.b", 200, {**INFO, "field_indexes": ["genre"]})
    assert col.drop_field_index("a.b").field_indexes == ["genre"]

    # Old servers that do not report the field read as no indexes.
    mock.on("GET", "/collections/docs", 200, INFO)
    assert col.info().field_indexes == []


def test_quantization(client: Client, mock: Recorder) -> None:
    mock.on("POST", "/collections", 201, {**INFO, "quantization": "scalar"})
    client.create_collection("docs", 3, quantization="scalar")
    assert mock.last_json()["quantization"] == "scalar"
    client.create_collection("docs", 3)
    assert "quantization" not in mock.last_json()

    mock.on("GET", "/collections/docs", 200, {**INFO, "quantization": "scalar"})
    assert client.collection("docs").info().quantization == "scalar"
    mock.on("GET", "/collections/docs", 200, INFO)
    assert client.collection("docs").info().quantization == "none"

    mock.on("POST", "/collections", 201, {**INFO, "quantization": "binary"})
    client.create_collection("docs", 3, quantization="binary", on_disk=True)
    assert mock.last_json()["quantization"] == "binary"


def test_expiry(client: Client, mock: Recorder) -> None:
    expiry = {"field": "exp", "default_ttl": 60}
    mock.on("POST", "/collections", 201, {**INFO, "expiry": expiry})
    client.create_collection("docs", 3, expiry=Expiry("exp", default_ttl=60))
    assert mock.last_json()["expiry"] == expiry
    client.create_collection("docs", 3, expiry=Expiry("meta.exp"))
    assert mock.last_json()["expiry"] == {"field": "meta.exp"}
    client.create_collection("docs", 3)
    assert "expiry" not in mock.last_json()

    mock.on("GET", "/collections/docs", 200, {**INFO, "expiry": expiry})
    col = client.collection("docs")
    assert col.info().expiry == Expiry("exp", 60)
    mock.on("GET", "/collections/docs", 200, INFO)
    assert col.info().expiry is None

    mock.on("PATCH", "/collections/docs", 200, INFO)
    col.update(expiry=Expiry("exp"))
    assert mock.last_json() == {"expiry": {"field": "exp"}}
    col.update(expiry=False)
    assert mock.last_json() == {"expiry": None}
    col.update(on_disk=True)
    assert "expiry" not in mock.last_json()


def test_tls_options(tmp_path: Path) -> None:
    from quiver._client import _verify

    assert _verify(None, None, None) is True
    with pytest.raises(ValueError):
        _verify(None, None, tmp_path / "key.pem")
    with pytest.raises(OSError):
        Client("https://localhost:6900", ca_cert=tmp_path / "missing.pem")
    with pytest.raises(OSError):
        AsyncClient("https://localhost:6900", client_cert=tmp_path / "missing.pem")


def test_unfiltered_scroll_keeps_the_get_endpoint(client: Client, mock: Recorder) -> None:
    mock.on("GET", "/collections/docs/points", 200, {"points": [], "next_cursor": None})
    client.collection("docs").scroll(limit=5)
    assert mock.requests[-1].method == "GET"


def test_iter_points_passes_ordering_through(client: Client, mock: Recorder) -> None:
    pages = iter(
        [
            {"points": [{"id": "x"}], "next_cursor": '[1,"x"]'},
            {"points": [{"id": "y"}], "next_cursor": None},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        mock.requests.append(request)
        return httpx.Response(200, json=next(pages))

    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    col = Client(http_client=http).collection("docs")
    ids = [p.id for p in col.iter_points(page_size=1, order_by=OrderBy("n", "desc"))]
    assert ids == ["x", "y"]
    assert mock.last_json() == {
        "limit": 1,
        "with_vectors": False,
        "cursor": '[1,"x"]',
        "order_by": {"field": "n", "direction": "desc"},
    }


def test_new_searches_validate_their_targets(client: Client) -> None:
    col = client.collection("docs")
    with pytest.raises(ValueError):
        col.search_groups(group_by="g")
    with pytest.raises(ValueError):
        col.search_groups([1, 0], id="a", group_by="g")
    with pytest.raises(ValueError):
        col.search_groups(sparse={1: 1.0}, group_by="g")
