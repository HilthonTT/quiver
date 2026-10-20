"""Sans-I/O request building and response parsing shared by both clients.

Each operation returns a :class:`Call`: what to send and how to turn the
JSON response into a Python value. The sync and async clients only differ in
how they execute a call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from urllib.parse import quote

from ._errors import error_for_status
from ._models import (
    HNSW,
    CollectionInfo,
    Filter,
    Index,
    Metric,
    Point,
    PointLike,
    SearchResult,
    SnapshotInfo,
    Vector,
    as_list,
    to_point_dict,
)

T = TypeVar("T")


@dataclass(frozen=True)
class Call(Generic[T]):
    method: str
    path: str
    parse: Callable[[Any], T]
    json: Any = None
    params: dict[str, Any] | None = None
    # Raw request body (instead of JSON) and its content type.
    content: bytes | None = None
    content_type: str | None = None
    # Hand the raw response bytes to `parse` instead of decoded JSON.
    raw: bool = False
    # Statuses that are mapped to a value instead of raising.
    allow: tuple[int, ...] = ()

    def headers(self) -> dict[str, str] | None:
        return {"content-type": self.content_type} if self.content_type else None


def _none(_: Any) -> None:
    return None


def _seg(value: str) -> str:
    return quote(value, safe="")


def raise_for_response(status: int, body: Any, text: str) -> None:
    if status < 400:
        return
    code: str | None = None
    message = text or f"HTTP {status}"
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        code = body["error"].get("code")
        message = body["error"].get("message", message)
    raise error_for_status(status, code, message)


def health() -> Call[dict[str, Any]]:
    return Call("GET", "/health", lambda b: dict(b))


def info() -> Call[dict[str, Any]]:
    return Call("GET", "/info", lambda b: dict(b))


def list_collections() -> Call[list[CollectionInfo]]:
    return Call(
        "GET",
        "/collections",
        lambda b: [CollectionInfo.from_dict(c) for c in b["collections"]],
    )


def create_collection(
    name: str, dimension: int, metric: Metric, index: Index | None
) -> Call[CollectionInfo]:
    body = {
        "name": name,
        "dimension": dimension,
        "metric": metric,
        "index": (index or HNSW()).to_dict(),
    }
    return Call("POST", "/collections", CollectionInfo.from_dict, json=body)


def get_collection(name: str) -> Call[CollectionInfo]:
    return Call("GET", f"/collections/{_seg(name)}", CollectionInfo.from_dict)


def delete_collection(name: str) -> Call[None]:
    return Call("DELETE", f"/collections/{_seg(name)}", _none)


def flush(name: str) -> Call[None]:
    return Call("POST", f"/collections/{_seg(name)}/flush", _none)


def compact(name: str) -> Call[CollectionInfo]:
    return Call("POST", f"/collections/{_seg(name)}/compact", CollectionInfo.from_dict)


def upsert(name: str, points: Sequence[dict[str, Any]]) -> Call[int]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points",
        lambda b: int(b["upserted"]),
        json={"points": list(points)},
    )


def insert(name: str, point: dict[str, Any]) -> Call[str]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points",
        lambda b: str(b["ids"][0]),
        json={"points": [point]},
    )


def get_point(name: str, point_id: str, with_vectors: bool) -> Call[Point | None]:
    return Call(
        "GET",
        f"/collections/{_seg(name)}/points/{_seg(point_id)}",
        lambda b: None if b is None else Point.from_dict(b),
        params={"with_vectors": str(with_vectors).lower()},
        allow=(404,),
    )


def lookup(name: str, ids: Sequence[str], with_vectors: bool) -> Call[list[Point]]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points/lookup",
        lambda b: [Point.from_dict(p) for p in b["points"]],
        json={"ids": list(ids), "with_vectors": with_vectors},
    )


def scroll(
    name: str, offset: int, limit: int, with_vectors: bool
) -> Call[tuple[list[Point], int | None]]:
    return Call(
        "GET",
        f"/collections/{_seg(name)}/points",
        lambda b: ([Point.from_dict(p) for p in b["points"]], b.get("next_offset")),
        params={"offset": offset, "limit": limit, "with_vectors": str(with_vectors).lower()},
    )


def delete_points(name: str, ids: Sequence[str]) -> Call[int]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points/delete",
        lambda b: int(b["deleted"]),
        json={"ids": list(ids)},
    )


def search_body(
    vector: Vector,
    k: int,
    filter: Filter | None,
    ef: int | None,
    with_vectors: bool,
    with_metadata: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "vector": as_list(vector),
        "k": k,
        "with_vectors": with_vectors,
        "with_metadata": with_metadata,
    }
    if filter is not None:
        body["filter"] = dict(filter)
    if ef is not None:
        body["ef"] = ef
    return body


def _hits(raw: Iterable[Any]) -> list[SearchResult]:
    return [SearchResult.from_dict(h) for h in raw]


def search(name: str, body: dict[str, Any]) -> Call[list[SearchResult]]:
    return Call(
        "POST", f"/collections/{_seg(name)}/search", lambda b: _hits(b["results"]), json=body
    )


def search_batch(name: str, bodies: Sequence[dict[str, Any]]) -> Call[list[list[SearchResult]]]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/search/batch",
        lambda b: [_hits(r) for r in b["results"]],
        json={"searches": list(bodies)},
    )


def chunked(items: Iterable[PointLike], size: int) -> Iterable[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(to_point_dict(item))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# --- snapshots ---------------------------------------------------------------


def _snapshots(b: Any) -> list[SnapshotInfo]:
    return [SnapshotInfo.from_dict(s) for s in b["snapshots"]]


def create_snapshot(name: str) -> Call[SnapshotInfo]:
    return Call("POST", f"/collections/{_seg(name)}/snapshots", SnapshotInfo.from_dict)


def list_snapshots(collection: str | None) -> Call[list[SnapshotInfo]]:
    params = {"collection": collection} if collection else None
    return Call("GET", "/snapshots", _snapshots, params=params)


def get_snapshot(snapshot_id: str) -> Call[SnapshotInfo]:
    return Call("GET", f"/snapshots/{_seg(snapshot_id)}", SnapshotInfo.from_dict)


def delete_snapshot(snapshot_id: str) -> Call[None]:
    return Call("DELETE", f"/snapshots/{_seg(snapshot_id)}", _none)


def restore_snapshot(
    snapshot_id: str, collection: str | None, replace: bool
) -> Call[CollectionInfo]:
    return Call(
        "POST",
        f"/snapshots/{_seg(snapshot_id)}/restore",
        CollectionInfo.from_dict,
        json={"collection": collection, "replace": replace},
    )


def download_snapshot(snapshot_id: str) -> Call[bytes]:
    return Call("GET", f"/snapshots/{_seg(snapshot_id)}/download", bytes, raw=True)


def upload_snapshot(archive: bytes) -> Call[SnapshotInfo]:
    return Call(
        "POST",
        "/snapshots",
        SnapshotInfo.from_dict,
        content=archive,
        content_type="application/gzip",
    )
