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
    MetadataMode,
    Metric,
    Point,
    PointLike,
    ScrollPage,
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
    # Error codes that are mapped to a value instead of raising.
    allow_codes: tuple[str, ...] = ()

    def headers(self) -> dict[str, str] | None:
        return {"content-type": self.content_type} if self.content_type else None


def _none(_: Any) -> None:
    return None


def _seg(value: str) -> str:
    # `.` and `..` would be removed as dot-segments when the URL is normalised.
    if value in (".", ".."):
        return "%2E" * len(value)
    return quote(value, safe="")


def error_code(body: Any) -> str | None:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        code = body["error"].get("code")
        return code if isinstance(code, str) else None
    return None


def is_allowed(call: Call[Any], status: int, body: Any) -> bool:
    """Whether an error response means "no value" for this call."""
    return status >= 400 and error_code(body) in call.allow_codes


def raise_for_response(status: int, body: Any, text: str) -> None:
    if status < 400:
        return
    message = text or f"HTTP {status}"
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message", message)
    raise error_for_status(status, error_code(body), message)


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
    name: str,
    dimension: int,
    metric: Metric,
    index: Index | None,
    field_indexes: Sequence[str] = (),
) -> Call[CollectionInfo]:
    body: dict[str, Any] = {
        "name": name,
        "dimension": dimension,
        "metric": metric,
        "index": (index or HNSW()).to_dict(),
    }
    if field_indexes:
        body["field_indexes"] = list(field_indexes)
    return Call("POST", "/collections", CollectionInfo.from_dict, json=body)


def create_field_index(name: str, field: str) -> Call[CollectionInfo]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/indexes",
        CollectionInfo.from_dict,
        json={"field": field},
    )


def drop_field_index(name: str, field: str) -> Call[CollectionInfo]:
    return Call(
        "DELETE", f"/collections/{_seg(name)}/indexes/{_seg(field)}", CollectionInfo.from_dict
    )


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


# Ids that `GET /points/{id}` cannot reach, because the server's fixed
# `/points/...` routes take those paths; they are fetched through lookup.
_ROUTED_POINT_IDS = frozenset({"lookup", "delete", "count", "metadata"})


def get_point(name: str, point_id: str, with_vectors: bool) -> Call[Point | None]:
    if point_id in _ROUTED_POINT_IDS:
        return Call(
            "POST",
            f"/collections/{_seg(name)}/points/lookup",
            lambda b: next((Point.from_dict(p) for p in b["points"]), None),
            json={"ids": [point_id], "with_vectors": with_vectors},
        )
    return Call(
        "GET",
        f"/collections/{_seg(name)}/points/{_seg(point_id)}",
        lambda b: None if b is None else Point.from_dict(b),
        params={"with_vectors": str(with_vectors).lower()},
        # A missing point; a missing collection still raises NotFoundError.
        allow_codes=("point_not_found",),
    )


def lookup(name: str, ids: Sequence[str], with_vectors: bool) -> Call[list[Point]]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points/lookup",
        lambda b: [Point.from_dict(p) for p in b["points"]],
        json={"ids": list(ids), "with_vectors": with_vectors},
    )


def scroll(name: str, cursor: str | None, limit: int, with_vectors: bool) -> Call[ScrollPage]:
    params: dict[str, Any] = {"limit": limit, "with_vectors": str(with_vectors).lower()}
    if cursor is not None:
        params["cursor"] = cursor
    return Call(
        "GET",
        f"/collections/{_seg(name)}/points",
        lambda b: ScrollPage([Point.from_dict(p) for p in b["points"]], b.get("next_cursor")),
        params=params,
    )


def delete_points(name: str, ids: Sequence[str]) -> Call[int]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points/delete",
        lambda b: int(b["deleted"]),
        json={"ids": list(ids)},
    )


def _ids(ids: str | Iterable[str]) -> list[str]:
    # A lone string is one id, not a sequence of one-character ids.
    return [ids] if isinstance(ids, str) else list(ids)


def search_options(
    k: int,
    filter: Filter | None,
    ef: int | None,
    max_distance: float | None,
    with_vectors: bool,
    with_metadata: bool,
) -> dict[str, Any]:
    """The settings shared by searches and recommendations."""
    body: dict[str, Any] = {"k": k, "with_vectors": with_vectors, "with_metadata": with_metadata}
    if filter is not None:
        body["filter"] = dict(filter)
    if ef is not None:
        body["ef"] = ef
    if max_distance is not None:
        body["max_distance"] = max_distance
    return body


def search_body(
    vector: Vector,
    k: int,
    filter: Filter | None,
    ef: int | None,
    with_vectors: bool,
    with_metadata: bool,
    max_distance: float | None = None,
) -> dict[str, Any]:
    options = search_options(k, filter, ef, max_distance, with_vectors, with_metadata)
    return {"vector": as_list(vector), **options}


def search_by_id_body(point_id: str, options: dict[str, Any]) -> dict[str, Any]:
    return {"id": point_id, **options}


def recommend(
    name: str, positive: str | Iterable[str], negative: str | Iterable[str], options: dict[str, Any]
) -> Call[list[SearchResult]]:
    body = {"positive": _ids(positive), "negative": _ids(negative), **options}
    return Call(
        "POST", f"/collections/{_seg(name)}/recommend", lambda b: _hits(b["results"]), json=body
    )


def count(name: str, filter: Filter | None) -> Call[int]:
    body = {} if filter is None else {"filter": dict(filter)}
    return Call(
        "POST", f"/collections/{_seg(name)}/points/count", lambda b: int(b["count"]), json=body
    )


def delete_where(name: str, filter: Filter) -> Call[int]:
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points/delete",
        lambda b: int(b["deleted"]),
        json={"filter": dict(filter)},
    )


def update_metadata(
    name: str,
    metadata: Any,
    ids: str | Iterable[str] | None,
    filter: Filter | None,
    mode: MetadataMode,
) -> Call[int]:
    if (ids is None) == (filter is None):
        raise ValueError("pass exactly one of ids= and filter=")
    body: dict[str, Any] = {"metadata": metadata, "mode": mode}
    if ids is not None:
        body["ids"] = _ids(ids)
    if filter is not None:
        body["filter"] = dict(filter)
    return Call(
        "POST",
        f"/collections/{_seg(name)}/points/metadata",
        lambda b: int(b["updated"]),
        json=body,
    )


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
