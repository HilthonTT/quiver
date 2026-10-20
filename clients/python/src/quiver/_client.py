"""Synchronous client."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

import httpx

from . import _protocol as proto
from ._errors import ConflictError, NotFoundError, QuiverConnectionError
from ._models import (
    CollectionInfo,
    Filter,
    Index,
    Metadata,
    Metric,
    Point,
    PointLike,
    SearchResult,
    SnapshotInfo,
    Vector,
)

T = TypeVar("T")

DEFAULT_URL = "http://localhost:6900"
DEFAULT_TIMEOUT = 30.0


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"user-agent": "quiver-python/0.2.0"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _decode(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


class Client:
    """Synchronous Quiver client.

    >>> client = Client("http://localhost:6900")
    >>> docs = client.get_or_create_collection("docs", dimension=3)
    >>> docs.upsert([("a", [1, 0, 0], {"lang": "en"})])
    1
    >>> docs.search([1, 0, 0], k=1)[0].id
    'a'

    Args:
        url: Server base URL.
        api_key: Sent as ``Authorization: Bearer <api_key>`` if given.
        timeout: Request timeout in seconds.
        http_client: Bring your own :class:`httpx.Client` (``url``, ``api_key``
            and ``timeout`` are then ignored).
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._http = http_client or httpx.Client(
            base_url=url.rstrip("/"), headers=_headers(api_key), timeout=timeout
        )

    # --- plumbing -------------------------------------------------------

    def _run(self, call: proto.Call[T]) -> T:
        try:
            response = self._http.request(
                call.method,
                call.path,
                json=call.json,
                params=call.params,
                content=call.content,
                headers=call.headers(),
            )
        except httpx.TransportError as exc:
            raise QuiverConnectionError(f"could not reach Quiver: {exc}") from exc
        if response.status_code in call.allow:
            return call.parse(None)
        if call.raw and response.status_code < 400:
            return call.parse(response.content)
        body = _decode(response)
        proto.raise_for_response(response.status_code, body, response.text)
        return call.parse(body)

    def close(self) -> None:
        """Closes the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- server & collections -------------------------------------------

    def health(self) -> dict[str, Any]:
        """Returns the server status and version."""
        return self._run(proto.health())

    def info(self) -> dict[str, Any]:
        """Returns server-wide statistics (version, storage engine, totals)."""
        return self._run(proto.info())

    def list_collections(self) -> list[CollectionInfo]:
        """Lists all collections."""
        return self._run(proto.list_collections())

    def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: Metric = "cosine",
        index: Index | None = None,
    ) -> Collection:
        """Creates a collection. Raises :class:`ConflictError` if it exists."""
        self._run(proto.create_collection(name, dimension, metric, index))
        return Collection(self, name)

    def get_collection(self, name: str) -> Collection:
        """Returns a handle to an existing collection. Raises :class:`NotFoundError`."""
        self._run(proto.get_collection(name))
        return Collection(self, name)

    def get_or_create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: Metric = "cosine",
        index: Index | None = None,
    ) -> Collection:
        """Returns the collection, creating it first if it does not exist."""
        try:
            return self.get_collection(name)
        except NotFoundError:
            pass
        try:
            return self.create_collection(name, dimension, metric=metric, index=index)
        except ConflictError:
            return Collection(self, name)

    def collection(self, name: str) -> Collection:
        """Returns a handle without checking that the collection exists."""
        return Collection(self, name)

    def delete_collection(self, name: str) -> None:
        """Deletes a collection and all of its points."""
        self._run(proto.delete_collection(name))

    # --- snapshots --------------------------------------------------------

    def list_snapshots(self, collection: str | None = None) -> list[SnapshotInfo]:
        """Lists snapshots, oldest first, optionally only one collection's."""
        return self._run(proto.list_snapshots(collection))

    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo:
        """Fetches one snapshot's metadata."""
        return self._run(proto.get_snapshot(snapshot_id))

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Deletes a snapshot archive."""
        self._run(proto.delete_snapshot(snapshot_id))

    def restore_snapshot(
        self, snapshot_id: str, *, collection: str | None = None, replace: bool = False
    ) -> Collection:
        """Restores a snapshot as a collection.

        Args:
            collection: Restore under this name instead of the original one.
            replace: Overwrite an existing collection with the same name.
        """
        info = self._run(proto.restore_snapshot(snapshot_id, collection, replace))
        return Collection(self, info.name)

    def download_snapshot(self, snapshot_id: str, path: str | os.PathLike[str]) -> Path:
        """Downloads a snapshot archive (``.tar.gz``) to ``path``."""
        data = self._run(proto.download_snapshot(snapshot_id))
        target = Path(path)
        target.write_bytes(data)
        return target

    def upload_snapshot(self, archive: bytes | str | os.PathLike[str]) -> SnapshotInfo:
        """Uploads a snapshot archive (bytes or a file path) to the server."""
        data = archive if isinstance(archive, bytes) else Path(archive).read_bytes()
        return self._run(proto.upload_snapshot(data))


class Collection:
    """Handle to one collection. Obtain it from a :class:`Client`."""

    def __init__(self, client: Client, name: str) -> None:
        self._client = client
        self.name = name

    def __repr__(self) -> str:
        return f"Collection({self.name!r})"

    def __len__(self) -> int:
        return self.info().count

    def info(self) -> CollectionInfo:
        """Fetches configuration and point count."""
        return self._client._run(proto.get_collection(self.name))

    def insert(
        self, vector: Vector, metadata: Metadata | None = None, *, id: str | None = None
    ) -> str:
        """Inserts one point and returns its id (generated by the server if not given)."""
        point = Point(id or "", vector, metadata).to_dict()
        if not id:
            del point["id"]
        return self._client._run(proto.insert(self.name, point))

    def create_snapshot(self) -> SnapshotInfo:
        """Writes a backup archive of this collection on the server."""
        return self._client._run(proto.create_snapshot(self.name))

    def snapshots(self) -> list[SnapshotInfo]:
        """Lists this collection's snapshots, oldest first."""
        return self._client._run(proto.list_snapshots(self.name))

    def upsert(self, points: Iterable[PointLike], *, batch_size: int = 1000) -> int:
        """Inserts or replaces points, sending them in batches.

        Each item may be a :class:`Point`, a dict, or an ``(id, vector[, metadata])``
        tuple. Vectors can be lists or numpy arrays. Returns the number written.
        """
        return sum(
            self._client._run(proto.upsert(self.name, batch))
            for batch in proto.chunked(points, batch_size)
        )

    def get(self, id: str, *, with_vectors: bool = True) -> Point | None:
        """Fetches one point, or ``None`` if it does not exist."""
        return self._client._run(proto.get_point(self.name, id, with_vectors))

    def get_many(self, ids: Sequence[str], *, with_vectors: bool = True) -> list[Point]:
        """Fetches several points; missing ids are skipped."""
        return self._client._run(proto.lookup(self.name, ids, with_vectors))

    def scroll(self, *, page_size: int = 500, with_vectors: bool = False) -> Iterator[Point]:
        """Iterates over every point in the collection, page by page."""
        offset: int | None = 0
        while offset is not None:
            page, offset = self._client._run(
                proto.scroll(self.name, offset, page_size, with_vectors)
            )
            yield from page

    def delete(self, ids: Sequence[str]) -> int:
        """Deletes points by id. Returns how many existed."""
        return self._client._run(proto.delete_points(self.name, ids))

    def search(
        self,
        vector: Vector,
        k: int = 10,
        *,
        filter: Filter | None = None,
        ef: int | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
    ) -> list[SearchResult]:
        """Finds the ``k`` nearest neighbours of ``vector``.

        Args:
            vector: Query vector (list or numpy array).
            k: Number of results.
            filter: MongoDB-style metadata filter, e.g. ``{"year": {"$gte": 2000}}``.
            ef: HNSW beam width for this query; higher = better recall, slower.
            with_vectors: Include stored vectors in results.
            with_metadata: Include metadata in results.
        """
        body = proto.search_body(vector, k, filter, ef, with_vectors, with_metadata)
        return self._client._run(proto.search(self.name, body))

    def search_batch(
        self,
        vectors: Sequence[Vector],
        k: int = 10,
        *,
        filter: Filter | None = None,
        ef: int | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
    ) -> list[list[SearchResult]]:
        """Runs several searches in one round trip (max 256)."""
        bodies = [proto.search_body(v, k, filter, ef, with_vectors, with_metadata) for v in vectors]
        return self._client._run(proto.search_batch(self.name, bodies))

    def flush(self) -> None:
        """Forces a snapshot to disk and truncates the write-ahead log."""
        self._client._run(proto.flush(self.name))

    def compact(self) -> CollectionInfo:
        """Reclaims space held by deleted points and rebuilds the index."""
        return self._client._run(proto.compact(self.name))
