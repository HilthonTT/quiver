"""Asynchronous client (mirrors :mod:`quiver._client`)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

import httpx

from . import _protocol as proto
from ._client import DEFAULT_TIMEOUT, DEFAULT_URL, _decode, _headers
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


class AsyncClient:
    """Asynchronous Quiver client.

    >>> async with AsyncClient("http://localhost:6900") as client:
    ...     docs = await client.get_or_create_collection("docs", dimension=3)
    ...     await docs.upsert([("a", [1, 0, 0])])
    ...     hits = await docs.search([1, 0, 0], k=1)

    Args:
        url: Server base URL.
        api_key: Sent as ``Authorization: Bearer <api_key>`` if given.
        timeout: Request timeout in seconds.
        http_client: Bring your own :class:`httpx.AsyncClient` (``url``, ``api_key``
            and ``timeout`` are then ignored).
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._http = http_client or httpx.AsyncClient(
            base_url=url.rstrip("/"), headers=_headers(api_key), timeout=timeout
        )

    # --- plumbing -------------------------------------------------------

    async def _run(self, call: proto.Call[T]) -> T:
        try:
            response = await self._http.request(
                call.method,
                call.path,
                json=call.json,
                params=call.params,
                content=call.content,
                headers=call.headers(),
            )
        except httpx.TransportError as exc:
            raise QuiverConnectionError(f"could not reach Quiver: {exc}") from exc
        if call.raw and response.status_code < 400:
            return call.parse(response.content)
        body = _decode(response)
        if proto.is_allowed(call, response.status_code, body):
            return call.parse(None)
        proto.raise_for_response(response.status_code, body, response.text)
        return call.parse(body)

    async def aclose(self) -> None:
        """Closes the underlying HTTP connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # --- server & collections -------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Returns the server status and version."""
        return await self._run(proto.health())

    async def info(self) -> dict[str, Any]:
        """Returns server-wide statistics (version, storage engine, totals)."""
        return await self._run(proto.info())

    async def list_collections(self) -> list[CollectionInfo]:
        """Lists all collections."""
        return await self._run(proto.list_collections())

    async def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: Metric = "cosine",
        index: Index | None = None,
    ) -> AsyncCollection:
        """Creates a collection. Raises :class:`ConflictError` if it exists."""
        await self._run(proto.create_collection(name, dimension, metric, index))
        return AsyncCollection(self, name)

    async def get_collection(self, name: str) -> AsyncCollection:
        """Returns a handle to an existing collection. Raises :class:`NotFoundError`."""
        await self._run(proto.get_collection(name))
        return AsyncCollection(self, name)

    async def get_or_create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: Metric = "cosine",
        index: Index | None = None,
    ) -> AsyncCollection:
        """Returns the collection, creating it first if it does not exist."""
        try:
            return await self.get_collection(name)
        except NotFoundError:
            pass
        try:
            return await self.create_collection(name, dimension, metric=metric, index=index)
        except ConflictError:
            return AsyncCollection(self, name)

    async def collection(self, name: str) -> AsyncCollection:
        """Returns a handle without checking that the collection exists."""
        return AsyncCollection(self, name)

    async def delete_collection(self, name: str) -> None:
        """Deletes a collection and all of its points."""
        await self._run(proto.delete_collection(name))

    # --- snapshots --------------------------------------------------------

    async def list_snapshots(self, collection: str | None = None) -> list[SnapshotInfo]:
        """Lists snapshots, oldest first, optionally only one collection's."""
        return await self._run(proto.list_snapshots(collection))

    async def get_snapshot(self, snapshot_id: str) -> SnapshotInfo:
        """Fetches one snapshot's metadata."""
        return await self._run(proto.get_snapshot(snapshot_id))

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Deletes a snapshot archive."""
        await self._run(proto.delete_snapshot(snapshot_id))

    async def restore_snapshot(
        self, snapshot_id: str, *, collection: str | None = None, replace: bool = False
    ) -> AsyncCollection:
        """Restores a snapshot as a collection.

        Args:
            collection: Restore under this name instead of the original one.
            replace: Overwrite an existing collection with the same name.
        """
        info = await self._run(proto.restore_snapshot(snapshot_id, collection, replace))
        return AsyncCollection(self, info.name)

    async def download_snapshot(self, snapshot_id: str, path: str | os.PathLike[str]) -> Path:
        """Downloads a snapshot archive (``.tar.gz``) to ``path``."""
        data = await self._run(proto.download_snapshot(snapshot_id))
        target = Path(path)
        target.write_bytes(data)
        return target

    async def upload_snapshot(self, archive: bytes | str | os.PathLike[str]) -> SnapshotInfo:
        """Uploads a snapshot archive (bytes or a file path) to the server."""
        data = archive if isinstance(archive, bytes) else Path(archive).read_bytes()
        return await self._run(proto.upload_snapshot(data))


class AsyncCollection:
    """Handle to one collection. Obtain it from an :class:`AsyncClient`."""

    def __init__(self, client: AsyncClient, name: str) -> None:
        self._client = client
        self.name = name

    def __repr__(self) -> str:
        return f"AsyncCollection({self.name!r})"

    async def count(self) -> int:
        """Number of points in the collection."""
        return (await self.info()).count

    async def info(self) -> CollectionInfo:
        """Fetches configuration and point count."""
        return await self._client._run(proto.get_collection(self.name))

    async def insert(
        self, vector: Vector, metadata: Metadata | None = None, *, id: str | None = None
    ) -> str:
        """Inserts one point and returns its id (generated by the server if not given)."""
        point = Point(id or "", vector, metadata).to_dict()
        if not id:
            del point["id"]
        return await self._client._run(proto.insert(self.name, point))

    async def create_snapshot(self) -> SnapshotInfo:
        """Writes a backup archive of this collection on the server."""
        return await self._client._run(proto.create_snapshot(self.name))

    async def snapshots(self) -> list[SnapshotInfo]:
        """Lists this collection's snapshots, oldest first."""
        return await self._client._run(proto.list_snapshots(self.name))

    async def upsert(self, points: Iterable[PointLike], *, batch_size: int = 1000) -> int:
        """Inserts or replaces points, sending them in batches.

        Each item may be a :class:`Point`, a dict, or an ``(id, vector[, metadata])``
        tuple. Vectors can be lists or numpy arrays. Returns the number written.
        """
        total = 0
        for batch in proto.chunked(points, batch_size):
            total += await self._client._run(proto.upsert(self.name, batch))
        return total

    async def get(self, id: str, *, with_vectors: bool = True) -> Point | None:
        """Fetches one point, or ``None`` if it does not exist."""
        return await self._client._run(proto.get_point(self.name, id, with_vectors))

    async def get_many(self, ids: Sequence[str], *, with_vectors: bool = True) -> list[Point]:
        """Fetches several points; missing ids are skipped."""
        return await self._client._run(proto.lookup(self.name, ids, with_vectors))

    async def scroll(
        self, *, page_size: int = 500, with_vectors: bool = False
    ) -> AsyncIterator[Point]:
        """Iterates over every point in the collection, page by page."""
        offset: int | None = 0
        while offset is not None:
            page, offset = await self._client._run(
                proto.scroll(self.name, offset, page_size, with_vectors)
            )
            for point in page:
                yield point

    async def delete(self, ids: Sequence[str]) -> int:
        """Deletes points by id. Returns how many existed."""
        return await self._client._run(proto.delete_points(self.name, ids))

    async def search(
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
        return await self._client._run(proto.search(self.name, body))

    async def search_batch(
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
        return await self._client._run(proto.search_batch(self.name, bodies))

    async def flush(self) -> None:
        """Forces a snapshot to disk and truncates the write-ahead log."""
        await self._client._run(proto.flush(self.name))

    async def compact(self) -> CollectionInfo:
        """Reclaims space held by deleted points and rebuilds the index."""
        return await self._client._run(proto.compact(self.name))
