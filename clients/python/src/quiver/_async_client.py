"""Asynchronous client (mirrors :mod:`quiver._client`)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

import httpx

from . import _protocol as proto
from ._client import DEFAULT_TIMEOUT, DEFAULT_URL, _decode, _headers, _verify
from ._errors import ConflictError, NotFoundError, QuiverConnectionError
from ._models import (
    CollectionInfo,
    Expiry,
    Filter,
    Index,
    Metadata,
    MetadataMode,
    Metric,
    Point,
    PointLike,
    Prefetch,
    Quantization,
    ScoredResult,
    ScrollPage,
    SearchResult,
    SnapshotInfo,
    SparseLike,
    SparseVectorParams,
    Vector,
    VectorParams,
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
        ca_cert: PEM file of CA certificates to trust for an ``https://`` server
            whose certificate a private CA signed.
        client_cert: PEM client certificate (optionally followed by its key),
            for servers that require one (mutual TLS).
        client_key: PEM private key for ``client_cert``, if kept separately.
        http_client: Bring your own :class:`httpx.AsyncClient` (``url``, ``api_key``,
            ``timeout`` and the TLS options are then ignored).
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        ca_cert: str | os.PathLike[str] | None = None,
        client_cert: str | os.PathLike[str] | None = None,
        client_key: str | os.PathLike[str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._http = http_client or httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers=_headers(api_key),
            timeout=timeout,
            verify=_verify(ca_cert, client_cert, client_key),
        )

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
        field_indexes: Sequence[str] = (),
        quantization: Quantization = "none",
        on_disk: bool = False,
        vectors: Mapping[str, VectorParams] | None = None,
        sparse_vectors: Mapping[str, SparseVectorParams] | Sequence[str] | None = None,
        expiry: Expiry | None = None,
    ) -> AsyncCollection:
        """Creates a collection. Raises :class:`ConflictError` if it exists.

        ``field_indexes`` names metadata fields (dot-separated paths) to index,
        so that filters on them only look at matching points. ``quantization="scalar"``
        stores vectors in about a quarter of the memory; vectors read back are
        then close to, but not exactly, what was written; ``quantization="binary"``
        searches 1-bit codes and re-ranks with the exact vectors. ``on_disk=True``
        keeps the vectors in a file instead of memory.

        ``expiry`` lets points expire at the time in a metadata field (see
        :class:`Expiry`).

        ``vectors`` adds named dense vectors every point must carry, each with
        its own dimension and metric; ``sparse_vectors`` adds named sparse
        vectors (for keyword search) points may carry.
        """
        await self._run(
            proto.create_collection(
                name,
                dimension,
                metric,
                index,
                field_indexes,
                quantization,
                on_disk,
                vectors,
                sparse_vectors,
                expiry,
            )
        )
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
        field_indexes: Sequence[str] = (),
        quantization: Quantization = "none",
        on_disk: bool = False,
        vectors: Mapping[str, VectorParams] | None = None,
        sparse_vectors: Mapping[str, SparseVectorParams] | Sequence[str] | None = None,
        expiry: Expiry | None = None,
    ) -> AsyncCollection:
        """Returns the collection, creating it first if it does not exist."""
        try:
            return await self.get_collection(name)
        except NotFoundError:
            pass
        try:
            return await self.create_collection(
                name,
                dimension,
                metric=metric,
                index=index,
                field_indexes=field_indexes,
                quantization=quantization,
                on_disk=on_disk,
                vectors=vectors,
                sparse_vectors=sparse_vectors,
                expiry=expiry,
            )
        except ConflictError:
            return AsyncCollection(self, name)

    def collection(self, name: str) -> AsyncCollection:
        """Returns a handle without checking that the collection exists."""
        return AsyncCollection(self, name)

    async def delete_collection(self, name: str) -> None:
        """Deletes a collection and all of its points."""
        await self._run(proto.delete_collection(name))

    async def rename_collection(self, name: str, new_name: str) -> AsyncCollection:
        """Renames a collection (by its own name, not an alias); aliases follow it."""
        info = await self._run(proto.rename_collection(name, new_name))
        return AsyncCollection(self, info.name)

    async def list_aliases(self) -> dict[str, str]:
        """Every alias, mapped to the collection it points to."""
        return await self._run(proto.list_aliases())

    async def set_alias(self, alias: str, collection: str) -> None:
        """Creates ``alias``, or atomically points it at ``collection``.

        An alias works wherever a collection name does, except for deleting
        and renaming collections.
        """
        await self._run(proto.set_alias(alias, collection))

    async def delete_alias(self, alias: str) -> None:
        """Deletes an alias; the collection stays."""
        await self._run(proto.delete_alias(alias))

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
        self, cursor: str | None = None, *, limit: int = 100, with_vectors: bool = False
    ) -> ScrollPage:
        """Fetches up to ``limit`` points with ids after ``cursor``, in ascending id order.

        Pass the page's ``next_cursor`` to get the next page; it is ``None``
        after the last one. Points changed between pages are never skipped or
        repeated.
        """
        return await self._client._run(proto.scroll(self.name, cursor, limit, with_vectors))

    async def iter_points(
        self, *, page_size: int = 500, with_vectors: bool = False
    ) -> AsyncIterator[Point]:
        """Iterates over every point in the collection in id order, page by page."""
        cursor: str | None = None
        while True:
            page = await self.scroll(cursor, limit=page_size, with_vectors=with_vectors)
            for point in page.points:
                yield point
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    async def delete(self, ids: Sequence[str]) -> int:
        """Deletes points by id. Returns how many existed."""
        return await self._client._run(proto.delete_points(self.name, ids))

    async def count(self, filter: Filter | None = None) -> int:
        """Counts the points, or only those whose metadata matches ``filter``."""
        return await self._client._run(proto.count(self.name, filter))

    async def delete_where(self, filter: Filter) -> int:
        """Deletes every point whose metadata matches ``filter``. Returns how many.

        An empty filter is rejected; delete the collection to remove everything.
        """
        return await self._client._run(proto.delete_where(self.name, filter))

    async def update_metadata(
        self,
        metadata: Any,
        *,
        ids: str | Sequence[str] | None = None,
        filter: Filter | None = None,
        mode: MetadataMode = "merge",
    ) -> int:
        """Changes metadata without resending vectors. Returns how many points changed.

        Select points with exactly one of ``ids`` (missing ones are skipped) and
        ``filter``. ``mode="merge"`` applies ``metadata`` as a JSON Merge Patch
        (RFC 7386: nested objects merge, ``None`` removes a key);
        ``mode="replace"`` replaces the whole value (``None`` removes it).
        """
        call = proto.update_metadata(self.name, metadata, ids, filter, mode)
        return await self._client._run(call)

    async def search(
        self,
        vector: Vector,
        k: int = 10,
        *,
        filter: Filter | None = None,
        ef: int | None = None,
        max_distance: float | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
        using: str | None = None,
    ) -> list[SearchResult]:
        """Finds the ``k`` nearest neighbours of ``vector``.

        Args:
            vector: Query vector (list or numpy array).
            k: Number of results.
            filter: MongoDB-style metadata filter, e.g. ``{"year": {"$gte": 2000}}``.
            ef: HNSW beam width for this query; higher = better recall, slower.
            max_distance: Drop hits farther than this. With HNSW it applies to
                the approximate top ``k``.
            with_vectors: Include stored vectors in results.
            with_metadata: Include metadata in results.
            using: Search this named vector instead of the main one.
        """
        body = proto.search_body(
            vector,
            k,
            filter,
            ef,
            with_vectors,
            with_metadata,
            max_distance=max_distance,
            using=using,
        )
        return await self._client._run(proto.search(self.name, body))

    async def search_by_id(
        self,
        id: str,
        k: int = 10,
        *,
        filter: Filter | None = None,
        ef: int | None = None,
        max_distance: float | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
        using: str | None = None,
    ) -> list[SearchResult]:
        """Finds the ``k`` nearest neighbours of point ``id``, leaving it out.

        With ``using``, compares that named vector of the points.
        Raises :class:`NotFoundError` (code ``point_not_found``) if it does not exist.
        """
        options = proto.search_options(
            k, filter, ef, max_distance, with_vectors, with_metadata, using
        )
        body = proto.search_by_id_body(id, options)
        return await self._client._run(proto.search(self.name, body))

    async def recommend(
        self,
        positive: str | Sequence[str],
        negative: str | Sequence[str] = (),
        k: int = 10,
        *,
        filter: Filter | None = None,
        ef: int | None = None,
        max_distance: float | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
        using: str | None = None,
    ) -> list[SearchResult]:
        """Finds points like the ``positive`` example ids and unlike the ``negative`` ones.

        The query is the mean of the positives (in the named vector ``using``
        if given), pushed away from the mean of the negatives. The examples
        themselves are never returned.
        """
        options = proto.search_options(
            k, filter, ef, max_distance, with_vectors, with_metadata, using
        )
        return await self._client._run(proto.recommend(self.name, positive, negative, options))

    async def search_batch(
        self,
        vectors: Sequence[Vector],
        k: int = 10,
        *,
        filter: Filter | None = None,
        ef: int | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
        using: str | None = None,
    ) -> list[list[SearchResult]]:
        """Runs several searches in one round trip (max 256)."""
        bodies = [
            proto.search_body(v, k, filter, ef, with_vectors, with_metadata, using=using)
            for v in vectors
        ]
        return await self._client._run(proto.search_batch(self.name, bodies))

    async def search_sparse(
        self,
        vector: SparseLike,
        k: int = 10,
        *,
        using: str,
        filter: Filter | None = None,
        max_distance: float | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
    ) -> list[SearchResult]:
        """Finds the ``k`` points scoring highest against a sparse vector.

        Args:
            vector: A :class:`SparseVector` or an ``{index: value}`` mapping.
            using: The sparse vector of the points to score against.

        Scores are dot products (IDF-weighted if the sparse vector's modifier
        is ``"idf"``); each hit's ``distance`` is its negated score.
        """
        options = proto.search_options(
            k, filter, None, max_distance, with_vectors, with_metadata, using
        )
        body = proto.sparse_search_body(vector, options)
        return await self._client._run(proto.search(self.name, body))

    async def query(
        self,
        prefetch: Sequence[Prefetch],
        k: int = 10,
        *,
        filter: Filter | None = None,
        rrf_k: int = 60,
        with_vectors: bool = False,
        with_metadata: bool = True,
    ) -> list[ScoredResult]:
        """Hybrid search: runs each :class:`Prefetch` search and fuses their rankings.

        Reciprocal rank fusion scores a point at rank ``r`` (from 1) of a search
        ``weight / (rrf_k + r)`` there, summed over the searches.

        >>> docs.query([Prefetch(vector=embedding), Prefetch(sparse={17: 1.0}, using="text")])
        """
        call = proto.query(self.name, prefetch, k, filter, rrf_k, with_vectors, with_metadata)
        return await self._client._run(call)

    async def update(
        self,
        *,
        index: Index | Mapping[str, Any] | None = None,
        quantization: Quantization | None = None,
        on_disk: bool | None = None,
        vectors: Mapping[str, Mapping[str, Any]] | None = None,
        sparse_vectors: Mapping[str, SparseVectorParams] | None = None,
        expiry: Expiry | Literal[False] | None = None,
    ) -> CollectionInfo:
        """Changes index, quantization, on-disk or expiry settings; may take a while.

        ``index`` is an index object (replacing the current settings) or a
        dict merged into them, e.g. ``{"ef_search": 128}``. ``vectors`` maps
        named vectors to dicts of the same keys. Anything but an ``ef_search``
        change rebuilds the index. ``expiry`` replaces the expiry settings,
        and ``expiry=False`` turns expiry off; points keep their expiry times.
        """
        call = proto.update_collection(
            self.name, index, quantization, on_disk, vectors, sparse_vectors, expiry
        )
        return await self._client._run(call)

    async def rename(self, new_name: str) -> None:
        """Renames this collection; this handle follows it."""
        info = await self._client._run(proto.rename_collection(self.name, new_name))
        self.name = info.name

    async def flush(self) -> None:
        """Forces a snapshot to disk and truncates the write-ahead log."""
        await self._client._run(proto.flush(self.name))

    async def compact(self) -> CollectionInfo:
        """Reclaims space held by deleted points and rebuilds the index."""
        return await self._client._run(proto.compact(self.name))

    async def create_field_index(self, field: str) -> CollectionInfo:
        """Indexes a metadata field (a dot-separated path, as in filters).

        Filters on indexed fields only look at matching points. Indexing a
        field that already has an index does nothing.
        """
        return await self._client._run(proto.create_field_index(self.name, field))

    async def drop_field_index(self, field: str) -> CollectionInfo:
        """Drops a field index. Raises :class:`NotFoundError` if there is none."""
        return await self._client._run(proto.drop_field_index(self.name, field))
