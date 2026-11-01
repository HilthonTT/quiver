"""Synchronous client."""

from __future__ import annotations

import os
import ssl
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

import httpx

from . import _protocol as proto
from ._errors import ConflictError, NotFoundError, QuiverConnectionError
from ._models import (
    CollectionInfo,
    Datatype,
    Example,
    Expiry,
    FacetValue,
    Filter,
    Fusion,
    Group,
    Index,
    Metadata,
    MetadataMode,
    Metric,
    OrderBy,
    Point,
    PointLike,
    Prefetch,
    Quantization,
    Rescore,
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

DEFAULT_URL = "http://localhost:6900"
DEFAULT_TIMEOUT = 30.0


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"user-agent": "quiver-python/0.2.0"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _verify(
    ca_cert: str | os.PathLike[str] | None,
    client_cert: str | os.PathLike[str] | None,
    client_key: str | os.PathLike[str] | None,
) -> ssl.SSLContext | bool:
    """TLS verification for httpx: the system defaults unless a private CA
    or a client certificate is given."""
    if ca_cert is None and client_cert is None:
        if client_key is not None:
            raise ValueError("client_key needs client_cert")
        return True
    context = ssl.create_default_context(cafile=None if ca_cert is None else os.fspath(ca_cert))
    if client_cert is not None:
        context.load_cert_chain(
            os.fspath(client_cert), None if client_key is None else os.fspath(client_key)
        )
    elif client_key is not None:
        raise ValueError("client_key needs client_cert")
    return context


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
        ca_cert: PEM file of CA certificates to trust for an ``https://`` server
            whose certificate a private CA signed.
        client_cert: PEM client certificate (optionally followed by its key),
            for servers that require one (mutual TLS).
        client_key: PEM private key for ``client_cert``, if kept separately.
        http_client: Bring your own :class:`httpx.Client` (``url``, ``api_key``,
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
        http_client: httpx.Client | None = None,
    ) -> None:
        self._http = http_client or httpx.Client(
            base_url=url.rstrip("/"),
            headers=_headers(api_key),
            timeout=timeout,
            verify=_verify(ca_cert, client_cert, client_key),
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
        if call.raw and response.status_code < 400:
            return call.parse(response.content)
        body = _decode(response)
        if proto.is_allowed(call, response.status_code, body):
            return call.parse(None)
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
        field_indexes: Sequence[str] = (),
        quantization: Quantization = "none",
        on_disk: bool = False,
        vectors: Mapping[str, VectorParams] | None = None,
        sparse_vectors: Mapping[str, SparseVectorParams] | Sequence[str] | None = None,
        expiry: Expiry | None = None,
        datatype: Datatype = "float32",
    ) -> Collection:
        """Creates a collection. Raises :class:`ConflictError` if it exists.

        ``field_indexes`` names metadata fields (dot-separated paths) to index,
        so that filters on them only look at matching points. ``quantization="scalar"``
        stores vectors in about a quarter of the memory; vectors read back are
        then close to, but not exactly, what was written; ``quantization="binary"``
        searches 1-bit codes and re-ranks with the exact vectors. ``on_disk=True``
        keeps the vectors in a file instead of memory. ``datatype="float16"`` or
        ``"uint8"`` stores components in half or a quarter of the memory (see
        :data:`Datatype`).

        ``expiry`` lets points expire at the time in a metadata field (see
        :class:`Expiry`).

        ``vectors`` adds named dense vectors every point must carry, each with
        its own dimension and metric; ``sparse_vectors`` adds named sparse
        vectors (for keyword search) points may carry.
        """
        self._run(
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
                datatype,
            )
        )
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
        field_indexes: Sequence[str] = (),
        quantization: Quantization = "none",
        on_disk: bool = False,
        vectors: Mapping[str, VectorParams] | None = None,
        sparse_vectors: Mapping[str, SparseVectorParams] | Sequence[str] | None = None,
        expiry: Expiry | None = None,
        datatype: Datatype = "float32",
    ) -> Collection:
        """Returns the collection, creating it first if it does not exist."""
        try:
            return self.get_collection(name)
        except NotFoundError:
            pass
        try:
            return self.create_collection(
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
                datatype=datatype,
            )
        except ConflictError:
            return Collection(self, name)

    def collection(self, name: str) -> Collection:
        """Returns a handle without checking that the collection exists."""
        return Collection(self, name)

    def delete_collection(self, name: str) -> None:
        """Deletes a collection and all of its points."""
        self._run(proto.delete_collection(name))

    def rename_collection(self, name: str, new_name: str) -> Collection:
        """Renames a collection (by its own name, not an alias); aliases follow it."""
        info = self._run(proto.rename_collection(name, new_name))
        return Collection(self, info.name)

    def list_aliases(self) -> dict[str, str]:
        """Every alias, mapped to the collection it points to."""
        return self._run(proto.list_aliases())

    def set_alias(self, alias: str, collection: str) -> None:
        """Creates ``alias``, or atomically points it at ``collection``.

        An alias works wherever a collection name does, except for deleting
        and renaming collections.
        """
        self._run(proto.set_alias(alias, collection))

    def delete_alias(self, alias: str) -> None:
        """Deletes an alias; the collection stays."""
        self._run(proto.delete_alias(alias))

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

    def get_many(self, ids: str | Iterable[str], *, with_vectors: bool = True) -> list[Point]:
        """Fetches several points; missing ids are skipped."""
        return self._client._run(proto.lookup(self.name, ids, with_vectors))

    def scroll(
        self,
        cursor: str | None = None,
        *,
        limit: int = 100,
        with_vectors: bool = False,
        filter: Filter | None = None,
        order_by: str | OrderBy | None = None,
    ) -> ScrollPage:
        """Fetches up to ``limit`` points with ids after ``cursor``, in ascending id order.

        Pass the page's ``next_cursor`` to get the next page; it is ``None``
        after the last one. Points changed between pages are never skipped or
        repeated.
        """
        if filter is None and order_by is None:
            return self._client._run(proto.scroll(self.name, cursor, limit, with_vectors))
        order = OrderBy(order_by) if isinstance(order_by, str) else order_by
        call = proto.scroll_with(self.name, cursor, limit, with_vectors, filter, order)
        return self._client._run(call)

    def iter_points(
        self,
        *,
        page_size: int = 500,
        with_vectors: bool = False,
        filter: Filter | None = None,
        order_by: str | OrderBy | None = None,
    ) -> Iterator[Point]:
        """Iterates over every point in the collection in id order, page by page."""
        cursor: str | None = None
        while True:
            page = self.scroll(
                cursor,
                limit=page_size,
                with_vectors=with_vectors,
                filter=filter,
                order_by=order_by,
            )
            yield from page.points
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def delete(self, ids: str | Iterable[str]) -> int:
        """Deletes points by id. Returns how many existed."""
        return self._client._run(proto.delete_points(self.name, ids))

    def count(self, filter: Filter | None = None) -> int:
        """Counts the points, or only those whose metadata matches ``filter``."""
        return self._client._run(proto.count(self.name, filter))

    def delete_where(self, filter: Filter) -> int:
        """Deletes every point whose metadata matches ``filter``. Returns how many.

        An empty filter is rejected; delete the collection to remove everything.
        """
        return self._client._run(proto.delete_where(self.name, filter))

    def update_metadata(
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
        return self._client._run(call)

    def search(
        self,
        vector: Vector,
        k: int = 10,
        *,
        offset: int = 0,
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
            offset=offset,
        )
        return self._client._run(proto.search(self.name, body))

    def search_by_id(
        self,
        id: str,
        k: int = 10,
        *,
        offset: int = 0,
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
            k, filter, ef, max_distance, with_vectors, with_metadata, using, offset
        )
        body = proto.search_by_id_body(id, options)
        return self._client._run(proto.search(self.name, body))

    def recommend(
        self,
        positive: str | Sequence[str],
        negative: str | Sequence[str] = (),
        k: int = 10,
        *,
        offset: int = 0,
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
            k, filter, ef, max_distance, with_vectors, with_metadata, using, offset
        )
        return self._client._run(proto.recommend(self.name, positive, negative, options))

    def search_batch(
        self,
        vectors: Sequence[Vector],
        k: int = 10,
        *,
        offset: int = 0,
        filter: Filter | None = None,
        ef: int | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
        using: str | None = None,
    ) -> list[list[SearchResult]]:
        """Runs several searches in one round trip (max 256)."""
        bodies = [
            proto.search_body(
                v, k, filter, ef, with_vectors, with_metadata, using=using, offset=offset
            )
            for v in vectors
        ]
        return self._client._run(proto.search_batch(self.name, bodies))

    def search_sparse(
        self,
        vector: SparseLike,
        k: int = 10,
        *,
        offset: int = 0,
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
            k, filter, None, max_distance, with_vectors, with_metadata, using, offset
        )
        body = proto.sparse_search_body(vector, options)
        return self._client._run(proto.search(self.name, body))

    def query(
        self,
        prefetch: Sequence[Prefetch],
        k: int = 10,
        *,
        offset: int = 0,
        filter: Filter | None = None,
        rrf_k: int = 60,
        fusion: Fusion = "rrf",
        rescore: Rescore | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
    ) -> list[ScoredResult]:
        """Hybrid search: runs each :class:`Prefetch` search and fuses their rankings.

        Reciprocal rank fusion scores a point at rank ``r`` (from 1) of a search
        ``weight / (rrf_k + r)`` there, summed over the searches.

        >>> docs.query([Prefetch(vector=embedding), Prefetch(sparse={17: 1.0}, using="text")])
        """
        call = proto.query(
            self.name,
            prefetch,
            k,
            filter,
            rrf_k,
            with_vectors,
            with_metadata,
            fusion=fusion,
            rescore=rescore,
            offset=offset,
        )
        return self._client._run(call)

    def facet(
        self, field: str, *, filter: Filter | None = None, limit: int = 10
    ) -> list[FacetValue]:
        """Counts the points per value of ``field``, most common first."""
        return self._client._run(proto.facet(self.name, field, filter, limit))

    def search_groups(
        self,
        vector: Vector | None = None,
        *,
        group_by: str,
        groups: int = 10,
        group_size: int = 3,
        id: str | None = None,
        sparse: SparseLike | None = None,
        using: str | None = None,
        filter: Filter | None = None,
        ef: int | None = None,
        max_distance: float | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
    ) -> list[Group]:
        """The best ``group_size`` hits for each of the ``groups`` best ``group_by`` values."""
        target = proto.search_target(vector, id, sparse)
        options = proto.search_options(
            1, filter, ef, max_distance, with_vectors, with_metadata, using
        )
        call = proto.search_groups(self.name, target, group_by, groups, group_size, options)
        return self._client._run(call)

    def discover(
        self,
        target: Example | None = None,
        context: Sequence[tuple[Example, Example]] = (),
        k: int = 10,
        *,
        offset: int = 0,
        filter: Filter | None = None,
        ef: int | None = None,
        max_distance: float | None = None,
        with_vectors: bool = False,
        with_metadata: bool = True,
        using: str | None = None,
    ) -> list[SearchResult]:
        """Finds points on the positive side of each context pair, nearest ``target`` first."""
        options = proto.search_options(
            k, filter, ef, max_distance, with_vectors, with_metadata, using, offset
        )
        return self._client._run(proto.discover(self.name, target, context, options))

    def update(
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
        return self._client._run(call)

    def rename(self, new_name: str) -> None:
        """Renames this collection; this handle follows it."""
        info = self._client._run(proto.rename_collection(self.name, new_name))
        self.name = info.name

    def flush(self) -> None:
        """Forces a snapshot to disk and truncates the write-ahead log."""
        self._client._run(proto.flush(self.name))

    def compact(self) -> CollectionInfo:
        """Reclaims space held by deleted points and rebuilds the index."""
        return self._client._run(proto.compact(self.name))

    def create_field_index(self, field: str) -> CollectionInfo:
        """Indexes a metadata field (a dot-separated path, as in filters).

        Filters on indexed fields only look at matching points. Indexing a
        field that already has an index does nothing.
        """
        return self._client._run(proto.create_field_index(self.name, field))

    def drop_field_index(self, field: str) -> CollectionInfo:
        """Drops a field index. Raises :class:`NotFoundError` if there is none."""
        return self._client._run(proto.drop_field_index(self.name, field))
