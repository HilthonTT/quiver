"""Data types exchanged with the server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

Metric = Literal["cosine", "euclidean", "dot_product", "manhattan", "hamming"]
"""Similarity metric. Distances are always "lower is closer":

* ``cosine`` → ``1 - cosine_similarity`` (vectors are stored normalised)
* ``euclidean`` → L2 distance
* ``dot_product`` → ``-(a · b)``
* ``manhattan`` → L1 distance
* ``hamming`` → number of differing components (for binary/categorical vectors)
"""


MetadataMode = Literal["merge", "replace"]
"""How :meth:`Collection.update_metadata` applies new metadata:

* ``merge`` → RFC 7386 JSON Merge Patch: nested objects merge, ``None`` removes a key
* ``replace`` → the whole value is replaced (``None`` removes the metadata)
"""

Quantization = Literal["none", "scalar", "binary"]
"""How vectors are held in memory:

* ``none`` → full 32-bit floats
* ``scalar`` → 8-bit codes, about a quarter of the memory; lossy
* ``binary`` → 1 bit per component, searched instead of the vectors, which are
  kept exactly and re-rank the results (combine with ``on_disk=True`` to save
  memory); for high-dimensional embeddings
"""

SparseModifier = Literal["none", "idf"]
"""How sparse query weights are adjusted before scoring:

* ``none`` → scores are plain dot products
* ``idf`` → each query weight is multiplied by its index's inverse document
  frequency in the collection (with term frequencies as values: BM25)
"""


class ArrayLike(Protocol):
    """Anything with a numpy-style ``tolist()`` method."""

    def tolist(self) -> Any: ...


Vector = Sequence[float] | ArrayLike
"""A list/tuple of floats or a 1-D numpy array."""

Metadata = dict[str, Any]
Filter = Mapping[str, Any]


@dataclass(frozen=True)
class HNSW:
    """Approximate nearest-neighbour index (the default).

    Args:
        m: Max neighbours per graph node. Higher = better recall, more memory.
        ef_construction: Beam width while building. Higher = better graph, slower inserts.
        ef_search: Default beam width while searching (override per query with ``ef=``).
        seed: Seed for reproducible graphs.
    """

    m: int = 16
    ef_construction: int = 128
    ef_search: int = 64
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": "hnsw",
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        return body


@dataclass(frozen=True)
class Flat:
    """Exact brute-force index. Best for small collections or perfect recall."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "flat"}


@dataclass(frozen=True)
class KDTree:
    """Exact k-d tree index. Fast for low-dimensional data (roughly < 20 dims).

    Not available with the ``dot_product`` metric.

    Args:
        balance_factor: The tree is rebuilt once its depth exceeds
            ``balance_factor * log2(n)``.
    """

    balance_factor: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {"type": "kd_tree", "balance_factor": self.balance_factor}


Index = HNSW | Flat | KDTree


@dataclass(frozen=True)
class VectorParams:
    """Settings of a named dense vector.

    Args:
        dimension: Number of components.
        metric: Distance metric.
        index: Index type (HNSW by default).
        quantization: ``"scalar"`` keeps 8-bit codes, about a quarter of the memory;
            ``"binary"`` searches 1-bit codes and re-ranks with the exact vectors.
        on_disk: Keep the vectors in a file instead of memory (durable storage only).
    """

    dimension: int
    metric: Metric = "cosine"
    index: Index | None = None
    quantization: Quantization = "none"
    on_disk: bool = False

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "dimension": self.dimension,
            "metric": self.metric,
            "index": (self.index or HNSW()).to_dict(),
        }
        if self.quantization != "none":
            body["quantization"] = self.quantization
        if self.on_disk:
            body["on_disk"] = True
        return body


@dataclass(frozen=True)
class Expiry:
    """Lets a collection's points expire.

    A point expires once the number in its metadata field ``field`` (a Unix
    time in seconds; a dot-separated path, as in filters) is not after the
    current time. Expired points disappear from every read at once and are
    deleted by a periodic sweep.

    Args:
        field: The metadata field holding each point's expiry time.
        default_ttl: Seconds to keep points written without ``field``: they
            get it set to the time they were written plus this. Without it,
            such points never expire.
    """

    field: str
    default_ttl: int | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"field": self.field}
        if self.default_ttl is not None:
            body["default_ttl"] = self.default_ttl
        return body

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Expiry:
        return cls(field=data["field"], default_ttl=data.get("default_ttl"))


@dataclass(frozen=True)
class SparseVectorParams:
    """Settings of a named sparse vector."""

    modifier: SparseModifier = "none"

    def to_dict(self) -> dict[str, Any]:
        return {} if self.modifier == "none" else {"modifier": self.modifier}


@dataclass(frozen=True)
class SparseVector:
    """A vector given by its non-zero components: ``values[i]`` at ``indices[i]``.

    Typically term weights, one index per vocabulary term.
    """

    indices: list[int]
    values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": [int(i) for i in self.indices],
            "values": [float(v) for v in self.values],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SparseVector:
        return cls(indices=list(data["indices"]), values=list(data["values"]))


SparseLike = SparseVector | Mapping[int, float]
"""A :class:`SparseVector`, or a ``{index: value}`` mapping."""


def as_sparse(vector: SparseLike) -> dict[str, Any]:
    if isinstance(vector, SparseVector):
        return vector.to_dict()
    return SparseVector(list(vector.keys()), list(vector.values())).to_dict()


@dataclass
class Point:
    """A vector with an id, optional JSON-serialisable metadata, and the
    collection's named dense and sparse vectors, if it has any."""

    id: str
    vector: Vector
    metadata: Metadata | None = None
    vectors: dict[str, Vector] = field(default_factory=dict)
    sparse_vectors: dict[str, SparseLike] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"id": self.id, "vector": as_list(self.vector)}
        if self.metadata is not None:
            body["metadata"] = self.metadata
        if self.vectors:
            body["vectors"] = {n: as_list(v) for n, v in self.vectors.items()}
        if self.sparse_vectors:
            body["sparse_vectors"] = {n: as_sparse(v) for n, v in self.sparse_vectors.items()}
        return body

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Point:
        return cls(
            id=data["id"],
            vector=data.get("vector", []),
            metadata=data.get("metadata"),
            vectors=dict(data.get("vectors", {})),
            sparse_vectors={
                n: SparseVector.from_dict(v) for n, v in data.get("sparse_vectors", {}).items()
            },
        )


PointLike = Point | Mapping[str, Any] | tuple[str, Vector] | tuple[str, Vector, Metadata | None]
"""Anything :meth:`Collection.upsert` accepts: a :class:`Point`, a dict with
``vector`` and optional ``id``/``metadata``/``vectors``/``sparse_vectors`` keys
(a missing id is generated by the server), or an ``(id, vector[, metadata])`` tuple."""


@dataclass(frozen=True)
class ScrollPage:
    """One page of :meth:`Collection.scroll`: points in ascending id order."""

    points: list[Point]
    next_cursor: str | None = None
    """Pass as ``cursor=`` to fetch the next page; ``None`` after the last page."""


def _named(data: Mapping[str, Any]) -> dict[str, list[float]]:
    return dict(data.get("vectors", {}))


def _sparse(data: Mapping[str, Any]) -> dict[str, SparseVector]:
    return {n: SparseVector.from_dict(v) for n, v in data.get("sparse_vectors", {}).items()}


@dataclass(frozen=True)
class SearchResult:
    """One search hit. For sparse searches, ``distance`` is the negated score."""

    id: str
    distance: float
    metadata: Metadata | None = None
    vector: list[float] | None = None
    vectors: dict[str, list[float]] = field(default_factory=dict)
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchResult:
        return cls(
            id=data["id"],
            distance=data["distance"],
            metadata=data.get("metadata"),
            vector=data.get("vector"),
            vectors=_named(data),
            sparse_vectors=_sparse(data),
        )


@dataclass(frozen=True)
class ScoredResult:
    """One result of a hybrid :meth:`Collection.query`; higher scores are better."""

    id: str
    score: float
    metadata: Metadata | None = None
    vector: list[float] | None = None
    vectors: dict[str, list[float]] = field(default_factory=dict)
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoredResult:
        return cls(
            id=data["id"],
            score=data["score"],
            metadata=data.get("metadata"),
            vector=data.get("vector"),
            vectors=_named(data),
            sparse_vectors=_sparse(data),
        )


@dataclass(frozen=True)
class Prefetch:
    """One search whose ranking :meth:`Collection.query` fuses with the others.

    Give exactly one of ``vector`` (dense) and ``sparse``.

    Args:
        vector: A dense query vector.
        sparse: A sparse query vector; ``using`` must then name the sparse vector.
        using: The named vector to search (the main one if ``None``, for dense).
        limit: How many of its best hits take part (server default: ``max(k, 50)``).
        weight: Multiplies its contribution to the fused score.
        ef: HNSW beam width.
        max_distance: Hits farther than this take no part.
    """

    vector: Vector | None = None
    sparse: SparseLike | None = None
    using: str | None = None
    limit: int | None = None
    weight: float = 1.0
    ef: int | None = None
    max_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        if (self.vector is None) == (self.sparse is None):
            raise ValueError("give a Prefetch exactly one of vector= and sparse=")
        body: dict[str, Any] = {"weight": self.weight}
        if self.vector is not None:
            body["vector"] = as_list(self.vector)
        if self.sparse is not None:
            body["sparse"] = as_sparse(self.sparse)
        for key in ("using", "limit", "ef", "max_distance"):
            value = getattr(self, key)
            if value is not None:
                body[key] = value
        return body


Fusion = Literal["rrf", "relative_score", "dbsf"]
"""How :meth:`Collection.query` combines its searches."""

Direction = Literal["asc", "desc"]

Example = str | Vector
"""A point id, or a dense vector."""


def as_example(example: Example) -> str | list[float]:
    return example if isinstance(example, str) else as_list(example)


@dataclass(frozen=True)
class Rescore:
    """Re-ranks the points :meth:`Collection.query` gathered by their distance to ``vector``."""

    vector: Vector
    using: str | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"vector": as_list(self.vector)}
        if self.using is not None:
            body["using"] = self.using
        return body


@dataclass(frozen=True)
class OrderBy:
    """Orders :meth:`Collection.scroll` by a metadata field, then by id."""

    field: str
    direction: Direction = "asc"

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "direction": self.direction}


@dataclass(frozen=True)
class Group:
    """The best hits sharing one value of the field :meth:`Collection.search_groups` groups by."""

    key: Any
    hits: list[SearchResult]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Group:
        return cls(key=data["key"], hits=[SearchResult.from_dict(h) for h in data["hits"]])


@dataclass(frozen=True)
class FacetValue:
    """How many points have one value of the field :meth:`Collection.facet` counts."""

    value: Any
    count: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FacetValue:
        return cls(value=data["value"], count=int(data["count"]))


@dataclass(frozen=True)
class CollectionInfo:
    """Configuration and statistics of a collection."""

    name: str
    dimension: int
    metric: str
    index: dict[str, Any] = field(default_factory=dict)
    count: int = 0
    tombstones: int = 0
    field_indexes: list[str] = field(default_factory=list)
    quantization: str = "none"
    on_disk: bool = False
    vectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Named dense vectors' settings, by name."""
    sparse_vectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Named sparse vectors' settings, by name."""
    expiry: Expiry | None = None
    """How points expire, if they do."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CollectionInfo:
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            metric=data["metric"],
            index=dict(data.get("index", {})),
            count=data.get("count", 0),
            tombstones=data.get("tombstones", 0),
            field_indexes=list(data.get("field_indexes", [])),
            quantization=data.get("quantization", "none"),
            on_disk=data.get("on_disk", False),
            vectors=dict(data.get("vectors", {})),
            sparse_vectors=dict(data.get("sparse_vectors", {})),
            expiry=Expiry.from_dict(data["expiry"]) if data.get("expiry") else None,
        )


@dataclass(frozen=True)
class SnapshotInfo:
    """A backup archive of one collection."""

    id: str
    collection: str
    created_at_ms: int
    points: int
    dimension: int
    metric: str
    index: dict[str, Any] = field(default_factory=dict)
    archive_bytes: int = 0
    data_sha256: str = ""
    quiver_version: str = ""
    quantization: str = "none"
    field_indexes: list[str] = field(default_factory=list)

    @property
    def created_at(self) -> datetime:
        """Creation time as an aware UTC datetime."""
        return datetime.fromtimestamp(self.created_at_ms / 1000, tz=timezone.utc)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SnapshotInfo:
        config = data.get("config", {})
        return cls(
            id=data["id"],
            collection=data["collection"],
            created_at_ms=data["created_at_ms"],
            points=data["points"],
            dimension=config.get("dimension", 0),
            metric=config.get("metric", ""),
            index=dict(config.get("index", {})),
            archive_bytes=data.get("archive_bytes", 0),
            data_sha256=data.get("data_sha256", ""),
            quiver_version=data.get("quiver_version", ""),
            quantization=config.get("quantization", "none"),
            field_indexes=list(data.get("field_indexes", [])),
        )


def as_list(vector: Vector) -> list[float]:
    """Converts lists, tuples and numpy arrays to a plain list of floats."""
    values = vector.tolist() if hasattr(vector, "tolist") else vector
    return [float(x) for x in values]


def to_point_dict(item: PointLike) -> dict[str, Any]:
    if isinstance(item, Point):
        return item.to_dict()
    if isinstance(item, Mapping):
        body: dict[str, Any] = {"vector": as_list(item["vector"])}
        if item.get("id") is not None:
            body["id"] = item["id"]
        if item.get("metadata") is not None:
            body["metadata"] = item["metadata"]
        if item.get("vectors"):
            body["vectors"] = {n: as_list(v) for n, v in item["vectors"].items()}
        if item.get("sparse_vectors"):
            body["sparse_vectors"] = {n: as_sparse(v) for n, v in item["sparse_vectors"].items()}
        return body
    if isinstance(item, tuple) and len(item) in (2, 3):
        return Point(*item).to_dict()
    raise TypeError(f"cannot convert {type(item).__name__} to a point")
