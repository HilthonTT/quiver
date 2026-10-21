"""Python client for Quiver, a vector database written in Rust.

Quick start::

    from quiver import Client

    client = Client("http://localhost:6900")
    docs = client.get_or_create_collection("docs", dimension=384)
    docs.upsert([("doc-1", embedding, {"lang": "en"})])
    hits = docs.search(query_embedding, k=5, filter={"lang": "en"})
"""

from ._async_client import AsyncClient, AsyncCollection
from ._client import Client, Collection
from ._errors import (
    AuthenticationError,
    ConflictError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    QuiverConnectionError,
    QuiverError,
    ServerError,
)
from ._models import (
    HNSW,
    CollectionInfo,
    Flat,
    KDTree,
    MetadataMode,
    Metric,
    Point,
    Prefetch,
    Quantization,
    ScoredResult,
    ScrollPage,
    SearchResult,
    SnapshotInfo,
    SparseModifier,
    SparseVector,
    SparseVectorParams,
    Vector,
    VectorParams,
)

__version__ = "0.2.0"

__all__ = [
    "HNSW",
    "AsyncClient",
    "AsyncCollection",
    "AuthenticationError",
    "Client",
    "Collection",
    "CollectionInfo",
    "ConflictError",
    "Flat",
    "InvalidRequestError",
    "KDTree",
    "MetadataMode",
    "Metric",
    "NotFoundError",
    "PermissionDeniedError",
    "Point",
    "Prefetch",
    "Quantization",
    "QuiverConnectionError",
    "QuiverError",
    "ScoredResult",
    "ScrollPage",
    "SearchResult",
    "ServerError",
    "SnapshotInfo",
    "SparseModifier",
    "SparseVector",
    "SparseVectorParams",
    "Vector",
    "VectorParams",
    "__version__",
]
