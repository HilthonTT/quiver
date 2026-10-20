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
    Metric,
    Point,
    SearchResult,
    SnapshotInfo,
    Vector,
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
    "Metric",
    "NotFoundError",
    "PermissionDeniedError",
    "Point",
    "QuiverConnectionError",
    "QuiverError",
    "SearchResult",
    "ServerError",
    "SnapshotInfo",
    "Vector",
    "__version__",
]
