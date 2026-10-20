from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from quiver import AsyncClient, Client

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """Mock transport that records requests and replays canned responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], tuple[int, Any]] = {}

    def on(self, method: str, path: str, status: int = 200, body: Any = None) -> None:
        self.routes[(method, path)] = (status, body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.raw_path.decode().split("?")[0])
        if key not in self.routes:
            return httpx.Response(
                404, json={"error": {"code": "not_found", "message": "no such route"}}
            )
        status, body = self.routes[key]
        return httpx.Response(status, json=body) if body is not None else httpx.Response(status)

    def last_json(self) -> Any:
        return json.loads(self.requests[-1].content)


@pytest.fixture
def mock() -> Recorder:
    return Recorder()


@pytest.fixture
def client(mock: Recorder) -> Client:
    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(mock))
    return Client(http_client=http)


@pytest.fixture
def async_client(mock: Recorder) -> AsyncClient:
    http = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(mock))
    return AsyncClient(http_client=http)
