"""Exception hierarchy."""

from __future__ import annotations


class QuiverError(Exception):
    """Base class for every error raised by the client."""

    def __init__(
        self, message: str, *, status_code: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}({self.message!r}, status_code={self.status_code}, code={self.code!r})"


class QuiverConnectionError(QuiverError):
    """The server could not be reached or the connection failed."""


class InvalidRequestError(QuiverError):
    """The request was rejected as invalid (HTTP 400/413/415/422)."""


class AuthenticationError(QuiverError):
    """The API key is missing or wrong (HTTP 401)."""


class PermissionDeniedError(QuiverError):
    """The API key is valid but read-only, and the request writes (HTTP 403)."""


class NotFoundError(QuiverError):
    """The collection, point or route does not exist (HTTP 404)."""


class ConflictError(QuiverError):
    """The resource already exists (HTTP 409)."""


class ServerError(QuiverError):
    """The server failed to handle the request (HTTP 5xx)."""


def error_for_status(status: int, code: str | None, message: str) -> QuiverError:
    cls: type[QuiverError]
    if status == 401:
        cls = AuthenticationError
    elif status == 403:
        cls = PermissionDeniedError
    elif status == 404:
        cls = NotFoundError
    elif status == 409:
        cls = ConflictError
    elif status >= 500:
        cls = ServerError
    else:
        cls = InvalidRequestError
    return cls(message, status_code=status, code=code)
