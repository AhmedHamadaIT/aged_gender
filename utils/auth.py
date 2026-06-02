"""
utils/auth.py
-------------
Security: optional API_AUTH_TOKEN bearer dependency on mutating routes.

When API_AUTH_TOKEN is set in the environment, all mutating HTTP methods
(POST, PUT, PATCH, DELETE) must supply an Authorization: Bearer <token>
header matching that token.  Read-only routes (GET, HEAD, OPTIONS) are
not affected, preserving backward compatibility for frontends and external
integrations that poll or stream without credentials.

When API_AUTH_TOKEN is NOT set (default), the dependency is a no-op and
every request passes through unchanged — zero breaking change.

Usage in FastAPI routes:
    from utils.auth import require_auth

    @app.post("/cameras", dependencies=[Depends(require_auth)])
    async def add_camera(...):
        ...

Or applied to an entire router:
    router = APIRouter(dependencies=[Depends(require_auth)])
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_TOKEN: str | None = os.getenv("API_AUTH_TOKEN")

# FastAPI's built-in bearer extractor — auto_error=False so we can return
# a custom 401 rather than a generic 403.
_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """
    FastAPI dependency: verify bearer token on mutating routes.

    When API_AUTH_TOKEN is not set this is a no-op.
    When it IS set, the request must carry a matching Bearer token or a 401
    is raised.
    """
    if not _TOKEN:
        return  # Auth disabled — pass through.

    if credentials is None or credentials.credentials != _TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Upload size guard ──────────────────────────────────────────────────────────

_UPLOAD_MAX_BYTES: int = int(os.getenv("UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB


async def check_upload_size(request: Request) -> None:
    """
    FastAPI dependency: reject requests whose Content-Length exceeds UPLOAD_MAX_BYTES.

    Applied to image-upload endpoints.  If Content-Length is absent the body is
    not pre-read so this only protects against declared oversize uploads; that is
    sufficient for standard HTTP clients.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            size = int(content_length)
        except ValueError:
            size = 0
        if size > _UPLOAD_MAX_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload too large: {size} bytes (max {_UPLOAD_MAX_BYTES})",
            )
