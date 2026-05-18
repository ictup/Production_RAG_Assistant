import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.request_id import get_request_id

REQUEST_LOGGER_NAME = "backend.request"


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    logging.getLogger("backend").setLevel(log_level.upper())


def build_request_log_payload(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "event": "http_request",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": latency_ms,
    }


def serialize_log_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        logger = logging.getLogger(REQUEST_LOGGER_NAME)
        request_id = get_request_id(request)
        started_at = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
            logger.info(
                serialize_log_payload(
                    build_request_log_payload(
                        request_id=request_id,
                        method=request.method,
                        path=request.url.path,
                        status_code=status_code,
                        latency_ms=latency_ms,
                    )
                )
            )
