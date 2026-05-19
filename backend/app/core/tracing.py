import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.logging import serialize_log_payload
from backend.app.core.request_id import get_request_id

TRACE_LOGGER_NAME = "backend.trace"
TRACE_ID_HEADER = "X-Trace-ID"

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_stack: ContextVar[tuple[str, ...]] = ContextVar("span_stack", default=())


@dataclass(frozen=True)
class TraceContextToken:
    trace_id_token: Token[str | None]
    span_stack_token: Token[tuple[str, ...]]


def generate_span_id() -> str:
    return uuid.uuid4().hex[:16]


def normalize_trace_id(trace_id: str | None) -> str | None:
    if trace_id is None:
        return None

    normalized = trace_id.strip()
    return normalized or None


def get_trace_id() -> str | None:
    return _trace_id.get()


def get_current_span_id() -> str | None:
    span_stack = _span_stack.get()
    return span_stack[-1] if span_stack else None


def start_trace(trace_id: str) -> TraceContextToken:
    return TraceContextToken(
        trace_id_token=_trace_id.set(trace_id),
        span_stack_token=_span_stack.set(()),
    )


def reset_trace(token: TraceContextToken) -> None:
    _span_stack.reset(token.span_stack_token)
    _trace_id.reset(token.trace_id_token)


@contextmanager
def trace_context(trace_id: str) -> Iterator[None]:
    token = start_trace(trace_id)
    try:
        yield
    finally:
        reset_trace(token)


def normalize_span_attributes(
    attributes: Mapping[str, object] | None,
) -> dict[str, object]:
    if attributes is None:
        return {}

    normalized: dict[str, object] = {}
    for key, value in attributes.items():
        if isinstance(value, str | int | float | bool) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)

    return normalized


def build_span_log_payload(
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    status: str,
    latency_ms: int,
    attributes: Mapping[str, object] | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "trace_span",
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "status": status,
        "latency_ms": latency_ms,
        "attributes": normalize_span_attributes(attributes),
    }
    if error_type is not None:
        payload["error_type"] = error_type

    return payload


@contextmanager
def trace_span(
    name: str,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[str | None]:
    trace_id = get_trace_id()
    if trace_id is None:
        yield None
        return

    span_id = generate_span_id()
    parent_span_id = get_current_span_id()
    span_stack = _span_stack.get()
    span_stack_token = _span_stack.set((*span_stack, span_id))
    started_at = time.perf_counter()
    status = "ok"
    error_type: str | None = None

    try:
        yield span_id
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        _span_stack.reset(span_stack_token)
        logging.getLogger(TRACE_LOGGER_NAME).info(
            serialize_log_payload(
                build_span_log_payload(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    name=name,
                    status=status,
                    latency_ms=latency_ms,
                    attributes=attributes,
                    error_type=error_type,
                )
            )
        )


class TraceContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = get_request_id(request)
        trace_id = normalize_trace_id(request.headers.get(TRACE_ID_HEADER))
        trace_id = trace_id or request_id
        request.state.trace_id = trace_id
        span_attributes: dict[str, object] = {
            "method": request.method,
            "path": request.url.path,
        }

        with trace_context(trace_id):
            with trace_span("http.request", span_attributes):
                response = await call_next(request)
                span_attributes["status_code"] = response.status_code
                response.headers[TRACE_ID_HEADER] = trace_id
                return response
