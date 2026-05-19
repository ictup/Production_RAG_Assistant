import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

RetryableCall = Callable[[], Awaitable[httpx.Response]]


@dataclass(frozen=True)
class OpenAIErrorInfo:
    operation: str
    category: str
    message: str
    retryable: bool
    status_code: int | None = None


class OpenAIProviderError(RuntimeError):
    def __init__(self, info: OpenAIErrorInfo | str) -> None:
        if isinstance(info, str):
            info = OpenAIErrorInfo(
                operation="OpenAI provider",
                category="provider_error",
                message=info,
                retryable=False,
            )
        super().__init__(info.message)
        self.operation = info.operation
        self.category = info.category
        self.retryable = info.retryable
        self.status_code = info.status_code


def classify_openai_status(status_code: int) -> tuple[str, bool]:
    if status_code == 401:
        return "authentication", False
    if status_code == 403:
        return "permission", False
    if status_code == 404:
        return "not_found", False
    if status_code == 408:
        return "timeout", True
    if status_code == 409:
        return "conflict", True
    if status_code == 429:
        return "rate_limit", True
    if 400 <= status_code < 500:
        return "invalid_request", False
    if status_code >= 500:
        return "server_error", True
    return "http_error", False


def build_openai_status_error(
    *,
    operation: str,
    status_code: int,
    response_text: str,
) -> OpenAIErrorInfo:
    category, retryable = classify_openai_status(status_code)
    text = response_text.strip()
    suffix = f": {text}" if text else ""
    return OpenAIErrorInfo(
        operation=operation,
        category=category,
        message=(
            f"{operation} failed with status {status_code} "
            f"({category}){suffix}"
        ),
        retryable=retryable,
        status_code=status_code,
    )


def build_openai_transport_error(
    *,
    operation: str,
    exc: httpx.HTTPError,
) -> OpenAIErrorInfo:
    category = "timeout" if isinstance(exc, httpx.TimeoutException) else "network"
    return OpenAIErrorInfo(
        operation=operation,
        category=category,
        message=f"{operation} failed due to {category} error: {exc}",
        retryable=True,
        status_code=None,
    )


async def post_with_retries(
    *,
    operation: str,
    call: RetryableCall,
    max_retries: int,
    retry_delay_seconds: float,
    error_cls: type[OpenAIProviderError],
) -> httpx.Response:
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must not be negative")

    last_error: OpenAIProviderError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await call()
        except httpx.HTTPError as exc:
            error = error_cls(
                build_openai_transport_error(operation=operation, exc=exc)
            )
        else:
            if response.is_success:
                return response
            error = error_cls(
                build_openai_status_error(
                    operation=operation,
                    status_code=response.status_code,
                    response_text=response.text,
                )
            )

        last_error = error
        if not error.retryable or attempt >= max_retries:
            raise error
        if retry_delay_seconds:
            await asyncio.sleep(retry_delay_seconds)

    if last_error is not None:
        raise last_error
    raise error_cls(
        OpenAIErrorInfo(
            operation=operation,
            category="unknown",
            message=f"{operation} failed for an unknown reason",
            retryable=False,
        )
    )
