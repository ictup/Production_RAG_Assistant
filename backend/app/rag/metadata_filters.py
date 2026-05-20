import math
from collections.abc import Mapping
from typing import Any


def normalize_metadata_filter(
    metadata_filter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if metadata_filter is None:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if not isinstance(key, str):
            raise ValueError("metadata_filter keys must be strings")
        key = key.strip()
        if not key:
            raise ValueError("metadata_filter keys must not be blank")
        normalized[key] = normalize_metadata_filter_value(value, path=key)
    return normalized


def normalize_metadata_filter_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata_filter value at {path} must be finite")
        return value
    if isinstance(value, list):
        return [
            normalize_metadata_filter_value(item, path=f"{path}[]")
            for item in value
        ]
    if isinstance(value, dict):
        return normalize_metadata_filter(value)

    raise ValueError(
        f"metadata_filter value at {path} must be JSON-compatible"
    )
