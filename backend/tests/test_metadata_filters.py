import pytest

from backend.app.rag.metadata_filters import normalize_metadata_filter


def test_normalize_metadata_filter_trims_keys_and_keeps_json_values() -> None:
    assert normalize_metadata_filter(
        {
            " topic ": "attention",
            "nested": {" difficulty ": "advanced"},
            "tags": ["gpu", "memory"],
            "published": True,
            "score": 1.5,
            "missing": None,
        }
    ) == {
        "topic": "attention",
        "nested": {"difficulty": "advanced"},
        "tags": ["gpu", "memory"],
        "published": True,
        "score": 1.5,
        "missing": None,
    }


def test_normalize_metadata_filter_rejects_blank_keys() -> None:
    with pytest.raises(ValueError, match="keys must not be blank"):
        normalize_metadata_filter({"  ": "attention"})


def test_normalize_metadata_filter_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        normalize_metadata_filter({"topic": object()})


def test_normalize_metadata_filter_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_metadata_filter({"score": float("nan")})
