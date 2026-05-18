import pytest
from fastapi import HTTPException

from backend.app.api.security import (
    extract_bearer_token,
    parse_api_keys,
    require_api_key,
)
from backend.app.core.config import Settings


def test_parse_api_keys_trims_and_ignores_empty_values() -> None:
    assert parse_api_keys(" dev-key, second-key, , ") == {
        "dev-key",
        "second-key",
    }


def test_extract_bearer_token_accepts_valid_authorization_header() -> None:
    assert extract_bearer_token("Bearer dev-key") == "dev-key"


@pytest.mark.parametrize(
    "authorization",
    [None, "", "dev-key", "Basic dev-key", "Bearer   "],
)
def test_extract_bearer_token_rejects_missing_or_malformed_header(
    authorization: str | None,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        extract_bearer_token(authorization)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "missing api key"


@pytest.mark.asyncio
async def test_require_api_key_accepts_configured_key() -> None:
    token = await require_api_key(
        settings=Settings(api_keys="dev-key,second-key"),
        authorization="Bearer second-key",
    )

    assert token == "second-key"


@pytest.mark.asyncio
async def test_require_api_key_rejects_unknown_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(
            settings=Settings(api_keys="dev-key"),
            authorization="Bearer wrong-key",
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid api key"
