from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from backend.app.core.config import Settings, get_settings


def parse_api_keys(raw_api_keys: str) -> set[str]:
    return {
        api_key.strip()
        for api_key in raw_api_keys.split(",")
        if api_key.strip()
    }


def unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def extract_bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise unauthorized("missing api key")

    scheme, separator, token = authorization.partition(" ")
    if separator == "" or scheme.lower() != "bearer" or not token.strip():
        raise unauthorized("missing api key")

    return token.strip()


async def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    token = extract_bearer_token(authorization)
    allowed_api_keys = parse_api_keys(settings.api_keys)

    if token not in allowed_api_keys:
        raise unauthorized("invalid api key")

    return token
