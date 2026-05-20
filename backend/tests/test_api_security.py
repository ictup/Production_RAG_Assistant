import pytest
from fastapi import HTTPException

from backend.app.api.security import (
    ApiPrincipal,
    extract_bearer_token,
    parse_api_key_roles,
    parse_api_key_workspace_access,
    parse_api_keys,
    parse_workspace_access_list,
    require_api_key,
    require_api_role,
    resolve_workspace_id,
)
from backend.app.core.config import Settings


def test_parse_api_keys_trims_and_ignores_empty_values() -> None:
    assert parse_api_keys(" dev-key, second-key, , ") == {
        "dev-key",
        "second-key",
    }


def test_parse_api_key_roles_supports_multiple_roles() -> None:
    assert parse_api_key_roles(
        "admin-key=admin; operator-key=operator; viewer-key=viewer"
    ) == {
        "admin-key": "admin",
        "operator-key": "operator",
        "viewer-key": "viewer",
    }


@pytest.mark.parametrize("raw_roles", ["dev-key", "=admin", "dev-key=owner"])
def test_parse_api_key_roles_rejects_invalid_entries(raw_roles: str) -> None:
    with pytest.raises(ValueError):
        parse_api_key_roles(raw_roles)


def test_parse_workspace_access_list_supports_wildcard() -> None:
    assert parse_workspace_access_list("*") is None


def test_parse_workspace_access_list_trims_named_workspaces() -> None:
    assert parse_workspace_access_list(" tenant-a | public | ") == frozenset(
        {"tenant-a", "public"}
    )


@pytest.mark.parametrize("raw_workspaces", ["", " | ", "public|*"])
def test_parse_workspace_access_list_rejects_invalid_values(
    raw_workspaces: str,
) -> None:
    with pytest.raises(ValueError):
        parse_workspace_access_list(raw_workspaces)


def test_parse_api_key_workspace_access_supports_multiple_keys() -> None:
    assert parse_api_key_workspace_access(
        "dev-key=*; tenant-key=tenant-a|tenant-b"
    ) == {
        "dev-key": None,
        "tenant-key": frozenset({"tenant-a", "tenant-b"}),
    }


@pytest.mark.parametrize("raw_access", ["dev-key", "=public", "dev-key="])
def test_parse_api_key_workspace_access_rejects_invalid_entries(
    raw_access: str,
) -> None:
    with pytest.raises(ValueError):
        parse_api_key_workspace_access(raw_access)


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
    principal = await require_api_key(
        settings=Settings(api_keys="dev-key,second-key"),
        authorization="Bearer second-key",
    )

    assert principal.token == "second-key"
    assert principal.role == "admin"
    assert principal.allowed_workspaces is None


@pytest.mark.asyncio
async def test_require_api_key_attaches_configured_role() -> None:
    principal = await require_api_key(
        settings=Settings(
            api_keys="admin-key,operator-key,viewer-key",
            api_key_roles=(
                "admin-key=admin;operator-key=operator;viewer-key=viewer"
            ),
        ),
        authorization="Bearer operator-key",
    )

    assert principal.token == "operator-key"
    assert principal.role == "operator"


@pytest.mark.asyncio
async def test_require_api_key_defaults_unmapped_role_to_viewer() -> None:
    principal = await require_api_key(
        settings=Settings(
            api_keys="viewer-key,unmapped-key",
            api_key_roles="viewer-key=viewer",
        ),
        authorization="Bearer unmapped-key",
    )

    assert principal.role == "viewer"


@pytest.mark.asyncio
async def test_require_api_key_attaches_configured_workspace_access() -> None:
    principal = await require_api_key(
        settings=Settings(
            api_keys="dev-key,tenant-key",
            api_key_workspace_access="tenant-key=tenant-a|tenant-b",
        ),
        authorization="Bearer tenant-key",
    )

    assert principal.token == "tenant-key"
    assert principal.allowed_workspaces == frozenset({"tenant-a", "tenant-b"})


@pytest.mark.asyncio
async def test_require_api_key_denies_unmapped_key_when_access_rules_exist() -> None:
    principal = await require_api_key(
        settings=Settings(
            api_keys="dev-key,tenant-key",
            api_key_workspace_access="tenant-key=tenant-a",
        ),
        authorization="Bearer dev-key",
    )

    assert principal.allowed_workspaces == frozenset()


@pytest.mark.asyncio
async def test_require_api_key_rejects_unknown_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(
            settings=Settings(api_keys="dev-key"),
            authorization="Bearer wrong-key",
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid api key"


@pytest.mark.asyncio
async def test_require_api_key_rejects_invalid_workspace_access_config() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(
            settings=Settings(
                api_keys="dev-key",
                api_key_workspace_access="dev-key",
            ),
            authorization="Bearer dev-key",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "invalid api key workspace configuration"


@pytest.mark.asyncio
async def test_require_api_key_rejects_invalid_role_config() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(
            settings=Settings(
                api_keys="dev-key",
                api_key_roles="dev-key=owner",
            ),
            authorization="Bearer dev-key",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "invalid api key role configuration"


def test_require_api_role_accepts_sufficient_role() -> None:
    require_api_role(ApiPrincipal(token="operator-key", role="operator"), "viewer")
    require_api_role(ApiPrincipal(token="admin-key", role="admin"), "operator")


def test_require_api_role_rejects_insufficient_role() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_api_role(ApiPrincipal(token="viewer-key", role="viewer"), "operator")

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "insufficient api role"


def test_resolve_workspace_id_defaults_and_trims_allowed_workspace() -> None:
    principal = ApiPrincipal(
        token="tenant-key",
        allowed_workspaces=frozenset({"public", "tenant-a"}),
    )

    assert resolve_workspace_id(principal, None) == "public"
    assert resolve_workspace_id(principal, " tenant-a ") == "tenant-a"


def test_resolve_workspace_id_rejects_forbidden_workspace() -> None:
    principal = ApiPrincipal(
        token="tenant-key",
        allowed_workspaces=frozenset({"tenant-a"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_workspace_id(principal, "tenant-b")

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "workspace access denied"
