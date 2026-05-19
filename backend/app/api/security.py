from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from backend.app.core.config import Settings, get_settings

WorkspaceAccess = frozenset[str] | None


@dataclass(frozen=True)
class ApiPrincipal:
    token: str
    allowed_workspaces: WorkspaceAccess = None

    def can_access_workspace(self, workspace_id: str) -> bool:
        if self.allowed_workspaces is None:
            return True
        return workspace_id in self.allowed_workspaces


def parse_api_keys(raw_api_keys: str) -> set[str]:
    return {
        api_key.strip()
        for api_key in raw_api_keys.split(",")
        if api_key.strip()
    }


def parse_workspace_access_list(raw_workspaces: str) -> WorkspaceAccess:
    raw_workspaces = raw_workspaces.strip()
    if raw_workspaces == "*":
        return None

    workspaces = frozenset(
        workspace.strip()
        for workspace in raw_workspaces.split("|")
        if workspace.strip()
    )
    if not workspaces:
        raise ValueError("workspace access entry must include at least one workspace")
    if "*" in workspaces:
        raise ValueError("wildcard workspace access must not be mixed")
    return workspaces


def parse_api_key_workspace_access(raw_access: str) -> dict[str, WorkspaceAccess]:
    access_by_key: dict[str, WorkspaceAccess] = {}
    for entry in raw_access.split(";"):
        entry = entry.strip()
        if not entry:
            continue

        api_key, separator, raw_workspaces = entry.partition("=")
        api_key = api_key.strip()
        if separator == "" or not api_key:
            raise ValueError(
                "workspace access entries must use api-key=workspace syntax"
            )

        access_by_key[api_key] = parse_workspace_access_list(raw_workspaces)

    return access_by_key


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


def build_api_principal(*, token: str, settings: Settings) -> ApiPrincipal:
    try:
        access_by_key = parse_api_key_workspace_access(
            settings.api_key_workspace_access
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="invalid api key workspace configuration",
        ) from exc

    if not access_by_key:
        return ApiPrincipal(token=token)

    return ApiPrincipal(
        token=token,
        allowed_workspaces=access_by_key.get(token, frozenset()),
    )


async def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> ApiPrincipal:
    token = extract_bearer_token(authorization)
    allowed_api_keys = parse_api_keys(settings.api_keys)

    if token not in allowed_api_keys:
        raise unauthorized("invalid api key")

    return build_api_principal(token=token, settings=settings)


def normalize_workspace_id(workspace_id: str | None) -> str:
    if workspace_id is None:
        return "public"

    normalized = workspace_id.strip()
    return normalized or "public"


def resolve_workspace_id(
    principal: ApiPrincipal,
    workspace_id: str | None,
) -> str:
    normalized_workspace_id = normalize_workspace_id(workspace_id)
    if not principal.can_access_workspace(normalized_workspace_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="workspace access denied",
        )
    return normalized_workspace_id
