from datetime import UTC, datetime

from fastapi.testclient import TestClient

from backend.app.api import routes_workspaces
from backend.app.core.config import Settings, get_settings
from backend.app.db.models import Workspace
from backend.app.db.repositories import (
    CreateWorkspaceInput,
    CreateWorkspaceResult,
    WorkspaceListResult,
)
from backend.app.main import create_app

AUTH_HEADERS = {"Authorization": "Bearer dev-key"}


class FakeWorkspaceRepository:
    def __init__(
        self,
        *,
        create_result: CreateWorkspaceResult | None = None,
        list_result: WorkspaceListResult | None = None,
        detail_workspace: Workspace | None = None,
    ) -> None:
        self.create_result = create_result or CreateWorkspaceResult(
            workspace=make_workspace_model(),
            created=True,
        )
        self.list_result = list_result or WorkspaceListResult(
            total=0,
            workspaces=[],
        )
        self.detail_workspace = detail_workspace
        self.create_calls: list[tuple[CreateWorkspaceInput, bool]] = []
        self.list_calls: list[tuple[frozenset[str] | None, int, int]] = []
        self.detail_calls: list[str] = []

    async def create_workspace(
        self,
        workspace_input: CreateWorkspaceInput,
        *,
        commit: bool = False,
    ) -> CreateWorkspaceResult:
        self.create_calls.append((workspace_input, commit))
        return self.create_result

    async def list_workspaces(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        workspace_ids: frozenset[str] | None = None,
    ) -> WorkspaceListResult:
        self.list_calls.append((workspace_ids, limit, offset))
        return self.list_result

    async def get_workspace(self, *, workspace_id: str) -> Workspace | None:
        self.detail_calls.append(workspace_id)
        return self.detail_workspace


def make_workspace_model() -> Workspace:
    return Workspace(
        id="tenant-a",
        name="Tenant A",
        description="GPU systems team",
        metadata_={"tier": "internal"},
        created_at=datetime(2026, 5, 18, 8, 0, tzinfo=UTC),
        updated_at=datetime(2026, 5, 18, 9, 0, tzinfo=UTC),
    )


def build_client(
    fake_repository: FakeWorkspaceRepository,
    settings: Settings | None = None,
) -> TestClient:
    settings = settings or Settings(api_keys="dev-key")
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[routes_workspaces.get_workspace_repository] = (
        lambda: fake_repository
    )
    return TestClient(app)


def test_create_workspace_route_creates_workspace() -> None:
    fake_repository = FakeWorkspaceRepository()
    client = build_client(fake_repository)

    response = client.post(
        "/workspaces",
        headers=AUTH_HEADERS,
        json={
            "id": " tenant-a ",
            "name": " Tenant A ",
            "description": " GPU systems team ",
            "metadata": {"tier": "internal"},
        },
    )

    assert response.status_code == 201
    workspace_input, commit = fake_repository.create_calls[0]
    assert workspace_input.id == "tenant-a"
    assert workspace_input.name == "Tenant A"
    assert workspace_input.description == "GPU systems team"
    assert workspace_input.metadata == {"tier": "internal"}
    assert commit is True
    assert response.json()["created"] is True
    assert response.json()["workspace"]["id"] == "tenant-a"


def test_create_workspace_route_returns_200_for_existing_workspace() -> None:
    fake_repository = FakeWorkspaceRepository(
        create_result=CreateWorkspaceResult(
            workspace=make_workspace_model(),
            created=False,
        )
    )
    client = build_client(fake_repository)

    response = client.post(
        "/workspaces",
        headers=AUTH_HEADERS,
        json={"id": "tenant-a"},
    )

    assert response.status_code == 200
    assert response.json()["created"] is False


def test_create_workspace_route_rejects_forbidden_workspace() -> None:
    fake_repository = FakeWorkspaceRepository()
    client = build_client(
        fake_repository,
        settings=Settings(
            api_keys="tenant-key",
            api_key_workspace_access="tenant-key=tenant-a",
        ),
    )

    response = client.post(
        "/workspaces",
        headers={"Authorization": "Bearer tenant-key"},
        json={"id": "tenant-b"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "workspace access denied"}
    assert fake_repository.create_calls == []


def test_list_workspaces_route_returns_paginated_workspaces() -> None:
    fake_repository = FakeWorkspaceRepository(
        list_result=WorkspaceListResult(total=1, workspaces=[make_workspace_model()])
    )
    client = build_client(fake_repository)

    response = client.get(
        "/workspaces",
        headers=AUTH_HEADERS,
        params={"limit": 10, "offset": 5},
    )

    assert response.status_code == 200
    assert fake_repository.list_calls == [(None, 10, 5)]
    body = response.json()
    assert body["total"] == 1
    assert body["count"] == 1
    assert body["limit"] == 10
    assert body["offset"] == 5
    assert body["workspaces"][0]["id"] == "tenant-a"


def test_list_workspaces_route_filters_to_principal_allowed_workspaces() -> None:
    fake_repository = FakeWorkspaceRepository()
    client = build_client(
        fake_repository,
        settings=Settings(
            api_keys="tenant-key",
            api_key_workspace_access="tenant-key=tenant-a|tenant-b",
        ),
    )

    response = client.get(
        "/workspaces",
        headers={"Authorization": "Bearer tenant-key"},
    )

    assert response.status_code == 200
    assert fake_repository.list_calls == [(frozenset({"tenant-a", "tenant-b"}), 20, 0)]


def test_get_workspace_route_returns_workspace() -> None:
    fake_repository = FakeWorkspaceRepository(detail_workspace=make_workspace_model())
    client = build_client(fake_repository)

    response = client.get("/workspaces/tenant-a", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert fake_repository.detail_calls == ["tenant-a"]
    assert response.json()["workspace"]["id"] == "tenant-a"
    assert response.json()["workspace"]["metadata"] == {"tier": "internal"}


def test_get_workspace_route_returns_404_for_missing_workspace() -> None:
    fake_repository = FakeWorkspaceRepository(detail_workspace=None)
    client = build_client(fake_repository)

    response = client.get("/workspaces/tenant-a", headers=AUTH_HEADERS)

    assert response.status_code == 404
    assert response.json() == {"detail": "workspace not found"}
    assert fake_repository.detail_calls == ["tenant-a"]


def test_get_workspace_route_rejects_forbidden_workspace() -> None:
    fake_repository = FakeWorkspaceRepository(detail_workspace=make_workspace_model())
    client = build_client(
        fake_repository,
        settings=Settings(
            api_keys="tenant-key",
            api_key_workspace_access="tenant-key=tenant-a",
        ),
    )

    response = client.get(
        "/workspaces/tenant-b",
        headers={"Authorization": "Bearer tenant-key"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "workspace access denied"}
    assert fake_repository.detail_calls == []


def test_workspaces_routes_require_api_key() -> None:
    fake_repository = FakeWorkspaceRepository()
    client = build_client(fake_repository)

    response = client.get("/workspaces")

    assert response.status_code == 401
    assert response.json() == {"detail": "missing api key"}
    assert fake_repository.list_calls == []


def test_openapi_exposes_workspace_routes() -> None:
    fake_repository = FakeWorkspaceRepository()
    client = build_client(fake_repository)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/workspaces" in paths
    assert "/workspaces/{workspace_id}" in paths
