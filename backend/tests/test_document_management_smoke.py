from pathlib import Path

import httpx
import pytest

from backend.app.core.config import Settings
from evals.document_management_smoke import (
    DocumentManagementSmokeError,
    assert_json_response,
    build_smoke_markdown,
    build_smoke_settings,
    run_document_management_smoke,
    select_api_key,
)


class FakeDocumentManagementClient:
    def __init__(self, *, fail_list: bool = False) -> None:
        self.fail_list = fail_list
        self.document_id = "11111111-1111-1111-1111-111111111111"
        self.calls: list[tuple[str, str]] = []
        self.created_source_uri: str | None = None
        self.chunks_inserted = 2

    def post(self, url: str, *, headers=None, json=None, params=None):
        self.calls.append(("POST", url))
        if url == "/workspaces":
            return httpx.Response(201, json={"created": True})

        if url == "/documents":
            self.created_source_uri = json["source_uri"]
            return httpx.Response(
                201,
                json={
                    "workspace_id": headers["X-Workspace-ID"],
                    "document_id": self.document_id,
                    "content_hash": "abc123",
                    "inserted": True,
                    "chunks_inserted": self.chunks_inserted,
                    "reason": None,
                },
            )

        if url == "/documents/reindex":
            return httpx.Response(
                200,
                json={
                    "workspace_id": headers["X-Workspace-ID"],
                    "source_uri": json["source_uri"],
                    "model": "not-used-dry-run",
                    "chunks_matched": self.chunks_inserted,
                    "chunks_embedded": 0,
                    "chunks_updated": 0,
                    "dry_run": True,
                    "elapsed_seconds": 0.01,
                },
            )

        return httpx.Response(404, json={"detail": "not found"})

    def get(self, url: str, *, headers=None, params=None):
        self.calls.append(("GET", url))
        if url == "/documents":
            if self.fail_list:
                return httpx.Response(500, text="database unavailable")
            return httpx.Response(
                200,
                json={
                    "workspace_id": headers["X-Workspace-ID"],
                    "total": 1,
                    "count": 1,
                    "limit": params["limit"],
                    "offset": params["offset"],
                    "documents": [
                        {
                            "id": self.document_id,
                            "source_uri": self.created_source_uri,
                        }
                    ],
                },
            )

        if url == f"/documents/{self.document_id}":
            deleted = ("DELETE", url) in self.calls
            if deleted:
                return httpx.Response(404, json={"detail": "document not found"})
            return httpx.Response(
                200,
                json={
                    "workspace_id": headers["X-Workspace-ID"],
                    "document": {
                        "id": self.document_id,
                        "source_uri": self.created_source_uri,
                    },
                    "chunks": [
                        {"chunk_index": 0, "text": "one"},
                        {"chunk_index": 1, "text": "two"},
                    ],
                },
            )

        return httpx.Response(404, json={"detail": "not found"})

    def delete(self, url: str, *, headers=None):
        self.calls.append(("DELETE", url))
        if url == f"/documents/{self.document_id}":
            return httpx.Response(
                200,
                json={
                    "workspace_id": headers["X-Workspace-ID"],
                    "document_id": self.document_id,
                    "deleted": True,
                },
            )
        return httpx.Response(404, json={"detail": "not found"})


def test_run_document_management_smoke_exercises_document_lifecycle() -> None:
    client = FakeDocumentManagementClient()

    result = run_document_management_smoke(
        client,
        workspace_id="public",
        api_key="dev-key",
        run_id="test-run",
        source_uri="uploads/test-doc.md",
    )

    assert result.workspace_id == "public"
    assert result.source_uri == "uploads/test-doc.md"
    assert result.document_id == client.document_id
    assert result.chunks_inserted == 2
    assert result.chunks_matched_for_reindex == 2
    assert client.calls == [
        ("POST", "/workspaces"),
        ("POST", "/documents"),
        ("GET", "/documents"),
        ("GET", f"/documents/{client.document_id}"),
        ("POST", "/documents/reindex"),
        ("DELETE", f"/documents/{client.document_id}"),
        ("GET", f"/documents/{client.document_id}"),
    ]


def test_run_document_management_smoke_cleans_up_after_failure() -> None:
    client = FakeDocumentManagementClient(fail_list=True)

    with pytest.raises(DocumentManagementSmokeError, match="list documents"):
        run_document_management_smoke(
            client,
            workspace_id="public",
            api_key="dev-key",
            run_id="test-run",
            source_uri="uploads/test-doc.md",
        )

    assert ("DELETE", f"/documents/{client.document_id}") in client.calls


def test_assert_json_response_reports_status_errors() -> None:
    response = httpx.Response(500, text="database unavailable")

    with pytest.raises(DocumentManagementSmokeError, match="expected"):
        assert_json_response(
            response,
            expected_status=200,
            label="list documents",
        )


def test_select_api_key_uses_first_configured_key() -> None:
    assert select_api_key(" dev-key, tenant-key ") == "dev-key"


def test_build_smoke_settings_supplies_local_api_key_when_missing() -> None:
    settings = build_smoke_settings(Settings(api_keys=""))

    assert settings.api_keys == "dev-key"
    assert settings.log_level == "ERROR"
    assert settings.embedding_provider == "fake"
    assert settings.rate_limit_enabled is False


def test_build_smoke_markdown_contains_unique_run_id() -> None:
    markdown = build_smoke_markdown("abc123")

    assert 'run_id: "abc123"' in markdown
    assert "Document Management Smoke" in markdown


def test_makefile_exposes_document_management_smoke_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "document-management-smoke:" in makefile
    assert "python -m evals.document_management_smoke" in makefile
