import argparse
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi.testclient import TestClient
from httpx import Response

from backend.app.api import routes_documents
from backend.app.api.security import parse_api_keys
from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app
from backend.app.rag.embeddings import FakeEmbeddingClient


class DocumentManagementSmokeError(RuntimeError):
    pass


class SyncApiClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        pass

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        pass

    def delete(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> Response:
        pass


@dataclass(frozen=True)
class DocumentManagementSmokeResult:
    workspace_id: str
    source_uri: str
    document_id: str
    chunks_inserted: int
    chunks_matched_for_reindex: int


def build_smoke_settings(settings: Settings | None = None) -> Settings:
    settings = settings or get_settings()
    api_keys = settings.api_keys if parse_api_keys(settings.api_keys) else "dev-key"
    return Settings(
        **{
            **settings.model_dump(),
            "log_level": "ERROR",
            "api_keys": api_keys,
            "embedding_provider": "fake",
            "embedding_model": "fake-embedding",
            "generator_provider": "fake",
            "query_rewriter_provider": "none",
            "reranker_provider": "none",
            "rate_limit_enabled": False,
        }
    )


def build_smoke_app(settings: Settings):
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[routes_documents.get_embedding_client] = (
        lambda: FakeEmbeddingClient(
            dimension=settings.embedding_dimension,
            model_name=settings.embedding_model,
        )
    )
    return app


def select_api_key(raw_api_keys: str) -> str:
    api_keys = sorted(parse_api_keys(raw_api_keys))
    if not api_keys:
        raise DocumentManagementSmokeError("API_KEYS must include at least one key")
    return api_keys[0]


def build_smoke_markdown(run_id: str) -> str:
    return f"""---
topic: "document-management"
run_id: "{run_id}"
---

# Document Management Smoke

This temporary document verifies upload, listing, detail lookup, dry-run
reindexing, and deletion for the document management API.
"""


def assert_json_response(
    response: Response,
    *,
    expected_status: int | set[int],
    label: str,
) -> dict[str, Any]:
    expected_statuses = (
        {expected_status} if isinstance(expected_status, int) else expected_status
    )
    if response.status_code not in expected_statuses:
        raise DocumentManagementSmokeError(
            f"{label} failed: expected {sorted(expected_statuses)}, "
            f"got {response.status_code}: {response.text}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise DocumentManagementSmokeError(
            f"{label} failed: response was not JSON"
        ) from exc
    if not isinstance(body, dict):
        raise DocumentManagementSmokeError(
            f"{label} failed: response JSON must be an object"
        )
    return body


def run_document_management_smoke(
    client: SyncApiClient,
    *,
    workspace_id: str,
    api_key: str,
    run_id: str | None = None,
    source_uri: str | None = None,
) -> DocumentManagementSmokeResult:
    run_id = run_id or uuid.uuid4().hex
    source_uri = source_uri or f"uploads/document-management-smoke-{run_id}.md"
    headers = {"Authorization": f"Bearer {api_key}", "X-Workspace-ID": workspace_id}
    document_id: str | None = None
    deleted = False

    assert_json_response(
        client.post(
            "/workspaces",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "id": workspace_id,
                "name": f"Document Management Smoke {workspace_id}",
                "description": "Workspace used by document management smoke tests.",
                "metadata": {"smoke": "document-management"},
            },
        ),
        expected_status={200, 201},
        label="ensure workspace",
    )

    try:
        create_body = assert_json_response(
            client.post(
                "/documents",
                headers=headers,
                json={
                    "source_uri": source_uri,
                    "markdown": build_smoke_markdown(run_id),
                    "title": "Document Management Smoke",
                    "metadata": {"smoke": "document-management", "run_id": run_id},
                    "chunk_size_tokens": 80,
                    "chunk_overlap_tokens": 10,
                },
            ),
            expected_status=201,
            label="create document",
        )
        if create_body.get("inserted") is not True:
            raise DocumentManagementSmokeError("create document did not insert")
        document_id = str(create_body.get("document_id") or "")
        if not document_id:
            raise DocumentManagementSmokeError("create document omitted document_id")
        chunks_inserted = int(create_body.get("chunks_inserted") or 0)
        if chunks_inserted <= 0:
            raise DocumentManagementSmokeError("create document inserted no chunks")

        list_body = assert_json_response(
            client.get(
                "/documents",
                headers=headers,
                params={"limit": 100, "offset": 0},
            ),
            expected_status=200,
            label="list documents",
        )
        listed_ids = {
            str(document.get("id"))
            for document in list_body.get("documents", [])
            if isinstance(document, dict)
        }
        if document_id not in listed_ids:
            raise DocumentManagementSmokeError(
                "created document was not returned by list documents"
            )

        detail_body = assert_json_response(
            client.get(f"/documents/{document_id}", headers=headers),
            expected_status=200,
            label="get document detail",
        )
        detail_document = detail_body.get("document")
        if not isinstance(detail_document, dict):
            raise DocumentManagementSmokeError("document detail omitted document")
        if detail_document.get("source_uri") != source_uri:
            raise DocumentManagementSmokeError("document detail source_uri mismatch")
        chunks = detail_body.get("chunks")
        if not isinstance(chunks, list) or len(chunks) < chunks_inserted:
            raise DocumentManagementSmokeError("document detail omitted chunks")

        reindex_body = assert_json_response(
            client.post(
                "/documents/reindex",
                headers=headers,
                json={"source_uri": source_uri, "dry_run": True},
            ),
            expected_status=200,
            label="dry-run document reindex",
        )
        chunks_matched = int(reindex_body.get("chunks_matched") or 0)
        if reindex_body.get("dry_run") is not True:
            raise DocumentManagementSmokeError("reindex response was not dry-run")
        if chunks_matched < chunks_inserted:
            raise DocumentManagementSmokeError(
                "reindex matched fewer chunks than upload"
            )

        delete_body = assert_json_response(
            client.delete(f"/documents/{document_id}", headers=headers),
            expected_status=200,
            label="delete document",
        )
        deleted = delete_body.get("deleted") is True
        if not deleted:
            raise DocumentManagementSmokeError("delete document did not delete")

        confirm_response = client.get(f"/documents/{document_id}", headers=headers)
        if confirm_response.status_code != 404:
            raise DocumentManagementSmokeError(
                "deleted document detail did not return 404"
            )

        return DocumentManagementSmokeResult(
            workspace_id=workspace_id,
            source_uri=source_uri,
            document_id=document_id,
            chunks_inserted=chunks_inserted,
            chunks_matched_for_reindex=chunks_matched,
        )
    finally:
        if document_id is not None and not deleted:
            client.delete(f"/documents/{document_id}", headers=headers)


def format_result(result: DocumentManagementSmokeResult) -> str:
    return "\n".join(
        [
            "document management smoke passed",
            f"workspace_id: {result.workspace_id}",
            f"source_uri: {result.source_uri}",
            f"document_id: {result.document_id}",
            f"chunks_inserted: {result.chunks_inserted}",
            f"chunks_matched_for_reindex: {result.chunks_matched_for_reindex}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a document management API smoke test."
    )
    parser.add_argument("--workspace-id", default="public")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--source-uri", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = build_smoke_settings()
    api_key = args.api_key or select_api_key(settings.api_keys)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with TestClient(build_smoke_app(settings)) as client:
        result = run_document_management_smoke(
            client,
            workspace_id=args.workspace_id,
            api_key=api_key,
            source_uri=args.source_uri,
        )
    print(format_result(result))


if __name__ == "__main__":
    main()
