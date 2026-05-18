from fastapi.testclient import TestClient

from backend.app.api import routes_chat
from backend.app.main import create_app
from backend.app.rag.citations import Source
from backend.app.rag.pipeline import (
    ChatPipelineRequest,
    ChatPipelineResponse,
    RetrievalInfo,
    UsageInfo,
)

AUTH_HEADERS = {"Authorization": "Bearer dev-key"}


class FakePipeline:
    def __init__(self) -> None:
        self.requests: list[ChatPipelineRequest] = []

    async def answer_question(
        self,
        request: ChatPipelineRequest,
    ) -> ChatPipelineResponse:
        self.requests.append(request)
        return ChatPipelineResponse(
            answer="FlashAttention reduces memory traffic. [1]",
            sources=[
                Source(
                    source_id="1",
                    title="FlashAttention Notes",
                    section="FlashAttention",
                    source_uri="llm_systems/flashattention.md",
                    chunk_id="chunk-1",
                    score=0.42,
                )
            ],
            retrieval=RetrievalInfo(
                mode="hybrid_rrf_rerank",
                vector_top_k=request.vector_top_k or 20,
                sparse_top_k=request.sparse_top_k or 20,
                fused_count=1,
                used_count=1,
                top_score=0.42,
            ),
            usage=UsageInfo(
                model="test-fake-llm",
                embedding_model="test-fake-embedding",
                latency_ms=12,
                input_tokens=10,
                output_tokens=5,
            ),
            citation_valid=True,
            refusal=None,
        )


def build_client(fake_pipeline: FakePipeline) -> TestClient:
    app = create_app()
    app.dependency_overrides[routes_chat.get_rag_pipeline] = lambda: fake_pipeline
    return TestClient(app)


def test_chat_route_returns_answer_sources_and_metadata() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.post(
        "/chat",
        headers={
            **AUTH_HEADERS,
            "X-Workspace-ID": "  tenant-a  ",
        },
        json={
            "question": "  What problem does FlashAttention solve?  ",
            "vector_top_k": 3,
            "sparse_top_k": 4,
            "fused_top_k": 5,
            "rerank_top_n": 2,
            "rerank": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "FlashAttention reduces memory traffic. [1]"
    assert body["sources"][0]["source_id"] == "1"
    assert body["retrieval"]["mode"] == "hybrid_rrf_rerank"
    assert body["usage"]["model"] == "test-fake-llm"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["citation_valid"] is True

    assert len(fake_pipeline.requests) == 1
    pipeline_request = fake_pipeline.requests[0]
    assert pipeline_request.question == "What problem does FlashAttention solve?"
    assert pipeline_request.workspace_id == "tenant-a"
    assert pipeline_request.vector_top_k == 3
    assert pipeline_request.sparse_top_k == 4
    assert pipeline_request.fused_top_k == 5
    assert pipeline_request.rerank_top_n == 2
    assert pipeline_request.rerank is False


def test_chat_route_defaults_workspace_to_public() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"question": "What is FlashAttention?"},
    )

    assert response.status_code == 200
    assert fake_pipeline.requests[0].workspace_id == "public"


def test_chat_route_uses_client_request_id() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.post(
        "/chat",
        headers={
            **AUTH_HEADERS,
            "X-Request-ID": "  client-request-123  ",
        },
        json={"question": "What is FlashAttention?"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "client-request-123"
    assert response.headers["X-Request-ID"] == "client-request-123"


def test_chat_route_rejects_blank_question() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"question": "   "},
    )

    assert response.status_code == 422
    assert fake_pipeline.requests == []


def test_chat_route_rejects_missing_api_key() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.post(
        "/chat",
        json={"question": "What is FlashAttention?"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing api key"}
    assert fake_pipeline.requests == []


def test_chat_route_rejects_invalid_api_key() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.post(
        "/chat",
        headers={"Authorization": "Bearer wrong-key"},
        json={"question": "What is FlashAttention?"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid api key"}
    assert fake_pipeline.requests == []


def test_openapi_exposes_chat_route() -> None:
    fake_pipeline = FakePipeline()
    client = build_client(fake_pipeline)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/chat" in response.json()["paths"]
