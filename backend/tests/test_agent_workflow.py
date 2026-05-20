import pytest

from backend.app.agent.workflow import (
    build_agent_run_id,
    run_support_triage_skeleton,
)
from backend.app.rag.citations import Source
from backend.app.rag.pipeline import RagRetrievalContext, RetrievalInfo
from backend.app.schemas.agent import SupportTicketRequest


class FakeSupportTicketRepository:
    def __init__(self) -> None:
        self.calls = []

    async def list_similar_support_tickets(self, **kwargs):
        self.calls.append(dict(kwargs))
        return []


class FakeRagPipeline:
    def __init__(self) -> None:
        self.requests = []

    async def retrieve_context(self, request):
        self.requests.append(request)
        return RagRetrievalContext(
            sources=[
                Source(
                    source_id="1",
                    title="Citation Debugging",
                    section="Validation",
                    source_uri="docs/citations.md",
                    chunk_id="chunk-1",
                    score=0.92,
                )
            ],
            context="[1] Citation Debugging\nInspect retrieved chunks.",
            retrieval=RetrievalInfo(
                mode="hybrid_rrf_rerank",
                vector_top_k=5,
                sparse_top_k=5,
                fused_count=1,
                used_count=1,
                top_score=0.92,
            ),
        )


def test_build_agent_run_id_uses_request_id() -> None:
    assert build_agent_run_id(" request-1 ") == "agent_request-1"


@pytest.mark.asyncio
async def test_support_triage_skeleton_finalizes_safe_ticket() -> None:
    fake_pipeline = FakeRagPipeline()
    fake_ticket_repository = FakeSupportTicketRepository()

    response = await run_support_triage_skeleton(
        SupportTicketRequest(
            ticket_id="TICKET-1",
            customer_message="How can I debug citation validation failures?",
        ),
        rag_pipeline=fake_pipeline,  # type: ignore[arg-type]
        support_ticket_repository=fake_ticket_repository,  # type: ignore[arg-type]
        request_id="request-1",
        trace_id="trace-1",
    )

    assert response.run_id == "agent_request-1"
    assert response.status == "finalized"
    assert response.category == "rag_failure"
    assert response.risk_level == "low"
    assert response.approval_required is False
    assert response.final_answer is not None
    assert response.draft_answer == response.final_answer
    assert "Citation Debugging" in response.final_answer
    assert "[1]" in response.final_answer
    assert response.trace_id == "trace-1"
    assert response.sources[0]["chunk_id"] == "chunk-1"
    assert response.retrieval_context == (
        "[1] Citation Debugging\nInspect retrieved chunks."
    )
    assert response.retrieval["top_score"] == 0.92
    assert response.historical_cases == []
    assert response.cited_source_ids == ["1"]
    assert response.cited_case_ids == []
    assert response.metrics["tool_count"] == 5
    assert response.metrics["citation_valid"] is True
    assert response.metrics["retrieved_source_count"] == 1
    assert response.metrics["historical_case_count"] == 0
    assert response.metrics["cited_source_count"] == 1
    assert response.metrics["cited_case_count"] == 0
    assert response.metrics["node_count"] == 5
    assert [tool_call["tool_name"] for tool_call in response.tool_calls] == [
        "classify_ticket_tool",
        "risk_check_tool",
        "rag_search_tool",
        "ticket_lookup_tool",
        "draft_response_tool",
    ]
    assert [node_run["node_name"] for node_run in response.node_runs] == [
        "classify_ticket",
        "risk_check",
        "rag_search",
        "ticket_lookup",
        "draft_response",
    ]
    assert all(node_run["success"] for node_run in response.node_runs)
    assert len(fake_pipeline.requests) == 1
    assert fake_pipeline.requests[0].workspace_id == "public"
    assert fake_pipeline.requests[0].rerank_top_n == 5
    assert fake_ticket_repository.calls == [
        {
            "query": "How can I debug citation validation failures?",
            "workspace_id": "public",
            "category": "rag_failure",
            "limit": 5,
        }
    ]


@pytest.mark.asyncio
async def test_support_triage_skeleton_routes_high_risk_ticket_to_approval() -> None:
    fake_pipeline = FakeRagPipeline()
    fake_ticket_repository = FakeSupportTicketRepository()

    response = await run_support_triage_skeleton(
        SupportTicketRequest(
            ticket_id="TICKET-2",
            customer_message="Delete all logs containing customer prompts.",
        ),
        rag_pipeline=fake_pipeline,  # type: ignore[arg-type]
        support_ticket_repository=fake_ticket_repository,  # type: ignore[arg-type]
        request_id="request-2",
    )

    assert response.status == "approval_required"
    assert response.category == "data_privacy"
    assert response.risk_level == "high"
    assert response.approval_required is True
    assert response.approval_id is None
    assert response.final_answer is None
    assert response.draft_answer is not None
    assert response.reason is not None
    assert "customer prompt" in response.reason
    assert response.sources == []
    assert response.retrieval == {}
    assert response.metrics["node_count"] == 2
    assert [node_run["node_name"] for node_run in response.node_runs] == [
        "classify_ticket",
        "risk_check",
    ]
    assert fake_pipeline.requests == []
    assert fake_ticket_repository.calls == []
