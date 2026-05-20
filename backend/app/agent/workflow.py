import time

from backend.app.agent.draft_response import (
    DraftResponseInput,
    draft_response_tool,
)
from backend.app.agent.graph import (
    AgentGraphNode,
    AgentGraphRunner,
    AgentNodeResult,
    serialize_node_runs,
)
from backend.app.agent.policies import check_support_risk, classify_ticket
from backend.app.agent.rag_search import RAGSearchInput, rag_search_tool
from backend.app.agent.state import AgentState
from backend.app.agent.ticket_lookup import TicketLookupInput, ticket_lookup_tool
from backend.app.agent.tools import ToolCallRecord
from backend.app.core.tracing import trace_span
from backend.app.db.repositories import SupportTicketRepository
from backend.app.rag.pipeline import RagPipeline
from backend.app.schemas.agent import AgentTriageResponse, SupportTicketRequest


def build_agent_run_id(request_id: str) -> str:
    return f"agent_{request_id.strip()}"


async def run_support_triage_skeleton(
    request: SupportTicketRequest,
    *,
    rag_pipeline: RagPipeline,
    support_ticket_repository: SupportTicketRepository,
    request_id: str,
    trace_id: str | None = None,
) -> AgentTriageResponse:
    started_at = time.perf_counter()
    run_id = build_agent_run_id(request_id)
    state = request.to_initial_state(run_id=run_id)
    tool_calls: list[ToolCallRecord] = []
    classification = None
    risk = None
    rag_search = None
    ticket_lookup = None
    draft_response = None

    def sync_tool_calls() -> list[dict[str, object]]:
        tool_call_payloads = [
            tool_call.model_dump(mode="json")
            for tool_call in tool_calls
        ]
        state["tool_calls"] = tool_call_payloads
        return tool_call_payloads

    def set_metrics(**extra: object) -> None:
        state["metrics"] = {
            "latency_ms": elapsed_ms(started_at),
            "tool_count": len(tool_calls),
            **extra,
        }

    async def classify_node(state: AgentState) -> AgentNodeResult:
        nonlocal classification

        classification_started_at = time.perf_counter()
        with trace_span(
            "agent.classify_ticket",
            {
                "workspace_id": request.workspace_id,
                "priority": request.priority,
                "message_length": len(request.customer_message),
            },
        ):
            classification = classify_ticket(request.customer_message)
        tool_calls.append(
            ToolCallRecord(
                tool_name="classify_ticket_tool",
                input_summary={
                    "message_length": len(request.customer_message),
                    "priority": request.priority,
                },
                output_summary={
                    "category": classification.category,
                    "risk_level": classification.risk_level,
                    "matched_terms": classification.matched_terms,
                },
                latency_ms=elapsed_ms(classification_started_at),
                success=True,
            )
        )
        state["category"] = classification.category
        sync_tool_calls()
        set_metrics(citation_valid=None)
        return AgentNodeResult(
            output_summary={
                "category": classification.category,
                "risk_level": classification.risk_level,
            }
        )

    async def risk_check_node(state: AgentState) -> AgentNodeResult:
        nonlocal risk
        if classification is None:
            raise RuntimeError("classification result missing")

        risk_started_at = time.perf_counter()
        with trace_span(
            "agent.risk_check",
            {
                "workspace_id": request.workspace_id,
                "category": classification.category,
                "priority": request.priority,
            },
        ):
            risk = check_support_risk(
                customer_message=request.customer_message,
                priority=request.priority,
            )
        tool_calls.append(
            ToolCallRecord(
                tool_name="risk_check_tool",
                input_summary={
                    "message_length": len(request.customer_message),
                    "priority": request.priority,
                    "category": classification.category,
                },
                output_summary={
                    "risk_level": risk.risk_level,
                    "approval_required": risk.approval_required,
                },
                latency_ms=elapsed_ms(risk_started_at),
                success=True,
            )
        )

        state["risk_level"] = risk.risk_level
        state["approval_required"] = risk.approval_required
        state["approval_status"] = (
            "pending" if risk.approval_required else "not_required"
        )
        state["final_action"] = (
            "approval_required" if risk.approval_required else "finalized"
        )
        if risk.approval_required:
            state["draft_answer"] = (
                "This support request requires human review before a response "
                "is finalized."
            )
        sync_tool_calls()
        set_metrics(citation_valid=None)
        return AgentNodeResult(
            continue_run=not risk.approval_required,
            output_summary={
                "risk_level": risk.risk_level,
                "approval_required": risk.approval_required,
            },
        )

    async def rag_search_node(state: AgentState) -> AgentNodeResult:
        nonlocal rag_search
        if classification is None:
            raise RuntimeError("classification result missing")

        rag_started_at = time.perf_counter()
        with trace_span(
            "agent.rag_search",
            {
                "workspace_id": request.workspace_id,
                "category": classification.category,
            },
        ):
            rag_search = await rag_search_tool(
                RAGSearchInput(
                    query=request.customer_message,
                    workspace_id=request.workspace_id,
                    top_k=5,
                ),
                pipeline=rag_pipeline,
            )
        tool_calls.append(
            ToolCallRecord(
                tool_name="rag_search_tool",
                input_summary={
                    "query_length": len(request.customer_message),
                    "workspace_id": request.workspace_id,
                    "top_k": 5,
                },
                output_summary={
                    "source_count": len(rag_search.sources),
                    "top_score": rag_search.top_score,
                    "refusal_recommended": rag_search.refusal_recommended,
                },
                latency_ms=elapsed_ms(rag_started_at),
                success=True,
            )
        )
        state["retrieved_sources"] = rag_search.sources
        sync_tool_calls()
        return AgentNodeResult(
            output_summary={
                "source_count": len(rag_search.sources),
                "refusal_recommended": rag_search.refusal_recommended,
            }
        )

    async def ticket_lookup_node(state: AgentState) -> AgentNodeResult:
        nonlocal ticket_lookup
        if classification is None:
            raise RuntimeError("classification result missing")

        ticket_lookup_started_at = time.perf_counter()
        with trace_span(
            "agent.ticket_lookup",
            {
                "workspace_id": request.workspace_id,
                "category": classification.category,
            },
        ):
            ticket_lookup = await ticket_lookup_tool(
                TicketLookupInput(
                    query=request.customer_message,
                    workspace_id=request.workspace_id,
                    category=classification.category,
                    limit=5,
                ),
                repository=support_ticket_repository,
            )
        tool_calls.append(
            ToolCallRecord(
                tool_name="ticket_lookup_tool",
                input_summary={
                    "query_length": len(request.customer_message),
                    "workspace_id": request.workspace_id,
                    "category": classification.category,
                    "limit": 5,
                },
                output_summary={
                    "case_count": len(ticket_lookup.cases),
                },
                latency_ms=elapsed_ms(ticket_lookup_started_at),
                success=True,
            )
        )
        state["historical_cases"] = ticket_lookup.cases
        sync_tool_calls()
        return AgentNodeResult(
            output_summary={
                "case_count": len(ticket_lookup.cases),
            }
        )

    async def draft_response_node(state: AgentState) -> AgentNodeResult:
        nonlocal draft_response
        if classification is None:
            raise RuntimeError("classification result missing")
        if rag_search is None:
            raise RuntimeError("rag search result missing")
        if ticket_lookup is None:
            raise RuntimeError("ticket lookup result missing")

        draft_started_at = time.perf_counter()
        with trace_span(
            "agent.draft_response",
            {
                "workspace_id": request.workspace_id,
                "category": classification.category,
                "source_count": len(rag_search.sources),
                "historical_case_count": len(ticket_lookup.cases),
            },
        ):
            draft_response = await draft_response_tool(
                DraftResponseInput(
                    customer_message=request.customer_message,
                    category=classification.category,
                    sources=rag_search.sources,
                    retrieval_context=rag_search.context,
                    historical_cases=ticket_lookup.cases,
                )
            )
        tool_calls.append(
            ToolCallRecord(
                tool_name="draft_response_tool",
                input_summary={
                    "message_length": len(request.customer_message),
                    "category": classification.category,
                    "source_count": len(rag_search.sources),
                    "historical_case_count": len(ticket_lookup.cases),
                },
                output_summary={
                    "citation_valid": draft_response.citation_valid,
                    "cited_source_count": len(draft_response.cited_source_ids),
                    "cited_case_count": len(draft_response.cited_case_ids),
                },
                latency_ms=elapsed_ms(draft_started_at),
                success=True,
            )
        )
        state["draft_answer"] = draft_response.draft
        state["cited_source_ids"] = draft_response.cited_source_ids
        state["cited_case_ids"] = draft_response.cited_case_ids
        state["final_answer"] = draft_response.draft
        sync_tool_calls()
        set_metrics(
            citation_valid=draft_response.citation_valid,
            retrieved_source_count=len(rag_search.sources),
            historical_case_count=len(ticket_lookup.cases),
            cited_source_count=len(draft_response.cited_source_ids),
            cited_case_count=len(draft_response.cited_case_ids),
            rag_refusal_recommended=rag_search.refusal_recommended,
        )
        return AgentNodeResult(
            output_summary={
                "citation_valid": draft_response.citation_valid,
                "cited_source_count": len(draft_response.cited_source_ids),
                "cited_case_count": len(draft_response.cited_case_ids),
            }
        )

    graph_result = await AgentGraphRunner().run(
        state=state,
        nodes=[
            AgentGraphNode("classify_ticket", classify_node),
            AgentGraphNode("risk_check", risk_check_node),
            AgentGraphNode("rag_search", rag_search_node),
            AgentGraphNode("ticket_lookup", ticket_lookup_node),
            AgentGraphNode("draft_response", draft_response_node),
        ],
    )
    state["node_runs"] = serialize_node_runs(graph_result.node_runs)
    state["metrics"] = {
        **state["metrics"],
        "node_count": len(graph_result.node_runs),
    }
    tool_call_payloads = sync_tool_calls()

    if classification is None or risk is None:
        raise RuntimeError("support triage graph did not produce risk state")

    if risk.approval_required:
        return AgentTriageResponse(
            run_id=run_id,
            status="approval_required",
            category=classification.category,
            risk_level=risk.risk_level,
            approval_required=True,
            draft_answer=state["draft_answer"],
            reason=risk.reason,
            tool_calls=tool_call_payloads,
            node_runs=state["node_runs"],
            metrics=state["metrics"],
            trace_id=trace_id,
        )

    if rag_search is None or ticket_lookup is None or draft_response is None:
        raise RuntimeError("support triage graph did not produce final state")

    return AgentTriageResponse(
        run_id=run_id,
        status="finalized",
        category=classification.category,
        risk_level=risk.risk_level,
        approval_required=False,
        final_answer=draft_response.draft,
        draft_answer=draft_response.draft,
        reason=risk.reason,
        sources=rag_search.sources,
        retrieval_context=rag_search.context,
        retrieval=rag_search.retrieval,
        historical_cases=ticket_lookup.cases,
        cited_source_ids=draft_response.cited_source_ids,
        cited_case_ids=draft_response.cited_case_ids,
        tool_calls=tool_call_payloads,
        node_runs=state["node_runs"],
        metrics=state["metrics"],
        trace_id=trace_id,
    )


def elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))
