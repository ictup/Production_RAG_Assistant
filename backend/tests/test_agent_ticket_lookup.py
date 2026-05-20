import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.agent.ticket_lookup import TicketLookupInput, ticket_lookup_tool
from backend.app.db.repositories import SupportTicketSummary


def make_ticket_summary() -> SupportTicketSummary:
    return SupportTicketSummary(
        id=uuid.UUID("66666666-6666-6666-6666-666666666666"),
        ticket_id="TICKET-1",
        workspace_id="public",
        category="rag_failure",
        customer_message="The answer has wrong citations.",
        resolution_summary="Inspect retrieved chunks and citation validation.",
        final_response="Check retrieved chunks and citation mappings.",
        tags=["citations", "rag"],
        risk_level="low",
        metadata={"source": "seed"},
        created_at=datetime(2026, 5, 20, 8, 0, tzinfo=UTC),
    )


class FakeSupportTicketRepository:
    def __init__(self) -> None:
        self.calls = []

    async def list_similar_support_tickets(self, **kwargs):
        self.calls.append(dict(kwargs))
        return [make_ticket_summary()]


@pytest.mark.asyncio
async def test_ticket_lookup_tool_returns_serialized_cases() -> None:
    repository = FakeSupportTicketRepository()

    output = await ticket_lookup_tool(
        TicketLookupInput(
            query=" wrong citations ",
            workspace_id=" public ",
            category=" rag_failure ",
            limit=3,
        ),
        repository=repository,  # type: ignore[arg-type]
    )

    assert repository.calls == [
        {
            "query": "wrong citations",
            "workspace_id": "public",
            "category": "rag_failure",
            "limit": 3,
        }
    ]
    assert output.cases == [
        {
            "id": "66666666-6666-6666-6666-666666666666",
            "ticket_id": "TICKET-1",
            "workspace_id": "public",
            "category": "rag_failure",
            "customer_message": "The answer has wrong citations.",
            "resolution_summary": (
                "Inspect retrieved chunks and citation validation."
            ),
            "final_response": "Check retrieved chunks and citation mappings.",
            "tags": ["citations", "rag"],
            "risk_level": "low",
            "metadata": {"source": "seed"},
            "created_at": "2026-05-20T08:00:00+00:00",
        }
    ]


def test_ticket_lookup_input_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        TicketLookupInput(query="", limit=0)
