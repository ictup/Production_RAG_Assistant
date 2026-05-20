from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.app.db.repositories import SupportTicketRepository, SupportTicketSummary


class TicketLookupInput(BaseModel):
    query: str
    workspace_id: str = "public"
    category: str | None = None
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query", "workspace_id")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("category")
    @classmethod
    def category_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class TicketLookupOutput(BaseModel):
    cases: list[dict[str, Any]]


async def ticket_lookup_tool(
    inp: TicketLookupInput,
    *,
    repository: SupportTicketRepository,
) -> TicketLookupOutput:
    tickets = await repository.list_similar_support_tickets(
        query=inp.query,
        workspace_id=inp.workspace_id,
        category=inp.category,
        limit=inp.limit,
    )
    return TicketLookupOutput(
        cases=[
            serialize_support_ticket_summary(ticket)
            for ticket in tickets
        ]
    )


def serialize_support_ticket_summary(
    ticket: SupportTicketSummary,
) -> dict[str, Any]:
    return {
        "id": str(ticket.id),
        "ticket_id": ticket.ticket_id,
        "workspace_id": ticket.workspace_id,
        "category": ticket.category,
        "customer_message": ticket.customer_message,
        "resolution_summary": ticket.resolution_summary,
        "final_response": ticket.final_response,
        "tags": list(ticket.tags),
        "risk_level": ticket.risk_level,
        "metadata": dict(ticket.metadata),
        "created_at": ticket.created_at.isoformat(),
    }
