import pytest
from pydantic import ValidationError

from backend.app.agent.draft_response import (
    DraftResponseInput,
    draft_response_tool,
)


@pytest.mark.asyncio
async def test_draft_response_tool_builds_cited_support_draft() -> None:
    output = await draft_response_tool(
        DraftResponseInput(
            customer_message="How can I debug citation validation failures?",
            category="rag_failure",
            sources=[
                {
                    "source_id": "1",
                    "title": "Citation Debugging",
                    "section": "Validation",
                    "source_uri": "docs/citations.md",
                    "chunk_id": "chunk-1",
                    "score": 0.92,
                }
            ],
            retrieval_context=(
                "[1] Citation Debugging\n"
                "Inspect retrieved chunks and citation validation output."
            ),
            historical_cases=[
                {
                    "ticket_id": "TICKET-123",
                    "resolution_summary": (
                        "Reindexed the affected chunks and verified the "
                        "citation validator result."
                    ),
                }
            ],
        )
    )

    assert "Citation Debugging" in output.draft
    assert "[1]" in output.draft
    assert "TICKET-123" in output.draft
    assert output.cited_source_ids == ["1"]
    assert output.cited_case_ids == ["TICKET-123"]
    assert output.citation_valid is True


@pytest.mark.asyncio
async def test_draft_response_tool_handles_missing_sources() -> None:
    output = await draft_response_tool(
        DraftResponseInput(
            customer_message="How can I debug citation validation failures?",
            category="rag_failure",
            sources=[],
            historical_cases=[],
        )
    )

    assert "do not have enough grounded knowledge base context" in output.draft
    assert output.cited_source_ids == []
    assert output.cited_case_ids == []
    assert output.citation_valid is False


def test_draft_response_input_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        DraftResponseInput(customer_message=" ")


def test_draft_response_input_normalizes_blank_optional_fields() -> None:
    inp = DraftResponseInput(
        customer_message=" Message ",
        category=" ",
        retrieval_context=" ",
    )

    assert inp.customer_message == "Message"
    assert inp.category == "unknown"
    assert inp.retrieval_context is None
