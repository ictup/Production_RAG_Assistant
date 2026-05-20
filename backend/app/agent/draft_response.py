from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.app.rag.citations import validate_citations

MAX_CONTEXT_SNIPPET_CHARS = 180
MAX_CASE_SNIPPET_CHARS = 140
MAX_CITED_SOURCES = 3
MAX_CITED_CASES = 2

CATEGORY_GUIDANCE: dict[str, str] = {
    "rag_failure": (
        "reproduce the exact question, inspect the retrieved chunks and scores, "
        "then verify citation validation before changing documents or retrieval "
        "settings"
    ),
    "serving_latency": (
        "capture request timing, compare p95 latency across retrieval and "
        "generation, then isolate the slow stage before tuning capacity"
    ),
    "rate_limit": (
        "confirm the provider quota, check request bursts, then apply backoff "
        "or reduce concurrency before asking for a quota increase"
    ),
    "deployment": (
        "verify the active configuration, migration state, container health, "
        "and rollback path before changing production deployment settings"
    ),
    "evaluation": (
        "re-run the failing eval case, compare expected sources and keywords, "
        "then inspect the retrieval and generation traces"
    ),
    "security": (
        "avoid sharing secrets, preserve the request id, and escalate for "
        "security review before taking action"
    ),
    "data_privacy": (
        "avoid exposing customer data, preserve audit context, and escalate "
        "for privacy review before taking action"
    ),
}


class DraftResponseInput(BaseModel):
    customer_message: str
    category: str = "unknown"
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_context: str | None = None
    historical_cases: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("customer_message")
    @classmethod
    def customer_message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("customer_message must not be blank")
        return value

    @field_validator("category")
    @classmethod
    def category_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        return value or "unknown"

    @field_validator("retrieval_context")
    @classmethod
    def retrieval_context_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class DraftResponseOutput(BaseModel):
    draft: str
    cited_source_ids: list[str] = Field(default_factory=list)
    cited_case_ids: list[str] = Field(default_factory=list)
    citation_valid: bool


async def draft_response_tool(inp: DraftResponseInput) -> DraftResponseOutput:
    cited_sources = inp.sources[:MAX_CITED_SOURCES]
    cited_source_ids = [
        source_id_for(source, index)
        for index, source in enumerate(cited_sources, start=1)
    ]
    cited_cases = inp.historical_cases[:MAX_CITED_CASES]
    cited_case_ids = [
        case_id
        for case in cited_cases
        if (case_id := text_value(case.get("ticket_id")))
    ]
    draft = build_draft_response(
        inp=inp,
        cited_sources=cited_sources,
        cited_cases=cited_cases,
    )
    return DraftResponseOutput(
        draft=draft,
        cited_source_ids=cited_source_ids,
        cited_case_ids=cited_case_ids,
        citation_valid=validate_citations(draft, len(inp.sources)),
    )


def build_draft_response(
    *,
    inp: DraftResponseInput,
    cited_sources: list[dict[str, Any]],
    cited_cases: list[dict[str, Any]],
) -> str:
    category = format_category(inp.category)
    guidance = CATEGORY_GUIDANCE.get(
        inp.category,
        (
            "collect the request id, workspace id, reproduction steps, and "
            "relevant logs before making changes"
        ),
    )
    if not cited_sources:
        return (
            f"Thanks for the details. This looks like a {category} support "
            "issue, but I do not have enough grounded knowledge base context "
            "to provide a final fix. Please collect the request id, workspace "
            "id, reproduction steps, relevant logs, and the top retrieved "
            "chunks before escalating."
        )

    primary_source = cited_sources[0]
    source_reference = format_source_reference(primary_source)
    context_snippet = summarize_context(inp.retrieval_context)

    paragraphs = [
        (
            f"Thanks for the details. This looks like a {category} support "
            f"issue. I would start from {source_reference} because it is the "
            "strongest retrieved knowledge source for this request [1]."
        ),
        f"Suggested response: {guidance} [1].",
    ]
    if context_snippet:
        paragraphs.append(f"Relevant retrieved context: {context_snippet} [1].")

    for case in cited_cases:
        case_summary = format_case_summary(case)
        if case_summary:
            paragraphs.append(case_summary)

    paragraphs.append(
        "If the issue persists after these checks, capture the request id, "
        "workspace id, top retrieved chunks, and citation validation result "
        "before escalating."
    )
    return "\n\n".join(paragraphs)


def format_source_reference(source: dict[str, Any]) -> str:
    title = text_value(source.get("title")) or "the retrieved source"
    section = text_value(source.get("section"))
    source_uri = text_value(source.get("source_uri"))
    if section and source_uri:
        return f"{title}, section {section} ({source_uri})"
    if section:
        return f"{title}, section {section}"
    if source_uri:
        return f"{title} ({source_uri})"
    return title


def format_case_summary(case: dict[str, Any]) -> str | None:
    ticket_id = text_value(case.get("ticket_id"))
    summary = text_value(case.get("resolution_summary")) or text_value(
        case.get("final_response")
    )
    if not ticket_id or not summary:
        return None
    summary = truncate_text(summary, MAX_CASE_SNIPPET_CHARS)
    return (
        f"Similar historical case {ticket_id} was resolved by: {summary}. "
        "Use it as internal context, but verify the current customer details "
        "before reusing the same resolution."
    )


def summarize_context(value: str | None) -> str | None:
    if not value:
        return None
    lines = [
        line.strip()
        for line in value.splitlines()
        if line.strip() and not line.strip().startswith("[")
    ]
    snippet = " ".join(lines) or value
    return truncate_text(snippet, MAX_CONTEXT_SNIPPET_CHARS)


def format_category(value: str) -> str:
    return " ".join(value.replace("_", " ").split()) or "unknown"


def source_id_for(source: dict[str, Any], fallback_index: int) -> str:
    return text_value(source.get("source_id")) or str(fallback_index)


def text_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def truncate_text(value: str, max_chars: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3].rstrip()}..."
