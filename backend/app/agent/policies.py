from pydantic import BaseModel, Field

from backend.app.agent.state import RiskLevel, TicketCategory

CATEGORY_KEYWORDS: dict[TicketCategory, tuple[str, ...]] = {
    "data_privacy": (
        "customer prompt",
        "customer data",
        "personal data",
        "pii",
        "privacy",
        "gdpr",
        "delete all logs",
        "export logs",
    ),
    "security": (
        "ignore previous instructions",
        "prompt injection",
        "jailbreak",
        "secret",
        "api key",
        "credential",
        "token leak",
    ),
    "rate_limit": (
        "429",
        "rate limit",
        "too many requests",
        "quota",
        "requests per minute",
    ),
    "serving_latency": (
        "latency",
        "p95",
        "slow",
        "timeout",
        "throughput",
    ),
    "rag_failure": (
        "citation",
        "retrieved chunk",
        "retrieval",
        "hallucination",
        "answer does not match",
        "wrong answer",
    ),
    "deployment": (
        "deploy",
        "docker",
        "compose",
        "migration",
        "postgres",
        "production config",
    ),
    "evaluation": (
        "eval",
        "evaluation",
        "regression",
        "dataset",
        "benchmark",
    ),
    "unknown": (),
}

CATEGORY_PRIORITY: tuple[TicketCategory, ...] = (
    "data_privacy",
    "security",
    "serving_latency",
    "rate_limit",
    "rag_failure",
    "deployment",
    "evaluation",
)

HIGH_RISK_TERMS = (
    "delete",
    "remove logs",
    "export customer",
    "customer data",
    "customer prompt",
    "personal data",
    "pii",
    "refund",
    "account change",
    "api key",
    "credential",
    "secret",
    "production immediately",
)

MEDIUM_RISK_TERMS = (
    "change config",
    "update config",
    "production config",
    "restart",
    "deploy",
    "migration",
    "increase limit",
    "decrease limit",
    "rate limit",
)


class TicketClassification(BaseModel):
    category: TicketCategory
    risk_level: RiskLevel
    matched_terms: list[str] = Field(default_factory=list)


class RiskCheckResult(BaseModel):
    risk_level: RiskLevel
    approval_required: bool
    reason: str


def classify_ticket(message: str) -> TicketClassification:
    normalized = normalize_policy_text(message)

    for category in CATEGORY_PRIORITY:
        keywords = CATEGORY_KEYWORDS[category]
        matched_terms = [term for term in keywords if term in normalized]
        if matched_terms:
            return TicketClassification(
                category=category,
                risk_level=default_risk_for_category(category),
                matched_terms=matched_terms,
            )

    return TicketClassification(category="unknown", risk_level="low")


def check_support_risk(
    *,
    customer_message: str,
    draft_answer: str = "",
    priority: str = "normal",
) -> RiskCheckResult:
    combined_text = normalize_policy_text(f"{customer_message}\n{draft_answer}")
    classification = classify_ticket(customer_message)

    high_risk_matches = [term for term in HIGH_RISK_TERMS if term in combined_text]
    if classification.category in {"data_privacy", "security"} or high_risk_matches:
        reason = build_reason(
            "high-risk support request",
            high_risk_matches or classification.matched_terms,
        )
        return RiskCheckResult(
            risk_level="high",
            approval_required=True,
            reason=reason,
        )

    medium_risk_matches = [
        term for term in MEDIUM_RISK_TERMS if term in combined_text
    ]
    if medium_risk_matches:
        if priority in {"high", "urgent"}:
            return RiskCheckResult(
                risk_level="high",
                approval_required=True,
                reason=build_reason(
                    "production-impacting guidance on high-priority ticket",
                    medium_risk_matches,
                ),
            )
        return RiskCheckResult(
            risk_level="medium",
            approval_required=False,
            reason=build_reason("operational guidance", medium_risk_matches),
        )

    return RiskCheckResult(
        risk_level=classification.risk_level,
        approval_required=False,
        reason=f"{classification.category} ticket can be handled without approval.",
    )


def default_risk_for_category(category: TicketCategory) -> RiskLevel:
    if category in {"data_privacy", "security"}:
        return "high"
    if category in {"deployment", "serving_latency", "rate_limit"}:
        return "medium"
    return "low"


def normalize_policy_text(value: str) -> str:
    return " ".join(value.lower().split())


def build_reason(prefix: str, matched_terms: list[str]) -> str:
    if not matched_terms:
        return prefix
    terms = ", ".join(sorted(set(matched_terms)))
    return f"{prefix}: matched {terms}"
