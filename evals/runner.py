from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from backend.app.rag.pipeline import ChatPipelineResponse
from evals.models import EvalCase, EvalCaseType, EvalSuite

AnswerEvalCase = Callable[[EvalCase], Awaitable[ChatPipelineResponse]]


class EvalCaseResult(BaseModel):
    dataset_name: str
    id: str
    case_type: EvalCaseType
    question: str
    passed: bool
    failure_reasons: list[str]
    answer: str
    refused: bool
    refusal_reason: str | None
    citation_valid: bool | None
    source_match: bool | None
    keyword_match: bool | None
    refusal_match: bool | None
    missing_sources: list[str]
    missing_keywords: list[str]
    sources: list[str]
    retrieval: dict[str, Any]
    usage: dict[str, Any]


class EvalDatasetResult(BaseModel):
    name: str
    case_type: EvalCaseType
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float


class EvalRunReport(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    datasets: list[EvalDatasetResult]
    results: list[EvalCaseResult]


async def run_eval_suite(
    suite: EvalSuite,
    *,
    answer_case: AnswerEvalCase,
) -> EvalRunReport:
    results: list[EvalCaseResult] = []

    for dataset in suite.datasets:
        for eval_case in dataset.cases:
            try:
                response = await answer_case(eval_case)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    build_error_result(
                        eval_case,
                        dataset_name=dataset.name,
                        exc=exc,
                    )
                )
                continue

            results.append(
                score_eval_case(
                    eval_case,
                    response=response,
                    dataset_name=dataset.name,
                )
            )

    return build_eval_report(suite, results)


def score_eval_case(
    eval_case: EvalCase,
    *,
    response: ChatPipelineResponse,
    dataset_name: str,
) -> EvalCaseResult:
    failure_reasons: list[str] = []
    refused = response.refusal is not None
    refusal_match = None
    source_match = None
    keyword_match = None
    missing_sources: list[str] = []
    missing_keywords: list[str] = []

    if eval_case.case_type == "rag":
        if refused:
            failure_reasons.append("unexpected_refusal")

        missing_sources = find_missing_expected_sources(
            eval_case.expected_sources,
            response,
        )
        source_match = not missing_sources
        if missing_sources:
            failure_reasons.append(
                "missing_expected_sources: " + ", ".join(missing_sources)
            )

        missing_keywords = find_missing_answer_keywords(
            eval_case.expected_keywords,
            response.answer,
        )
        keyword_match = not missing_keywords
        if missing_keywords:
            failure_reasons.append(
                "missing_expected_keywords: " + ", ".join(missing_keywords)
            )

        if eval_case.must_cite and response.citation_valid is not True:
            failure_reasons.append("expected_valid_citation")

    if eval_case.should_refuse:
        refusal_match = refused
        if not refused:
            failure_reasons.append("expected_refusal")

    if (
        eval_case.case_type == "security"
        and eval_case.should_not_follow_retrieved_instruction
        and not eval_case.should_refuse
    ):
        failure_reasons.append("unsupported_security_expectation")

    return EvalCaseResult(
        dataset_name=dataset_name,
        id=eval_case.id,
        case_type=eval_case.case_type,
        question=eval_case.question,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        answer=response.answer,
        refused=refused,
        refusal_reason=response.refusal.reason if response.refusal else None,
        citation_valid=response.citation_valid,
        source_match=source_match,
        keyword_match=keyword_match,
        refusal_match=refusal_match,
        missing_sources=missing_sources,
        missing_keywords=missing_keywords,
        sources=[format_source(source.model_dump()) for source in response.sources],
        retrieval=response.retrieval.model_dump(),
        usage=response.usage.model_dump(),
    )


def build_error_result(
    eval_case: EvalCase,
    *,
    dataset_name: str,
    exc: Exception,
) -> EvalCaseResult:
    return EvalCaseResult(
        dataset_name=dataset_name,
        id=eval_case.id,
        case_type=eval_case.case_type,
        question=eval_case.question,
        passed=False,
        failure_reasons=[f"runner_error: {type(exc).__name__}: {exc}"],
        answer="",
        refused=False,
        refusal_reason=None,
        citation_valid=None,
        source_match=None,
        keyword_match=None,
        refusal_match=None,
        missing_sources=[],
        missing_keywords=[],
        sources=[],
        retrieval={},
        usage={},
    )


def build_eval_report(
    suite: EvalSuite,
    results: list[EvalCaseResult],
) -> EvalRunReport:
    dataset_results: list[EvalDatasetResult] = []
    for dataset in suite.datasets:
        dataset_case_results = [
            result for result in results if result.dataset_name == dataset.name
        ]
        dataset_results.append(
            EvalDatasetResult(
                name=dataset.name,
                case_type=dataset.case_type,
                total_cases=len(dataset_case_results),
                passed_cases=count_passed(dataset_case_results),
                failed_cases=count_failed(dataset_case_results),
                pass_rate=calculate_pass_rate(dataset_case_results),
            )
        )

    return EvalRunReport(
        total_cases=len(results),
        passed_cases=count_passed(results),
        failed_cases=count_failed(results),
        pass_rate=calculate_pass_rate(results),
        datasets=dataset_results,
        results=results,
    )


def find_missing_expected_sources(
    expected_sources: list[str],
    response: ChatPipelineResponse,
) -> list[str]:
    source_text = "\n".join(
        format_source(source.model_dump()) for source in response.sources
    )
    normalized_source_text = normalize_for_match(source_text)
    return [
        expected_source
        for expected_source in expected_sources
        if normalize_for_match(expected_source) not in normalized_source_text
    ]


def find_missing_answer_keywords(
    expected_keywords: list[str],
    answer: str,
) -> list[str]:
    normalized_answer = normalize_for_match(answer)
    return [
        keyword
        for keyword in expected_keywords
        if normalize_for_match(keyword) not in normalized_answer
    ]


def normalize_for_match(value: str) -> str:
    return value.casefold()


def format_source(source: dict[str, Any]) -> str:
    title = source.get("title") or ""
    section = source.get("section") or ""
    source_uri = source.get("source_uri") or ""
    return f"{source_uri} {title} {section}".strip()


def count_passed(results: list[EvalCaseResult]) -> int:
    return sum(1 for result in results if result.passed)


def count_failed(results: list[EvalCaseResult]) -> int:
    return sum(1 for result in results if not result.passed)


def calculate_pass_rate(results: list[EvalCaseResult]) -> float:
    if not results:
        return 0.0
    return count_passed(results) / len(results)
