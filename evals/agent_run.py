import argparse
import asyncio
import json
from pathlib import Path

from evals.agent_loaders import (
    DEFAULT_AGENT_EVAL_DATASET,
    load_agent_eval_dataset,
)
from evals.agent_runner import (
    AgentEvalCaseResult,
    AgentEvalRunReport,
    run_agent_eval_dataset,
)

DEFAULT_AGENT_REPORT_OUTPUT = Path("evals/reports/agent_support_triage.json")
DEFAULT_AGENT_MARKDOWN_REPORT_OUTPUT = Path("docs/agent_eval_report.md")


def serialize_agent_report(report: AgentEvalRunReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )


def write_agent_report(
    report: AgentEvalRunReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_agent_report(report) + "\n", encoding="utf-8")


def write_agent_markdown_report(
    report: AgentEvalRunReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_agent_report_markdown(report) + "\n",
        encoding="utf-8",
    )


def format_agent_report_summary(report: AgentEvalRunReport) -> str:
    lines = [
        (
            f"agent eval cases: {report.passed_cases}/"
            f"{report.total_cases} passed ({report.pass_rate:.1%})"
        )
    ]
    for dataset in report.datasets:
        lines.append(
            f"- {dataset.name}: {dataset.passed_cases}/"
            f"{dataset.total_cases} passed ({dataset.pass_rate:.1%})"
        )
    status_counts = ", ".join(
        f"{status}={count}"
        for status, count in report.status_counts.items()
    )
    risk_counts = ", ".join(
        f"{risk_level}={count}"
        for risk_level, count in report.risk_counts.items()
    )
    lines.append(f"- statuses: {status_counts}")
    lines.append(f"- risks: {risk_counts}")
    lines.append(
        "- metrics: "
        f"tools={report.metrics.tool_selection_accuracy:.1%}, "
        f"approval={report.metrics.approval_required_accuracy:.1%}, "
        "unsafe_block="
        f"{format_optional_percent(report.metrics.unsafe_action_block_rate)}, "
        f"citation={format_optional_percent(report.metrics.citation_valid_rate)}"
    )
    return "\n".join(lines)


def format_agent_report_markdown(report: AgentEvalRunReport) -> str:
    lines = [
        "# Agent Support Triage Eval Report",
        "",
        "This report summarizes the deterministic support triage eval gate for "
        "the Agentic RAG workflow. It is generated from "
        "`evals/datasets/agent_support_triage.jsonl` and the real backend "
        "workflow runner with fake RAG, fake historical ticket lookup, and "
        "fake approval persistence.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {report.total_cases} |",
        f"| Passed cases | {report.passed_cases} |",
        f"| Failed cases | {report.failed_cases} |",
        f"| Pass rate | {report.pass_rate:.1%} |",
        f"| Unsafe action cases | {report.unsafe_action_cases} |",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(format_metric_rows(report))
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| Dimension | Distribution |",
            "| --- | --- |",
            f"| Status | {format_count_map(report.status_counts)} |",
            f"| Category | {format_count_map(report.category_counts)} |",
            f"| Risk level | {format_count_map(report.risk_counts)} |",
            "",
            "## Representative Cases",
            "",
            "| Case | Category | Risk | Status | Tools | Passed |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for result in select_representative_results(report):
        lines.append(
            "| "
            f"`{result.id}` | {result.expected_category} | "
            f"{result.expected_risk_level} | {result.expected_status} | "
            f"{len(result.tool_names)} | {format_bool(result.passed)} |"
        )

    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
        ]
    )
    if report.failed_cases:
        lines.extend(format_failed_case_rows(report))
    else:
        lines.extend(
            [
                "Current deterministic run has 0 failing cases. The gate is "
                "designed to catch these regression classes:",
                "",
                "| Regression class | What would fail |",
                "| --- | --- |",
                "| Wrong classification | Category accuracy and expected "
                "category checks |",
                "| Missed high-risk action | Unsafe action block rate and "
                "approval checks |",
                "| Tool routing drift | Tool sequence accuracy |",
                "| Graph routing drift | Node sequence accuracy |",
                "| Ungrounded final answer | Citation validity and answer "
                "keyword checks |",
            ]
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "uv run python -m evals.agent_run --format summary "
            "--fail-on-failure --no-output",
            "uv run python -m evals.agent_run --format summary "
            "--fail-on-failure --markdown-output docs/agent_eval_report.md",
            "```",
        ]
    )
    return "\n".join(lines)


def format_metric_rows(report: AgentEvalRunReport) -> list[str]:
    metrics = report.metrics
    return [
        f"| Task success rate | {metrics.task_success_rate:.1%} |",
        f"| Status accuracy | {metrics.status_accuracy:.1%} |",
        f"| Category accuracy | {metrics.category_accuracy:.1%} |",
        f"| Risk level accuracy | {metrics.risk_level_accuracy:.1%} |",
        (
            "| Approval required accuracy | "
            f"{metrics.approval_required_accuracy:.1%} |"
        ),
        (
            "| Tool selection accuracy | "
            f"{metrics.tool_selection_accuracy:.1%} |"
        ),
        (
            "| Node sequence accuracy | "
            f"{metrics.node_sequence_accuracy:.1%} |"
        ),
        (
            "| Answer keyword accuracy | "
            f"{format_optional_percent(metrics.answer_keyword_accuracy)} |"
        ),
        (
            "| Approval reason keyword accuracy | "
            f"{format_optional_percent(metrics.reason_keyword_accuracy)} |"
        ),
        (
            "| Citation valid rate | "
            f"{format_optional_percent(metrics.citation_valid_rate)} |"
        ),
        (
            "| Unsafe action block rate | "
            f"{format_optional_percent(metrics.unsafe_action_block_rate)} |"
        ),
        (
            "| Average tool calls per task | "
            f"{metrics.avg_tool_calls_per_task:.2f} |"
        ),
        f"| P95 agent latency | {metrics.p95_agent_latency_ms:.0f} ms |",
    ]


def format_failed_case_rows(report: AgentEvalRunReport) -> list[str]:
    lines = [
        "| Case | Failure reasons |",
        "| --- | --- |",
    ]
    for result in report.results:
        if result.passed:
            continue
        lines.append(
            f"| `{result.id}` | {'; '.join(result.failure_reasons)} |"
        )
    return lines


def select_representative_results(
    report: AgentEvalRunReport,
) -> list[AgentEvalCaseResult]:
    finalized = [
        result for result in report.results
        if result.expected_status == "finalized"
    ][:3]
    approvals = [
        result for result in report.results
        if result.expected_status == "approval_required"
    ][:3]
    return finalized + approvals


def format_count_map(values: dict[str, int]) -> str:
    return ", ".join(f"`{name}`={count}" for name, count in values.items())


def format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Agent support triage evals."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_AGENT_EVAL_DATASET,
        help="Path to the Agent support triage JSONL dataset.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "summary"),
        default="json",
        help="Report format for stdout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AGENT_REPORT_OUTPUT,
        help="Path for the full JSON report. Use --no-output to disable.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write a report file.",
    )
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="Exit with code 1 when any Agent eval case fails.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=(
            "Optional path for a portfolio-friendly Markdown report, for "
            f"example {DEFAULT_AGENT_MARKDOWN_REPORT_OUTPUT}."
        ),
    )
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_agent_eval_dataset(args.dataset)
    report = await run_agent_eval_dataset(dataset)

    if not args.no_output:
        write_agent_report(report, args.output)
    if args.markdown_output is not None:
        write_agent_markdown_report(report, args.markdown_output)

    if args.format == "json":
        print(serialize_agent_report(report))
    else:
        print(format_agent_report_summary(report))

    if args.fail_on_failure and report.failed_cases:
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
