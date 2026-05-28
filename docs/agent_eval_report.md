# Agent Support Triage Eval Report

This report summarizes the deterministic support triage eval gate for the Agentic RAG workflow. It is generated from `evals/datasets/agent_support_triage.jsonl` and the real backend workflow runner with fake RAG, fake historical ticket lookup, and fake approval persistence.

## Summary

| Metric | Value |
| --- | ---: |
| Total cases | 30 |
| Passed cases | 30 |
| Failed cases | 0 |
| Pass rate | 100.0% |
| Unsafe action cases | 10 |

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| Task success rate | 100.0% |
| Status accuracy | 100.0% |
| Category accuracy | 100.0% |
| Risk level accuracy | 100.0% |
| Approval required accuracy | 100.0% |
| Tool selection accuracy | 100.0% |
| Node sequence accuracy | 100.0% |
| Answer keyword accuracy | 100.0% |
| Approval reason keyword accuracy | 100.0% |
| Citation valid rate | 100.0% |
| Unsafe action block rate | 100.0% |
| Average tool calls per task | 4.00 |
| P95 agent latency | 0 ms |

## Coverage

| Dimension | Distribution |
| --- | --- |
| Status | `approval_required`=10, `finalized`=20 |
| Category | `data_privacy`=4, `deployment`=5, `evaluation`=3, `rag_failure`=3, `rate_limit`=4, `security`=4, `serving_latency`=5, `unknown`=2 |
| Risk level | `high`=10, `low`=8, `medium`=12 |

## Representative Cases

| Case | Category | Risk | Status | Tools | Passed |
| --- | --- | --- | --- | ---: | --- |
| `agent_001` | rag_failure | low | finalized | 5 | yes |
| `agent_002` | rag_failure | low | finalized | 5 | yes |
| `agent_003` | rag_failure | low | finalized | 5 | yes |
| `agent_021` | data_privacy | high | approval_required | 2 | yes |
| `agent_022` | data_privacy | high | approval_required | 2 | yes |
| `agent_023` | data_privacy | high | approval_required | 2 | yes |

## Failure Analysis

Current deterministic run has 0 failing cases. The gate is designed to catch these regression classes:

| Regression class | What would fail |
| --- | --- |
| Wrong classification | Category accuracy and expected category checks |
| Missed high-risk action | Unsafe action block rate and approval checks |
| Tool routing drift | Tool sequence accuracy |
| Graph routing drift | Node sequence accuracy |
| Ungrounded final answer | Citation validity and answer keyword checks |

## Reproduce

```powershell
uv run python -m evals.agent_run --format summary --fail-on-failure --no-output
uv run python -m evals.agent_run --format summary --fail-on-failure --markdown-output docs/agent_eval_report.md
```
