# Agentic RAG Support Workflow

This document tracks the V3 extension of Production RAG Assistant. The goal is
to turn the existing RAG backend into a controlled, observable, evaluable
support triage workflow.

The extension is not an unrestricted autonomous agent. It is a backend-owned
state machine with explicit tools, schemas, risk policies, human approval for
high-risk paths, tracing, metrics, and evals.

## Target Workflow

```text
support ticket
-> classify ticket
-> retrieve grounded knowledge
-> search historical support cases
-> draft cited response
-> run risk check
-> route high-risk drafts to human approval
-> finalize safe responses
```

## Current Foundation

The repository already provides the production RAG layer that this workflow can
reuse:

- FastAPI API surface.
- Workspace isolation and API key roles.
- Postgres and pgvector data model.
- Hybrid vector and sparse retrieval.
- Query rewrite, reranking, citation validation, and refusal guards.
- Chat logs, export jobs, audit logs, metrics, evals, Docker, and CI.

## Step 1 Scope

The first implementation step adds only the stable contracts for the future
workflow:

- `backend.app.agent.state.AgentState`
- `backend.app.schemas.agent.SupportTicketRequest`
- rule-based ticket classification policy
- rule-based support risk policy
- MCP-style tool specs and tool call records

This step intentionally does not add LangGraph, new database tables, or public
agent endpoints. Those will be added after the contracts are tested.

## Tool Registry

The MVP workflow will use these backend-controlled tools:

| Tool | Purpose | Risk |
| --- | --- | --- |
| `rag_search_tool` | Search the internal RAG knowledge base | low |
| `ticket_lookup_tool` | Find similar historical support tickets | low |
| `draft_response_tool` | Draft a cited support response | medium |
| `risk_check_tool` | Classify risk and approval need | low |
| `human_approval_tool` | Create an internal approval request | high |

Tools must have explicit input and output schemas. The workflow must not execute
arbitrary code, run unrestricted SQL, send external messages, or call tools that
are not registered by the backend.

## Risk Policy

High-risk requests require human approval before finalization. Examples:

- deleting or exporting customer data
- handling private customer prompts or PII
- secrets, credentials, API keys, or prompt injection attempts
- refunds or account state changes
- production-impacting actions on urgent or high-priority tickets

Medium-risk requests can be drafted without approval but should remain careful
and auditable. Examples include deployment guidance, rate-limit tuning,
migration advice, and latency troubleshooting.

## Planned Next Steps

1. Add the support triage route skeleton under `/agent/support-triage`.
2. Add the graph runner abstraction, initially without LangGraph.
3. Implement `rag_search_tool` by reusing the existing RAG pipeline.
4. Add `support_tickets` and `agent_approvals` database tables.
5. Add approval API endpoints.
6. Add agent-specific Prometheus metrics.
7. Add 30 support eval cases and an agent eval report.

