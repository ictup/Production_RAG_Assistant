import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.app.agent.state import AgentState
from backend.app.core.tracing import trace_span

AgentNodeHandler = Callable[[AgentState], Awaitable["AgentNodeResult | None"]]


class AgentNodeResult(BaseModel):
    continue_run: bool = True
    output_summary: dict[str, Any] = Field(default_factory=dict)


class AgentNodeRunRecord(BaseModel):
    node_name: str
    output_summary: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(ge=0)
    success: bool
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentGraphRunResult(BaseModel):
    node_runs: list[AgentNodeRunRecord]


@dataclass(frozen=True)
class AgentGraphNode:
    name: str
    handler: AgentNodeHandler

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("agent graph node name must not be blank")
        object.__setattr__(self, "name", name)


class AgentGraphRunner:
    async def run(
        self,
        *,
        state: AgentState,
        nodes: Sequence[AgentGraphNode],
    ) -> AgentGraphRunResult:
        node_runs: list[AgentNodeRunRecord] = []
        for node in nodes:
            started_at = time.perf_counter()
            try:
                with trace_span(
                    "agent.graph.node",
                    {
                        "run_id": state["run_id"],
                        "workspace_id": state["workspace_id"],
                        "node_name": node.name,
                    },
                ):
                    result = await node.handler(state)
                result = result or AgentNodeResult()
                node_runs.append(
                    AgentNodeRunRecord(
                        node_name=node.name,
                        output_summary=result.output_summary,
                        latency_ms=elapsed_ms(started_at),
                        success=True,
                    )
                )
                state["node_runs"] = serialize_node_runs(node_runs)
                if not result.continue_run:
                    break
            except Exception as exc:
                node_runs.append(
                    AgentNodeRunRecord(
                        node_name=node.name,
                        latency_ms=elapsed_ms(started_at),
                        success=False,
                        error=exc.__class__.__name__,
                    )
                )
                state["node_runs"] = serialize_node_runs(node_runs)
                state["errors"].append(
                    {
                        "node_name": node.name,
                        "error": exc.__class__.__name__,
                    }
                )
                state["final_action"] = "failed"
                raise
        return AgentGraphRunResult(node_runs=node_runs)


def serialize_node_runs(
    node_runs: Sequence[AgentNodeRunRecord],
) -> list[dict[str, Any]]:
    return [
        node_run.model_dump(mode="json")
        for node_run in node_runs
    ]


def elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))
