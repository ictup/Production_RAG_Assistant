import pytest

from backend.app.agent.graph import (
    AgentGraphNode,
    AgentGraphRunner,
    AgentNodeResult,
)
from backend.app.agent.state import build_initial_agent_state


@pytest.mark.asyncio
async def test_agent_graph_runner_executes_nodes_in_order() -> None:
    state = build_initial_agent_state(
        run_id="run-1",
        ticket_id="TICKET-1",
        customer_message="How do I debug citations?",
    )
    visited: list[str] = []

    async def first_node(state):
        visited.append("first")
        state["metadata"]["first"] = True
        return AgentNodeResult(output_summary={"step": 1})

    async def second_node(state):
        visited.append("second")
        state["metadata"]["second"] = True
        return AgentNodeResult(output_summary={"step": 2})

    result = await AgentGraphRunner().run(
        state=state,
        nodes=[
            AgentGraphNode("first", first_node),
            AgentGraphNode("second", second_node),
        ],
    )

    assert visited == ["first", "second"]
    assert [node.node_name for node in result.node_runs] == ["first", "second"]
    assert state["metadata"] == {"first": True, "second": True}
    assert state["node_runs"][0]["node_name"] == "first"
    assert state["node_runs"][1]["output_summary"] == {"step": 2}


@pytest.mark.asyncio
async def test_agent_graph_runner_stops_when_node_requests_stop() -> None:
    state = build_initial_agent_state(
        run_id="run-1",
        ticket_id="TICKET-1",
        customer_message="Delete customer data.",
    )
    visited: list[str] = []

    async def stop_node(state):
        visited.append("stop")
        return AgentNodeResult(
            continue_run=False,
            output_summary={"approval_required": True},
        )

    async def skipped_node(state):
        visited.append("skipped")
        return AgentNodeResult()

    result = await AgentGraphRunner().run(
        state=state,
        nodes=[
            AgentGraphNode("stop", stop_node),
            AgentGraphNode("skipped", skipped_node),
        ],
    )

    assert visited == ["stop"]
    assert [node.node_name for node in result.node_runs] == ["stop"]
    assert state["node_runs"][0]["output_summary"] == {
        "approval_required": True,
    }


@pytest.mark.asyncio
async def test_agent_graph_runner_records_failed_node_before_reraising() -> None:
    state = build_initial_agent_state(
        run_id="run-1",
        ticket_id="TICKET-1",
        customer_message="How do I debug citations?",
    )

    async def failing_node(state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await AgentGraphRunner().run(
            state=state,
            nodes=[AgentGraphNode("failing", failing_node)],
        )

    assert state["final_action"] == "failed"
    assert state["errors"] == [
        {
            "node_name": "failing",
            "error": "RuntimeError",
        }
    ]
    assert state["node_runs"][0]["node_name"] == "failing"
    assert state["node_runs"][0]["success"] is False
    assert state["node_runs"][0]["error"] == "RuntimeError"


def test_agent_graph_node_rejects_blank_name() -> None:
    async def handler(state):
        return AgentNodeResult()

    with pytest.raises(ValueError, match="node name"):
        AgentGraphNode(" ", handler)
