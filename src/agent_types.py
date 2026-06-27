from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    query: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentStep:
    thought: str
    action: ToolCall
    observation: ToolObservation


@dataclass(frozen=True)
class AgentRunResult:
    question: str
    steps: list[AgentStep]
    final_answer: str
    loop_count: int
    reflection_passed: bool | None = None


def tool_observation_to_dict(observation: ToolObservation) -> dict[str, Any]:
    return {
        "tool_name": observation.tool_name,
        "query": observation.query,
        "filters": observation.filters,
        "results": observation.results,
        "output": observation.output,
    }


def agent_step_to_dict(step: AgentStep) -> dict[str, Any]:
    return {
        "thought": step.thought,
        "action": {
            "name": step.action.name,
            "arguments": step.action.arguments,
        },
        "observation": tool_observation_to_dict(step.observation),
    }


def agent_run_result_to_dict(result: AgentRunResult) -> dict[str, Any]:
    return {
        "question": result.question,
        "final_answer": result.final_answer,
        "loop_count": result.loop_count,
        "reflection_passed": result.reflection_passed,
        "steps": [agent_step_to_dict(step) for step in result.steps],
    }
