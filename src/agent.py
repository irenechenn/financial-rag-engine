from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.agent_types import AgentRunResult, AgentStep, ToolCall, ToolObservation, tool_observation_to_dict
from src.interfaces import LLMProvider


ToolHandler = Callable[..., Awaitable[ToolObservation]]


@dataclass(frozen=True)
class AgentDecision:
    thought: str
    action: ToolCall | None = None
    final_answer: str | None = None


class ReActAgent:
    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: dict[str, ToolHandler],
        max_loops: int = 3,
    ) -> None:
        if max_loops <= 0:
            raise ValueError("max_loops must be positive.")

        self.llm_provider = llm_provider
        self.tools = tools
        self.max_loops = max_loops

    async def run(self, question: str) -> AgentRunResult:
        steps: list[AgentStep] = []

        for _ in range(self.max_loops):
            decision = parse_agent_decision(
                await self.llm_provider.generate(
                    system_prompt=build_react_system_prompt(self.tools),
                    user_prompt=build_react_user_prompt(question, steps),
                )
            )

            if decision.final_answer is not None:
                return AgentRunResult(
                    question=question,
                    steps=steps,
                    final_answer=decision.final_answer,
                    loop_count=len(steps),
                )

            if decision.action is None:
                raise ValueError("Agent decision must include either action or final_answer.")

            observation = await self._run_tool(decision.action)
            steps.append(
                AgentStep(
                    thought=decision.thought,
                    action=decision.action,
                    observation=observation,
                )
            )

        return AgentRunResult(
            question=question,
            steps=steps,
            final_answer="Unable to complete within max_loops.",
            loop_count=len(steps),
            reflection_passed=False,
        )

    async def _run_tool(self, tool_call: ToolCall) -> ToolObservation:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolObservation(
                tool_name=tool_call.name,
                output={
                    "error": f"Unknown tool: {tool_call.name}",
                    "available_tools": sorted(self.tools),
                    "recovery_hint": "Call one of the available tools or return final_answer as a top-level field.",
                },
            )
        try:
            return await tool(**tool_call.arguments)
        except TypeError as exc:
            return ToolObservation(
                tool_name=tool_call.name,
                output={
                    "error": f"Invalid arguments for tool {tool_call.name}: {exc}",
                    "arguments": tool_call.arguments,
                    "recovery_hint": "Retry with the tool schema exactly as specified in the system prompt.",
                },
            )


def parse_agent_decision(raw_response: str) -> AgentDecision:
    try:
        payload = _load_first_json_object(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent response must be valid JSON: {raw_response}") from exc

    thought = str(payload.get("thought", "")).strip()
    if not thought:
        raise ValueError("Agent response must include a non-empty thought.")

    final_answer = payload.get("final_answer")
    action_payload = payload.get("action")

    if final_answer is not None and action_payload is not None:
        raise ValueError("Agent response cannot include both action and final_answer.")
    if final_answer is None and action_payload is None:
        raise ValueError("Agent response must include action or final_answer.")

    action = None
    if action_payload is not None:
        if not isinstance(action_payload, dict):
            raise ValueError("Agent action must be an object.")

        name = action_payload.get("name")
        arguments = action_payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Agent action must include a tool name.")
        if not isinstance(arguments, dict):
            raise ValueError("Agent action arguments must be an object.")
        if name == "final_answer":
            coerced_final_answer = _coerce_final_answer_action(arguments)
            if coerced_final_answer:
                return AgentDecision(thought=thought, final_answer=coerced_final_answer)

        action = ToolCall(name=name, arguments=arguments)

    return AgentDecision(
        thought=thought,
        action=action,
        final_answer=str(final_answer).strip() if final_answer is not None else None,
    )


def _coerce_final_answer_action(arguments: dict[str, Any]) -> str | None:
    for key in ("final_answer", "answer", "text", "content"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in arguments.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    if arguments:
        return json.dumps(arguments, ensure_ascii=False)
    return None


def build_react_system_prompt(tools: dict[str, ToolHandler]) -> str:
    tool_names = ", ".join(sorted(tools))
    return f"""You are a financial analysis agent.
You must solve questions through a ReAct loop: Thought -> Action -> Observation -> Final Answer.
Available tools: {tool_names}.
Do not call final_answer as a tool. final_answer is only a top-level field used when finishing.

Tool schemas:
- search_transcript_tool(query: string, ticker: string | null, year: integer | null, quarter: number | null)
- compare_financial_metrics_tool(metric_name: string, values: list of objects with label and value)

Use stock tickers such as AAPL, TSLA, MSFT, NVDA, GOOGL, META, or AMZN when calling search_transcript_tool.
For numeric comparisons across periods, call compare_financial_metrics_tool after retrieving the source values.
If a question mentions multiple companies, tickers, years, or periods, call search_transcript_tool once for each distinct company/year or ticker/year pair before final_answer.
Do not finish until the observations cover every company and year requested by the question.

Return exactly one JSON object on every turn. Do not include markdown, Observation blocks, or extra text.
To call a tool:
{{
  "thought": "why this tool is needed",
  "action": {{
    "name": "tool_name",
    "arguments": {{}}
  }}
}}

To finish:
{{
  "thought": "why the available observations are enough",
  "final_answer": "answer grounded only in observations"
}}

Keep final_answer concise: 1-2 short paragraphs or up to 5 bullets.
Do not invent financial numbers. Use observations from tools as evidence."""


def build_react_user_prompt(question: str, steps: list[AgentStep]) -> str:
    return json.dumps(
        {
            "question": question,
            "previous_steps": [
                {
                    "thought": step.thought,
                    "action": {
                        "name": step.action.name,
                        "arguments": step.action.arguments,
                    },
                    "observation": _observation_to_dict(step.observation),
                }
                for step in steps
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _observation_to_dict(observation: ToolObservation) -> dict[str, Any]:
    return tool_observation_to_dict(observation)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        return stripped.removeprefix("```json").removesuffix("```").strip()
    if stripped.startswith("```"):
        return stripped.removeprefix("```").removesuffix("```").strip()
    return stripped


def _load_first_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("Agent response JSON must be an object.", cleaned, index)
        return payload
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Agent response must contain a JSON object.", cleaned, 0)
