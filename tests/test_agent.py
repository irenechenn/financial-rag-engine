from __future__ import annotations

import json

import pytest

from src.agent import ReActAgent, parse_agent_decision
from src.agent_types import ToolObservation
from src.interfaces import LLMProvider


class ScriptedLLMProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if not self.responses:
            raise AssertionError("No scripted LLM responses left.")
        return self.responses.pop(0)


async def fake_search_transcript_tool(
    query: str,
    ticker: str | None = None,
    year: int | None = None,
    quarter: int | float | None = None,
) -> ToolObservation:
    return ToolObservation(
        tool_name="search_transcript_tool",
        query=query,
        filters={"ticker": ticker, "year": year, "quarter": quarter},
        results=[
            {
                "text": f"{ticker} {year} gross margin was {'18%' if year == 2024 else '16%'} due to cost pressure.",
                "metadata": {"ticker": ticker, "year": year, "quarter": quarter},
                "score": 0.9,
            }
        ],
    )


async def fake_compare_financial_metrics_tool(metric_name: str, values: list[dict[str, float]]) -> ToolObservation:
    first = values[0]
    last = values[-1]
    return ToolObservation(
        tool_name="compare_financial_metrics_tool",
        output={
            "metric_name": metric_name,
            "absolute_change": last["value"] - first["value"],
            "direction": "decrease",
            "values": values,
        },
    )


@pytest.mark.asyncio
async def test_react_agent_runs_tools_and_returns_final_answer() -> None:
    llm = ScriptedLLMProvider(
        responses=[
            json.dumps(
                {
                    "thought": "I need Tesla 2024 gross margin first.",
                    "action": {
                        "name": "search_transcript_tool",
                        "arguments": {"query": "gross margin", "ticker": "TSLA", "year": 2024},
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "Now I need Tesla 2025 gross margin.",
                    "action": {
                        "name": "search_transcript_tool",
                        "arguments": {"query": "gross margin decline", "ticker": "TSLA", "year": 2025},
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "The observations contain both years and the decline reason.",
                    "final_answer": "Tesla gross margin declined from 18% in 2024 to 16% in 2025 due to cost pressure.",
                }
            ),
        ]
    )
    agent = ReActAgent(
        llm_provider=llm,
        tools={"search_transcript_tool": fake_search_transcript_tool},
        max_loops=3,
    )

    result = await agent.run("Compare Tesla gross margin in 2024 and 2025.")

    assert result.loop_count == 2
    assert len(result.steps) == 2
    assert result.steps[0].action.name == "search_transcript_tool"
    assert result.steps[0].observation.filters["year"] == 2024
    assert "18%" in result.final_answer
    assert "16%" in result.final_answer
    assert len(llm.calls) == 3
    assert "previous_steps" in llm.calls[-1]["user_prompt"]


@pytest.mark.asyncio
async def test_react_agent_can_search_then_compare_then_answer() -> None:
    llm = ScriptedLLMProvider(
        responses=[
            json.dumps(
                {
                    "thought": "I need Tesla 2024 gross margin first.",
                    "action": {
                        "name": "search_transcript_tool",
                        "arguments": {"query": "gross margin", "ticker": "TSLA", "year": 2024},
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "Now I need Tesla 2025 gross margin.",
                    "action": {
                        "name": "search_transcript_tool",
                        "arguments": {"query": "gross margin", "ticker": "TSLA", "year": 2025},
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "I should calculate the change instead of doing mental math.",
                    "action": {
                        "name": "compare_financial_metrics_tool",
                        "arguments": {
                            "metric_name": "gross_margin",
                            "values": [
                                {"label": "2024", "value": 18.0},
                                {"label": "2025", "value": 16.0},
                            ],
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "I have retrieved both values and calculated the change.",
                    "final_answer": "Tesla gross margin decreased by 2 percentage points, from 18% to 16%.",
                }
            ),
        ]
    )
    agent = ReActAgent(
        llm_provider=llm,
        tools={
            "search_transcript_tool": fake_search_transcript_tool,
            "compare_financial_metrics_tool": fake_compare_financial_metrics_tool,
        },
        max_loops=4,
    )

    result = await agent.run("Compare Tesla gross margin in 2024 and 2025.")

    assert result.loop_count == 3
    assert [step.action.name for step in result.steps] == [
        "search_transcript_tool",
        "search_transcript_tool",
        "compare_financial_metrics_tool",
    ]
    assert result.steps[-1].observation.output is not None
    assert result.steps[-1].observation.output["absolute_change"] == -2.0
    assert "2 percentage points" in result.final_answer


@pytest.mark.asyncio
async def test_react_agent_returns_failure_after_max_loops() -> None:
    llm = ScriptedLLMProvider(
        responses=[
            json.dumps(
                {
                    "thought": "I need more evidence.",
                    "action": {
                        "name": "search_transcript_tool",
                        "arguments": {"query": "gross margin", "ticker": "TSLA", "year": 2024},
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "I still need more evidence.",
                    "action": {
                        "name": "search_transcript_tool",
                        "arguments": {"query": "gross margin", "ticker": "TSLA", "year": 2025},
                    },
                }
            ),
        ]
    )
    agent = ReActAgent(
        llm_provider=llm,
        tools={"search_transcript_tool": fake_search_transcript_tool},
        max_loops=2,
    )

    result = await agent.run("Compare Tesla gross margin in 2024 and 2025.")

    assert result.final_answer == "Unable to complete within max_loops."
    assert result.loop_count == 2
    assert result.reflection_passed is False


@pytest.mark.asyncio
async def test_react_agent_turns_unknown_tool_into_observation() -> None:
    llm = ScriptedLLMProvider(
        responses=[
            json.dumps(
                {
                    "thought": "I mistakenly call a non-tool.",
                    "action": {"name": "final_answer", "arguments": {}},
                }
            ),
            json.dumps(
                {
                    "thought": "The previous observation says final_answer is not a tool.",
                    "final_answer": "I should answer through the top-level final_answer field.",
                }
            ),
        ]
    )
    agent = ReActAgent(
        llm_provider=llm,
        tools={"search_transcript_tool": fake_search_transcript_tool},
        max_loops=3,
    )

    result = await agent.run("Answer with evidence.")

    assert result.loop_count == 1
    assert result.steps[0].observation.output is not None
    assert result.steps[0].observation.output["error"] == "Unknown tool: final_answer"
    assert result.final_answer == "I should answer through the top-level final_answer field."


@pytest.mark.asyncio
async def test_react_agent_turns_invalid_tool_arguments_into_observation() -> None:
    llm = ScriptedLLMProvider(
        responses=[
            json.dumps(
                {
                    "thought": "I forgot the compare tool arguments.",
                    "action": {"name": "compare_financial_metrics_tool", "arguments": {}},
                }
            ),
            json.dumps(
                {
                    "thought": "The observation says the arguments were invalid.",
                    "final_answer": "I need valid metric_name and values before comparing.",
                }
            ),
        ]
    )
    agent = ReActAgent(
        llm_provider=llm,
        tools={"compare_financial_metrics_tool": fake_compare_financial_metrics_tool},
        max_loops=3,
    )

    result = await agent.run("Compare two metrics.")

    assert result.loop_count == 1
    assert result.steps[0].observation.output is not None
    assert "Invalid arguments for tool compare_financial_metrics_tool" in result.steps[0].observation.output["error"]
    assert result.final_answer == "I need valid metric_name and values before comparing."


def test_parse_agent_decision_accepts_fenced_json() -> None:
    decision = parse_agent_decision(
        """```json
        {"thought": "Enough evidence exists.", "final_answer": "Final."}
        ```"""
    )

    assert decision.thought == "Enough evidence exists."
    assert decision.final_answer == "Final."


def test_parse_agent_decision_uses_first_json_object_when_model_adds_extra_text() -> None:
    decision = parse_agent_decision(
        """{
          "thought": "I need source evidence first.",
          "action": {
            "name": "search_transcript_tool",
            "arguments": {"query": "services revenue", "ticker": "AAPL", "year": 2023}
          }
        }

        **Observation:**
        ```json
        {"fake": "model-generated observation that should be ignored"}
        ```

        {"thought": "I can answer now.", "final_answer": "Unsupported shortcut."}
        """
    )

    assert decision.thought == "I need source evidence first."
    assert decision.action is not None
    assert decision.action.name == "search_transcript_tool"
    assert decision.final_answer is None


def test_parse_agent_decision_coerces_final_answer_action() -> None:
    decision = parse_agent_decision(
        json.dumps(
            {
                "thought": "The observations are enough to answer.",
                "action": {
                    "name": "final_answer",
                    "arguments": {
                        "answer": "Nvidia described strong data center demand from accelerated computing."
                    },
                },
            }
        )
    )

    assert decision.action is None
    assert decision.final_answer == "Nvidia described strong data center demand from accelerated computing."


def test_parse_agent_decision_coerces_final_answer_action_with_nonstandard_key() -> None:
    decision = parse_agent_decision(
        json.dumps(
            {
                "thought": "The observations are enough to answer.",
                "action": {
                    "name": "final_answer",
                    "arguments": {"result": "Meta advertising demand improved in the retrieved commentary."},
                },
            }
        )
    )

    assert decision.action is None
    assert decision.final_answer == "Meta advertising demand improved in the retrieved commentary."


def test_parse_agent_decision_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_agent_decision("Thought: I should search.")


def test_parse_agent_decision_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="action or final_answer"):
        parse_agent_decision(json.dumps({"thought": "I need to do something."}))
