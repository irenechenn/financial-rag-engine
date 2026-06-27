from __future__ import annotations

from src.agent_types import AgentRunResult, AgentStep, ToolCall, ToolObservation
from src.guardrails import check_numeric_faithfulness, extract_numbers, number_supported_by_observations


def test_extract_numbers_preserves_financial_units() -> None:
    assert extract_numbers("Revenue was $21.2 billion, up 8%, with 935 million subscriptions and ~$21B run-rate.") == [
        "$21.2 billion",
        "8%",
        "935 million",
        "$21B",
    ]


def test_check_numeric_faithfulness_passes_when_numbers_are_observed() -> None:
    result = AgentRunResult(
        question="Question?",
        steps=[
            AgentStep(
                thought="Need evidence.",
                action=ToolCall(name="search_transcript_tool", arguments={}),
                observation=ToolObservation(
                    tool_name="search_transcript_tool",
                    results=[
                        {
                            "text": "Services revenue reached $21.2 billion and grew 8%.",
                            "metadata": {"ticker": "AAPL", "year": 2023},
                            "score": 0.9,
                        }
                    ],
                ),
            )
        ],
        final_answer="Services revenue was $21.2 billion and grew 8%.",
        loop_count=1,
    )

    guardrail = check_numeric_faithfulness(result)

    assert guardrail.passed is True
    assert guardrail.unsupported_numbers == []


def test_check_numeric_faithfulness_fails_when_number_is_missing() -> None:
    result = AgentRunResult(
        question="Question?",
        steps=[
            AgentStep(
                thought="Need evidence.",
                action=ToolCall(name="search_transcript_tool", arguments={}),
                observation=ToolObservation(
                    tool_name="search_transcript_tool",
                    results=[
                        {
                            "text": "Services revenue reached $21.2 billion.",
                            "metadata": {"ticker": "AAPL", "year": 2023},
                            "score": 0.9,
                        }
                    ],
                ),
            )
        ],
        final_answer="Services revenue was $21.2 billion and grew 9%.",
        loop_count=1,
    )

    guardrail = check_numeric_faithfulness(result)

    assert guardrail.passed is False
    assert guardrail.unsupported_numbers == ["9%"]
    assert "9%" in guardrail.issues[0]
    assert "9%" in guardrail.unsupported_contexts["9%"]


def test_number_supported_by_observations_accepts_rounded_money_units() -> None:
    observation_text = "Services revenue was $21.2 billion in 2023 and $24.2 billion in 2024."

    assert number_supported_by_observations("$21B", observation_text)
    assert number_supported_by_observations("$24", observation_text)


def test_number_supported_by_observations_accepts_percent_from_calculated_output() -> None:
    observation_text = "{'relative_change_pct': 19.617224880382782}"

    assert number_supported_by_observations("19%", observation_text)
