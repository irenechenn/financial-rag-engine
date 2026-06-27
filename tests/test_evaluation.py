from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent_types import AgentRunResult, AgentStep, ToolCall, ToolObservation
from src.evaluation import (
    EvalCase,
    eval_case_result_to_dict,
    has_expected_metadata_hit,
    has_expected_tool_call_queries,
    has_expected_tool_calls,
    load_eval_cases,
    missing_expected_tool_calls,
    run_eval_cases,
    select_eval_cases,
    summarize_eval_results,
)
from src.guardrails import GuardrailResult


def test_load_eval_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "aapl_case",
                    "question": "Question?",
                    "category": "Simple",
                    "expected_tickers": ["aapl"],
                    "expected_years": [2023],
                    "expected_tool_calls": [
                        {
                            "tool_name": "search_transcript_tool",
                            "ticker": "aapl",
                            "year": 2023,
                            "query_terms": ["Services"],
                        }
                    ],
                    "required_answer_terms": ["Services"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases == [
        EvalCase(
            case_id="aapl_case",
            question="Question?",
            category="simple",
            expected_tickers=["AAPL"],
            expected_years=[2023],
            expected_tool_calls=[
                {
                    "tool_name": "search_transcript_tool",
                    "ticker": "AAPL",
                    "year": 2023,
                    "query_terms": ["services"],
                }
            ],
            required_answer_terms=["services"],
        )
    ]


def test_has_expected_metadata_hit_checks_ticker_and_year() -> None:
    result = AgentRunResult(
        question="Question?",
        steps=[
            AgentStep(
                thought="Search Apple.",
                action=ToolCall(name="search_transcript_tool", arguments={}),
                observation=ToolObservation(
                    tool_name="search_transcript_tool",
                    results=[
                        {
                            "text": "Services revenue grew 8%.",
                            "metadata": {"ticker": "AAPL", "year": 2023.0},
                            "score": 0.9,
                        }
                    ],
                ),
            )
        ],
        final_answer="Services revenue grew 8%.",
        loop_count=1,
    )

    assert has_expected_metadata_hit(
        result,
        EvalCase(case_id="case", question="Question?", expected_tickers=["AAPL"], expected_years=[2023]),
    )
    assert not has_expected_metadata_hit(
        result,
        EvalCase(case_id="case", question="Question?", expected_tickers=["TSLA"], expected_years=[2023]),
    )


def test_select_eval_cases_filters_by_category_and_limit() -> None:
    cases = [
        EvalCase(case_id="simple_one", question="Question?", category="simple"),
        EvalCase(case_id="comparison_one", question="Question?", category="comparison"),
        EvalCase(case_id="comparison_two", question="Question?", category="comparison"),
    ]

    selected = select_eval_cases(cases, category="comparison", limit=1)

    assert [case.case_id for case in selected] == ["comparison_one"]


def test_select_eval_cases_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        select_eval_cases([EvalCase(case_id="case", question="Question?")], limit=-1)


def test_has_expected_tool_calls_checks_planning_coverage() -> None:
    result = AgentRunResult(
        question="Question?",
        steps=[
            AgentStep(
                thought="Search Apple 2023.",
                action=ToolCall(
                    name="search_transcript_tool",
                    arguments={"query": "services revenue", "ticker": "AAPL", "year": 2023},
                ),
                observation=ToolObservation(tool_name="search_transcript_tool"),
            ),
            AgentStep(
                thought="Search Apple 2024.",
                action=ToolCall(
                    name="search_transcript_tool",
                    arguments={"query": "services revenue", "ticker": "AAPL", "year": 2024},
                ),
                observation=ToolObservation(tool_name="search_transcript_tool"),
            ),
        ],
        final_answer="Answer.",
        loop_count=2,
    )

    assert has_expected_tool_calls(
        result,
        EvalCase(
            case_id="case",
            question="Question?",
            expected_tool_calls=[
                {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2023, "query_terms": ["services"]},
                {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2024, "query_terms": ["services"]},
            ],
        ),
    )
    assert has_expected_tool_call_queries(
        result,
        EvalCase(
            case_id="case",
            question="Question?",
            expected_tool_calls=[
                {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2023, "query_terms": ["services"]},
                {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2024, "query_terms": ["services"]},
            ],
        ),
    )
    assert not has_expected_tool_calls(
        result,
        EvalCase(
            case_id="case",
            question="Question?",
            expected_tool_calls=[
                {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2023},
                {"tool_name": "search_transcript_tool", "ticker": "TSLA", "year": 2024},
            ],
        ),
    )
    assert missing_expected_tool_calls(
        result,
        EvalCase(
            case_id="case",
            question="Question?",
            expected_tool_calls=[
                {"tool_name": "search_transcript_tool", "ticker": "TSLA", "year": 2024},
            ],
        ),
    ) == [{"tool_name": "search_transcript_tool", "ticker": "TSLA", "year": 2024}]
    assert not has_expected_tool_call_queries(
        result,
        EvalCase(
            case_id="case",
            question="Question?",
            expected_tool_calls=[
                {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2024, "query_terms": ["margin"]},
            ],
        ),
    )


@pytest.mark.asyncio
async def test_run_eval_cases_summarizes_agentic_metrics() -> None:
    async def fake_answer_fn(question: str):
        return (
            AgentRunResult(
                question=question,
                steps=[
                    AgentStep(
                        thought="Search Apple.",
                        action=ToolCall(
                            name="search_transcript_tool",
                            arguments={"query": "services revenue", "ticker": "AAPL", "year": 2023},
                        ),
                        observation=ToolObservation(
                            tool_name="search_transcript_tool",
                            results=[
                                {
                                    "text": "Services revenue grew 8%.",
                                    "metadata": {"ticker": "AAPL", "year": 2023},
                                    "score": 0.9,
                                }
                            ],
                        ),
                    )
                ],
                final_answer="Services revenue grew 8%.",
                loop_count=1,
            ),
            GuardrailResult(passed=True, checked_numbers=["8%"]),
            1,
        )

    results = await run_eval_cases(
        [
            EvalCase(
                case_id="aapl_case",
                question="Question?",
                category="simple",
                expected_tickers=["AAPL"],
                expected_years=[2023],
                expected_tool_calls=[
                    {"tool_name": "search_transcript_tool", "ticker": "AAPL", "year": 2023, "query_terms": ["services"]}
                ],
                required_answer_terms=["services"],
            )
        ],
        fake_answer_fn,
    )
    summary = summarize_eval_results(results)

    assert summary["case_count"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["guardrail_pass_rate"] == 1.0
    assert summary["expected_metadata_hit_rate"] == 1.0
    assert summary["expected_tool_calls_hit_rate"] == 1.0
    assert summary["expected_tool_call_queries_hit_rate"] == 1.0
    assert summary["average_agent_loops"] == 1.0
    assert summary["average_correction_attempts"] == 1.0
    assert summary["by_category"]["simple"]["case_count"] == 1
    assert summary["by_category"]["simple"]["success_rate"] == 1.0
    assert summary["by_category"]["simple"]["expected_tool_calls_hit_rate"] == 1.0
    assert summary["by_category"]["simple"]["expected_tool_call_queries_hit_rate"] == 1.0
    assert eval_case_result_to_dict(results[0])["category"] == "simple"
    assert eval_case_result_to_dict(results[0])["expected_tool_calls_hit"] is True
    assert eval_case_result_to_dict(results[0])["expected_tool_call_queries_hit"] is True
    assert eval_case_result_to_dict(results[0])["missing_expected_tool_calls"] == []
    assert eval_case_result_to_dict(results[0])["unsupported_contexts"] == {}


@pytest.mark.asyncio
async def test_run_eval_cases_captures_case_errors() -> None:
    async def failing_answer_fn(question: str):
        raise RuntimeError(f"boom for {question}")

    results = await run_eval_cases(
        [EvalCase(case_id="broken_case", question="Question?", category="comparison")],
        failing_answer_fn,
    )
    summary = summarize_eval_results(results)
    payload = eval_case_result_to_dict(results[0])

    assert summary["case_count"] == 1
    assert summary["success_rate"] == 0.0
    assert summary["by_category"]["comparison"]["success_rate"] == 0.0
    assert results[0].error_message == "boom for Question?"
    assert payload["passed"] is False
    assert payload["error_message"] == "boom for Question?"
