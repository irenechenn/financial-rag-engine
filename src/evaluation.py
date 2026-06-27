"""Agent evaluation helpers.

AI Intuition / Why This Exists
Evaluation is the bridge between "the agent can answer one demo question" and
"the system is improving in a measurable way." Agentic RAG adds loops, tools,
guardrails, and correction attempts, so we need a structured harness that can
measure both answer quality proxies and operational cost proxies.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from src.agent_types import AgentRunResult, agent_run_result_to_dict
from src.guardrails import GuardrailResult
from src.tools import normalize_ticker


@dataclass(frozen=True)
class EvalCase:
    """A single agent evaluation question.

    AI Intuition / Why This Exists
    A benchmark case should describe the user's question plus the minimum
    evidence pattern we expect the agent to touch. We avoid hard-coding one
    exact final answer because financial transcript wording can vary, but we
    still check whether retrieval hit the intended ticker/year and whether the
    answer contains important terms.
    """

    case_id: str
    question: str
    category: str = "uncategorized"
    expected_tickers: list[str] = field(default_factory=list)
    expected_years: list[int] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    required_answer_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCaseResult:
    """Measured outcome for one evaluation case.

    AI Intuition / Why This Exists
    The agent's final answer alone is too thin for debugging. This result keeps
    both quality proxies, such as guardrail pass and expected metadata hits, and
    cost proxies, such as latency, loops, and correction attempts.
    """

    case: EvalCase
    result: AgentRunResult
    guardrail: GuardrailResult
    correction_count: int
    latency_seconds: float
    expected_metadata_hit: bool
    expected_tool_calls_hit: bool
    expected_tool_call_queries_hit: bool
    required_terms_hit: bool
    missing_expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None


AgentAnswerFn = Callable[[str], Awaitable[tuple[AgentRunResult, GuardrailResult, int]]]


def load_eval_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation case file must contain a JSON list.")

    return [eval_case_from_dict(item) for item in payload]


def select_eval_cases(cases: list[EvalCase], category: str | None = None, limit: int | None = None) -> list[EvalCase]:
    selected = cases
    if category is not None:
        normalized_category = category.lower()
        selected = [case for case in selected if case.category == normalized_category]

    if limit is None:
        return selected
    if limit < 0:
        raise ValueError("Evaluation case limit must be greater than or equal to 0.")
    return selected[:limit]


def eval_case_from_dict(payload: dict[str, Any]) -> EvalCase:
    return EvalCase(
        case_id=str(payload["case_id"]),
        question=str(payload["question"]),
        category=str(payload.get("category", "uncategorized")).lower(),
        expected_tickers=[str(value).upper() for value in payload.get("expected_tickers", [])],
        expected_years=[int(value) for value in payload.get("expected_years", [])],
        expected_tool_calls=[
            normalize_expected_tool_call(value) for value in payload.get("expected_tool_calls", [])
        ],
        required_answer_terms=[str(value).lower() for value in payload.get("required_answer_terms", [])],
    )


def normalize_expected_tool_call(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": str(payload.get("tool_name", "search_transcript_tool")),
        "ticker": normalize_ticker(payload.get("ticker") or payload.get("company")),
        "year": int(payload["year"]) if payload.get("year") is not None else None,
        "query_terms": [str(value).lower() for value in payload.get("query_terms", [])],
    }


async def run_eval_cases(cases: list[EvalCase], answer_fn: AgentAnswerFn) -> list[EvalCaseResult]:
    results: list[EvalCaseResult] = []
    for case in cases:
        start = perf_counter()
        try:
            result, guardrail, correction_count = await answer_fn(case.question)
            error_message = None
        except Exception as exc:
            result = AgentRunResult(
                question=case.question,
                steps=[],
                final_answer=f"Evaluation failed: {exc}",
                loop_count=0,
                reflection_passed=False,
            )
            guardrail = GuardrailResult(passed=False, issues=[str(exc)])
            correction_count = 0
            error_message = str(exc)
        latency_seconds = perf_counter() - start
        results.append(
            eval_case_result_from_run(
                case=case,
                result=result,
                guardrail=guardrail,
                correction_count=correction_count,
                latency_seconds=latency_seconds,
                error_message=error_message,
            )
        )
    return results


def eval_case_result_from_run(
    case: EvalCase,
    result: AgentRunResult,
    guardrail: GuardrailResult,
    correction_count: int,
    latency_seconds: float,
    error_message: str | None = None,
) -> EvalCaseResult:
    missing_tool_calls = missing_expected_tool_calls(result, case)
    return EvalCaseResult(
        case=case,
        result=result,
        guardrail=guardrail,
        correction_count=correction_count,
        latency_seconds=latency_seconds,
        expected_metadata_hit=has_expected_metadata_hit(result, case),
        expected_tool_calls_hit=not missing_tool_calls,
        expected_tool_call_queries_hit=has_expected_tool_call_queries(result, case),
        missing_expected_tool_calls=missing_tool_calls,
        required_terms_hit=has_required_answer_terms(result.final_answer, case.required_answer_terms),
        error_message=error_message,
    )


def has_expected_metadata_hit(result: AgentRunResult, case: EvalCase) -> bool:
    if not case.expected_tickers and not case.expected_years:
        return True

    ticker_hits: set[str] = set()
    year_hits: set[int] = set()
    for step in result.steps:
        for item in step.observation.results:
            metadata = item.get("metadata", {})
            ticker = metadata.get("ticker")
            year = metadata.get("year")
            if ticker is not None:
                ticker_hits.add(str(ticker).upper())
            if year is not None:
                year_hits.add(int(float(year)))

    ticker_ok = not case.expected_tickers or set(case.expected_tickers).issubset(ticker_hits)
    year_ok = not case.expected_years or set(case.expected_years).issubset(year_hits)
    return ticker_ok and year_ok


def has_expected_tool_calls(result: AgentRunResult, case: EvalCase) -> bool:
    return not missing_expected_tool_calls(result, case)


def missing_expected_tool_calls(result: AgentRunResult, case: EvalCase) -> list[dict[str, Any]]:
    if not case.expected_tool_calls:
        return []

    actual_calls = actual_tool_calls(result)
    return [
        expected
        for expected in case.expected_tool_calls
        if not any(tool_call_matches(expected, actual) for actual in actual_calls)
    ]


def actual_tool_calls(result: AgentRunResult) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": step.action.name,
            "ticker": normalize_ticker(step.action.arguments.get("ticker") or step.action.arguments.get("company")),
            "year": int(step.action.arguments["year"]) if step.action.arguments.get("year") is not None else None,
            "query": str(step.action.arguments.get("query", "")).lower(),
        }
        for step in result.steps
    ]


def has_expected_tool_call_queries(result: AgentRunResult, case: EvalCase) -> bool:
    expected_calls_with_query_terms = [
        expected for expected in case.expected_tool_calls if expected.get("query_terms")
    ]
    if not expected_calls_with_query_terms:
        return True

    actual_calls = actual_tool_calls(result)
    return all(
        any(tool_call_matches(expected, actual) and query_terms_match(expected, actual) for actual in actual_calls)
        for expected in expected_calls_with_query_terms
    )


def tool_call_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return (
        expected["tool_name"] == actual["tool_name"]
        and (expected["ticker"] is None or expected["ticker"] == actual["ticker"])
        and (expected["year"] is None or expected["year"] == actual["year"])
    )


def query_terms_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    query = actual.get("query", "")
    return all(str(term).lower() in query for term in expected.get("query_terms", []))


def has_required_answer_terms(answer: str, required_terms: list[str]) -> bool:
    lowered = answer.lower()
    return all(term in lowered for term in required_terms)


def summarize_eval_results(results: list[EvalCaseResult]) -> dict[str, Any]:
    if not results:
        return {
            "case_count": 0,
            "success_rate": 0.0,
            "guardrail_pass_rate": 0.0,
            "expected_metadata_hit_rate": 0.0,
            "expected_tool_calls_hit_rate": 0.0,
            "expected_tool_call_queries_hit_rate": 0.0,
            "required_terms_hit_rate": 0.0,
            "average_agent_loops": 0.0,
            "average_correction_attempts": 0.0,
            "average_latency_seconds": 0.0,
            "by_category": {},
        }

    case_count = len(results)
    success_count = sum(1 for result in results if case_result_passed(result))
    return {
        "case_count": case_count,
        "success_rate": success_count / case_count,
        "guardrail_pass_rate": average([1.0 if result.guardrail.passed else 0.0 for result in results]),
        "expected_metadata_hit_rate": average([1.0 if result.expected_metadata_hit else 0.0 for result in results]),
        "expected_tool_calls_hit_rate": average([1.0 if result.expected_tool_calls_hit else 0.0 for result in results]),
        "expected_tool_call_queries_hit_rate": average(
            [1.0 if result.expected_tool_call_queries_hit else 0.0 for result in results]
        ),
        "required_terms_hit_rate": average([1.0 if result.required_terms_hit else 0.0 for result in results]),
        "average_agent_loops": average([float(result.result.loop_count) for result in results]),
        "average_correction_attempts": average([float(result.correction_count) for result in results]),
        "average_latency_seconds": average([result.latency_seconds for result in results]),
        "by_category": summarize_results_by_category(results),
    }


def summarize_results_by_category(results: list[EvalCaseResult]) -> dict[str, dict[str, Any]]:
    categories: dict[str, list[EvalCaseResult]] = {}
    for result in results:
        categories.setdefault(result.case.category, []).append(result)

    return {
        category: {
            "case_count": len(category_results),
            "success_rate": average([1.0 if case_result_passed(result) else 0.0 for result in category_results]),
            "guardrail_pass_rate": average([1.0 if result.guardrail.passed else 0.0 for result in category_results]),
            "expected_tool_calls_hit_rate": average(
                [1.0 if result.expected_tool_calls_hit else 0.0 for result in category_results]
            ),
            "expected_tool_call_queries_hit_rate": average(
                [1.0 if result.expected_tool_call_queries_hit else 0.0 for result in category_results]
            ),
            "average_agent_loops": average([float(result.result.loop_count) for result in category_results]),
            "average_correction_attempts": average([float(result.correction_count) for result in category_results]),
            "average_latency_seconds": average([result.latency_seconds for result in category_results]),
        }
        for category, category_results in sorted(categories.items())
    }


def case_result_passed(result: EvalCaseResult) -> bool:
    return (
        result.error_message is None
        and result.guardrail.passed
        and result.expected_metadata_hit
        and result.expected_tool_calls_hit
        and result.expected_tool_call_queries_hit
        and result.required_terms_hit
    )


def eval_case_result_to_dict(result: EvalCaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case.case_id,
        "category": result.case.category,
        "question": result.case.question,
        "passed": case_result_passed(result),
        "guardrail_passed": result.guardrail.passed,
        "expected_metadata_hit": result.expected_metadata_hit,
        "expected_tool_calls_hit": result.expected_tool_calls_hit,
        "expected_tool_call_queries_hit": result.expected_tool_call_queries_hit,
        "missing_expected_tool_calls": result.missing_expected_tool_calls,
        "required_terms_hit": result.required_terms_hit,
        "loop_count": result.result.loop_count,
        "correction_count": result.correction_count,
        "latency_seconds": result.latency_seconds,
        "error_message": result.error_message,
        "unsupported_numbers": result.guardrail.unsupported_numbers,
        "unsupported_contexts": result.guardrail.unsupported_contexts,
        "final_answer": result.result.final_answer,
        "trace": agent_run_result_to_dict(result.result),
    }


def save_eval_report(results: list[EvalCaseResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_eval_results(results),
        "cases": [eval_case_result_to_dict(result) for result in results],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
