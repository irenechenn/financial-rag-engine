from __future__ import annotations

import pytest

from src.agent_types import AgentRunResult
from src.evaluation import EvalCase
from src.guardrails import GuardrailResult
from src.profiler import (
    parse_nvidia_smi_csv,
    percentile,
    profile_cases,
    profile_run_to_dict,
    profile_sweep,
    profile_sweep_to_dict,
)


@pytest.mark.asyncio
async def test_profile_cases_records_latency_and_agentic_metrics() -> None:
    async def fake_answer_fn(question: str):
        return (
            AgentRunResult(question=question, steps=[], final_answer="Answer", loop_count=2),
            GuardrailResult(passed=True),
            1,
        )

    run = await profile_cases(
        [
            EvalCase(case_id="case_1", question="Question 1?", category="simple"),
            EvalCase(case_id="case_2", question="Question 2?", category="simple"),
        ],
        fake_answer_fn,
        concurrency=2,
    )
    payload = profile_run_to_dict(run)

    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["concurrency"] == 2
    assert payload["summary"]["average_agent_loops"] == 2.0
    assert payload["summary"]["average_correction_attempts"] == 1.0
    assert payload["summary"]["guardrail_pass_rate"] == 1.0
    assert len(payload["gpu_memory_snapshots"]) == 2


@pytest.mark.asyncio
async def test_profile_cases_captures_case_errors() -> None:
    async def failing_answer_fn(question: str):
        raise RuntimeError(f"boom {question}")

    run = await profile_cases(
        [EvalCase(case_id="case_1", question="Question?", category="simple")],
        failing_answer_fn,
        concurrency=1,
    )
    payload = profile_run_to_dict(run)

    assert payload["summary"]["error_rate"] == 1.0
    assert payload["cases"][0]["error_message"] == "boom Question?"


@pytest.mark.asyncio
async def test_profile_cases_rejects_invalid_concurrency() -> None:
    async def fake_answer_fn(question: str):
        return (
            AgentRunResult(question=question, steps=[], final_answer="Answer", loop_count=0),
            GuardrailResult(passed=True),
            0,
        )

    with pytest.raises(ValueError, match="concurrency"):
        await profile_cases([], fake_answer_fn, concurrency=0)


@pytest.mark.asyncio
async def test_profile_sweep_runs_multiple_concurrency_values() -> None:
    async def fake_answer_fn(question: str):
        return (
            AgentRunResult(question=question, steps=[], final_answer="Answer", loop_count=1),
            GuardrailResult(passed=True),
            0,
        )

    sweep = await profile_sweep(
        [EvalCase(case_id="case_1", question="Question?", category="simple")],
        fake_answer_fn,
        concurrency_values=[1, 2],
    )
    payload = profile_sweep_to_dict(sweep)

    assert payload["summary"]["run_count"] == 2
    assert payload["summary"]["concurrency_values"] == [1, 2]
    assert [run["summary"]["concurrency"] for run in payload["runs"]] == [1, 2]


@pytest.mark.asyncio
async def test_profile_sweep_rejects_empty_concurrency_values() -> None:
    async def fake_answer_fn(question: str):
        return (
            AgentRunResult(question=question, steps=[], final_answer="Answer", loop_count=0),
            GuardrailResult(passed=True),
            0,
        )

    with pytest.raises(ValueError, match="concurrency_values"):
        await profile_sweep([], fake_answer_fn, concurrency_values=[])


def test_percentile_interpolates_values() -> None:
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0], 95) == pytest.approx(2.9)


def test_parse_nvidia_smi_csv() -> None:
    output = "0, NVIDIA GeForce RTX 5080, 1024, 16384, 55\n"

    assert parse_nvidia_smi_csv(output) == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5080",
            "memory_used_mb": 1024.0,
            "memory_total_mb": 16384.0,
            "utilization_gpu_pct": 55.0,
        }
    ]
