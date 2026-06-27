"""Agentic RAG profiling helpers.

AI Intuition / Why This Exists
Agentic RAG is slower than naive RAG because a single user question may require
multiple LLM turns, tool calls, and correction attempts. Profiling makes that
cost visible: latency, loop counts, correction counts, concurrency pressure,
and optional GPU memory snapshots all become measurable instead of anecdotal.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from src.agent_types import AgentRunResult
from src.evaluation import EvalCase
from src.guardrails import GuardrailResult


AgentAnswerFn = Callable[[str], Awaitable[tuple[AgentRunResult, GuardrailResult, int]]]


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """GPU memory state at one point in the profiler run.

    AI Intuition / Why This Exists
    Day 6 eventually targets RTX 5080 local inference. Even before deep KV cache
    attribution, a coarse GPU memory snapshot helps correlate concurrency with
    memory pressure.
    """

    label: str
    gpus: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ProfileCaseResult:
    """Measured runtime outcome for one profiled case.

    AI Intuition / Why This Exists
    Latency alone is not enough for Agentic RAG. We also need loops and
    correction attempts because those explain why latency increases.
    """

    case_id: str
    category: str
    question: str
    latency_seconds: float
    loop_count: int
    correction_count: int
    guardrail_passed: bool
    error_message: str | None = None


@dataclass(frozen=True)
class ProfileRunResult:
    """Complete profiler report for one concurrency setting.

    AI Intuition / Why This Exists
    A profiler run should preserve both aggregate metrics and raw case-level
    timings so we can inspect outliers instead of trusting averages.
    """

    concurrency: int
    total_latency_seconds: float
    cases: list[ProfileCaseResult]
    gpu_memory_snapshots: list[GpuMemorySnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class ProfileSweepResult:
    """Profiler results across multiple concurrency settings.

    AI Intuition / Why This Exists
    RTX 5080 profiling is most useful as a sweep: concurrency 1 may look fine,
    while concurrency 8 or 16 reveals p99 latency and memory pressure. A sweep
    keeps those runs in one report so the trade-off is visible.
    """

    runs: list[ProfileRunResult]


async def profile_cases(cases: list[EvalCase], answer_fn: AgentAnswerFn, concurrency: int) -> ProfileRunResult:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")

    semaphore = asyncio.Semaphore(concurrency)
    snapshots = [sample_gpu_memory("before")]
    start = perf_counter()
    results = await asyncio.gather(*[_profile_case(case, answer_fn, semaphore) for case in cases])
    total_latency_seconds = perf_counter() - start
    snapshots.append(sample_gpu_memory("after"))

    return ProfileRunResult(
        concurrency=concurrency,
        total_latency_seconds=total_latency_seconds,
        cases=list(results),
        gpu_memory_snapshots=snapshots,
    )


async def profile_sweep(
    cases: list[EvalCase],
    answer_fn: AgentAnswerFn,
    concurrency_values: list[int],
) -> ProfileSweepResult:
    if not concurrency_values:
        raise ValueError("concurrency_values must not be empty.")

    runs: list[ProfileRunResult] = []
    for concurrency in concurrency_values:
        runs.append(await profile_cases(cases, answer_fn, concurrency=concurrency))
    return ProfileSweepResult(runs=runs)


async def _profile_case(case: EvalCase, answer_fn: AgentAnswerFn, semaphore: asyncio.Semaphore) -> ProfileCaseResult:
    async with semaphore:
        start = perf_counter()
        try:
            result, guardrail, correction_count = await answer_fn(case.question)
            latency_seconds = perf_counter() - start
            return ProfileCaseResult(
                case_id=case.case_id,
                category=case.category,
                question=case.question,
                latency_seconds=latency_seconds,
                loop_count=result.loop_count,
                correction_count=correction_count,
                guardrail_passed=guardrail.passed,
            )
        except Exception as exc:
            latency_seconds = perf_counter() - start
            return ProfileCaseResult(
                case_id=case.case_id,
                category=case.category,
                question=case.question,
                latency_seconds=latency_seconds,
                loop_count=0,
                correction_count=0,
                guardrail_passed=False,
                error_message=str(exc),
            )


def summarize_profile_run(run: ProfileRunResult) -> dict[str, Any]:
    latencies = [case.latency_seconds for case in run.cases]
    return {
        "concurrency": run.concurrency,
        "case_count": len(run.cases),
        "total_latency_seconds": run.total_latency_seconds,
        "throughput_cases_per_second": safe_divide(len(run.cases), run.total_latency_seconds),
        "latency_seconds": {
            "average": average(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "average_agent_loops": average([float(case.loop_count) for case in run.cases]),
        "average_correction_attempts": average([float(case.correction_count) for case in run.cases]),
        "guardrail_pass_rate": average([1.0 if case.guardrail_passed else 0.0 for case in run.cases]),
        "error_rate": average([1.0 if case.error_message else 0.0 for case in run.cases]),
    }


def profile_run_to_dict(run: ProfileRunResult) -> dict[str, Any]:
    return {
        "summary": summarize_profile_run(run),
        "cases": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "latency_seconds": case.latency_seconds,
                "loop_count": case.loop_count,
                "correction_count": case.correction_count,
                "guardrail_passed": case.guardrail_passed,
                "error_message": case.error_message,
            }
            for case in run.cases
        ],
        "gpu_memory_snapshots": [
            {
                "label": snapshot.label,
                "gpus": snapshot.gpus,
                "error": snapshot.error,
            }
            for snapshot in run.gpu_memory_snapshots
        ],
    }


def summarize_profile_sweep(sweep: ProfileSweepResult) -> dict[str, Any]:
    return {
        "run_count": len(sweep.runs),
        "concurrency_values": [run.concurrency for run in sweep.runs],
        "runs": [summarize_profile_run(run) for run in sweep.runs],
    }


def profile_sweep_to_dict(sweep: ProfileSweepResult) -> dict[str, Any]:
    return {
        "summary": summarize_profile_sweep(sweep),
        "runs": [profile_run_to_dict(run) for run in sweep.runs],
    }


def save_profile_report(run: ProfileRunResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile_run_to_dict(run), file, ensure_ascii=False, indent=2)


def save_profile_sweep_report(sweep: ProfileSweepResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile_sweep_to_dict(sweep), file, ensure_ascii=False, indent=2)


def sample_gpu_memory(label: str) -> GpuMemorySnapshot:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return GpuMemorySnapshot(label=label, error="nvidia-smi not found")

    command = [
        nvidia_smi,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError) as exc:
        return GpuMemorySnapshot(label=label, error=str(exc))

    return GpuMemorySnapshot(label=label, gpus=parse_nvidia_smi_csv(completed.stdout))


def parse_nvidia_smi_csv(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, name, memory_used_mb, memory_total_mb, utilization_gpu_pct = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": float(memory_used_mb),
                "memory_total_mb": float(memory_total_mb),
                "utilization_gpu_pct": float(utilization_gpu_pct),
            }
        )
    return gpus


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (percentile_value / 100)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
