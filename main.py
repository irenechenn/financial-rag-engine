from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from src.agent import ReActAgent
from src.agent_types import AgentRunResult, ToolObservation, agent_run_result_to_dict
from src.data_loader import load_and_chunk
from src.evaluation import (
    EvalCaseResult,
    load_eval_cases,
    run_eval_cases,
    save_eval_report,
    select_eval_cases,
    summarize_eval_results,
)
from src.guardrails import GuardrailResult, check_numeric_faithfulness
from src.interfaces import EmbeddingProvider, LLMProvider
from src.providers.claude_prov import ClaudeLLMProvider
from src.providers.ollama_prov import OllamaLLMProvider
from src.providers.openai_prov import OpenAIEmbeddingProvider
from src.providers.voyage_prov import VoyageEmbeddingProvider
from src.profiler import (
    ProfileRunResult,
    ProfileSweepResult,
    profile_cases,
    profile_sweep,
    save_profile_report,
    save_profile_sweep_report,
    summarize_profile_run,
    summarize_profile_sweep,
)
from src.tools import compare_financial_metrics_tool, load_transcript_search_tool
from src.vector_store import FaissVectorStore


DEFAULT_DATA_PATH = Path("data/mini_sp500_transcripts.json")
DEFAULT_INDEX_PATH = Path("indexes/naive_voyage_finance")
DEFAULT_EVAL_CASES_PATH = Path("eval_cases/day3_agentic_smoke.json")


def get_embedding_provider(name: str) -> EmbeddingProvider:
    if name == "voyage":
        return VoyageEmbeddingProvider()
    if name == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {name}")


def get_llm_provider(name: str) -> LLMProvider:
    if name == "claude":
        return ClaudeLLMProvider()
    if name == "ollama":
        return OllamaLLMProvider()
    raise ValueError(f"Unsupported LLM provider: {name}")


async def build_index(
    data_path: Path,
    index_path: Path,
    chunk_size: int,
    overlap: int,
    embedding_provider_name: str,
) -> None:
    chunks = load_and_chunk(data_path, chunk_size=chunk_size, overlap=overlap)
    embeddings = get_embedding_provider(embedding_provider_name)
    store = await FaissVectorStore.from_chunks(chunks, embeddings)
    store.save(index_path)
    print(f"Built {embedding_provider_name} index with {len(chunks)} chunks at {index_path}")


async def search_index(
    index_path: Path,
    query: str,
    ticker: str | None,
    year: int | None,
    top_k: int,
    embedding_provider_name: str,
) -> None:
    embeddings = get_embedding_provider(embedding_provider_name)
    store = FaissVectorStore.load(index_path)
    filters = {"ticker": ticker, "year": year}
    results = await store.search(query, embeddings, top_k=top_k, filters=filters)

    for rank, result in enumerate(results, start=1):
        metadata = result.metadata
        print(f"\n[{rank}] score={result.score:.4f} {metadata.get('ticker')} {metadata.get('year')} Q{metadata.get('quarter')}")
        print(result.text[:900])


def create_react_agent(
    index_path: Path,
    embedding_provider_name: str,
    llm_provider_name: str,
    top_k: int,
    max_loops: int,
) -> ReActAgent:
    embeddings = get_embedding_provider(embedding_provider_name)
    llm_provider = get_llm_provider(llm_provider_name)
    search_tool = load_transcript_search_tool(index_path=index_path, embedding_provider=embeddings, top_k=top_k)
    return ReActAgent(
        llm_provider=llm_provider,
        tools={
            "search_transcript_tool": search_tool,
            "compare_financial_metrics_tool": _async_compare_financial_metrics_tool,
        },
        max_loops=max_loops,
    )


async def _async_compare_financial_metrics_tool(metric_name: str, values: list[dict]) -> ToolObservation:
    return compare_financial_metrics_tool(metric_name=metric_name, values=values)


async def ask_agent(
    question: str,
    index_path: Path,
    embedding_provider_name: str,
    llm_provider_name: str,
    top_k: int,
    max_loops: int,
) -> AgentRunResult:
    agent = create_react_agent(
        index_path=index_path,
        embedding_provider_name=embedding_provider_name,
        llm_provider_name=llm_provider_name,
        top_k=top_k,
        max_loops=max_loops,
    )
    return await agent.run(question)


async def ask_agent_with_correction(
    question: str,
    index_path: Path,
    embedding_provider_name: str,
    llm_provider_name: str,
    top_k: int,
    max_loops: int,
    max_corrections: int,
) -> tuple[AgentRunResult, GuardrailResult, int]:
    if max_corrections < 0:
        raise ValueError("max_corrections must be greater than or equal to 0.")

    correction_prompt = question
    correction_count = 0

    for attempt in range(max_corrections + 1):
        result = await ask_agent(
            question=correction_prompt,
            index_path=index_path,
            embedding_provider_name=embedding_provider_name,
            llm_provider_name=llm_provider_name,
            top_k=top_k,
            max_loops=max_loops,
        )
        guardrail = check_numeric_faithfulness(result)
        if guardrail.passed or attempt == max_corrections:
            return result, guardrail, correction_count

        correction_count += 1
        correction_prompt = build_correction_prompt(question, guardrail)

    raise RuntimeError("Unreachable correction loop state.")


def build_correction_prompt(question: str, guardrail: GuardrailResult) -> str:
    unsupported = ", ".join(guardrail.unsupported_numbers)
    issues = "\n".join(f"- {issue}" for issue in guardrail.issues)
    return f"""Original question:
{question}

Your previous answer failed numeric faithfulness validation.
Unsupported numbers: {unsupported}
Issues:
{issues}

Re-run the ReAct process. Search for direct evidence for any unsupported numbers before using them. If you cannot find support, omit those numbers from the final answer."""


def save_agent_trace(
    result: AgentRunResult,
    output_path: Path,
    guardrail: GuardrailResult | None = None,
    correction_count: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent_run_result_to_dict(result)
    if guardrail is not None:
        payload["guardrail"] = {
            "passed": guardrail.passed,
            "checked_numbers": guardrail.checked_numbers,
            "unsupported_numbers": guardrail.unsupported_numbers,
            "unsupported_contexts": guardrail.unsupported_contexts,
            "issues": guardrail.issues,
        }
    if correction_count is not None:
        payload["correction_count"] = correction_count
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def default_trace_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"agent_trace_{timestamp}.json"


def default_eval_report_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"eval_report_{timestamp}.json"


def default_profile_report_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"profile_report_{timestamp}.json"


def default_profile_sweep_report_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"profile_sweep_{timestamp}.json"


def parse_concurrency_values(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one concurrency value is required.")
    if any(item <= 0 for item in values):
        raise ValueError("Concurrency values must be positive.")
    return values


async def evaluate_agent(
    cases_path: Path,
    output_path: Path,
    index_path: Path,
    embedding_provider_name: str,
    llm_provider_name: str,
    top_k: int,
    max_loops: int,
    max_corrections: int,
    category: str | None = None,
    limit: int | None = None,
) -> list[EvalCaseResult]:
    cases = select_eval_cases(load_eval_cases(cases_path), category=category, limit=limit)

    async def answer_fn(question: str) -> tuple[AgentRunResult, GuardrailResult, int]:
        return await ask_agent_with_correction(
            question=question,
            index_path=index_path,
            embedding_provider_name=embedding_provider_name,
            llm_provider_name=llm_provider_name,
            top_k=top_k,
            max_loops=max_loops,
            max_corrections=max_corrections,
        )

    results = await run_eval_cases(cases, answer_fn)
    save_eval_report(results, output_path)
    print_eval_results(results, output_path)
    return results


async def profile_agent(
    cases_path: Path,
    output_path: Path,
    index_path: Path,
    embedding_provider_name: str,
    llm_provider_name: str,
    top_k: int,
    max_loops: int,
    max_corrections: int,
    concurrency: int,
    category: str | None = None,
    limit: int | None = None,
) -> ProfileRunResult:
    cases = select_eval_cases(load_eval_cases(cases_path), category=category, limit=limit)

    async def answer_fn(question: str) -> tuple[AgentRunResult, GuardrailResult, int]:
        return await ask_agent_with_correction(
            question=question,
            index_path=index_path,
            embedding_provider_name=embedding_provider_name,
            llm_provider_name=llm_provider_name,
            top_k=top_k,
            max_loops=max_loops,
            max_corrections=max_corrections,
        )

    run = await profile_cases(cases, answer_fn, concurrency=concurrency)
    save_profile_report(run, output_path)
    print_profile_results(run, output_path)
    return run


async def profile_agent_sweep(
    cases_path: Path,
    output_path: Path,
    index_path: Path,
    embedding_provider_name: str,
    llm_provider_name: str,
    top_k: int,
    max_loops: int,
    max_corrections: int,
    concurrency_values: list[int],
    category: str | None = None,
    limit: int | None = None,
) -> ProfileSweepResult:
    cases = select_eval_cases(load_eval_cases(cases_path), category=category, limit=limit)

    async def answer_fn(question: str) -> tuple[AgentRunResult, GuardrailResult, int]:
        return await ask_agent_with_correction(
            question=question,
            index_path=index_path,
            embedding_provider_name=embedding_provider_name,
            llm_provider_name=llm_provider_name,
            top_k=top_k,
            max_loops=max_loops,
            max_corrections=max_corrections,
        )

    sweep = await profile_sweep(cases, answer_fn, concurrency_values=concurrency_values)
    save_profile_sweep_report(sweep, output_path)
    print_profile_sweep_results(sweep, output_path)
    return sweep


def print_eval_results(results: list[EvalCaseResult], output_path: Path) -> None:
    summary = summarize_eval_results(results)
    print("\nEvaluation Summary")
    print(f"Cases: {summary['case_count']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Guardrail pass rate: {summary['guardrail_pass_rate']:.2%}")
    print(f"Expected metadata hit rate: {summary['expected_metadata_hit_rate']:.2%}")
    print(f"Expected tool calls hit rate: {summary['expected_tool_calls_hit_rate']:.2%}")
    print(f"Expected tool call query hit rate: {summary['expected_tool_call_queries_hit_rate']:.2%}")
    print(f"Required terms hit rate: {summary['required_terms_hit_rate']:.2%}")
    print(f"Average agent loops: {summary['average_agent_loops']:.2f}")
    print(f"Average correction attempts: {summary['average_correction_attempts']:.2f}")
    print(f"Average latency seconds: {summary['average_latency_seconds']:.2f}")
    print(f"Report: {output_path}")

    if summary["by_category"]:
        print("\nBy Category")
        for category, category_summary in summary["by_category"].items():
            print(
                f"- {category}: cases={category_summary['case_count']}, "
                f"success={category_summary['success_rate']:.2%}, "
                f"tool_calls={category_summary['expected_tool_calls_hit_rate']:.2%}, "
                f"queries={category_summary['expected_tool_call_queries_hit_rate']:.2%}, "
                f"loops={category_summary['average_agent_loops']:.2f}, "
                f"corrections={category_summary['average_correction_attempts']:.2f}"
            )

    if not results:
        return

    print("\nCases")
    for result in results:
        status = "PASS" if result.guardrail.passed and result.expected_metadata_hit and result.required_terms_hit else "FAIL"
        print(
            f"- {status} {result.case.case_id} [{result.case.category}]: loops={result.result.loop_count}, "
            f"corrections={result.correction_count}, guardrail={result.guardrail.passed}, "
            f"tool_calls={result.expected_tool_calls_hit}, queries={result.expected_tool_call_queries_hit}"
        )
        if result.missing_expected_tool_calls:
            print(f"  missing_tool_calls={json.dumps(result.missing_expected_tool_calls, ensure_ascii=False)}")


def print_profile_results(run: ProfileRunResult, output_path: Path) -> None:
    summary = summarize_profile_run(run)
    latency = summary["latency_seconds"]
    print("\nProfile Summary")
    print(f"Cases: {summary['case_count']}")
    print(f"Concurrency: {summary['concurrency']}")
    print(f"Total latency seconds: {summary['total_latency_seconds']:.2f}")
    print(f"Throughput cases/sec: {summary['throughput_cases_per_second']:.3f}")
    print(f"Latency p50/p95/p99: {latency['p50']:.2f}s / {latency['p95']:.2f}s / {latency['p99']:.2f}s")
    print(f"Average agent loops: {summary['average_agent_loops']:.2f}")
    print(f"Average correction attempts: {summary['average_correction_attempts']:.2f}")
    print(f"Guardrail pass rate: {summary['guardrail_pass_rate']:.2%}")
    print(f"Error rate: {summary['error_rate']:.2%}")
    print(f"Report: {output_path}")

    for snapshot in run.gpu_memory_snapshots:
        if snapshot.error:
            print(f"GPU memory [{snapshot.label}]: {snapshot.error}")
        else:
            print(f"GPU memory [{snapshot.label}]: {json.dumps(snapshot.gpus, ensure_ascii=False)}")


def print_profile_sweep_results(sweep: ProfileSweepResult, output_path: Path) -> None:
    summary = summarize_profile_sweep(sweep)
    print("\nProfile Sweep Summary")
    print(f"Runs: {summary['run_count']}")
    print(f"Concurrency values: {summary['concurrency_values']}")
    print(f"Report: {output_path}")

    for run_summary in summary["runs"]:
        latency = run_summary["latency_seconds"]
        print(
            f"- concurrency={run_summary['concurrency']}: "
            f"cases={run_summary['case_count']}, "
            f"total={run_summary['total_latency_seconds']:.2f}s, "
            f"p95={latency['p95']:.2f}s, "
            f"p99={latency['p99']:.2f}s, "
            f"loops={run_summary['average_agent_loops']:.2f}, "
            f"corrections={run_summary['average_correction_attempts']:.2f}, "
            f"errors={run_summary['error_rate']:.2%}"
        )


def print_agent_result(
    result: AgentRunResult,
    full_trace_path: Path | None = None,
    guardrail: GuardrailResult | None = None,
    correction_count: int = 0,
) -> None:
    print("\nFinal Answer")
    print(result.final_answer)
    print(f"\nAgent loops: {result.loop_count}")
    print(f"Correction attempts: {correction_count}")
    if result.reflection_passed is not None:
        print(f"Reflection passed: {result.reflection_passed}")
    if guardrail is not None:
        print(f"Numeric guardrail passed: {guardrail.passed}")
        if guardrail.unsupported_numbers:
            print("Unsupported numbers:")
            for number in guardrail.unsupported_numbers:
                print(f"- {number}")
    if full_trace_path is not None:
        print(f"Full trace: {full_trace_path}")

    if not result.steps:
        return

    print("\nTrace")
    for idx, step in enumerate(result.steps, start=1):
        print(f"\n[{idx}] Thought")
        print(step.thought)
        print("Action")
        print(json.dumps({"name": step.action.name, "arguments": step.action.arguments}, ensure_ascii=False, indent=2))
        print("Observation")
        print(json.dumps(_compact_observation(step.observation), ensure_ascii=False, indent=2))


def _compact_observation(observation: ToolObservation) -> dict:
    results = observation.results
    return {
        "tool_name": observation.tool_name,
        "query": observation.query,
        "filters": observation.filters,
        "result_count": len(results),
        "results": [
            {
                "metadata": result.get("metadata"),
                "score": result.get("score"),
                "text_preview": result.get("text", "")[:500],
            }
            for result in results
        ],
        "output": observation.output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial RAG engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="Build a FAISS index from local transcripts.")
    build.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    build.add_argument("--chunk-size", type=int, default=500)
    build.add_argument("--overlap", type=int, default=50)
    build.add_argument("--embedding-provider", choices=("voyage", "openai"), default="voyage")

    search = subparsers.add_parser("search", help="Search a built FAISS index.")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    search.add_argument("--ticker")
    search.add_argument("--year", type=int)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--embedding-provider", choices=("voyage", "openai"), default="voyage")

    ask = subparsers.add_parser("ask", help="Ask the agentic RAG engine a financial question.")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--max-loops", type=int, default=3)
    ask.add_argument("--embedding-provider", choices=("voyage", "openai"), default="voyage")
    ask.add_argument("--llm-provider", choices=("claude", "ollama"), default="claude")
    ask.add_argument("--trace-output", type=Path, help="Write the full agent trace to a JSON file.")
    ask.add_argument("--no-trace-file", action="store_true", help="Do not write a full agent trace JSON file.")
    ask.add_argument("--max-corrections", type=int, default=1, help="Retry with a correction prompt if numeric guardrail fails.")

    eval_parser = subparsers.add_parser("eval", help="Run a fixed evaluation set against the agentic RAG engine.")
    eval_parser.add_argument("--cases", type=Path, default=DEFAULT_EVAL_CASES_PATH)
    eval_parser.add_argument("--output", type=Path, default=None, help="Write the evaluation report JSON here.")
    eval_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--max-loops", type=int, default=3)
    eval_parser.add_argument("--max-corrections", type=int, default=1)
    eval_parser.add_argument("--category", help="Run only cases from this category, such as simple or comparison.")
    eval_parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    eval_parser.add_argument("--embedding-provider", choices=("voyage", "openai"), default="voyage")
    eval_parser.add_argument("--llm-provider", choices=("claude", "ollama"), default="claude")

    profile = subparsers.add_parser("profile", help="Profile agentic RAG latency, loops, corrections, and GPU memory.")
    profile.add_argument("--cases", type=Path, default=DEFAULT_EVAL_CASES_PATH)
    profile.add_argument("--output", type=Path, default=None, help="Write the profile report JSON here.")
    profile.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    profile.add_argument("--top-k", type=int, default=3)
    profile.add_argument("--max-loops", type=int, default=3)
    profile.add_argument("--max-corrections", type=int, default=1)
    profile.add_argument("--concurrency", type=int, default=1)
    profile.add_argument("--category", help="Profile only cases from this category.")
    profile.add_argument("--limit", type=int, help="Profile only the first N selected cases.")
    profile.add_argument("--embedding-provider", choices=("voyage", "openai"), default="voyage")
    profile.add_argument("--llm-provider", choices=("claude", "ollama"), default="claude")

    sweep = subparsers.add_parser("profile-sweep", help="Run profile across multiple concurrency settings.")
    sweep.add_argument("--cases", type=Path, default=DEFAULT_EVAL_CASES_PATH)
    sweep.add_argument("--output", type=Path, default=None, help="Write the profile sweep JSON here.")
    sweep.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    sweep.add_argument("--top-k", type=int, default=3)
    sweep.add_argument("--max-loops", type=int, default=3)
    sweep.add_argument("--max-corrections", type=int, default=1)
    sweep.add_argument("--concurrency-values", default="1,2,4,8,16")
    sweep.add_argument("--category", help="Profile only cases from this category.")
    sweep.add_argument("--limit", type=int, help="Profile only the first N selected cases.")
    sweep.add_argument("--embedding-provider", choices=("voyage", "openai"), default="voyage")
    sweep.add_argument("--llm-provider", choices=("claude", "ollama"), default="claude")

    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.command == "build-index":
        await build_index(args.data, args.index, args.chunk_size, args.overlap, args.embedding_provider)
    elif args.command == "search":
        await search_index(args.index, args.query, args.ticker, args.year, args.top_k, args.embedding_provider)
    elif args.command == "ask":
        result, guardrail, correction_count = await ask_agent_with_correction(
            question=args.question,
            index_path=args.index,
            embedding_provider_name=args.embedding_provider,
            llm_provider_name=args.llm_provider,
            top_k=args.top_k,
            max_loops=args.max_loops,
            max_corrections=args.max_corrections,
        )
        trace_path = None if args.no_trace_file else args.trace_output or default_trace_path()
        if trace_path is not None:
            save_agent_trace(result, trace_path, guardrail=guardrail, correction_count=correction_count)
        print_agent_result(result, full_trace_path=trace_path, guardrail=guardrail, correction_count=correction_count)
    elif args.command == "eval":
        await evaluate_agent(
            cases_path=args.cases,
            output_path=args.output or default_eval_report_path(),
            index_path=args.index,
            embedding_provider_name=args.embedding_provider,
            llm_provider_name=args.llm_provider,
            top_k=args.top_k,
            max_loops=args.max_loops,
            max_corrections=args.max_corrections,
            category=args.category,
            limit=args.limit,
        )
    elif args.command == "profile":
        await profile_agent(
            cases_path=args.cases,
            output_path=args.output or default_profile_report_path(),
            index_path=args.index,
            embedding_provider_name=args.embedding_provider,
            llm_provider_name=args.llm_provider,
            top_k=args.top_k,
            max_loops=args.max_loops,
            max_corrections=args.max_corrections,
            concurrency=args.concurrency,
            category=args.category,
            limit=args.limit,
        )
    elif args.command == "profile-sweep":
        await profile_agent_sweep(
            cases_path=args.cases,
            output_path=args.output or default_profile_sweep_report_path(),
            index_path=args.index,
            embedding_provider_name=args.embedding_provider,
            llm_provider_name=args.llm_provider,
            top_k=args.top_k,
            max_loops=args.max_loops,
            max_corrections=args.max_corrections,
            concurrency_values=parse_concurrency_values(args.concurrency_values),
            category=args.category,
            limit=args.limit,
        )


if __name__ == "__main__":
    asyncio.run(main())
