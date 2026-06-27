from __future__ import annotations

from pathlib import Path

import pytest

import main
from src.agent_types import AgentRunResult, AgentStep, ToolCall, ToolObservation


def test_parse_args_supports_ask_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "ask",
            "Compare Tesla margins.",
            "--index",
            "indexes/naive_voyage_finance",
            "--top-k",
            "2",
            "--max-loops",
            "3",
            "--embedding-provider",
            "voyage",
            "--llm-provider",
            "ollama",
        ],
    )

    args = main.parse_args()

    assert args.command == "ask"
    assert args.question == "Compare Tesla margins."
    assert args.index == Path("indexes/naive_voyage_finance")
    assert args.top_k == 2
    assert args.max_loops == 3
    assert args.embedding_provider == "voyage"
    assert args.llm_provider == "ollama"


def test_parse_args_supports_eval_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "eval",
            "--cases",
            "eval_cases/day3_agentic_smoke.json",
            "--output",
            "outputs/eval_report.json",
            "--index",
            "indexes/naive_voyage_finance",
            "--top-k",
            "2",
            "--max-loops",
            "4",
            "--max-corrections",
            "1",
            "--category",
            "simple",
            "--limit",
            "2",
            "--embedding-provider",
            "voyage",
            "--llm-provider",
            "ollama",
        ],
    )

    args = main.parse_args()

    assert args.command == "eval"
    assert args.cases == Path("eval_cases/day3_agentic_smoke.json")
    assert args.output == Path("outputs/eval_report.json")
    assert args.index == Path("indexes/naive_voyage_finance")
    assert args.top_k == 2
    assert args.max_loops == 4
    assert args.max_corrections == 1
    assert args.category == "simple"
    assert args.limit == 2
    assert args.embedding_provider == "voyage"
    assert args.llm_provider == "ollama"


def test_parse_args_supports_profile_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "profile",
            "--cases",
            "eval_cases/day3_agentic_smoke.json",
            "--output",
            "outputs/profile_report.json",
            "--index",
            "indexes/naive_voyage_finance",
            "--top-k",
            "2",
            "--max-loops",
            "4",
            "--max-corrections",
            "1",
            "--concurrency",
            "2",
            "--category",
            "simple",
            "--limit",
            "2",
            "--embedding-provider",
            "voyage",
            "--llm-provider",
            "ollama",
        ],
    )

    args = main.parse_args()

    assert args.command == "profile"
    assert args.cases == Path("eval_cases/day3_agentic_smoke.json")
    assert args.output == Path("outputs/profile_report.json")
    assert args.index == Path("indexes/naive_voyage_finance")
    assert args.top_k == 2
    assert args.max_loops == 4
    assert args.max_corrections == 1
    assert args.concurrency == 2
    assert args.category == "simple"
    assert args.limit == 2
    assert args.embedding_provider == "voyage"
    assert args.llm_provider == "ollama"


def test_parse_args_supports_profile_sweep_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "profile-sweep",
            "--cases",
            "eval_cases/day3_agentic_smoke.json",
            "--output",
            "outputs/profile_sweep.json",
            "--index",
            "indexes/naive_voyage_finance",
            "--top-k",
            "2",
            "--max-loops",
            "4",
            "--max-corrections",
            "1",
            "--concurrency-values",
            "1,2,4",
            "--category",
            "simple",
            "--limit",
            "2",
            "--embedding-provider",
            "voyage",
            "--llm-provider",
            "ollama",
        ],
    )

    args = main.parse_args()

    assert args.command == "profile-sweep"
    assert args.output == Path("outputs/profile_sweep.json")
    assert args.concurrency_values == "1,2,4"
    assert args.category == "simple"
    assert args.limit == 2
    assert args.llm_provider == "ollama"


def test_parse_concurrency_values() -> None:
    assert main.parse_concurrency_values("1, 2,4") == [1, 2, 4]


def test_parse_concurrency_values_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        main.parse_concurrency_values("1,0")


def test_get_llm_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        main.get_llm_provider("unknown")


@pytest.mark.asyncio
async def test_ask_agent_runs_created_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        async def run(self, question: str) -> AgentRunResult:
            return AgentRunResult(question=question, steps=[], final_answer="done", loop_count=0)

    def fake_create_react_agent(
        index_path: Path,
        embedding_provider_name: str,
        llm_provider_name: str,
        top_k: int,
        max_loops: int,
    ) -> FakeAgent:
        assert index_path == Path("indexes/naive_voyage_finance")
        assert embedding_provider_name == "voyage"
        assert llm_provider_name == "ollama"
        assert top_k == 3
        assert max_loops == 3
        return FakeAgent()

    monkeypatch.setattr(main, "create_react_agent", fake_create_react_agent)

    result = await main.ask_agent(
        question="Question?",
        index_path=Path("indexes/naive_voyage_finance"),
        embedding_provider_name="voyage",
        llm_provider_name="ollama",
        top_k=3,
        max_loops=3,
    )

    assert result.final_answer == "done"
    assert result.question == "Question?"


@pytest.mark.asyncio
async def test_main_eval_passes_category_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "eval",
            "--cases",
            "eval_cases/day3_agentic_smoke.json",
            "--output",
            "outputs/eval_report.json",
            "--category",
            "simple",
            "--limit",
            "2",
        ],
    )

    async def fake_evaluate_agent(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "evaluate_agent", fake_evaluate_agent)

    await main.main()

    assert captured["cases_path"] == Path("eval_cases/day3_agentic_smoke.json")
    assert captured["output_path"] == Path("outputs/eval_report.json")
    assert captured["llm_provider_name"] == "claude"
    assert captured["category"] == "simple"
    assert captured["limit"] == 2


@pytest.mark.asyncio
async def test_main_profile_passes_concurrency_category_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "profile",
            "--cases",
            "eval_cases/day3_agentic_smoke.json",
            "--output",
            "outputs/profile_report.json",
            "--concurrency",
            "2",
            "--category",
            "simple",
            "--limit",
            "2",
        ],
    )

    async def fake_profile_agent(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(main, "profile_agent", fake_profile_agent)

    await main.main()

    assert captured["cases_path"] == Path("eval_cases/day3_agentic_smoke.json")
    assert captured["output_path"] == Path("outputs/profile_report.json")
    assert captured["llm_provider_name"] == "claude"
    assert captured["concurrency"] == 2
    assert captured["category"] == "simple"
    assert captured["limit"] == 2


@pytest.mark.asyncio
async def test_main_profile_sweep_passes_concurrency_values(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "profile-sweep",
            "--cases",
            "eval_cases/day3_agentic_smoke.json",
            "--output",
            "outputs/profile_sweep.json",
            "--concurrency-values",
            "1,2,4",
            "--category",
            "simple",
            "--limit",
            "2",
        ],
    )

    async def fake_profile_agent_sweep(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(main, "profile_agent_sweep", fake_profile_agent_sweep)

    await main.main()

    assert captured["cases_path"] == Path("eval_cases/day3_agentic_smoke.json")
    assert captured["output_path"] == Path("outputs/profile_sweep.json")
    assert captured["llm_provider_name"] == "claude"
    assert captured["concurrency_values"] == [1, 2, 4]
    assert captured["category"] == "simple"
    assert captured["limit"] == 2


def test_print_agent_result_includes_trace(capsys: pytest.CaptureFixture[str]) -> None:
    result = AgentRunResult(
        question="Question?",
        steps=[
            AgentStep(
                thought="Need evidence.",
                action=ToolCall(name="search_transcript_tool", arguments={"ticker": "TSLA", "year": 2025}),
                observation=ToolObservation(
                    tool_name="search_transcript_tool",
                    query="gross margin",
                    filters={"ticker": "TSLA", "year": 2025},
                    results=[
                        {
                            "text": "Tesla gross margin was 16%.",
                            "metadata": {"ticker": "TSLA", "year": 2025},
                            "score": 0.9,
                        }
                    ],
                ),
            )
        ],
        final_answer="Tesla gross margin was 16%.",
        loop_count=1,
    )

    main.print_agent_result(result)

    output = capsys.readouterr().out
    assert "Final Answer" in output
    assert "Tesla gross margin was 16%." in output
    assert "Agent loops: 1" in output
    assert "search_transcript_tool" in output
    assert "text_preview" in output


def test_save_agent_trace_includes_guardrail(tmp_path: Path) -> None:
    result = AgentRunResult(
        question="Question?",
        steps=[],
        final_answer="No numbers.",
        loop_count=0,
    )
    output_path = tmp_path / "trace.json"

    guardrail = main.check_numeric_faithfulness(result)
    main.save_agent_trace(result, output_path, guardrail=guardrail)

    content = output_path.read_text(encoding="utf-8")
    assert '"question": "Question?"' in content
    assert '"guardrail"' in content
    assert '"passed": true' in content
    assert '"unsupported_contexts"' in content


def test_save_agent_trace_includes_correction_count(tmp_path: Path) -> None:
    result = AgentRunResult(question="Question?", steps=[], final_answer="Answer", loop_count=0)
    output_path = tmp_path / "trace.json"

    main.save_agent_trace(result, output_path, correction_count=2)

    content = output_path.read_text(encoding="utf-8")
    assert '"correction_count": 2' in content


@pytest.mark.asyncio
async def test_ask_agent_with_correction_retries_after_guardrail_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    failed_result = AgentRunResult(
        question="Question?",
        steps=[
            AgentStep(
                thought="Need evidence.",
                action=ToolCall(name="search_transcript_tool", arguments={}),
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
        final_answer="Services revenue grew 9%.",
        loop_count=1,
    )
    corrected_result = AgentRunResult(
        question="Question?",
        steps=failed_result.steps,
        final_answer="Services revenue grew 8%.",
        loop_count=1,
    )

    async def fake_ask_agent(
        question: str,
        index_path: Path,
        embedding_provider_name: str,
        llm_provider_name: str,
        top_k: int,
        max_loops: int,
    ) -> AgentRunResult:
        calls.append(question)
        return failed_result if len(calls) == 1 else corrected_result

    monkeypatch.setattr(main, "ask_agent", fake_ask_agent)

    result, guardrail, correction_count = await main.ask_agent_with_correction(
        question="Question?",
        index_path=Path("indexes/naive_voyage_finance"),
        embedding_provider_name="voyage",
        llm_provider_name="ollama",
        top_k=3,
        max_loops=3,
        max_corrections=1,
    )

    assert result.final_answer == "Services revenue grew 8%."
    assert guardrail.passed is True
    assert correction_count == 1
    assert len(calls) == 2
    assert "Unsupported numbers: 9%" in calls[1]


@pytest.mark.asyncio
async def test_ask_agent_with_correction_stops_at_max_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ask_agent(
        question: str,
        index_path: Path,
        embedding_provider_name: str,
        llm_provider_name: str,
        top_k: int,
        max_loops: int,
    ) -> AgentRunResult:
        return AgentRunResult(
            question=question,
            steps=[
                AgentStep(
                    thought="Need evidence.",
                    action=ToolCall(name="search_transcript_tool", arguments={}),
                    observation=ToolObservation(
                        tool_name="search_transcript_tool",
                        results=[{"text": "Services revenue grew 8%.", "metadata": {}, "score": 0.9}],
                    ),
                )
            ],
            final_answer="Services revenue grew 9%.",
            loop_count=1,
        )

    monkeypatch.setattr(main, "ask_agent", fake_ask_agent)

    result, guardrail, correction_count = await main.ask_agent_with_correction(
        question="Question?",
        index_path=Path("indexes/naive_voyage_finance"),
        embedding_provider_name="voyage",
        llm_provider_name="ollama",
        top_k=3,
        max_loops=3,
        max_corrections=1,
    )

    assert result.final_answer == "Services revenue grew 9%."
    assert guardrail.passed is False
    assert correction_count == 1


@pytest.mark.asyncio
async def test_ask_agent_with_correction_rejects_negative_corrections() -> None:
    with pytest.raises(ValueError, match="max_corrections"):
        await main.ask_agent_with_correction(
            question="Question?",
            index_path=Path("indexes/naive_voyage_finance"),
            embedding_provider_name="voyage",
            llm_provider_name="ollama",
            top_k=3,
            max_loops=3,
            max_corrections=-1,
        )
