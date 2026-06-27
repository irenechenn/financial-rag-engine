# Demo Guide

This guide is the shortest path to showing the project end to end.

## AI Intuition / Why This Exists

The demo should prove three things quickly: the system can retrieve grounded transcript evidence, the agent can plan with tools instead of answering directly, and the profiler can measure local multi-loop inference on real hardware.

## 1. Build The Index

```powershell
python main.py build-index --embedding-provider voyage
```

Expected result:

```text
Built voyage index with ... chunks at indexes/naive_voyage_finance
```

## 2. Show Retrieval

```powershell
python main.py search "Apple services revenue growth" --ticker AAPL --year 2023
```

What to point out:

- The search is filtered by metadata.
- Returned chunks include transcript text and ticker/year/quarter metadata.
- This is the evidence layer the agent will use.

## 3. Run Agentic RAG

```powershell
python main.py ask "Compare Tesla gross margin in 2024 and 2025, and explain the reason using only retrieved evidence." --llm-provider ollama --embedding-provider voyage --max-loops 3 --max-corrections 1
```

What to point out:

- The trace shows Thought -> Action -> Observation.
- The agent calls `search_transcript_tool`.
- The final answer is checked by numeric guardrails.
- A full trace JSON is written to `outputs/`.

## 4. Run Evaluation

```powershell
python main.py eval --llm-provider ollama --embedding-provider voyage
```

What to point out:

- Evaluation is not just pass/fail.
- It checks metadata hits, expected tool calls, required answer terms, guardrails, loops, corrections, and latency.

## 5. Show RTX 5080 Profiling

```powershell
python main.py profile-sweep --llm-provider ollama --embedding-provider voyage --concurrency-values 1,2,4,8,16 --category simple
```

What to point out:

- Agentic RAG has multi-loop latency, not one model-call latency.
- The profiler measures throughput, p95/p99 latency, average agent loops, correction attempts, runtime errors, and GPU memory snapshots.
- Curated benchmark reports are in `benchmarks/5080/`.

## Final Talking Point

The project demonstrates a complete path from RAG retrieval to Agentic tool use, guardrail validation, evaluation, and local GPU profiling.

