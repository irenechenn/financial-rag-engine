# Project Report

## AI Intuition / Why This Exists

This project was built to move beyond naive RAG. A financial question often requires multiple pieces of evidence, period-specific retrieval, numeric comparison, and answer validation. The system therefore treats retrieval and comparison as tools inside an agent loop rather than stuffing context into one prompt and hoping the model answers correctly.

## Objective

Build a financial Agentic RAG engine that can:

- retrieve earnings-call transcript evidence with metadata filters,
- plan multi-step financial analysis through ReAct,
- call tools for retrieval and metric comparison,
- validate generated financial numbers against observations,
- evaluate retrieval/answer behavior across curated cases,
- profile local multi-loop inference on RTX 5080 hardware.

## System Design

The system has four main layers:

1. Data and indexing
   `src/data_loader.py` chunks transcript records and preserves metadata. `src/vector_store.py` builds and loads a FAISS index.

2. Provider abstraction
   `src/interfaces.py` defines `EmbeddingProvider` and `LLMProvider`. Voyage handles embeddings; Claude and Ollama handle generation through interchangeable providers.

3. Agentic RAG
   `src/agent.py` runs the ReAct loop. `src/tools.py` exposes transcript search and financial metric comparison as callable tools.

4. Safety, evaluation, and profiling
   `src/guardrails.py` checks numeric faithfulness. `src/evaluation.py` measures answer behavior. `src/profiler.py` measures latency, loops, corrections, error rate, and GPU memory across concurrency settings.

## Key Engineering Decisions

### Tool Calls Instead Of Direct Answering

The model must call tools before answering. This makes evidence collection inspectable and allows evaluation to check whether the agent actually searched the right ticker/year.

### Runtime Errors Become Observations

Local Llama occasionally emits imperfect tool calls. Instead of letting bad tool calls crash profiling runs, the runtime converts unknown tools and invalid arguments into Observation errors. This gives the agent a chance to self-correct on the next loop.

### Numeric Guardrails

Financial answers are high-risk when numbers are hallucinated. The guardrail extracts numbers from the final answer and checks whether they are supported by retrieved observations.

### Curated Benchmark Artifacts

Raw `outputs/` are ignored. Selected results are committed under `benchmarks/5080/` so the project retains reproducible evidence without turning the repo into a scratch log.

## Final Benchmark Summary

RTX 5080 + Ollama Llama 3.1 8B:

- Simple sweep: `profile_sweep_20260627_005946.json`
- Comparison sweep: `profile_sweep_20260627_010307.json`
- Best concurrency: `4`
- Runtime errors: `0%`
- Numeric guardrail pass rate: `100%`

Interpretation:

`concurrency=4` delivered the best throughput/latency trade-off. Increasing to `8` or `16` did not improve throughput and increased tail latency in the simple sweep.

## Limitations

- The current sample dataset is intentionally small.
- Embeddings still depend on Voyage API.
- Comparison evaluation has only two curated cases.
- `errors=0%` means the runtime did not crash; it does not prove every answer is semantically perfect.

## Future Work

- Add local embedding provider for a fully offline path.
- Expand eval cases across more companies, years, and financial metrics.
- Add richer answer-quality scoring beyond numeric guardrails.
- Compare multiple local Ollama models on the same profiler sweep.
- Add a small web UI for trace inspection.

## Final Status

The project is complete as a portfolio-grade Agentic RAG prototype with documented setup, demo commands, curated benchmarks, and final RTX 5080 profiling results.

