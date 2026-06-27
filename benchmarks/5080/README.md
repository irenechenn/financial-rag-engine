# RTX 5080 Benchmark Notes

These benchmark artifacts are curated profile sweep reports from local Ollama inference on an RTX 5080.

## AI Intuition / Why This Exists

Agentic RAG latency is not a single model-call number. Each user question can trigger multiple ReAct loops, multiple transcript searches, and optional correction passes. The profiler records how this multi-step workload behaves as concurrent agent sessions increase.

`outputs/` remains ignored because it is scratch output. This directory stores selected benchmark JSON files that are useful for analysis and project documentation.

## Final Baseline

Model path:

```text
Voyage embeddings -> FAISS retrieval -> Ollama Llama 3.1 8B generation
```

Latest stable simple sweep:

```text
profile_sweep_20260627_005946.json
```

Latest stable comparison sweep:

```text
profile_sweep_20260627_010307.json
```

## Simple Cases

`profile_sweep_20260627_005946.json`

| Concurrency | Cases | Total | Throughput | p95 | p99 | Avg Loops | Errors | Guardrail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 26.89s | 0.223/s | 4.60s | 4.60s | 3.00 | 0% | 100% |
| 2 | 6 | 17.00s | 0.353/s | 6.11s | 6.17s | 3.00 | 0% | 100% |
| 4 | 6 | 15.72s | 0.382/s | 10.44s | 10.67s | 3.00 | 0% | 100% |
| 8 | 6 | 16.09s | 0.373/s | 15.81s | 16.01s | 3.00 | 0% | 100% |
| 16 | 6 | 16.04s | 0.374/s | 15.73s | 15.96s | 3.00 | 0% | 100% |

## Comparison Cases

`profile_sweep_20260627_010307.json`

| Concurrency | Cases | Total | Throughput | p95 | p99 | Avg Loops | Errors | Guardrail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 8.41s | 0.238/s | 4.31s | 4.32s | 3.00 | 0% | 100% |
| 2 | 2 | 5.60s | 0.357/s | 5.55s | 5.59s | 3.00 | 0% | 100% |
| 4 | 2 | 5.45s | 0.367/s | 5.39s | 5.43s | 3.00 | 0% | 100% |
| 8 | 2 | 5.82s | 0.344/s | 5.75s | 5.79s | 3.00 | 0% | 100% |
| 16 | 2 | 5.68s | 0.352/s | 5.62s | 5.66s | 3.00 | 0% | 100% |

## Readout

Concurrency `4` is the current local inference sweet spot:

- It delivers the best throughput in both simple and comparison sweeps.
- It keeps runtime errors at `0%`.
- It keeps numeric guardrail pass rate at `100%`.
- Higher concurrency does not improve throughput and increases tail latency on the simple sweep.

## Commands

```powershell
python main.py profile-sweep --llm-provider ollama --embedding-provider voyage --concurrency-values 1,2,4,8,16 --category simple
```

```powershell
python main.py profile-sweep --llm-provider ollama --embedding-provider voyage --concurrency-values 1,2,4,8,16 --category comparison
```
