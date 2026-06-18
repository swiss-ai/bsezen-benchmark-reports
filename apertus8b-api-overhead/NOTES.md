# Apertus-8B API Overhead Experiment Notes

## Research Question

Does routing requests through `https://api.swissai.svc.cscs.ch` add measurable latency or throughput overhead compared with hitting the same Apertus-8B SGLang server directly from the benchmarker allocation?

## Hypothesis

The SwissAI API path may add a small fixed request overhead visible in TTFT, but should not materially change TPOT or the saturation point if the gateway is not applying meaningful streaming backpressure.

## Design

Use one Apertus-8B SGLang server and alternate benchmark paths:

1. Direct internal endpoint, replicate 1
2. SwissAI API endpoint, replicate 1
3. Direct internal endpoint, replicate 2
4. SwissAI API endpoint, replicate 2

Only the endpoint path should change. Model, server job, prompt pool, load schedule, and SLOs should remain fixed.

## Serving Configuration

| Field | Value |
|---|---|
| Model | `swiss-ai/Apertus-8B-Instruct-2509` |
| Engine | SGLang |
| Nodes | 1 Clariden GH200 node |
| Cache policy | `--disable-radix-cache` |
| Metrics | `--enable-metrics` |
| Account/reservation | `infra01`, `SD-69241-apertus-1-5-0` |
| Served job | `2561679` |
| Served model name | `swiss-ai/Apertus-8B-Instruct-2509-api-overhead-sglang-brachium-20260618-183436` |
| Direct endpoint | `http://172.28.40.248:8080` |
| API endpoint | `https://api.swissai.svc.cscs.ch` |

The exact launch script is stored under `launch-scripts/`.

## Benchmark Runs

| Path | Replicate | Benchmark job | Run DB |
|---|---:|---:|---|
| Direct | 1 | 2561761 | `data/direct_run1.db` |
| API | 1 | 2561998 | `data/api_run1.db` |
| Direct | 2 | 2562891 | `data/direct_run2.db` |
| API | 2 | 2563417 | `data/api_run2.db` |

All four runs used the same served model, prompt pool shape, arrival process, rate sweep, and SLOs. Only the endpoint path changed.

## DCGM Telemetry

The report should query DCGM by served-model SLURM job ID, not benchmark job ID:

```text
slurm_job_id="2561679"
```

Use the benchmark DB measurement windows to align telemetry per rate level. Include at minimum GPU utilization, SM active percentage, tensor active percentage, memory-copy utilization, framebuffer used, and total GPU power.

## Report Focus

Compare direct vs API for TTFT, TPOT, E2E latency, throughput, error rate, and server/DCGM telemetry. Randomness/quality is not part of this experiment; the model is real, but the benchmark is a serving-path overhead measurement.

## Preliminary Outcome

Both direct and API paths completed two clean replicates and early-stopped at `λ=24` due TTFT p95 saturation. The quick summary suggests the API path adds a small TTFT shift at low load, while TPOT and saturation behavior are similar.
