#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

PROM_BASE = "https://metrics.swissai.svc.cscs.ch/api/datasources/proxy/uid/PBFA97CFB590B2093/api/v1/query"

SOURCE_PROMPT_DISTRIBUTION = {
    "source_file": "/capstor/scratch/cscs/bsezen/loadtest/prompts-apertus.json",
    "label": "medium",
    "count": 1000,
    "input_tokens": {
        "min": 403,
        "p25": 542,
        "median": 591,
        "p75": 635,
        "p95": 683,
        "max": 700,
        "mean": 585.615,
        "std": 65.269,
    },
    "max_tokens": {
        "min": 2,
        "p25": 157,
        "median": 263,
        "p75": 382,
        "p95": 533,
        "max": 799,
        "mean": 275.085,
        "std": 154.270,
    },
    "filler_units": [" a", " the", " x", " A", "hello", " token"],
}

SERVING_CONFIG = {
    "vLLM": {
        "model_name": "swiss-ai/Apertus-70B-Instruct-2509-vllm-pure-brachium-20260618-115341",
        "slurm_job_id": "2558545",
        "node": "nid006189",
        "launch_script": "sml/model-launch/local/apertus70b-vllm-vs-sglang/01_vllm_pure.sh",
        "comparison_flags": ["--no-enable-prefix-caching"],
        "cache_policy": "prefix caching disabled",
    },
    "SGLang": {
        "model_name": "swiss-ai/Apertus-70B-Instruct-2509-sglang-pure-brachium-20260618-115342",
        "slurm_job_id": "2558544",
        "node": "nid006134",
        "launch_script": "sml/model-launch/local/apertus70b-vllm-vs-sglang/02_sglang_pure.sh",
        "comparison_flags": ["--disable-radix-cache", "--enable-metrics"],
        "cache_policy": "radix cache disabled; metrics enabled for scraping",
    },
}


@dataclass(frozen=True)
class Run:
    name: str
    engine: str
    replicate: int
    db: Path
    slurm_job_id: str
    node: str


RUNS = [
    Run("vLLM #1", "vLLM", 1, DATA / "vllm_run1.db", "2558545", "nid006189"),
    Run("vLLM #2", "vLLM", 2, DATA / "vllm_run2.db", "2558545", "nid006189"),
    Run("SGLang #1", "SGLang", 1, DATA / "sglang_run1.db", "2558544", "nid006134"),
    Run("SGLang #2", "SGLang", 2, DATA / "sglang_run2.db", "2558544", "nid006134"),
]


def pct(values: list[float], p: float) -> float | None:
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def load_run(run: Run) -> dict:
    con = sqlite3.connect(run.db)
    con.row_factory = sqlite3.Row
    exp = dict(con.execute("select * from experiments limit 1").fetchone())
    rates = []
    for rate, start, end in con.execute(
        "select rate_lambda, min(ts), max(ts) from server_stats group by rate_lambda order by rate_lambda"
    ):
        rows = con.execute(
            "select success,error,ttft_ms,tpot_ms,e2e_ms,input_tokens,output_tokens from requests where rate_lambda=?",
            (rate,),
        ).fetchall()
        ok = [r for r in rows if r["success"]]
        errors = len(rows) - len(ok)
        rates.append(
            {
                "rate": float(rate),
                "start": start,
                "end": end,
                "requests": len(rows),
                "success": len(ok),
                "error_pct": 100 * errors / len(rows) if rows else None,
                "ttft_p50_ms": pct([r["ttft_ms"] for r in ok], 50),
                "ttft_p95_ms": pct([r["ttft_ms"] for r in ok], 95),
                "ttft_p99_ms": pct([r["ttft_ms"] for r in ok], 99),
                "tpot_p50_ms": pct([r["tpot_ms"] for r in ok], 50),
                "tpot_p95_ms": pct([r["tpot_ms"] for r in ok], 95),
                "tpot_p99_ms": pct([r["tpot_ms"] for r in ok], 99),
                "e2e_p95_ms": pct([r["e2e_ms"] for r in ok], 95),
                "input_tokens_avg": sum(r["input_tokens"] for r in ok) / len(ok) if ok else None,
                "output_tokens_avg": sum(r["output_tokens"] for r in ok) / len(ok) if ok else None,
                "input_tokens_s": sum(r["input_tokens"] for r in ok) / 180,
                "output_tokens_s": sum(r["output_tokens"] for r in ok) / 180,
            }
        )
    con.close()
    return {"run": run.__dict__ | {"db": str(run.db.name)}, "experiment": exp, "rates": rates}


def iso_to_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def prom_query(query: str, ts: float) -> list[float]:
    params = urllib.parse.urlencode({"query": query, "time": f"{ts:.0f}"})
    try:
        with urllib.request.urlopen(f"{PROM_BASE}?{params}", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    if payload.get("status") != "success":
        return []
    values: list[float] = []
    for item in payload.get("data", {}).get("result", []):
        try:
            values.append(float(item["value"][1]))
        except Exception:
            pass
    return [value for value in values if math.isfinite(value)]


def scalar(query: str, ts: float) -> float | None:
    try:
        vals = prom_query(query, ts)
        return max(vals) if vals else None
    except Exception:
        return None


def add_dcgm(data: dict) -> None:
    metrics = {
        "gpu_util_pct": "avg(avg_over_time(DCGM_FI_DEV_GPU_UTIL{{{sel}}}[{dur}s]))",
        "sm_active_pct": "100 * avg(avg_over_time(DCGM_FI_PROF_SM_ACTIVE{{{sel}}}[{dur}s]))",
        "tensor_active_pct": "100 * avg(avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{sel}}}[{dur}s]))",
        "mem_copy_util_pct": "avg(avg_over_time(DCGM_FI_DEV_MEM_COPY_UTIL{{{sel}}}[{dur}s]))",
        "fb_used_gib": "avg(avg_over_time(DCGM_FI_DEV_FB_USED{{{sel}}}[{dur}s])) / 1024",
        "power_total_w": "sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE{{{sel}}}[{dur}s]))",
    }
    for item in data["runs"]:
        sel = f'slurm_job_id="{item["run"]["slurm_job_id"]}"'
        for rate in item["rates"]:
            start = iso_to_ts(rate["start"])
            end = iso_to_ts(rate["end"])
            dur = max(1, int(end - start))
            rate["dcgm"] = {
                name: scalar(template.format(sel=sel, dur=dur), end)
                for name, template in metrics.items()
            }


def grouped(data: dict) -> dict[str, dict[int, dict[float, dict]]]:
    out: dict[str, dict[int, dict[float, dict]]] = {}
    for item in data["runs"]:
        engine = item["run"]["engine"]
        rep = item["run"]["replicate"]
        out.setdefault(engine, {})[rep] = {r["rate"]: r for r in item["rates"]}
    return out


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def generate_plots(data: dict) -> None:
    if plt is None:
        return
    g = grouped(data)
    colors = {"vLLM": "#0072B2", "SGLang": "#D55E00"}
    rates = [2.0, 4.0, 8.0, 12.0]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for metric, ax, slo, label in [
        ("ttft_p95_ms", axes[0], 10000, "TTFT p95 (ms)"),
        ("tpot_p95_ms", axes[1], 200, "TPOT p95 (ms)"),
    ]:
        for engine, reps in g.items():
            means, lows, highs = [], [], []
            for rate in rates:
                m, s = mean_std([reps[r].get(rate, {}).get(metric) for r in reps])
                means.append(m)
                lows.append(s or 0)
                highs.append(s or 0)
            ax.errorbar(rates, means, yerr=[lows, highs], marker="o", linewidth=2, capsize=4, label=engine, color=colors[engine])
        ax.axhline(slo, color="#cc0000", linestyle="--", label=f"SLO {slo:g} ms")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "latency_p95_replicates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for metric, ax, label in [
        ("input_tokens_s", axes[0], "Input tokens/s"),
        ("output_tokens_s", axes[1], "Output tokens/s"),
    ]:
        for engine, reps in g.items():
            means, lows, highs = [], [], []
            for rate in rates:
                m, s = mean_std([reps[r].get(rate, {}).get(metric) for r in reps])
                means.append(m)
                lows.append(s or 0)
                highs.append(s or 0)
            ax.errorbar(rates, means, yerr=[lows, highs], marker="o", linewidth=2, capsize=4, label=engine, color=colors[engine])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "throughput_replicates.png", dpi=180)
    plt.close(fig)

    dcgm_metrics = [("gpu_util_pct", "GPU util %"), ("sm_active_pct", "SM active %"), ("tensor_active_pct", "Tensor active %"), ("power_total_w", "Total power W")]
    fig, axes = plt.subplots(len(dcgm_metrics), 1, figsize=(9, 11), sharex=True)
    for (metric, label), ax in zip(dcgm_metrics, axes):
        for engine, reps in g.items():
            means, lows, highs = [], [], []
            for rate in rates:
                vals = [reps[r].get(rate, {}).get("dcgm", {}).get(metric) for r in reps]
                m, s = mean_std(vals)
                means.append(m)
                lows.append(s or 0)
                highs.append(s or 0)
            ax.errorbar(rates, means, yerr=[lows, highs], marker="o", linewidth=2, capsize=4, label=engine, color=colors[engine])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "dcgm_replicates.png", dpi=180)
    plt.close(fig)


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def metric_table(data: dict, metric: str) -> str:
    g = grouped(data)
    lines = ["| Engine | λ=2 | λ=4 | λ=8 | λ=12 |", "|---|---:|---:|---:|---:|"]
    for engine in ["vLLM", "SGLang"]:
        vals = []
        for rate in [2.0, 4.0, 8.0, 12.0]:
            m, s = mean_std([g[engine][rep][rate].get(metric) for rep in g[engine]])
            vals.append(f"{fmt(m)} ± {fmt(s)}")
        lines.append(f"| {engine} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def dcgm_table(data: dict, metric: str) -> str:
    g = grouped(data)
    lines = ["| Engine | λ=2 | λ=4 | λ=8 | λ=12 |", "|---|---:|---:|---:|---:|"]
    for engine in ["vLLM", "SGLang"]:
        vals = []
        for rate in [2.0, 4.0, 8.0, 12.0]:
            m, s = mean_std([g[engine][rep][rate].get("dcgm", {}).get(metric) for rep in g[engine]])
            vals.append(f"{fmt(m)} ± {fmt(s)}")
        lines.append(f"| {engine} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(data: dict) -> None:
    report = f"""# Apertus-70B vLLM vs SGLang: pure-cache-disabled comparison

**Date:** 2026-06-18  
**Model:** `swiss-ai/Apertus-70B-Instruct-2509`  
**Hardware:** single Clariden GH200 node per engine, TP4  
**Replicates:** N=2 per engine  
**Benchmark nodes:** `infra02` with `power_throttling` reservation

## Executive Summary

This report compares vLLM and SGLang on Apertus-70B after correcting the initial invalid comparison. vLLM prefix caching and SGLang radix caching are both disabled. The exact same prompt pool, arrival process, phases, SLOs, and served models were reused across the two replicates.

| Finding | Result |
|---|---|
| Maximum passing swept rate | Both engines pass through **λ=8 req/s** |
| Saturation point | Both engines early-stop at **λ=12 req/s** |
| vLLM saturation reason | TPOT p95 crosses 200 ms at λ=12 |
| SGLang saturation reason | TTFT p95 jumps to ~49 s and TPOT p95 ~355 ms at λ=12 |
| Replicability | The λ=12 failure mode reproduced in both runs |

## Methodology

| Attribute | Value |
|---|---|
| Scenario | `thesis-apertus-medium` |
| Prompt source | `/capstor/scratch/cscs/bsezen/loadtest/prompts-apertus.json`, label `medium` only |
| Source input-token shape | 1,000 medium prompts; min 403, p25 542, median 591, p75 635, p95 683, max 700, mean 585.6 ± 65.3 |
| Source output budget shape (`max_tokens`) | min 2, p25 157, median 263, p75 382, p95 533, max 799, mean 275.1 ± 154.3 |
| Observed successful output length | ~269-278 output chunks/tokens/request depending on λ and engine |
| Total prompts | 20,000 generated from 1,000 medium prompts with recycling |
| Arrival process | Poisson |
| Sweep | `[2, 4, 8, 12, 16, 20, 24]` req/s with early stop |
| Phases | 60 s warmup, 180 s measurement, 300 s drain |
| SLOs | TTFT p95 ≤ 10,000 ms, TPOT p95 ≤ 200 ms, error ≤ 1% |

The exact same offered load was used for both engines and both replicates. The corpus is intentionally synthetic: each prompt is filler text (`" a"`, `" the"`, `" x"`, `" A"`, `"hello"`, or `" token"`) chosen so the Apertus tokenizer produces exact target token counts. The benchmark uses only the `medium` label, not the long-input or XL-input prompts. This makes the workload suitable for controlled throughput/latency stress, but not for quality conclusions.

### Serving Launch Configuration

| Engine | Served model | SLURM job | Node | Relevant launch flags | Launch script |
|---|---|---:|---|---|---|
| vLLM | `swiss-ai/Apertus-70B-Instruct-2509-vllm-pure-brachium-20260618-115341` | 2558545 | nid006189 | `--no-enable-prefix-caching` | `sml/model-launch/local/apertus70b-vllm-vs-sglang/01_vllm_pure.sh` |
| SGLang | `swiss-ai/Apertus-70B-Instruct-2509-sglang-pure-brachium-20260618-115342` | 2558544 | nid006134 | `--disable-radix-cache`, `--enable-metrics` | `sml/model-launch/local/apertus70b-vllm-vs-sglang/02_sglang_pure.sh` |

The corrected comparison intentionally disables both caching paths that would otherwise make the engine comparison asymmetric: vLLM prefix caching is disabled and SGLang radix caching is disabled. SGLang metrics are enabled so Prometheus/DCGM-related scraping remains available.

The benchmark databases also contain IBT `backend_config` metadata, but these runs used external SML-served SwissAI API endpoints. For serving behavior, the launch scripts and served-model job configuration above are authoritative.

### Prompt and Output Distribution

| Field | min | p25 | median | p75 | p95 | max | mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source `input_tokens` | 403 | 542 | 591 | 635 | 683 | 700 | 585.6 ± 65.3 |
| Source `max_tokens` | 2 | 157 | 263 | 382 | 533 | 799 | 275.1 ± 154.3 |

Observed successful response lengths closely track the source output budgets because `output_length_mode: forced` was used. Across the corrected runs, average observed output length is ~269-278 output chunks/tokens per request.

## Capacity

![Latency p95 replicates](images/latency_p95_replicates.png)

### TTFT p95 (ms, mean ± std)

{metric_table(data, "ttft_p95_ms")}

### TPOT p95 (ms, mean ± std)

{metric_table(data, "tpot_p95_ms")}

At λ=8, vLLM has lower TTFT and substantially lower TPOT. At λ=12 both engines breach the TPOT SLO, but SGLang also develops a large TTFT queueing spike while vLLM remains below the TTFT SLO.

## Token Throughput

![Throughput replicates](images/throughput_replicates.png)

### Output tokens/s (mean ± std)

{metric_table(data, "output_tokens_s")}

The throughput curves use successful request rows only. Error rate was 0% in all corrected and replicated rate levels.

## DCGM Telemetry

![DCGM replicates](images/dcgm_replicates.png)

### GPU utilization % (mean ± std)

{dcgm_table(data, "gpu_util_pct")}

### SM active % (mean ± std)

{dcgm_table(data, "sm_active_pct")}

### Total GPU power W, 4 GPUs (mean ± std)

{dcgm_table(data, "power_total_w")}

DCGM was queried from SwissAI Prometheus/Grafana for the served-model SLURM jobs:

| Engine | SLURM job | Node |
|---|---:|---|
| vLLM | 2558545 | nid006189 |
| SGLang | 2558544 | nid006134 |

The report stores per-rate DCGM values in `data.json`, including GPU utilization, SM-active, tensor-active, memory-copy utilization, framebuffer used, and total GPU power. These counters are aligned to each benchmark level using the `server_stats` timestamps from the run DBs.

## Interpretation

The replicated result supports the corrected conclusion: vLLM is materially faster on this workload when both prefix/radix caching paths are disabled. The largest operational difference is at λ=12: vLLM is near the TPOT SLO boundary, while SGLang queues hard enough to push TTFT p95 to ~49 seconds.

This is a capacity result, not a quality result. Quality evaluation was disabled. The prompts are synthetic token-count prompts, so the run answers scheduler/throughput behavior under controlled token shapes.

## Disclosures & Limitations

- Initial invalid run was discarded because vLLM prefix caching was not disabled while SGLang radix cache was disabled.
- The IBT recycling path bug was fixed before the corrected runs; otherwise high-rate vLLM could crash when the pool recycled.
- SwissAI streaming does not reliably emit `[DONE]`; the benchmark accepts truncated streams with content via `IBT_ACCEPT_TRUNCATED_STREAM_WITH_CONTENT=1`.
- DCGM counters are external telemetry, aligned by wall-clock windows, not rows persisted inside the IBT DB.
- No quality gate or quality comparison was run.

## Provenance

| Item | Value |
|---|---|
| vLLM served model | `swiss-ai/Apertus-70B-Instruct-2509-vllm-pure-brachium-20260618-115341` |
| SGLang served model | `swiss-ai/Apertus-70B-Instruct-2509-sglang-pure-brachium-20260618-115342` |
| vLLM run DBs | `data/vllm_run1.db`, `data/vllm_run2.db` |
| SGLang run DBs | `data/sglang_run1.db`, `data/sglang_run2.db` |
| Generated | {datetime.now(timezone.utc).isoformat()} |
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    data = {"serving_config": SERVING_CONFIG, "source_prompt_distribution": SOURCE_PROMPT_DISTRIBUTION, "runs": [load_run(run) for run in RUNS]}
    add_dcgm(data)
    (ROOT / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    generate_plots(data)
    write_report(data)


if __name__ == "__main__":
    main()
