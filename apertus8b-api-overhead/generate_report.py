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


@dataclass(frozen=True)
class Run:
    name: str
    path: str
    replicate: int
    db: Path
    benchmark_job_id: str


SERVED = {
    "job_id": "2561679",
    "node": "nid007232",
    "direct_endpoint": "http://172.28.40.248:8080",
    "api_endpoint": "https://api.swissai.svc.cscs.ch",
    "model": "swiss-ai/Apertus-8B-Instruct-2509-api-overhead-sglang-brachium-20260618-183436",
    "launch_script": "launch-scripts/01_sglang_apertus8b_api_overhead.sh",
}

RUNS = [
    Run("Direct #1", "Direct", 1, DATA / "direct_run1.db", "2561761"),
    Run("Direct #2", "Direct", 2, DATA / "direct_run2.db", "2562891"),
    Run("API #1", "API", 1, DATA / "api_run1.db", "2561998"),
    Run("API #2", "API", 2, DATA / "api_run2.db", "2563417"),
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
            "select success,ttft_ms,tpot_ms,e2e_ms,input_tokens,output_tokens from requests where rate_lambda=?",
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
                "tpot_p50_ms": pct([r["tpot_ms"] for r in ok], 50),
                "tpot_p95_ms": pct([r["tpot_ms"] for r in ok], 95),
                "e2e_p95_ms": pct([r["e2e_ms"] for r in ok], 95),
                "input_tokens_avg": sum(r["input_tokens"] for r in ok) / len(ok) if ok else None,
                "output_tokens_avg": sum(r["output_tokens"] for r in ok) / len(ok) if ok else None,
                "input_tokens_s": sum(r["input_tokens"] for r in ok) / 180,
                "output_tokens_s": sum(r["output_tokens"] for r in ok) / 180,
            }
        )
    con.close()
    return {"run": run.__dict__ | {"db": run.db.name}, "experiment": exp, "rates": rates}


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
    out = []
    for item in payload.get("data", {}).get("result", []):
        try:
            value = float(item["value"][1])
        except Exception:
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def scalar(query: str, ts: float) -> float | None:
    values = prom_query(query, ts)
    return max(values) if values else None


def add_dcgm(data: dict) -> None:
    metrics = {
        "gpu_util_pct": "avg(avg_over_time(DCGM_FI_DEV_GPU_UTIL{{{sel}}}[{dur}s]))",
        "sm_active_pct": "100 * avg(avg_over_time(DCGM_FI_PROF_SM_ACTIVE{{{sel}}}[{dur}s]))",
        "tensor_active_pct": "100 * avg(avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{sel}}}[{dur}s]))",
        "mem_copy_util_pct": "avg(avg_over_time(DCGM_FI_DEV_MEM_COPY_UTIL{{{sel}}}[{dur}s]))",
        "fb_used_gib": "avg(avg_over_time(DCGM_FI_DEV_FB_USED{{{sel}}}[{dur}s])) / 1024",
        "power_total_w": "sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE{{{sel}}}[{dur}s]))",
    }
    sel = f'slurm_job_id="{SERVED["job_id"]}"'
    for item in data["runs"]:
        for rate in item["rates"]:
            start = iso_to_ts(rate["start"])
            end = iso_to_ts(rate["end"])
            dur = max(1, int(end - start))
            rate["dcgm"] = {name: scalar(template.format(sel=sel, dur=dur), end) for name, template in metrics.items()}


def grouped(data: dict) -> dict[str, dict[int, dict[float, dict]]]:
    out: dict[str, dict[int, dict[float, dict]]] = {}
    for item in data["runs"]:
        path = item["run"]["path"]
        rep = item["run"]["replicate"]
        out.setdefault(path, {})[rep] = {r["rate"]: r for r in item["rates"]}
    return out


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def table(data: dict, metric: str, dcgm: bool = False) -> str:
    g = grouped(data)
    rates = [8.0, 16.0, 24.0]
    lines = ["| Path | λ=8 | λ=16 | λ=24 |", "|---|---:|---:|---:|"]
    for path in ["Direct", "API"]:
        cells = []
        for rate in rates:
            vals = []
            for rep in g[path]:
                row = g[path][rep][rate]
                vals.append(row.get("dcgm", {}).get(metric) if dcgm else row.get(metric))
            m, s = mean_std(vals)
            cells.append(f"{fmt(m)} ± {fmt(s)}")
        lines.append(f"| {path} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_plots(data: dict) -> None:
    if plt is None:
        return
    g = grouped(data)
    rates = [8.0, 16.0, 24.0]
    colors = {"Direct": "#0072B2", "API": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for metric, ax, label in [("ttft_p95_ms", axes[0], "TTFT p95 (ms)"), ("tpot_p95_ms", axes[1], "TPOT p95 (ms)")]:
        for path in ["Direct", "API"]:
            means, errs = [], []
            for rate in rates:
                m, s = mean_std([g[path][rep][rate].get(metric) for rep in g[path]])
                means.append(m)
                errs.append(s or 0)
            ax.errorbar(rates, means, yerr=errs, marker="o", capsize=4, label=path, color=colors[path])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "latency_p95.png", dpi=180)
    plt.close(fig)


def write_report(data: dict) -> None:
    report = f"""# Apertus-8B SwissAI API Overhead

**Date:** 2026-06-18  
**Model:** `swiss-ai/Apertus-8B-Instruct-2509`  
**Engine:** SGLang, radix cache disabled, metrics enabled  
**Served job:** `{SERVED['job_id']}` on `{SERVED['node']}`  
**Replicates:** N=2 per endpoint path

## Research Question

Does routing requests through `https://api.swissai.svc.cscs.ch` add measurable latency or throughput overhead compared with hitting the same Apertus-8B SGLang server directly from the benchmarker allocation?

## Methodology

| Attribute | Value |
|---|---|
| Direct endpoint | `{SERVED['direct_endpoint']}` |
| API endpoint | `{SERVED['api_endpoint']}` |
| Served model name | `{SERVED['model']}` |
| Launch script | `{SERVED['launch_script']}` |
| Prompt scenario | `thesis-apertus-medium` |
| Arrival process | Poisson |
| Sweep | `[8, 16, 24, 32, 40, 48, 56, 64]` with early stop |
| Phases | 60 s warmup, 180 s measurement, 300 s drain |
| SLOs used | TTFT p95 ≤ 10,000 ms, TPOT p95 ≤ 200 ms, error ≤ 1% |

Only the endpoint path changed between paired runs. The same served model, prompt shape, benchmark node class, and rate schedule were used for direct and API runs.

## Results

![Latency p95](images/latency_p95.png)

### TTFT p95 (ms, mean ± std)

{table(data, 'ttft_p95_ms')}

### TPOT p95 (ms, mean ± std)

{table(data, 'tpot_p95_ms')}

### Output tokens/s (mean ± std)

{table(data, 'output_tokens_s')}

### Error rate % (mean ± std)

{table(data, 'error_pct')}

Both paths passed through λ=16 and early-stopped at λ=24 due TTFT p95 saturation. The API path shows a small low-load TTFT increase, while TPOT and the saturation point are similar in these two replicates.

## DCGM Telemetry

### GPU utilization % (mean ± std)

{table(data, 'gpu_util_pct', dcgm=True)}

### SM active % (mean ± std)

{table(data, 'sm_active_pct', dcgm=True)}

### Total GPU power W (mean ± std)

{table(data, 'power_total_w', dcgm=True)}

DCGM telemetry is queried with `slurm_job_id="{SERVED['job_id']}"` and aligned to each benchmark measurement window using timestamps from the run DBs.

## Provenance

| Path | Replicate | Benchmark job | DB |
|---|---:|---:|---|
| Direct | 1 | 2561761 | `data/direct_run1.db` |
| Direct | 2 | 2562891 | `data/direct_run2.db` |
| API | 1 | 2561998 | `data/api_run1.db` |
| API | 2 | 2563417 | `data/api_run2.db` |

## Limitations

- This is a serving-path overhead experiment, not a model quality evaluation.
- The workload uses the existing synthetic `thesis-apertus-medium` prompt shape.
- λ=24 is saturated in both paths, so API overhead should be interpreted mainly at λ=8 and λ=16.
- Client event-loop lag warnings appeared at λ=24, which reinforces treating the saturated point as overload evidence rather than a precise latency estimate.

Generated: {datetime.now(timezone.utc).isoformat()}
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    data = {"served": SERVED, "runs": [load_run(run) for run in RUNS]}
    add_dcgm(data)
    (ROOT / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    generate_plots(data)
    write_report(data)


if __name__ == "__main__":
    main()
