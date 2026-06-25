#!/usr/bin/env python3
"""Regenerate data.json and plots for the DSv3 non-latent expert-granularity sweep.

Reads the per-variant SQLite run databases from ../../experiments/dsv3-nonlatent-moe-sweep/
and writes:
  - data.json  : per-rate latency + throughput summary (means and stds across replicates)
  - images/    : TTFT p95 and TPOT p95 vs λ for the three variants

Usage:
  uv run python generate_report.py
  uv run --with matplotlib python generate_report.py
"""

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
EXPERIMENTS = ROOT.parent.parent / "experiments" / "dsv3-nonlatent-moe-sweep"
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

MEASUREMENT_S = 180  # configured per benchmark_config.yaml

PROM_BASE = "https://metrics.swissai.svc.cscs.ch/api/datasources/proxy/uid/PBFA97CFB590B2093/api/v1/query"

DCGM_METRICS = {
    "gpu_util_pct":       "avg(avg_over_time(DCGM_FI_DEV_GPU_UTIL{{{sel}}}[{dur}s]))",
    "sm_active_pct":      "100 * avg(avg_over_time(DCGM_FI_PROF_SM_ACTIVE{{{sel}}}[{dur}s]))",
    "tensor_active_pct":  "100 * avg(avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{sel}}}[{dur}s]))",
    "mem_copy_util_pct":  "avg(avg_over_time(DCGM_FI_DEV_MEM_COPY_UTIL{{{sel}}}[{dur}s]))",
    "fb_used_gib":        "avg(avg_over_time(DCGM_FI_DEV_FB_USED{{{sel}}}[{dur}s])) / 1024",
    "power_total_w":      "sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE{{{sel}}}[{dur}s]))",
    # Communication — NVLink and PCIe throughput. PROF metrics are bytes counters;
    # rate() gives bytes/s, summed across all GPUs in the job. GiB/s for readability.
    "nvlink_tx_gib_s":    "sum(rate(DCGM_FI_PROF_NVLINK_TX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
    "nvlink_rx_gib_s":    "sum(rate(DCGM_FI_PROF_NVLINK_RX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
    "pcie_tx_gib_s":      "sum(rate(DCGM_FI_PROF_PCIE_TX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
    "pcie_rx_gib_s":      "sum(rate(DCGM_FI_PROF_PCIE_RX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
}

SOURCE_PROMPT_DISTRIBUTION = {
    "source_file": "/capstor/scratch/cscs/bsezen/loadtest/prompts-deepseek-thesis.json",
    "label": "medium",
    "count": 1000,
    "input_tokens": {"min": 401, "median": 589, "p95": 682, "max": 700},
    "max_tokens": {"min": 2, "median": 270, "p95": 541, "max": 869},
}

SLOS = {"ttft_p95_ms": 3000, "tpot_p95_ms": 80, "error_rate_pct": 1.0}


@dataclass(frozen=True)
class Run:
    variant: str        # "N32-k2" etc — keyed for grouping
    replicate: int
    db: Path
    serving_slurm_job_id: str
    served_model: str
    launch_script: str


RUNS = [
    Run(
        "N32-k2", 1,
        EXPERIMENTS / "n32-k2" / "sglang_run1.db",
        "2560219",
        "swiss-ai/dsv3-nonlatent-N32-k2-tp16-brachium-20260618-145933",
        "sml/model-launch/local/dsv3-nonlatent-moe-sweep/01_sglang_n32_k2.sh",
    ),
    Run(
        "N32-k2", 2,
        EXPERIMENTS / "n32-k2" / "sglang_run2.db",
        "2560219",
        "swiss-ai/dsv3-nonlatent-N32-k2-tp16-brachium-20260618-145933",
        "sml/model-launch/local/dsv3-nonlatent-moe-sweep/01_sglang_n32_k2.sh",
    ),
    Run(
        "N64-k4", 1,
        EXPERIMENTS / "n64-k4" / "sglang_run1.db",
        "2561154",
        "swiss-ai/dsv3-nonlatent-N64-k4-tp16-brachium-20260618-164631",
        "sml/model-launch/local/dsv3-nonlatent-moe-sweep/02_sglang_n64_k4.sh",
    ),
    Run(
        "N64-k4", 2,
        EXPERIMENTS / "n64-k4" / "sglang_run2.db",
        "2561154",
        "swiss-ai/dsv3-nonlatent-N64-k4-tp16-brachium-20260618-164631",
        "sml/model-launch/local/dsv3-nonlatent-moe-sweep/02_sglang_n64_k4.sh",
    ),
    Run(
        "N128-k8", 1,
        EXPERIMENTS / "n128-k8" / "sglang_run1.db",
        "2561481",
        "swiss-ai/dsv3-nonlatent-N128-k8-tp16-brachium-20260618-174321",
        "sml/model-launch/local/dsv3-nonlatent-moe-sweep/03_sglang_n128_k8.sh",
    ),
    Run(
        "N128-k8", 2,
        EXPERIMENTS / "n128-k8" / "sglang_run2.db",
        "2561481",
        "swiss-ai/dsv3-nonlatent-N128-k8-tp16-brachium-20260618-174321",
        "sml/model-launch/local/dsv3-nonlatent-moe-sweep/03_sglang_n128_k8.sh",
    ),
]

VARIANT_ORDER = ["N32-k2", "N64-k4", "N128-k8"]
VARIANT_COLORS = {"N32-k2": "#0072B2", "N64-k4": "#009E73", "N128-k8": "#D55E00"}


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


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def iso_to_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def prom_scalar(query: str, ts: float) -> float | None:
    params = urllib.parse.urlencode({"query": query, "time": f"{ts:.0f}"})
    try:
        with urllib.request.urlopen(f"{PROM_BASE}?{params}", timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if payload.get("status") != "success":
        return None
    results = payload.get("data", {}).get("result", [])
    if not results:
        return None
    try:
        return float(results[0]["value"][1])
    except Exception:
        return None


def load_run(run: Run) -> dict:
    con = sqlite3.connect(run.db)
    con.row_factory = sqlite3.Row
    experiment = dict(con.execute("select * from experiments limit 1").fetchone())
    server_windows = {
        float(rate): (start, end)
        for rate, start, end in con.execute(
            "select rate_lambda, min(ts), max(ts) from server_stats "
            "group by rate_lambda order by rate_lambda"
        )
    }
    rates = []
    for (rate,) in con.execute(
        "select distinct rate_lambda from requests order by rate_lambda"
    ):
        rows = con.execute(
            "select success, ttft_ms, tpot_ms, e2e_ms, input_tokens, output_tokens "
            "from requests where rate_lambda=?",
            (rate,),
        ).fetchall()
        ok = [r for r in rows if r["success"]]
        errors = len(rows) - len(ok)
        start, end = server_windows.get(float(rate), (None, None))
        entry = {
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
            "input_tokens_s": sum(r["input_tokens"] for r in ok) / MEASUREMENT_S if ok else None,
            "output_tokens_s": sum(r["output_tokens"] for r in ok) / MEASUREMENT_S if ok else None,
        }
        if start and end:
            t_start, t_end = iso_to_ts(start), iso_to_ts(end)
            dur = max(1, int(t_end - t_start))
            sel = f'slurm_job_id="{run.serving_slurm_job_id}"'
            entry["dcgm"] = {
                name: prom_scalar(template.format(sel=sel, dur=dur), t_end)
                for name, template in DCGM_METRICS.items()
            }
        rates.append(entry)
    con.close()
    return {
        "variant": run.variant,
        "replicate": run.replicate,
        "db": str(run.db.relative_to(ROOT.parent.parent)),
        "serving_slurm_job_id": run.serving_slurm_job_id,
        "served_model": run.served_model,
        "launch_script": run.launch_script,
        "experiment": experiment,
        "rates": rates,
    }


def grouped(data: dict) -> dict[str, dict[int, dict[float, dict]]]:
    out: dict[str, dict[int, dict[float, dict]]] = {}
    for item in data["runs"]:
        out.setdefault(item["variant"], {})[item["replicate"]] = {r["rate"]: r for r in item["rates"]}
    return out


def all_rates(data: dict) -> list[float]:
    rates: set[float] = set()
    for item in data["runs"]:
        for r in item["rates"]:
            rates.add(r["rate"])
    return sorted(rates)


def _plot_variant_series(ax, g: dict, rates: list[float], metric_path: list[str]) -> None:
    """Plot mean±std series for each variant on the given axes.

    metric_path is the sequence of dict keys to reach the value, e.g.
    ["dcgm", "gpu_util_pct"] or ["output_tokens_s"].
    """
    for variant in VARIANT_ORDER:
        reps = g.get(variant, {})
        means, stds = [], []
        for rate in rates:
            vals: list[float | None] = []
            for rep in reps.values():
                val: object = rep.get(rate, {})
                for key in metric_path:
                    if not isinstance(val, dict):
                        val = None
                        break
                    val = val.get(key)
                vals.append(val if isinstance(val, (int, float)) else None)
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s or 0)
        ax.errorbar(
            rates, means, yerr=stds,
            marker="o", linewidth=2, capsize=4,
            label=variant, color=VARIANT_COLORS[variant],
        )


def _single_metric_plot(
    g: dict,
    rates: list[float],
    metric_path: list[str],
    ylabel: str,
    filename: str,
    xlabel: str = "λ (requests/s)",
    logy: bool = False,
) -> Path | None:
    """Create a single-panel plot and save it to images/filename."""
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_variant_series(ax, g, rates, metric_path)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = IMAGES / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_plots(data: dict) -> list[Path]:
    if plt is None:
        print("matplotlib not installed; skipping plots")
        return []
    g = grouped(data)
    rates = all_rates(data)
    written: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for metric, ax, slo, label in [
        ("ttft_p95_ms", axes[0], SLOS["ttft_p95_ms"], "TTFT p95 (ms)"),
        ("tpot_p95_ms", axes[1], SLOS["tpot_p95_ms"], "TPOT p95 (ms)"),
    ]:
        _plot_variant_series(ax, g, rates, [metric])
        ax.axhline(slo, color="#cc0000", linestyle="--", label=f"SLO {slo:g} ms")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    latency_path = IMAGES / "latency_p95_sweep.png"
    fig.savefig(latency_path, dpi=180)
    plt.close(fig)
    written.append(latency_path)

    tp_path = _single_metric_plot(
        g, rates, ["output_tokens_s"],
        "Output tokens/s", "throughput_sweep.png",
    )
    if tp_path:
        written.append(tp_path)

    dcgm_panels = [
        ("gpu_util_pct", "GPU util %", "dcgm_gpu_util.png"),
        ("sm_active_pct", "SM active %", "dcgm_sm_active.png"),
        ("tensor_active_pct", "Tensor active %", "dcgm_tensor_active.png"),
        ("mem_copy_util_pct", "Memory copy util %", "dcgm_mem_copy_util.png"),
        ("power_total_w", "Total GPU power (W, 16 GPUs)", "dcgm_power_total.png"),
    ]
    for metric, ylabel, filename in dcgm_panels:
        path = _single_metric_plot(
            g, rates, ["dcgm", metric],
            ylabel, filename,
        )
        if path:
            written.append(path)

    comm_panels = [
        ("nvlink_tx_gib_s", "NVLink TX (GiB/s, summed across 16 GPUs)", "comm_nvlink_tx.png"),
        ("nvlink_rx_gib_s", "NVLink RX (GiB/s, summed across 16 GPUs)", "comm_nvlink_rx.png"),
        ("pcie_tx_gib_s", "PCIe TX (GiB/s, summed across 16 GPUs)", "comm_pcie_tx.png"),
        ("pcie_rx_gib_s", "PCIe RX (GiB/s, summed across 16 GPUs)", "comm_pcie_rx.png"),
    ]
    for metric, ylabel, filename in comm_panels:
        path = _single_metric_plot(
            g, rates, ["dcgm", metric],
            ylabel, filename,
        )
        if path:
            written.append(path)

    return written


def main() -> None:
    runs = [load_run(r) for r in RUNS]
    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "slos": SLOS,
        "source_prompt_distribution": SOURCE_PROMPT_DISTRIBUTION,
        "variants": VARIANT_ORDER,
        "runs": runs,
    }
    out = ROOT / "data.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT.parent.parent)}")
    plot_paths = generate_plots(data)
    for path in plot_paths:
        print(f"wrote {path.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
