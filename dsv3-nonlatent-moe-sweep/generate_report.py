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


def load_run(run: Run) -> dict:
    con = sqlite3.connect(run.db)
    con.row_factory = sqlite3.Row
    experiment = dict(con.execute("select * from experiments limit 1").fetchone())
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
        rates.append({
            "rate": float(rate),
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
        })
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


def generate_plots(data: dict) -> None:
    if plt is None:
        print("matplotlib not installed; skipping plots")
        return
    g = grouped(data)
    rates = all_rates(data)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for metric, ax, slo, label in [
        ("ttft_p95_ms", axes[0], SLOS["ttft_p95_ms"], "TTFT p95 (ms)"),
        ("tpot_p95_ms", axes[1], SLOS["tpot_p95_ms"], "TPOT p95 (ms)"),
    ]:
        for variant in VARIANT_ORDER:
            reps = g.get(variant, {})
            means, stds = [], []
            for rate in rates:
                vals = [reps[r].get(rate, {}).get(metric) for r in reps]
                m, s = mean_std(vals)
                means.append(m)
                stds.append(s or 0)
            ax.errorbar(
                rates, means, yerr=stds,
                marker="o", linewidth=2, capsize=4,
                label=variant, color=VARIANT_COLORS[variant],
            )
        ax.axhline(slo, color="#cc0000", linestyle="--", label=f"SLO {slo:g} ms")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "latency_p95_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in VARIANT_ORDER:
        reps = g.get(variant, {})
        means, stds = [], []
        for rate in rates:
            vals = [reps[r].get(rate, {}).get("output_tokens_s") for r in reps]
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s or 0)
        ax.errorbar(
            rates, means, yerr=stds,
            marker="o", linewidth=2, capsize=4,
            label=variant, color=VARIANT_COLORS[variant],
        )
    ax.set_xlabel("λ (requests/s)")
    ax.set_ylabel("Output tokens/s")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "throughput_sweep.png", dpi=180)
    plt.close(fig)


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
    generate_plots(data)
    if plt is not None:
        print(f"wrote {IMAGES.relative_to(ROOT.parent.parent)}/latency_p95_sweep.png")
        print(f"wrote {IMAGES.relative_to(ROOT.parent.parent)}/throughput_sweep.png")


if __name__ == "__main__":
    main()
