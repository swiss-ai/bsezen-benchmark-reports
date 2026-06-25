#!/usr/bin/env python3
"""Generate data.json and plots for the DSv3 latent vs non-latent comparison.

Reads the per-replicate SQLite run databases from ./results/ and queries
SwissAI Prometheus for DCGM and Slingshot telemetry aligned to each rate level.

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
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

MEASUREMENT_S = 180

PROM_BASE = "https://metrics.swissai.svc.cscs.ch/api/datasources/proxy/uid/PBFA97CFB590B2093/api/v1/query"

DCGM_METRICS = {
    "gpu_util_pct": "avg(avg_over_time(DCGM_FI_DEV_GPU_UTIL{{{sel}}}[{dur}s]))",
    "sm_active_pct": "100 * avg(avg_over_time(DCGM_FI_PROF_SM_ACTIVE{{{sel}}}[{dur}s]))",
    "tensor_active_pct": "100 * avg(avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{sel}}}[{dur}s]))",
    "mem_copy_util_pct": "avg(avg_over_time(DCGM_FI_DEV_MEM_COPY_UTIL{{{sel}}}[{dur}s]))",
    "fb_used_gib": "avg(avg_over_time(DCGM_FI_DEV_FB_USED{{{sel}}}[{dur}s])) / 1024",
    "power_total_w": "sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE{{{sel}}}[{dur}s]))",
    "nvlink_tx_gib_s": "sum(rate(DCGM_FI_PROF_NVLINK_TX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
    "nvlink_rx_gib_s": "sum(rate(DCGM_FI_PROF_NVLINK_RX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
    "pcie_tx_gib_s": "sum(rate(DCGM_FI_PROF_PCIE_TX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
    "pcie_rx_gib_s": "sum(rate(DCGM_FI_PROF_PCIE_RX_BYTES{{{sel}}}[{dur}s])) / 1073741824",
}

SLINGSHOT_METRICS = {
    "slingshot_tx_gib_s": "sum(rate(slingshot_tx_bytes_total{{{sel}}}[{dur}s])) / 1073741824",
    "slingshot_rx_gib_s": "sum(rate(slingshot_rx_bytes_total{{{sel}}}[{dur}s])) / 1073741824",
    "slingshot_tx_packets_s": "sum(rate(slingshot_tx_packets_total{{{sel}}}[{dur}s]))",
    "slingshot_rx_packets_s": "sum(rate(slingshot_rx_packets_total{{{sel}}}[{dur}s]))",
}

SLOS = {"ttft_p95_ms": 3000, "tpot_p95_ms": 80, "error_rate_pct": 1.0}


@dataclass(frozen=True)
class Run:
    variant: str
    replicate: int
    db: Path
    serving_slurm_job_id: str
    served_model: str


RUNS = [
    Run(
        "nonlatent",
        1,
        RESULTS / "nonlatent_rep1.db",
        "2613594",
        "swiss-ai/dsv3-comparable-nonlatent-N64-k4-tp16-brachium-20260625-010705",
    ),
    Run(
        "nonlatent",
        2,
        RESULTS / "nonlatent_rep2.db",
        "2613594",
        "swiss-ai/dsv3-comparable-nonlatent-N64-k4-tp16-brachium-20260625-010705",
    ),
    Run(
        "latent",
        1,
        RESULTS / "latent_rep1.db",
        "2614134",
        "swiss-ai/dsv3-comparable-latent-N64-k4-tp16-brachium-20260625-023009",
    ),
    Run(
        "latent",
        2,
        RESULTS / "latent_rep2.db",
        "2614134",
        "swiss-ai/dsv3-comparable-latent-N64-k4-tp16-brachium-20260625-023009",
    ),
    Run(
        "latent_n224_k14",
        1,
        RESULTS / "n224-k14/latent_n224_k14_rep1.db",
        "2616868",
        "swiss-ai/dsv3-comparable-latent-N224-k14-tp16-brachium-20260625-133537",
    ),
    Run(
        "latent_n224_k14",
        2,
        RESULTS / "n224-k14/latent_n224_k14_rep2.db",
        "2616868",
        "swiss-ai/dsv3-comparable-latent-N224-k14-tp16-brachium-20260625-133537",
    ),
]

VARIANT_ORDER = ["nonlatent", "latent", "latent_n224_k14"]
VARIANT_LABELS = {
    "nonlatent": "Non-latent N=64 k=4",
    "latent": "Latent N=64 k=4 (wider experts)",
    "latent_n224_k14": "Latent N=224 k=14 (more experts)",
}
VARIANT_COLORS = {
    "nonlatent": "#0072B2",
    "latent": "#D55E00",
    "latent_n224_k14": "#009E73",
}


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
            entry["slingshot"] = {
                name: prom_scalar(template.format(sel=sel, dur=dur), t_end)
                for name, template in SLINGSHOT_METRICS.items()
            }
        rates.append(entry)
    con.close()
    return {
        "variant": run.variant,
        "replicate": run.replicate,
        "db": str(run.db.relative_to(ROOT.parent.parent)),
        "serving_slurm_job_id": run.serving_slurm_job_id,
        "served_model": run.served_model,
        "experiment": experiment,
        "rates": rates,
    }


def grouped(data: dict) -> dict[str, dict[int, dict[float, dict]]]:
    out: dict[str, dict[int, dict[float, dict]]] = {}
    for item in data["runs"]:
        out.setdefault(item["variant"], {})[item["replicate"]] = {
            r["rate"]: r for r in item["rates"]
        }
    return out


def all_rates(data: dict) -> list[float]:
    rates: set[float] = set()
    for item in data["runs"]:
        for r in item["rates"]:
            rates.add(r["rate"])
    return sorted(rates)


def _plot_metric_panel(
    g: dict,
    rates: list[float],
    metric: str,
    label: str,
    category: str | None,
    yscale: str | None,
    slo: float | None,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in VARIANT_ORDER:
        reps = g.get(variant, {})
        xs, means, stds = [], [], []
        for rate in rates:
            if category is None:
                vals = [reps[r].get(rate, {}).get(metric) for r in reps]
            else:
                vals = [
                    reps[r].get(rate, {}).get(category, {}).get(metric) for r in reps
                ]
            m, s = mean_std(vals)
            if m is not None:
                xs.append(rate)
                means.append(m)
                stds.append(s or 0)
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            label=VARIANT_LABELS[variant],
            color=VARIANT_COLORS[variant],
        )
    if slo is not None:
        ax.axhline(slo, color="#cc0000", linestyle="--", label=f"SLO {slo:g} ms")
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_xlabel("λ (requests/s)")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def generate_plots(data: dict) -> None:
    if plt is None:
        print("matplotlib not installed; skipping plots")
        return
    g = grouped(data)
    rates = all_rates(data)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for metric, ax, slo, label in [
        ("ttft_p95_ms", axes[0], SLOS["ttft_p95_ms"], "TTFT p95 (ms)"),
        ("tpot_p95_ms", axes[1], SLOS["tpot_p95_ms"], "TPOT p95 (ms)"),
    ]:
        for variant in VARIANT_ORDER:
            reps = g.get(variant, {})
            xs, means, stds = [], [], []
            for rate in rates:
                vals = [reps[r].get(rate, {}).get(metric) for r in reps]
                m, s = mean_std(vals)
                if m is not None:
                    xs.append(rate)
                    means.append(m)
                    stds.append(s or 0)
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
            )
        ax.axhline(slo, color="#cc0000", linestyle="--", label=f"SLO {slo:g} ms")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("λ (requests/s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "latency_p95.png", dpi=180)
    plt.close(fig)

    fig = _plot_metric_panel(
        g, rates, "output_tokens_s", "Output tokens/s", None, None, None
    )
    fig.savefig(IMAGES / "throughput.png", dpi=180)
    plt.close(fig)

    dcgm_panels = [
        ("gpu_util_pct", "GPU util %", "dcgm_gpu_util"),
        ("sm_active_pct", "SM active %", "dcgm_sm_active"),
        ("tensor_active_pct", "Tensor active %", "dcgm_tensor_active"),
        ("mem_copy_util_pct", "Mem copy util %", "dcgm_mem_copy_util"),
        ("power_total_w", "Total GPU power (W, 16 GPUs)", "dcgm_power"),
        ("fb_used_gib", "Framebuffer used (GiB/GPU)", "dcgm_fb_used"),
    ]
    for metric, label, filename in dcgm_panels:
        fig = _plot_metric_panel(g, rates, metric, label, "dcgm", None, None)
        fig.savefig(IMAGES / f"{filename}.png", dpi=180)
        plt.close(fig)

    comm_panels = [
        ("nvlink_tx_gib_s", "NVLink TX (GiB/s, summed across 16 GPUs)", "comm_nvlink_tx"),
        ("nvlink_rx_gib_s", "NVLink RX (GiB/s, summed across 16 GPUs)", "comm_nvlink_rx"),
        ("pcie_tx_gib_s", "PCIe TX (GiB/s, summed across 16 GPUs)", "comm_pcie_tx"),
        ("pcie_rx_gib_s", "PCIe RX (GiB/s, summed across 16 GPUs)", "comm_pcie_rx"),
    ]
    for metric, label, filename in comm_panels:
        fig = _plot_metric_panel(g, rates, metric, label, "dcgm", None, None)
        fig.savefig(IMAGES / f"{filename}.png", dpi=180)
        plt.close(fig)

    # Slingshot: packet rates are far more stable than byte counters on this cluster.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, metric, label in [
        (axes[0], "slingshot_tx_packets_s", "TX packets/s"),
        (axes[1], "slingshot_rx_packets_s", "RX packets/s"),
    ]:
        for variant in VARIANT_ORDER:
            reps = g.get(variant, {})
            xs, means, stds = [], [], []
            for rate in rates:
                vals = [
                    reps[r].get(rate, {}).get("slingshot", {}).get(metric)
                    for r in reps
                ]
                m, s = mean_std(vals)
                if m is not None:
                    xs.append(rate)
                    means.append(m)
                    stds.append(s or 0)
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
            )
        ax.set_xlabel("λ (requests/s)")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "comm_slingshot_packets.png", dpi=180)
    plt.close(fig)

    # Keep the raw byte plots as a diagnostic, but log-scale and with a note.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, metric, label in [
        (axes[0], "slingshot_tx_gib_s", "TX GiB/s"),
        (axes[1], "slingshot_rx_gib_s", "RX GiB/s"),
    ]:
        for variant in VARIANT_ORDER:
            reps = g.get(variant, {})
            xs, means, stds = [], [], []
            for rate in rates:
                vals = [
                    reps[r].get(rate, {}).get("slingshot", {}).get(metric)
                    for r in reps
                ]
                m, s = mean_std(vals)
                if m is not None:
                    xs.append(rate)
                    means.append(m)
                    stds.append(s or 0)
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
            )
        ax.set_xlabel("λ (requests/s)")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(
        "Slingshot byte counters are noisy/under-reported on this cluster (log scale)",
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(IMAGES / "comm_slingshot_bytes.png", dpi=180)
    plt.close(fig)


def main() -> int:
    runs = [load_run(r) for r in RUNS]
    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "slos": SLOS,
        "variants": VARIANT_ORDER,
        "variant_labels": VARIANT_LABELS,
        "runs": runs,
    }
    out = ROOT / "data.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT.parent.parent)}")
    generate_plots(data)
    if plt is not None:
        print(f"wrote {IMAGES.relative_to(ROOT.parent.parent)}/latency_p95.png")
        print(f"wrote {IMAGES.relative_to(ROOT.parent.parent)}/throughput.png")
        for filename in [
            "dcgm_gpu_util",
            "dcgm_sm_active",
            "dcgm_tensor_active",
            "dcgm_mem_copy_util",
            "dcgm_power",
            "dcgm_fb_used",
        ]:
            print(f"wrote {IMAGES.relative_to(ROOT.parent.parent)}/{filename}.png")
        for filename in [
            "comm_nvlink_tx",
            "comm_nvlink_rx",
            "comm_pcie_tx",
            "comm_pcie_rx",
            "comm_slingshot_packets",
            "comm_slingshot_bytes",
        ]:
            print(f"wrote {IMAGES.relative_to(ROOT.parent.parent)}/{filename}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
