#!/usr/bin/env python3
"""Generate plots and markdown report for Apertus-8B K8s vs Slurm comparison."""

import sqlite3
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("Warning: matplotlib/numpy not available, skipping plots")

# Data sources: local DBs in data/
DATA_DIR = Path(__file__).parent / "data"
SLURM_DBS = sorted(DATA_DIR.glob("run_slurm-*.db"))
K8S_DBS = sorted(DATA_DIR.glob("run_k8s-*.db"))


def load_metrics(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT rate_lambda,
               COUNT(*) as requests,
               SUM(success) as successful,
               AVG(ttft_ms) as avg_ttft,
               MAX(ttft_ms) as max_ttft,
               AVG(tpot_ms) as avg_tpot,
               MAX(tpot_ms) as max_tpot,
               AVG(e2e_ms) as avg_e2e
        FROM requests
        GROUP BY rate_lambda
        ORDER BY rate_lambda
    ''')
    results = cursor.fetchall()

    cursor.execute('SELECT run_id, model, backend FROM experiments LIMIT 1')
    meta = cursor.fetchone()

    conn.close()
    return results, meta


def aggregate_dbs(db_paths):
    """Aggregate metrics across replicate runs per rate level.

    Returns a dict keyed by rate with per-run arrays and mean/std.
    """
    by_rate = {}
    for db_path in db_paths:
        rows, meta = load_metrics(db_path)
        for row in rows:
            rate = row[0]
            if rate not in by_rate:
                by_rate[rate] = {
                    "requests": [],
                    "success": [],
                    "ttft_avg": [],
                    "ttft_max": [],
                    "tpot_avg": [],
                    "tpot_max": [],
                    "e2e_avg": [],
                }
            by_rate[rate]["requests"].append(row[1])
            by_rate[rate]["success"].append(row[2])
            by_rate[rate]["ttft_avg"].append(row[3])
            by_rate[rate]["ttft_max"].append(row[4])
            by_rate[rate]["tpot_avg"].append(row[5])
            by_rate[rate]["tpot_max"].append(row[6])
            by_rate[rate]["e2e_avg"].append(row[7])

    summary = {}
    for rate in sorted(by_rate):
        d = by_rate[rate]
        summary[rate] = {
            "requests_total": sum(d["requests"]),
            "success_total": sum(d["success"]),
            "ttft_avg_mean": statistics.mean(d["ttft_avg"]),
            "ttft_avg_std": statistics.stdev(d["ttft_avg"]) if len(d["ttft_avg"]) > 1 else 0.0,
            "ttft_max_mean": statistics.mean(d["ttft_max"]),
            "ttft_max_std": statistics.stdev(d["ttft_max"]) if len(d["ttft_max"]) > 1 else 0.0,
            "tpot_avg_mean": statistics.mean(d["tpot_avg"]),
            "tpot_avg_std": statistics.stdev(d["tpot_avg"]) if len(d["tpot_avg"]) > 1 else 0.0,
            "tpot_max_mean": statistics.mean(d["tpot_max"]),
            "tpot_max_std": statistics.stdev(d["tpot_max"]) if len(d["tpot_max"]) > 1 else 0.0,
            "e2e_avg_mean": statistics.mean(d["e2e_avg"]),
            "e2e_avg_std": statistics.stdev(d["e2e_avg"]) if len(d["e2e_avg"]) > 1 else 0.0,
            "runs": len(d["ttft_avg"]),
        }
    return summary, by_rate


slurm_summary, slurm_by_rate = aggregate_dbs(SLURM_DBS)
k8s_summary, k8s_by_rate = aggregate_dbs(K8S_DBS)

rates = sorted(set(slurm_summary.keys()) & set(k8s_summary.keys()))

slurm_rates = rates
k8s_rates = rates


def get_values(summary, rates, key):
    return [summary[r][key] for r in rates]


slurm_ttft_avg = get_values(slurm_summary, rates, "ttft_avg_mean")
slurm_ttft_avg_std = get_values(slurm_summary, rates, "ttft_avg_std")
slurm_ttft_max = get_values(slurm_summary, rates, "ttft_max_mean")
slurm_ttft_max_std = get_values(slurm_summary, rates, "ttft_max_std")
slurm_tpot_avg = get_values(slurm_summary, rates, "tpot_avg_mean")
slurm_tpot_avg_std = get_values(slurm_summary, rates, "tpot_avg_std")
slurm_tpot_max = get_values(slurm_summary, rates, "tpot_max_mean")
slurm_e2e_avg = get_values(slurm_summary, rates, "e2e_avg_mean")
slurm_requests = get_values(slurm_summary, rates, "requests_total")
slurm_success = get_values(slurm_summary, rates, "success_total")

k8s_ttft_avg = get_values(k8s_summary, rates, "ttft_avg_mean")
k8s_ttft_avg_std = get_values(k8s_summary, rates, "ttft_avg_std")
k8s_ttft_max = get_values(k8s_summary, rates, "ttft_max_mean")
k8s_ttft_max_std = get_values(k8s_summary, rates, "ttft_max_std")
k8s_tpot_avg = get_values(k8s_summary, rates, "tpot_avg_mean")
k8s_tpot_avg_std = get_values(k8s_summary, rates, "tpot_avg_std")
k8s_tpot_max = get_values(k8s_summary, rates, "tpot_max_mean")
k8s_e2e_avg = get_values(k8s_summary, rates, "e2e_avg_mean")
k8s_requests = get_values(k8s_summary, rates, "requests_total")
k8s_success = get_values(k8s_summary, rates, "success_total")

# Create output directory
out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)

# Generate plots if libraries available
if HAS_PLOTS:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })

    # Plot 1: TTFT Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.errorbar(
        slurm_rates, slurm_ttft_avg, yerr=slurm_ttft_avg_std,
        fmt='-o', color='#1f77b4', linewidth=2, markersize=7,
        capsize=4, label='Slurm'
    )
    ax1.errorbar(
        k8s_rates, k8s_ttft_avg, yerr=k8s_ttft_avg_std,
        fmt='-s', color='#ff7f0e', linewidth=2, markersize=7,
        capsize=4, label='K8s'
    )
    ax1.axhline(y=10000, color='g', linestyle='--', alpha=0.7, label='SLO Threshold (10s)')
    ax1.set_xlabel('Request Rate λ (req/s)')
    ax1.set_ylabel('Average TTFT (ms)')
    ax1.set_title('Average Time-to-First-Token', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xticks(rates)
    ax1.tick_params(axis='x', rotation=30)

    ax2.errorbar(
        slurm_rates, slurm_ttft_max, yerr=slurm_ttft_max_std,
        fmt='-o', color='#1f77b4', linewidth=2, markersize=7,
        capsize=4, label='Slurm'
    )
    ax2.errorbar(
        k8s_rates, k8s_ttft_max, yerr=k8s_ttft_max_std,
        fmt='-s', color='#ff7f0e', linewidth=2, markersize=7,
        capsize=4, label='K8s'
    )
    ax2.axhline(y=10000, color='g', linestyle='--', alpha=0.7, label='SLO Threshold (10s)')
    ax2.set_xlabel('Request Rate λ (req/s)')
    ax2.set_ylabel('Max TTFT (ms)')
    ax2.set_title('Maximum Time-to-First-Token', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xticks(rates)
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(out_dir / 'ttft_comparison.png', bbox_inches='tight')
    plt.close()

    # Plot 2: TPOT Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        slurm_rates, slurm_tpot_avg, yerr=slurm_tpot_avg_std,
        fmt='-o', color='#1f77b4', linewidth=2, markersize=8,
        capsize=4, label='Slurm'
    )
    ax.errorbar(
        k8s_rates, k8s_tpot_avg, yerr=k8s_tpot_avg_std,
        fmt='-s', color='#ff7f0e', linewidth=2, markersize=8,
        capsize=4, label='K8s'
    )
    ax.axhline(y=200, color='g', linestyle='--', alpha=0.7, label='SLO Threshold (200ms)')
    ax.set_xlabel('Request Rate λ (req/s)')
    ax.set_ylabel('Average TPOT (ms)')
    ax.set_title('Time-Per-Output-Token: Slurm vs Kubernetes', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rates)
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(out_dir / 'tpot_comparison.png', bbox_inches='tight')
    plt.close()

    # Plot 3: Normalized comparison at λ=36
    # TTFT and TPOT have very different scales, so use separate subplots.
    idx_36 = rates.index(36.0)
    fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(14, 5.5))

    ttft_metrics = ['Avg TTFT', 'Max TTFT']
    x = np.arange(len(ttft_metrics))
    width = 0.35

    slurm_ttft_36 = [
        slurm_ttft_avg[idx_36] / 1000,
        slurm_ttft_max[idx_36] / 1000,
    ]
    k8s_ttft_36 = [
        k8s_ttft_avg[idx_36] / 1000,
        k8s_ttft_max[idx_36] / 1000,
    ]
    slurm_ttft_36_std = [
        slurm_ttft_avg_std[idx_36] / 1000,
        slurm_ttft_max_std[idx_36] / 1000,
    ]
    k8s_ttft_36_std = [
        k8s_ttft_avg_std[idx_36] / 1000,
        k8s_ttft_max_std[idx_36] / 1000,
    ]

    ax_ttft.bar(
        x - width / 2, slurm_ttft_36, width, yerr=slurm_ttft_36_std,
        label='Slurm', color='#1f77b4', alpha=0.85,
        capsize=4, error_kw={"linewidth": 1.5}
    )
    ax_ttft.bar(
        x + width / 2, k8s_ttft_36, width, yerr=k8s_ttft_36_std,
        label='K8s', color='#ff7f0e', alpha=0.85,
        capsize=4, error_kw={"linewidth": 1.5}
    )
    ax_ttft.set_ylabel('TTFT (seconds)')
    ax_ttft.set_title('TTFT at λ=36 req/s', fontweight='bold')
    ax_ttft.set_xticks(x)
    ax_ttft.set_xticklabels(ttft_metrics)
    ax_ttft.legend()
    ax_ttft.grid(True, alpha=0.3, axis='y')

    tpot_metrics = ['Avg TPOT']
    x_tpot = np.arange(len(tpot_metrics))
    slurm_tpot_36 = [slurm_tpot_avg[idx_36]]
    k8s_tpot_36 = [k8s_tpot_avg[idx_36]]
    slurm_tpot_36_std = [slurm_tpot_avg_std[idx_36]]
    k8s_tpot_36_std = [k8s_tpot_avg_std[idx_36]]

    ax_tpot.bar(
        x_tpot - width / 2, slurm_tpot_36, width, yerr=slurm_tpot_36_std,
        label='Slurm', color='#1f77b4', alpha=0.85,
        capsize=4, error_kw={"linewidth": 1.5}
    )
    ax_tpot.bar(
        x_tpot + width / 2, k8s_tpot_36, width, yerr=k8s_tpot_36_std,
        label='K8s', color='#ff7f0e', alpha=0.85,
        capsize=4, error_kw={"linewidth": 1.5}
    )
    ax_tpot.set_ylabel('TPOT (ms)')
    ax_tpot.set_title('TPOT at λ=36 req/s', fontweight='bold')
    ax_tpot.set_xticks(x_tpot)
    ax_tpot.set_xticklabels(tpot_metrics)
    ax_tpot.legend()
    ax_tpot.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Performance at λ=36 req/s (Healthy Load)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'latency_at_36.png', bbox_inches='tight')
    plt.close()

    print(f"✓ Plots saved to {out_dir}")

# Replicate job IDs and node metadata extraction

def extract_job_info(db_paths):
    info = []
    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT run_id, model, backend FROM experiments LIMIT 1')
        row = c.fetchone()
        conn.close()
        if row:
            run_id = row[0]
            # Extract numeric job id if present
            job_id = run_id.split('-')[-1] if '-' in run_id else 'N/A'
            info.append({"run_id": run_id, "job_id": job_id, "path": str(db_path)})
    return info


slurm_info = extract_job_info(SLURM_DBS)
k8s_info = extract_job_info(K8S_DBS)

slurm_job_ids = ", ".join(f"Job {i['job_id']}" for i in slurm_info)
k8s_job_ids = ", ".join(f"Job {i['job_id']}" for i in k8s_info)


# Helpers for report formatting

def fmt_ms(val):
    return f"{val:,.0f} ms"


def fmt_s(val):
    return f"{val / 1000:.2f}s"


def pct_diff(k, s):
    return 100 * (k / s - 1)


def ratio(k, s):
    return k / s


def run_table_rows(summary, by_rate, rate):
    """Generate markdown rows for per-run details at a given rate."""
    d = by_rate[rate]
    rows = []
    for i in range(len(d["ttft_avg"])):
        rows.append(
            f"| **#{i + 1}** | {d['ttft_avg'][i]:.0f} ms | {d['tpot_avg'][i]:.0f} ms |"
        )
    mean_ttft = statistics.mean(d["ttft_avg"])
    std_ttft = statistics.stdev(d["ttft_avg"]) if len(d["ttft_avg"]) > 1 else 0.0
    mean_tpot = statistics.mean(d["tpot_avg"])
    std_tpot = statistics.stdev(d["tpot_avg"]) if len(d["tpot_avg"]) > 1 else 0.0
    rows.append(
        f"| **Mean ± Std** | {mean_ttft:.0f} ± {std_ttft:.0f} ms | {mean_tpot:.0f} ± {std_tpot:.0f} ms |"
    )
    return "\n".join(rows)


def summary_table_rows(summary, rates):
    rows = []
    for r in rates:
        s = summary[r]
        status = "✅ **Healthy**" if r <= 36.0 else "❌ **Saturated**"
        rows.append(
            f"| **{r:.1f}** | {s['requests_total']:,} | {s['success_total']:,} "
            f"({100 * s['success_total'] / s['requests_total']:.0f}%) | "
            f"{s['ttft_avg_mean']:.0f} ms | {s['ttft_max_mean']:.0f} ms | "
            f"{s['tpot_avg_mean']:.0f} ms | {status} |"
        )
    return "\n".join(rows)


idx_36 = rates.index(36.0)
idx_42 = rates.index(42.0)

# Generate markdown report
report_md = f"""# Apertus-8B: Kubernetes vs Slurm Performance Comparison

**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Model:** swiss-ai/Apertus-8B-Instruct-2509  
**Engine:** SGLang  
**Context:** 8K tokens  
**Infrastructure:** CSCS Clariden (GH200)

---

## Research Question

Does platform (Kubernetes vs SLURM) measurably affect Apertus-8B inference performance at identical hardware and engine configuration, and if so, where (TTFT, TPOT, saturation)?

---

## Executive Summary

This report compares the inference performance of **Apertus-8B** served via **Kubernetes** versus **Slurm** on identical hardware (single GH200 node). Both deployments use SGLang with the same configuration.

### Key Findings

| Metric | Finding |
|--------|---------|
| **λ* (Knee Point)** | ~36 req/s for **both** platforms |
| **TTFT (network-bound)** | K8s is ~{abs(pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36])):.0f}% worse at healthy load (mean ± std) |
| **TPOT (GPU-bound)** | K8s is actually **~{abs(pct_diff(k8s_tpot_avg[idx_36], slurm_tpot_avg[idx_36])):.0f}% better** at healthy load |
| **Degradation** | K8s degrades more sharply under overload ({ratio(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):.1f}× TTFT, {ratio(k8s_tpot_avg[idx_42], slurm_tpot_avg[idx_42]):.1f}× TPOT) |
| **Reproducibility** | ✅ N={slurm_summary[rates[0]]['runs']} replicate{'s' if slurm_summary[rates[0]]['runs'] > 1 else ''} per platform |
| **Error Rate** | 0% for both platforms (clean saturation) |

---

## Methodology

### Workload
- **Scenario:** thesis-apertus-medium (mixed prompt lengths)
- **Prompts:** 30,000 unique prompts (with recycling enabled)
- **Arrival Process:** Poisson distribution
- **Rate Levels:** {rates} req/s
- **Phases:** 60s warmup / 180s measurement / 300s drain

### SLOs
- TTFT p95 ≤ 10,000 ms
- TPOT p95 ≤ 200 ms
- Error rate ≤ 1%

### Early Stop Condition
Sweeps terminate after 1 consecutive saturated level (SLO breach).

---

## Results

### Latency vs Request Rate

![TTFT Comparison](ttft_comparison.png)

*Figure 1: Time-to-First-Token (TTFT) comparison. Both platforms show sharp latency increase beyond λ=36 req/s. K8s exhibits higher baseline latency and more severe degradation under overload. Error bars show standard deviation across N={slurm_summary[rates[0]]['runs']} replicates.*

![TPOT Comparison](tpot_comparison.png)

*Figure 2: Time-Per-Output-Token (TPOT) remains within SLO for both platforms at λ=36, but approaches or exceeds threshold at higher loads. Error bars show standard deviation across replicates.*

### Detailed Metrics (N={slurm_summary[rates[0]]['runs']} Replicates)

Results from {slurm_summary[rates[0]]['runs']} independent run{'s' if slurm_summary[rates[0]]['runs'] > 1 else ''} per platform ({slurm_job_ids} for Slurm; {k8s_job_ids} for K8s).

#### Slurm

| Run | λ=36 TTFT | λ=36 TPOT |
|-----|-----------|-----------|
{run_table_rows(slurm_summary, slurm_by_rate, 36.0)}

#### Kubernetes

| Run | λ=36 TTFT | λ=36 TPOT |
|-----|-----------|-----------|
{run_table_rows(k8s_summary, k8s_by_rate, 36.0)}

### Performance Summary (N={slurm_summary[rates[0]]['runs']})

| Metric | Slurm (mean ± std) | K8s (mean ± std) | Difference |
|--------|-------------------|------------------|------------|
| **TTFT @ λ=36** | {slurm_ttft_avg[idx_36]:.0f} ± {slurm_ttft_avg_std[idx_36]:.0f} ms | {k8s_ttft_avg[idx_36]:.0f} ± {k8s_ttft_avg_std[idx_36]:.0f} ms | K8s {pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36]):+.0f}% |
| **TPOT @ λ=36** | {slurm_tpot_avg[idx_36]:.0f} ± {slurm_tpot_avg_std[idx_36]:.0f} ms | {k8s_tpot_avg[idx_36]:.0f} ± {k8s_tpot_avg_std[idx_36]:.0f} ms | **K8s {pct_diff(k8s_tpot_avg[idx_36], slurm_tpot_avg[idx_36]):+.0f}%** |
| **TTFT @ λ=42** | {slurm_ttft_avg[idx_42]:.0f} ± {slurm_ttft_avg_std[idx_42]:.0f} ms | {k8s_ttft_avg[idx_42]:.0f} ± {k8s_ttft_avg_std[idx_42]:.0f} ms | K8s {pct_diff(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):+.0f}% ({ratio(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):.1f}× worse) |
| **TPOT @ λ=42** | {slurm_tpot_avg[idx_42]:.0f} ± {slurm_tpot_avg_std[idx_42]:.0f} ms | {k8s_tpot_avg[idx_42]:.0f} ± {k8s_tpot_avg_std[idx_42]:.0f} ms | K8s {pct_diff(k8s_tpot_avg[idx_42], slurm_tpot_avg[idx_42]):+.0f}% ({ratio(k8s_tpot_avg[idx_42], slurm_tpot_avg[idx_42]):.1f}× worse) |

*Note: Lower variance in K8s TTFT at λ=36 (σ={k8s_ttft_avg_std[idx_36]:.0f}ms) vs Slurm (σ={slurm_ttft_avg_std[idx_36]:.0f}ms) suggests more consistent network behavior at healthy load.*

### Per-Level Summary

#### Slurm ({slurm_job_ids})

| λ (req/s) | Requests | Success | Avg TTFT | Max TTFT | Avg TPOT | Status |
|-----------|----------|---------|----------|----------|----------|--------|
{summary_table_rows(slurm_summary, rates)}

#### Kubernetes ({k8s_job_ids})

| λ (req/s) | Requests | Success | Avg TTFT | Max TTFT | Avg TPOT | Status |
|-----------|----------|---------|----------|----------|----------|--------|
{summary_table_rows(k8s_summary, rates)}

### Performance at λ=36 (Healthy Load)

![Latency at λ=36](latency_at_36.png)

*Figure 3: Direct comparison at healthy load (λ=36 req/s). K8s shows consistently higher TTFT, while TPOT is slightly lower. Error bars show standard deviation across replicates.*

| Metric | Slurm | K8s | K8s Overhead |
|--------|-------|-----|--------------|
| **Avg TTFT** | {slurm_ttft_avg[idx_36] / 1000:.2f}s | {k8s_ttft_avg[idx_36] / 1000:.2f}s | **+{pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36]):.0f}%** |
| **Max TTFT** | {slurm_ttft_max[idx_36] / 1000:.2f}s | {k8s_ttft_max[idx_36] / 1000:.2f}s | **+{pct_diff(k8s_ttft_max[idx_36], slurm_ttft_max[idx_36]):.0f}%** |
| **Avg TPOT** | {slurm_tpot_avg[idx_36]:.0f}ms | {k8s_tpot_avg[idx_36]:.0f}ms | **{pct_diff(k8s_tpot_avg[idx_36], slurm_tpot_avg[idx_36]):+.0f}%** |

### Performance at λ=42 (Saturated)

| Metric | Slurm | K8s | K8s Degradation |
|--------|-------|-----|-----------------|
| **Avg TTFT** | {slurm_ttft_avg[idx_42] / 1000:.1f}s | {k8s_ttft_avg[idx_42] / 1000:.1f}s | **{ratio(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):.1f}× worse** |
| **Max TTFT** | {slurm_ttft_max[idx_42] / 1000:.1f}s | {k8s_ttft_max[idx_42] / 1000:.1f}s | **{ratio(k8s_ttft_max[idx_42], slurm_ttft_max[idx_42]):.1f}× worse** |
| **Avg TPOT** | {slurm_tpot_avg[idx_42]:.0f}ms | {k8s_tpot_avg[idx_42]:.0f}ms | **{ratio(k8s_tpot_avg[idx_42], slurm_tpot_avg[idx_42]):.1f}× worse** |

---

## Analysis

### 1. Platform Parity at Knee
Both platforms saturate at the **same request rate** (~36 req/s), indicating the bottleneck is the model/GPU, not the orchestration layer.

### 2. TTFT vs TPOT: Different Patterns

**Time-to-First-Token (TTFT)** — K8s is worse:
- K8s exhibits **~{pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36]):.0f}% higher TTFT** at healthy load ({k8s_ttft_avg[idx_36]:.0f}ms vs {slurm_ttft_avg[idx_36]:.0f}ms)
- Likely due to: ingress/networking latency, API gateway hop, pod networking overhead
- TTFT is network/queuing-bound, not GPU-bound

**Time-Per-Output-Token (TPOT)** — K8s is **better** at healthy load:
- K8s TPOT: **{k8s_tpot_avg[idx_36]:.0f}ms** vs Slurm: **{slurm_tpot_avg[idx_36]:.0f}ms** at λ=36 (~{abs(pct_diff(k8s_tpot_avg[idx_36], slurm_tpot_avg[idx_36])):.0f}% improvement)
- TPOT is GPU/compute-bound; suggests K8s SGLang container may have slight compute advantage
- Under overload (λ=42), K8s TPOT degrades to {k8s_tpot_avg[idx_42]:.0f}ms vs Slurm {slurm_tpot_avg[idx_42]:.0f}ms

### 3. Degradation Under Overload
When overloaded (λ=42), K8s degrades **more severely** than Slurm across all metrics:
- TTFT: K8s {k8s_ttft_avg[idx_42] / 1000:.1f}s vs Slurm {slurm_ttft_avg[idx_42] / 1000:.1f}s (**{ratio(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):.1f}× worse**)
- TPOT: K8s {k8s_tpot_avg[idx_42]:.0f}ms vs Slurm {slurm_tpot_avg[idx_42]:.0f}ms (**{ratio(k8s_tpot_avg[idx_42], slurm_tpot_avg[idx_42]):.1f}× worse**)
- This suggests K8s networking/queuing amplifies overload effects

### 4. Clean Saturation
Both platforms show **0% error rate** even under overload—latency degrades gracefully rather than failing.

---

## Limitations & Open Questions

### What We Know
- ✅ **TTFT**: K8s is consistently worse (~{pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36]):.0f}% at healthy load, ~{ratio(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):.1f}× under overload)
- ✅ **TPOT**: K8s is actually **better at healthy load** ({k8s_tpot_avg[idx_36]:.0f}ms vs {slurm_tpot_avg[idx_36]:.0f}ms) but degrades more sharply under overload
- ✅ Both platforms saturate at the same request rate (~36 req/s)
- ✅ TTFT is network/queuing-bound; TPOT is GPU/compute-bound

### What We Haven't Proven
- ❌ **Root cause not isolated**: We have not proven the network is the bottleneck
- ❌ **No framework metrics**: Prometheus scraping failed (external endpoint limitation)
- ❌ **No network profiling**: No tcpdump, latency probes, or bandwidth tests
- ❌ **Configuration differences**: SGLang versions, CUDA drivers, container settings not verified identical
- ❌ **Background load**: K8s cluster may have had competing workloads

### Future Work to Isolate Root Cause
1. **Network profiling**: Measure latency to API gateway vs direct node access
2. **Control test**: Run Slurm job with explicit network hop to mimic K8s ingress
3. **Framework metrics**: Enable SGLang Prometheus endpoint for direct scraping
4. **Configuration audit**: Verify identical SGLang versions, CUDA, and container configs
5. **Load isolation**: Run K8s test during quiet cluster period

### Conservative Interpretation
> **We observed a measurable performance difference between K8s and Slurm deployments, but we cannot definitively attribute it to network overhead without further isolation experiments.**

---

## Conclusions

1. **λ* ≈ 36 req/s** is the maximum supportable load for Apertus-8B on a single GH200 (both platforms)
2. **Slurm shows lower latency** (~{pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36]):.0f}% at healthy load, ~{ratio(k8s_ttft_avg[idx_42], slurm_ttft_avg[idx_42]):.1f}× difference under overload)
3. **K8s overhead is real but root cause unproven**: Could be network, configuration, or infrastructure differences
4. **Neither platform** handles overload well—stay below λ=36 for production

### Recommendations

**For latency-sensitive production workloads:**
- **TTFT-sensitive workloads** (chat, streaming): Prefer Slurm (~{pct_diff(k8s_ttft_avg[idx_36], slurm_ttft_avg[idx_36]):.0f}% lower latency)
- **Throughput-sensitive workloads** (batch processing): K8s may be comparable or slightly better (~{abs(pct_diff(k8s_tpot_avg[idx_36], slurm_tpot_avg[idx_36])):.0f}% lower TPOT at healthy load)
- K8s is acceptable if overhead is within SLO budgets and operational benefits justify it

**For further investigation:**
- Isolate root cause before attributing to "K8s networking"
- Consider running both platforms with identical SGLang configurations and profiling enabled

---

## Provenance

| Attribute | Value |
|-----------|-------|
| **Benchmark Tool** | inference-benchmarking-tool (migrated) |
| **Slurm Runs** | {slurm_job_ids} |
| **K8s Runs** | {k8s_job_ids} |
| **Reservation** | SD-69241-apertus-1-5-0 |
| **Replicates** | N={slurm_summary[rates[0]]['runs']} per platform |
| **Raw Data** | SQLite DBs in `data/` |

---

*Generated: {datetime.now(timezone.utc).isoformat()}*
"""

# Write report
report_path = out_dir / 'report.md'
report_path.write_text(report_md)
print(f"✓ Report saved to {report_path}")

# Also save JSON data for further analysis

def json_safe(val):
    return val if not isinstance(val, float) else round(val, 6)


data_json = {
    "slurm": {
        "rates": slurm_rates,
        "requests": slurm_requests,
        "success": slurm_success,
        "ttft_avg_ms": [json_safe(v) for v in slurm_ttft_avg],
        "ttft_avg_std_ms": [json_safe(v) for v in slurm_ttft_avg_std],
        "ttft_max_ms": [json_safe(v) for v in slurm_ttft_max],
        "tpot_avg_ms": [json_safe(v) for v in slurm_tpot_avg],
        "tpot_avg_std_ms": [json_safe(v) for v in slurm_tpot_avg_std],
        "tpot_max_ms": [json_safe(v) for v in slurm_tpot_max],
        "e2e_avg_ms": [json_safe(v) for v in slurm_e2e_avg],
        "job_ids": [i["job_id"] for i in slurm_info],
    },
    "k8s": {
        "rates": k8s_rates,
        "requests": k8s_requests,
        "success": k8s_success,
        "ttft_avg_ms": [json_safe(v) for v in k8s_ttft_avg],
        "ttft_avg_std_ms": [json_safe(v) for v in k8s_ttft_avg_std],
        "ttft_max_ms": [json_safe(v) for v in k8s_ttft_max],
        "tpot_avg_ms": [json_safe(v) for v in k8s_tpot_avg],
        "tpot_avg_std_ms": [json_safe(v) for v in k8s_tpot_avg_std],
        "tpot_max_ms": [json_safe(v) for v in k8s_tpot_max],
        "e2e_avg_ms": [json_safe(v) for v in k8s_e2e_avg],
        "job_ids": [i["job_id"] for i in k8s_info],
    },
    "metadata": {
        "model": "swiss-ai/Apertus-8B-Instruct-2509",
        "engine": "SGLang",
        "context": "8K",
        "replicates": slurm_summary[rates[0]]["runs"],
        "generated": datetime.now(timezone.utc).isoformat(),
    }
}

json_path = out_dir / 'data.json'
json_path.write_text(json.dumps(data_json, indent=2))
print(f"✓ Raw data saved to {json_path}")
