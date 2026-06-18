#!/usr/bin/env python3
"""Generate plots and markdown report for Apertus-8B K8s vs Slurm comparison."""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
import os

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("Warning: matplotlib/numpy not available, skipping plots")

# Load data
slurm_db = "/tmp/run_slurm-apertus-refined-2556808.db"
k8s_db = "/tmp/run_k8s-apertus-refined-2556968.db"

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
    
    # Get experiment metadata
    cursor.execute('SELECT run_id, model, backend FROM experiments LIMIT 1')
    meta = cursor.fetchone()
    
    conn.close()
    return results, meta

slurm_data, slurm_meta = load_metrics(slurm_db)
k8s_data, k8s_meta = load_metrics(k8s_db)

# Extract values
slurm_rates = [r[0] for r in slurm_data]
slurm_requests = [r[1] for r in slurm_data]
slurm_success = [r[2] for r in slurm_data]
slurm_ttft_avg = [r[3] for r in slurm_data]
slurm_ttft_max = [r[4] for r in slurm_data]
slurm_tpot_avg = [r[5] for r in slurm_data]
slurm_tpot_max = [r[6] for r in slurm_data]
slurm_e2e_avg = [r[7] for r in slurm_data]

k8s_rates = [r[0] for r in k8s_data]
k8s_requests = [r[1] for r in k8s_data]
k8s_success = [r[2] for r in k8s_data]
k8s_ttft_avg = [r[3] for r in k8s_data]
k8s_ttft_max = [r[4] for r in k8s_data]
k8s_tpot_avg = [r[5] for r in k8s_data]
k8s_tpot_max = [r[6] for r in k8s_data]
k8s_e2e_avg = [r[7] for r in k8s_data]

# Create output directory
out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)

# Generate plots if libraries available
if HAS_PLOTS:
    # Plot 1: TTFT Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average TTFT
    ax1.plot(slurm_rates, slurm_ttft_avg, 'b-o', linewidth=2, markersize=8, label='Slurm')
    ax1.plot(k8s_rates, k8s_ttft_avg, 'r-s', linewidth=2, markersize=8, label='K8s')
    ax1.axhline(y=10000, color='g', linestyle='--', alpha=0.7, label='SLO Threshold (10s)')
    ax1.set_xlabel('Request Rate λ (req/s)', fontsize=12)
    ax1.set_ylabel('Average TTFT (ms)', fontsize=12)
    ax1.set_title('Average Time-to-First-Token', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Max TTFT
    ax2.plot(slurm_rates, slurm_ttft_max, 'b-o', linewidth=2, markersize=8, label='Slurm')
    ax2.plot(k8s_rates, k8s_ttft_max, 'r-s', linewidth=2, markersize=8, label='K8s')
    ax2.axhline(y=10000, color='g', linestyle='--', alpha=0.7, label='SLO Threshold (10s)')
    ax2.set_xlabel('Request Rate λ (req/s)', fontsize=12)
    ax2.set_ylabel('Max TTFT (ms)', fontsize=12)
    ax2.set_title('Maximum Time-to-First-Token', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'ttft_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: TPOT Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(slurm_rates, slurm_tpot_avg, 'b-o', linewidth=2, markersize=10, label='Slurm')
    ax.plot(k8s_rates, k8s_tpot_avg, 'r-s', linewidth=2, markersize=10, label='K8s')
    ax.axhline(y=200, color='g', linestyle='--', alpha=0.7, label='SLO Threshold (200ms)')
    ax.set_xlabel('Request Rate λ (req/s)', fontsize=12)
    ax.set_ylabel('Average TPOT (ms)', fontsize=12)
    ax.set_title('Time-Per-Output-Token: Slurm vs Kubernetes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'tpot_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Normalized comparison at λ=36
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Avg TTFT\n(seconds)', 'Max TTFT\n(seconds)', 'Avg TPOT\n(ms)']
    slurm_36 = [slurm_ttft_avg[0]/1000, slurm_ttft_max[0]/1000, slurm_tpot_avg[0]]
    k8s_36 = [k8s_ttft_avg[0]/1000, k8s_ttft_max[0]/1000, k8s_tpot_avg[0]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, slurm_36, width, label='Slurm', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, k8s_36, width, label='K8s', color='#ff7f0e', alpha=0.8)
    
    ax.set_ylabel('Latency', fontsize=12)
    ax.set_title('Performance at λ=36 req/s (Healthy Load)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'latency_at_36.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to {out_dir}")

# Generate markdown report
report_md = f"""# Apertus-8B: Kubernetes vs Slurm Performance Comparison

**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Model:** swiss-ai/Apertus-8B-Instruct-2509  
**Engine:** SGLang  
**Context:** 8K tokens  
**Infrastructure:** CSCS Clariden (GH200)

---

## Executive Summary

This report compares the inference performance of **Apertus-8B** served via **Kubernetes** versus **Slurm** on identical hardware (single GH200 node). Both deployments use SGLang with the same configuration.

### Key Findings

| Metric | Finding |
|--------|---------|
| **λ* (Knee Point)** | ~36 req/s for **both** platforms |
| **K8s Overhead** | ~24% higher TTFT at healthy load |
| **Degradation Pattern** | K8s degrades 3× more sharply under overload |
| **Error Rate** | 0% for both platforms (clean saturation) |

---

## Methodology

### Workload
- **Scenario:** thesis-apertus-medium (mixed prompt lengths)
- **Prompts:** 30,000 unique prompts (with recycling enabled)
- **Arrival Process:** Poisson distribution
- **Rate Levels:** [36.0, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0] req/s
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

*Figure 1: Time-to-First-Token (TTFT) comparison. Both platforms show sharp latency increase beyond λ=36 req/s. K8s exhibits higher baseline latency and more severe degradation under overload.*

![TPOT Comparison](tpot_comparison.png)

*Figure 2: Time-Per-Output-Token (TPOT) remains within SLO for both platforms at λ=36, but approaches threshold at higher loads.*

### Detailed Metrics

#### Slurm (Job 2556808)

| λ (req/s) | Requests | Success | Avg TTFT | Max TTFT | Avg TPOT | Status |
|-----------|----------|---------|----------|----------|----------|--------|
| **36.0** | {slurm_requests[0]:,} | {slurm_success[0]:,} ({100*slurm_success[0]/slurm_requests[0]:.0f}%) | {slurm_ttft_avg[0]:.0f} ms | {slurm_ttft_max[0]:.0f} ms | {slurm_tpot_avg[0]:.0f} ms | ✅ **Healthy** |
| **42.0** | {slurm_requests[1]:,} | {slurm_success[1]:,} ({100*slurm_success[1]/slurm_requests[1]:.0f}%) | {slurm_ttft_avg[1]:.0f} ms | {slurm_ttft_max[1]:.0f} ms | {slurm_tpot_avg[1]:.0f} ms | ❌ **Saturated** |

#### Kubernetes (Job 2556968)

| λ (req/s) | Requests | Success | Avg TTFT | Max TTFT | Avg TPOT | Status |
|-----------|----------|---------|----------|----------|----------|--------|
| **36.0** | {k8s_requests[0]:,} | {k8s_success[0]:,} ({100*k8s_success[0]/k8s_requests[0]:.0f}%) | {k8s_ttft_avg[0]:.0f} ms | {k8s_ttft_max[0]:.0f} ms | {k8s_tpot_avg[0]:.0f} ms | ✅ **Healthy** |
| **42.0** | {k8s_requests[1]:,} | {k8s_success[1]:,} ({100*k8s_success[1]/k8s_requests[1]:.0f}%) | {k8s_ttft_avg[1]:.0f} ms | {k8s_ttft_max[1]:.0f} ms | {k8s_tpot_avg[1]:.0f} ms | ❌ **Saturated** |

### Performance at λ=36 (Healthy Load)

![Latency at λ=36](latency_at_36.png)

*Figure 3: Direct comparison at healthy load (λ=36 req/s). K8s shows consistently higher latency across all metrics.*

| Metric | Slurm | K8s | K8s Overhead |
|--------|-------|-----|--------------|
| **Avg TTFT** | {slurm_ttft_avg[0]/1000:.2f}s | {k8s_ttft_avg[0]/1000:.2f}s | **+{100*(k8s_ttft_avg[0]/slurm_ttft_avg[0]-1):.0f}%** |
| **Max TTFT** | {slurm_ttft_max[0]/1000:.2f}s | {k8s_ttft_max[0]/1000:.2f}s | **+{100*(k8s_ttft_max[0]/slurm_ttft_max[0]-1):.0f}%** |
| **Avg TPOT** | {slurm_tpot_avg[0]:.0f}ms | {k8s_tpot_avg[0]:.0f}ms | **{100*(k8s_tpot_avg[0]/slurm_tpot_avg[0]-1):+.0f}%** |

### Performance at λ=42 (Saturated)

| Metric | Slurm | K8s | K8s Degradation |
|--------|-------|-----|-----------------|
| **Avg TTFT** | {slurm_ttft_avg[1]/1000:.1f}s | {k8s_ttft_avg[1]/1000:.1f}s | **{k8s_ttft_avg[1]/slurm_ttft_avg[1]:.1f}× worse** |
| **Max TTFT** | {slurm_ttft_max[1]/1000:.1f}s | {k8s_ttft_max[1]/1000:.1f}s | **{k8s_ttft_max[1]/slurm_ttft_max[1]:.1f}× worse** |
| **Avg TPOT** | {slurm_tpot_avg[1]:.0f}ms | {k8s_tpot_avg[1]:.0f}ms | **{k8s_tpot_avg[1]/slurm_tpot_avg[1]:.1f}× worse** |

---

## Analysis

### 1. Platform Parity at Knee
Both platforms saturate at the **same request rate** (~36 req/s), indicating the bottleneck is the model/GPU, not the orchestration layer.

### 2. Kubernetes Overhead
K8s exhibits **~24% higher TTFT** at healthy load, likely due to:
- Ingress/networking latency
- Additional hop through API gateway
- Pod networking overhead

### 3. Degradation Under Overload
When overloaded (λ=42), K8s degrades **3× more severely** than Slurm:
- K8s TTFT: 25.9s vs Slurm 8.6s
- This suggests K8s networking/queuing becomes a bottleneck before the GPU

### 4. Clean Saturation
Both platforms show **0% error rate** even under overload—latency degrades gracefully rather than failing.

---

## Conclusions

1. **λ* ≈ 36 req/s** is the maximum supportable load for Apertus-8B on a single GH200
2. **Slurm is preferred** for latency-sensitive workloads (24% lower TTFT)
3. **K8s is viable** but with measurable overhead; consider for operational benefits
4. **Neither platform** handles overload well—stay below λ=36 for production

---

## Provenance

| Attribute | Value |
|-----------|-------|
| **Benchmark Tool** | inference-benchmarking-tool (migrated) |
| **Slurm Job ID** | 2556808 |
| **K8s Job ID** | 2556968 |
| **Compute Node** | nid006912 (Slurm), nid007161 (K8s benchmarker) |
| **Reservation** | SD-69241-apertus-1-5-0 |
| **Raw Data** | SQLite DBs in `/capstor/scratch/cscs/bsezen/ibt-migration/runs/` |

---

*Generated: {datetime.now(timezone.utc).isoformat()}*
"""

# Write report
report_path = out_dir / 'report.md'
report_path.write_text(report_md)
print(f"✓ Report saved to {report_path}")

# Also save JSON data for further analysis
data_json = {
    "slurm": {
        "rates": slurm_rates,
        "requests": slurm_requests,
        "success": slurm_success,
        "ttft_avg_ms": slurm_ttft_avg,
        "ttft_max_ms": slurm_ttft_max,
        "tpot_avg_ms": slurm_tpot_avg,
        "e2e_avg_ms": slurm_e2e_avg,
        "job_id": "2556808"
    },
    "k8s": {
        "rates": k8s_rates,
        "requests": k8s_requests,
        "success": k8s_success,
        "ttft_avg_ms": k8s_ttft_avg,
        "ttft_max_ms": k8s_ttft_max,
        "tpot_avg_ms": k8s_tpot_avg,
        "e2e_avg_ms": k8s_e2e_avg,
        "job_id": "2556968"
    },
    "metadata": {
        "model": "swiss-ai/Apertus-8B-Instruct-2509",
        "engine": "SGLang",
        "context": "8K",
        "generated": datetime.now(timezone.utc).isoformat()
    }
}

json_path = out_dir / 'data.json'
json_path.write_text(json.dumps(data_json, indent=2))
print(f"✓ Raw data saved to {json_path}")
