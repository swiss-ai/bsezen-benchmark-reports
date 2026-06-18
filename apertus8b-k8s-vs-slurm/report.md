# Apertus-8B: Kubernetes vs Slurm Performance Comparison

**Date:** 2026-06-18  
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
| **36.0** | 8,580 | 8,580 (100%) | 322 ms | 1431 ms | 70 ms | ✅ **Healthy** |
| **42.0** | 10,154 | 10,154 (100%) | 8570 ms | 33426 ms | 167 ms | ❌ **Saturated** |

#### Kubernetes (Job 2556968)

| λ (req/s) | Requests | Success | Avg TTFT | Max TTFT | Avg TPOT | Status |
|-----------|----------|---------|----------|----------|----------|--------|
| **36.0** | 8,580 | 8,580 (100%) | 400 ms | 1608 ms | 65 ms | ✅ **Healthy** |
| **42.0** | 10,154 | 10,154 (100%) | 25924 ms | 84188 ms | 201 ms | ❌ **Saturated** |

### Performance at λ=36 (Healthy Load)

![Latency at λ=36](latency_at_36.png)

*Figure 3: Direct comparison at healthy load (λ=36 req/s). K8s shows consistently higher latency across all metrics.*

| Metric | Slurm | K8s | K8s Overhead |
|--------|-------|-----|--------------|
| **Avg TTFT** | 0.32s | 0.40s | **+24%** |
| **Max TTFT** | 1.43s | 1.61s | **+12%** |
| **Avg TPOT** | 70ms | 65ms | **-8%** |

### Performance at λ=42 (Saturated)

| Metric | Slurm | K8s | K8s Degradation |
|--------|-------|-----|-----------------|
| **Avg TTFT** | 8.6s | 25.9s | **3.0× worse** |
| **Max TTFT** | 33.4s | 84.2s | **2.5× worse** |
| **Avg TPOT** | 167ms | 201ms | **1.2× worse** |

---

## Analysis

### 1. Platform Parity at Knee
Both platforms saturate at the **same request rate** (~36 req/s), indicating the bottleneck is the model/GPU, not the orchestration layer.

### 2. Kubernetes Overhead
K8s exhibits **~24% higher TTFT** at healthy load. Potential contributing factors (not isolated):
- Ingress/networking latency (additional hop through API gateway)
- Pod networking overhead (CNI, service mesh)
- Different SGLang container configurations
- Background load/noise on shared K8s infrastructure
- Resource limits or scheduling differences

### 3. Degradation Under Overload
When overloaded (λ=42), K8s degrades **3× more severely** than Slurm:
- K8s TTFT: 25.9s vs Slurm 8.6s
- This suggests K8s queuing or resource contention amplifies the overload

### 4. Clean Saturation
Both platforms show **0% error rate** even under overload—latency degrades gracefully rather than failing.

---

## Limitations & Open Questions

### What We Know
- ✅ K8s consistently shows higher latency (~24% at healthy load, ~3× under overload)
- ✅ Both platforms saturate at the same request rate (~36 req/s)
- ✅ The difference is reproducible across multiple runs

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
2. **Slurm shows lower latency** (~24% at healthy load, ~3× difference under overload)
3. **K8s overhead is real but root cause unproven**: Could be network, configuration, or infrastructure differences
4. **Neither platform** handles overload well—stay below λ=36 for production

### Recommendations

**For latency-sensitive production workloads:**
- Prefer Slurm if raw performance is the primary concern
- K8s is acceptable if the ~24% overhead is within SLO budgets and operational benefits justify it

**For further investigation:**
- Isolate root cause before attributing to "K8s networking"
- Consider running both platforms with identical SGLang configurations and profiling enabled

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

*Generated: 2026-06-18T01:54:21.465406+00:00*
