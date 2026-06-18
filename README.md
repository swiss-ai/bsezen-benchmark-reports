# Benchmark Reports

Collection of LLM inference benchmark reports.

## Reports

- [Apertus-70B: vLLM vs SGLang pure cache-disabled replicates](./apertus70b-vllm-vs-sglang-pure-replicates/report.md) - June 2026
- [Apertus-8B: K8s vs Slurm](./apertus8b-k8s-vs-slurm/report.md) - June 2026
- [DeepSeek-V3 non-latent MoE: expert-granularity sweep at constant compute](./dsv3-nonlatent-moe-sweep/report.md) - June 2026

## Structure

Each report directory contains:
- `report.md` - Full markdown report
- `*.png` - Plots and figures
- `data.json` - Raw metrics
- `generate_report.py` - Script to regenerate (optional)
