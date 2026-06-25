#!/usr/bin/env bash
set -euo pipefail

export SML_RESERVATION=SD-69241-apertus-1-5-0

served_model_name="swiss-ai/dsv3-comparable-nonlatent-N64-k4-tp16-$(whoami)-$(date +%Y%m%d-%H%M%S)"

uv run sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-account infra01 \
  --slurm-nodes-per-replica 4 \
  --slurm-time 04:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-nonlatent-N64-k4 \
    --served-model-name ${served_model_name} \
    --tp-size 16 \
    --ep-size 16 \
    --disable-radix-cache \
    --enable-metrics \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0"
