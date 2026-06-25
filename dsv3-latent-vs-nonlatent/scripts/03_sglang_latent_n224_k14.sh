#!/usr/bin/env bash
set -euo pipefail

export SML_RESERVATION=SD-69241-apertus-1-5-0

served_model_name="swiss-ai/dsv3-comparable-latent-N224-k14-tp16-$(whoami)-$(date +%Y%m%d-%H%M%S)"

uv run sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-account infra01 \
  --slurm-nodes-per-replica 4 \
  --slurm-time 04:00:00 \
  --serving-framework sglang \
  --slurm-environment local/dsv3-latent-vs-nonlatent/sglang_latent_moe.toml \
  --framework-args "--model-path /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-latent-N224-k14 \
    --served-model-name ${served_model_name} \
    --tp-size 16 \
    --ep-size 16 \
    --disable-radix-cache \
    --enable-metrics \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0"
