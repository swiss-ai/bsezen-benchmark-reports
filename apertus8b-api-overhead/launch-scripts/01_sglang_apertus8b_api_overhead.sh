#!/usr/bin/env bash
set -euo pipefail

export SML_RESERVATION=SD-69241-apertus-1-5-0

served_model_name="swiss-ai/Apertus-8B-Instruct-2509-api-overhead-sglang-$(whoami)-$(date +%Y%m%d-%H%M%S)"

uv run sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-account infra01 \
  --slurm-nodes-per-replica 1 \
  --slurm-time 04:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
    --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name ${served_model_name} \
    --host 0.0.0.0 \
    --disable-radix-cache \
    --enable-metrics"
