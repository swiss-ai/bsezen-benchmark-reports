#!/usr/bin/env bash
set -euo pipefail

# Generate the DeepSeek-V3-shaped prompt corpus used by all variants.
# Only the `medium` label is used in this benchmark; it is recycled to 20k requests.

cd "$(dirname "$0")/../../../loadtest-prompt-generation"

uv run prompt-generation \
  --output prompts-deepseek-thesis.json \
  --num-prompts 4000 \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --workload-config workloads/thesis.yaml \
  --seed 1234

# Copy to cluster prompt cache:
# rsync -avz prompts-deepseek-thesis.json cscs-clariden:/capstor/scratch/cscs/$USER/loadtest/
