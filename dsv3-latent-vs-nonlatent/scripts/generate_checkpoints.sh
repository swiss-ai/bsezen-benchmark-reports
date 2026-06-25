#!/usr/bin/env bash
set -euo pipefail

# Generate the three comparable DeepSeek-V3-shaped FP8 checkpoints.
# Run from /capstor/scratch/cscs/$USER on Clariden (or adapt paths).

cd "$(dirname "$0")/../../../model-generation"

# Non-latent baseline: N=64, k=4, intermediate=4096
uv run model.py \
    --out /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-nonlatent-N64-k4 \
    --n-routed-experts 64 \
    --num-experts-per-tok 4 \
    --hidden-size 7168 \
    --moe-intermediate-size 4096 \
    --fp8-serving-tp-size 16 \
    --fp8-serving-ep-size 16 \
    --weight-init-mode tile \
    --skip-tokenizer

# Latent via wider experts (original comparison): N=64, k=4, intermediate=14336
uv run model.py \
    --out /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-latent-N64-k4 \
    --n-routed-experts 64 \
    --num-experts-per-tok 4 \
    --hidden-size 7168 \
    --latent-moe-dim 2048 \
    --constant-total-params 344B \
    --fp8-serving-tp-size 16 \
    --fp8-serving-ep-size 16 \
    --weight-init-mode tile \
    --skip-tokenizer

# Latent via more experts (follow-up comparison): N=224, k=14, intermediate=4096
uv run model.py \
    --out /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-latent-N224-k14 \
    --n-routed-experts 224 \
    --num-experts-per-tok 14 \
    --hidden-size 7168 \
    --latent-moe-dim 2048 \
    --moe-intermediate-size 4096 \
    --fp8-serving-tp-size 16 \
    --fp8-serving-ep-size 16 \
    --weight-init-mode tile \
    --skip-tokenizer

# Tokenizer files are copied separately from a DeepSeek-V3 checkout, e.g.:
# rsync -av /capstor/scratch/cscs/bsezen/hf-dsv3-nonlatent-sweep-N64-k4-tp16/tokenizer* \
#   /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-{nonlatent,latent}-N64-k4/
# rsync -av ... /capstor/scratch/cscs/bsezen/hf-dsv3-comparable-latent-N224-k14/
