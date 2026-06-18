# DeepSeek-V3 Non-Latent MoE Sweep Notes

## Research Question

Holding total/active parameters and serving topology constant (TP16/EP16, k/N = 6.25%), does expert granularity (N=32/k=2 → 64/k=4 → 128/k=8) measurably affect throughput or latency?

## Goal

Benchmark a standard MoE sweep that mirrors the latent-MoE expert-granularity sweep, but without `--latent-moe-dim`.

The main comparison is latent MoE vs non-latent MoE at similar total and active parameter scale.

## Models To Include

Use only the three comparable `TP16/EP16` non-latent shapes for now:

| Variant | N routed experts | k active experts | k/N | Expected dry-run size |
|---|---:|---:|---:|---:|
| coarse | 32 | 2 | 6.25% | 346.40B total, 39.88B active/token |
| medium | 64 | 4 | 6.25% | 346.41B total, 39.89B active/token |
| fine | 128 | 8 | 6.25% | 346.44B total, 39.92B active/token |

Drop `N=256,k=16` for now.

## Why N=256 Is Excluded

Without latent MoE, the comparable ~346B `N=256,k=16` shape needs `moe_intermediate_size=1024`.

Strict `TP16` FP8 serving requires:

```text
moe_intermediate_size % (128 * TP) == 0
```

For `TP16`, that means:

```text
moe_intermediate_size % 2048 == 0
```

So `moe_intermediate_size=1024` does not pass strict `TP16` FP8 validation. Forcing the smallest valid `TP16` value, `2048`, makes the non-latent `N=256,k=16` checkpoint about `673B`, so it is not comparable to the 344B sweep.

The ~346B `N=256,k=16` shape is possible with `TP8` alignment, but that changes the serving topology and should not be mixed into the same primary TP16 comparison.

## Generation Command

Run from `model-generation/`:

```bash
for spec in 32:2 64:4 128:8; do
  N=${spec%%:*}
  K=${spec##*:}
  uv run model.py \
    --out /capstor/scratch/cscs/$USER/hf-dsv3-nonlatent-sweep-N${N}-k${K}-tp16 \
    --n-routed-experts ${N} \
    --num-experts-per-tok ${K} \
    --hidden-size 7168 \
    --constant-total-params 344B \
    --fp8-serving-tp-size 16 \
    --fp8-serving-ep-size 16 \
    --weight-init-mode tile
done
```

Do not pass `--latent-moe-dim`.

## Benchmark Plan

| Dimension | Plan |
|---|---|
| Serving engine | SGLang first |
| Serving topology | 4 nodes, `TP16/EP16` |
| Cache policy | Disable radix cache unless explicitly benchmarking cache behavior |
| Metrics | Use `--enable-metrics` |
| Benchmark node | `infra02` with `power_throttling` reservation |
| Load shape | Same arrival process, warmup, measurement, drain, SLOs, and lambda sweep for all variants |
| Replicates | Add at least one replicate if saturation behavior is noisy or important |

## Report Points To Preserve

1. The non-latent models omit `--latent-moe-dim` entirely.
2. The included variants all keep `k/N = 6.25%`.
3. The included variants are all near 346B total and ~39.9B active parameters/token.
4. `N=256,k=16` is excluded because comparable size conflicts with strict TP16 FP8 alignment.
5. Checkpoints use random/tiled weights and are for infrastructure benchmarking, not model quality evaluation.
