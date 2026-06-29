# Nemotron-H Synthetic Router Balance Sanity Check

Date: 2026-06-29

## Goal

Check whether the synthetic Nemotron-H router weights collapse tokens onto a small subset of experts. This is a sanity check for random synthetic checkpoints, not a measurement of trained-model expert utilization.

Runtime routed-expert capture in SGLang was attempted with `--enable-return-routed-experts`, but the current Nemotron-H path failed with `capturing routing experts but get layer_id None`. To avoid modifying SGLang or contaminating benchmark timing, this check was performed offline from the saved router weights.

## Method

For each checkpoint, the script reads every `model.layers.*.mixer.gate.weight` tensor, samples random hidden states, applies the Nemotron-H router scoring rule, and counts selected experts after top-k routing.

Experiment settings:

| Property | Value |
|---|---:|
| Hidden size | 1024 |
| Pattern | `(M*E) x 8` |
| MoE layers | 8 |
| Routed experts | 64 |
| Top-k | 4 |
| Sampled hidden states per layer/seed | 200,000 |
| Seeds | 1, 2, 3, 4, 5 |
| Measurements per variant | 40 |

Checkpoints:

| Variant | Path | Notes |
|---|---|---|
| Non-latent | `/capstor/scratch/cscs/bsezen/tmp/nemotron-h-router-nonlatent-h1024-e64-k4-s28` | `moe_latent_size = null` |
| Latent | `/capstor/scratch/cscs/bsezen/tmp/nemotron-h-router-latent-h1024-l256-e64-k4-s29` | `moe_latent_size = 256` |

Script:

```bash
uv run router_balance_check.py \
  --checkpoint <checkpoint> \
  --label <label> \
  --tokens 200000 \
  --seeds 1 2 3 4 5 \
  --out <output.csv>
```

## Results

Ratios are normalized by the uniform expected count per expert.

| Variant | Mean min ratio | Worst min ratio | Mean max ratio | Worst max ratio | Mean CV | Mean normalized entropy |
|---|---:|---:|---:|---:|---:|---:|
| Non-latent | 0.856 | 0.795 | 1.163 | 1.226 | 0.065 | 0.9995 |
| Latent | 0.853 | 0.798 | 1.173 | 1.234 | 0.064 | 0.9995 |

Raw CSVs:

- `nonlatent_h1024_e64_k4.csv`
- `latent_h1024_l256_e64_k4.csv`

## Interpretation

The synthetic routers do not collapse. Across 8 MoE layers and 5 random-hidden-state seeds, both latent and non-latent checkpoints stay close to uniform expert utilization. The hottest experts receive about `1.23x` the uniform expected count in the worst case, while the coldest receive about `0.80x`; mean coefficient of variation is about `0.064-0.065`.

This does not prove that trained Nemotron-H models have the same routing distribution. It only rules out a trivial synthetic-checkpoint artifact where most tokens route to a small subset of experts. Because the same offline method is applied to both latent and non-latent random checkpoints, the check supports using these synthetic checkpoints for comparative serving experiments without adding an SGLang instrumentation patch.
