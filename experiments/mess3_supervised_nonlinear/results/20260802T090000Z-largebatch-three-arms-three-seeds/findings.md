# Three decoder arms, three seeds — nonlinear prediction head under large-batch SGD

**Campaign** `20260802T090000Z-largebatch-three-arms-three-seeds` · 3 arms × 3 seeds
(42–44) × 62,500 updates · large-batch SGD (batch 16,384, LR 0.16, √ scaling) ·
same paper transformer backbone and affine block-3 belief probe · vast instance
`46568172`, experiment ref `bccba42`.

Machine-readable companion: `campaign_summary.json` (per-run metrics and local
paths). Checkpoints and full artifact trees remain on B2 under
`mess3_supervised_nonlinear-*-largebatch-*`.

---

## Executive summary

All **9/9** runs completed the budget, reached exact-validation loss at the
Bayesian floor, and passed the preregistered scientific gate (final pre-LN probe
MSE ≤ 1e-3, validation gap ≤ 0.005 nats).

Replacing the paper's linear `64 → 3` prediction head with a 2-layer or 4-layer
Swish MLP decoder **does not break** belief decodability from the final
transformer-block residual before the last LayerNorm. Affine probe R² stays
≈ 0.996–0.998 across arms.

The 4-layer head shows **slightly higher** probe MSE on average (3.68×10⁻⁴ vs
2.53×10⁻⁴ for linear control), with seed 42 the weakest point (R² = 0.9955).
**We do not treat that as evidence that depth hurts belief resolution:** n = 3
seeds per arm is too small to separate a real effect from seed noise, the
arm-level ranges overlap, and every run still passes the gate comfortably.

---

## What was held fixed vs varied

| component | setting |
| --- | --- |
| Backbone | Paper-faithful 4-block MESS3 transformer |
| Training | Large-batch variant of supervised SGD (62.5k steps, batch 16,384, LR 0.16) |
| Belief probe | Affine OLS on **block_3 pre-final-LayerNorm** (unchanged) |
| **Varied** | Next-token prediction head only |

| arm | prediction head |
| --- | --- |
| `linear_decoder_control` | `64 → 3` (paper) |
| `two_layer_decoder` | `64 → 64 → 3` with Swish |
| `four_layer_decoder` | `64 → 64 → 64 → 64 → 3` with Swish |

---

## Per-run detail

Probe targets exact Bayesian belief on length-10 contexts. Reported layer is
`block_3` (pre-final LayerNorm).

| arm | seed | steps | val loss (nats) | val gap | probe R² | probe MSE | gate |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| `linear_decoder_control` | 42 | 62,500 | 0.802542 | +3.04×10⁻⁵ | 0.9978 | 2.48×10⁻⁴ | PASS |
| `linear_decoder_control` | 43 | 62,500 | 0.802530 | +1.86×10⁻⁵ | 0.9972 | 3.15×10⁻⁴ | PASS |
| `linear_decoder_control` | 44 | 62,500 | 0.802563 | +5.08×10⁻⁵ | 0.9982 | 1.97×10⁻⁴ | PASS |
| `two_layer_decoder` | 42 | 62,500 | 0.802533 | +2.09×10⁻⁵ | 0.9976 | 2.63×10⁻⁴ | PASS |
| `two_layer_decoder` | 43 | 62,500 | 0.802531 | +1.96×10⁻⁵ | 0.9982 | 2.05×10⁻⁴ | PASS |
| `two_layer_decoder` | 44 | 62,500 | 0.802543 | +3.14×10⁻⁵ | 0.9975 | 2.77×10⁻⁴ | PASS |
| `four_layer_decoder` | 42 | 62,500 | 0.802533 | +2.16×10⁻⁵ | 0.9955 | 5.02×10⁻⁴ | PASS |
| `four_layer_decoder` | 43 | 62,500 | 0.802558 | +4.64×10⁻⁵ | 0.9966 | 3.74×10⁻⁴ | PASS |
| `four_layer_decoder` | 44 | 62,500 | 0.802546 | +3.43×10⁻⁵ | 0.9980 | 2.27×10⁻⁴ | PASS |

Bayesian floor: **0.802512 nats** (identical across runs).

---

## Arm-level aggregates (descriptive only)

| arm | mean R² | R² range | mean MSE | mean val gap | gate |
| --- | ---: | --- | ---: | ---: | --- |
| `linear_decoder_control` | 0.9977 | [0.9972, 0.9982] | 2.53×10⁻⁴ | +3.3×10⁻⁵ | 3/3 |
| `two_layer_decoder` | 0.9978 | [0.9975, 0.9982] | 2.48×10⁻⁴ | +2.4×10⁻⁵ | 3/3 |
| `four_layer_decoder` | 0.9967 | [0.9955, 0.9980] | 3.68×10⁻⁴ | +3.4×10⁻⁵ | 3/3 |

Mean training wall time ≈ **31 min/run** (~4.8 h for the full queue on one RTX
4090).

---

## What we can and cannot conclude

**Supported:** Nonlinear prediction-head depth, at least up to four Swish hidden
layers, is compatible with strong linear belief decodability at the usual probe
location under this training recipe. Next-token cross-entropy still drives the
backbone to encode belief-like structure even when the readout head is nonlinear.

**Not supported (yet):** That the 4-layer head *reduces* belief resolution
relative to linear or 2-layer heads. The arm means differ slightly, but with
three seeds the confidence intervals on those means overlap heavily and a single
seed (`four_layer_decoder` 42) could be driving the gap. A directional claim
would need more seeds, explicit variance reporting, and ideally a preregistered
test — not eyeballing arm averages.

---

## Operational notes

- First launch used the paper micro-batch recipe (1M steps, batch 64); it was
  stopped after the linear-control arm and restarted with large-batch settings.
- Compact results in this directory were salvaged from B2; the vast box
  self-destructed after the queue finished.
- Prior aborted paper-batch B2 paths (without the `largebatch` prefix) are
  superseded by this campaign.
