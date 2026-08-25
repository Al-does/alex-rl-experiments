# Cassandra targeted PPO: interventions and vf-clip grid (Aug 2026)

Concise findings from 5M-step runs at **entropy coefficient 0.03** (seed 42).
Earlier intervention runs at entropy 0.008 are excluded — they confounded
comparison with the control and are not reported here.

## Control

| Setting | Value |
|---|---|
| Recipe | `targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m` |
| Model | width 64, 4 layers, 1 head; context/BPTT 256/256 |
| vf_clip_param / vf_loss_coeff | 10 / 0.5 |
| Gamma / lambda | 0.990 / 0.95 |

**5M checkpoint:** mean episode return **352.98** (5,013,504 steps; final
policy entropy ≈ 0.94).

## RLlib vf-clip semantics (Ray 2.56)

`vf_clip_param` clips **squared value error**, not value predictions:

```text
vf_loss = (V - target)²
clipped = clamp(vf_loss, max=vf_clip_param)
```

Weighted critic contribution is `vf_loss_coeff × clipped_loss`. The baseline
(clip=10, coeff=0.5) rarely saturates the clip (~4–8% of iteration-level loss
is clipped). The failed intervention `vf_clip=100, coeff=0.5` allowed errors
up to ±10 with max weighted loss 50, dominating the first update and
collapsing learning.

## Phase 1: targeted interventions (entropy 0.03)

All leaves share the control recipe except for the named change.

| Condition | Change | Final return | Δ vs control | Notes |
|---|---|---:|---:|---|
| **bptt_64** | BPTT 64 (context 256) | **364.22** | +11.2 | Best intervention; shorter backprop through time |
| lambda_0.98 | GAE λ = 0.98 | 277.25 | −75.7 | Slower credit assignment hurts |
| vf_clip_100 | clip=100, coeff=**0.5** | 39.50 | −313.5 | Mis-scaled critic; see semantics above |
| previous_reward | +1-d previous reward feature | 31.60 | −321.4 | Policy entropy → ~0.01 by ~950K steps |

**Takeaway:** only **bptt_64** beat control among structural interventions.
`previous_reward` and mis-scaled vf-clip both collapse to the always-operate
baseline (~30–40 return).

## Phase 2: vf-clip / vf-loss-coeff grid (entropy 0.03)

Grid held entropy, gamma, lambda, and architecture fixed; varied critic
capacity via clip and coeff. Target max weighted critic loss per cell: ~5 or ~1.

| Cell | vf_clip | vf_coeff | Max weighted loss | Final return | Δ vs control |
|---|---:|---:|---:|---:|---:|
| **vf100_coeff001** | 100 | 0.01 | 1 | **495.30** | **+142.3** |
| **vf400_coeff00125** | 400 | 0.0125 | 5 | **448.92** | **+95.9** |
| vf100_coeff005 | 100 | 0.05 | 5 | 362.14 | +9.2 |
| vf100_coeff0002 | 100 | 0.002 | 0.2 | 32.54 | −320.4 |
| vf400_coeff00025 | 400 | 0.0025 | 1 | 32.03 | −320.9 |

**Takeaway:** loosening vf-clip while **reducing** vf_loss_coeff restores a
balanced actor–critic update. The sweet spot is near **clip=100, coeff=0.01**
(max weighted loss ≈ 1), which beat control by +40% return and beat bptt_64.
Cells with max weighted loss ≈ 0.2 (coeff 0.002–0.0025) collapse like
`previous_reward` — critic signal is too weak to learn.

## Phase 3: previous_reward rerun with best critic (entropy 0.03)

After the vf-clip grid identified **vf100_coeff001**, we re-ran the
`previous_reward` intervention (visible +1-d previous reward feature) with those
critic settings instead of the default clip=10 / coeff=0.5.

| Condition | vf_clip / coeff | Final return | Δ vs control | Notes |
|---|---|---:|---:|---|
| previous_reward (default critic) | 10 / 0.5 | 31.60 | −321.4 | Collapse; entropy → ~0.01 |
| **previous_reward + vf100_coeff001** | 100 / 0.01 | **492.36** | **+139.4** | Matches best baseline (495.30) |
| vf100_coeff001 (no reward feature) | 100 / 0.01 | 495.30 | +142.3 | Phase 2 best cell |

Run: `targeted_ppo_previous_reward_best_critic_5m` /
`20260825T023104Z-baed31de` (seed 42, 5M steps, RTX 4090).

**Takeaway:** the earlier `previous_reward` collapse was a **critic scaling**
failure, not an inherent incompatibility with the reward feature. With a
properly scaled critic, visible previous reward is neutral to slightly
negative (−2.9 return vs vf100_coeff001 without the feature).

## Recommended next step

Adopt **vf_clip_param=100, vf_loss_coeff=0.01** as the new critic default for
this recipe (replacing clip=10, coeff=0.5). Optionally combine with **bptt=64**
— not yet tested jointly.

## Artifacts

| Path | Contents |
|---|---|
| `targeted_ppo_small_interventions_5m/` | Intervention leaves + compact results |
| `targeted_ppo_vf_clip_grid_5m/` | Five-cell grid + compact results |
| `targeted_ppo_previous_reward_best_critic_5m/` | Phase 3 previous_reward rerun |
| `interventions_and_vf_clip_2026_08_reward_curves.png` | Return vs steps overlay |

Control curve source: `results/entropy_diagnostics_2026_08/compact_training_curves.json`
(`targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m`).
