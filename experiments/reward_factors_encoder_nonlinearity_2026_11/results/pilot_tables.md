# Eq. (4) encoder-only experiment

Mean ± sample SD across agent seeds 42–44 (pilot: seed 42 only).
Ladder/Full-State cells: occupancy / mirror actions changed. Gap normalization uses paired fresh intact/map references.
Fixed-λ affine/exact is the local historical control; any discrepancies from previously published remote controls require an exact unchanged-runner CPU recheck (recorded in the manifest).
Full-State mirror changes are a positive-control response, not a spillover-minimization objective.

## Held-out encoder R²

| Test / coordinates | Affine exact | Affine decoded | MLP exact | MLP decoded |
|---|---:|---:|---:|---:|
| ladder / marginal | 0.7153±0.0000 | 0.7273±0.0000 | 0.8524±0.0000 | 0.8494±0.0000 |
| ladder_full / marginal | 0.7781±0.0000 | 0.7950±0.0000 | 0.8464±0.0000 | 0.8573±0.0000 |
| moving / marginal | 0.6399±0.0000 | 0.7088±0.0000 | 0.8142±0.0000 | 0.8685±0.0000 |
| moving / joint | 0.6964±0.0000 | 0.8071±0.0000 | 0.8196±0.0000 | 0.8997±0.0000 |
| xor_eps02 / marginal | 0.5258±0.0000 | 0.6562±0.0000 | 0.6575±0.0000 | 0.7613±0.0000 |
| xor_eps02 / joint | 0.4342±0.0000 | 0.7880±0.0000 | 0.6545±0.0000 | 0.8205±0.0000 |
| xor_v2 / marginal | 0.5813±0.0000 | 0.6186±0.0000 | 0.7961±0.0000 | 0.8420±0.0000 |
| xor_v2 / joint | 0.6365±0.0000 | 0.8399±0.0000 | 0.7731±0.0000 | 0.8641±0.0000 |

## Archived λ held fixed: strict encoder-only comparison

| Encoder / source / coordinates | Ladder ↓ / ↓ | Moving reward ↑ | Moving gap ↑ | Moving F1 changed ↓ | XOR eps02 ↑ | XOR v2 ↑ | Full-State |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine/exact/marginal | 0.1282±0.0000 / 0.0105±0.0000 | 0.9600±0.0000 | — | 0.1313±0.0000 | 0.6406±0.0000 | 0.5518±0.0000 | 0.1406±0.0000 / 0.5596±0.0000 |
| affine/decoded/marginal | 0.0964±0.0000 / 0.0177±0.0000 | 0.9751±0.0000 | — | 0.1464±0.0000 | 0.6514±0.0000 | 0.5552±0.0000 | 0.0767±0.0000 / 0.5610±0.0000 |
| mlp/exact/marginal | 0.1042±0.0000 / 0.0101±0.0000 | 0.9784±0.0000 | — | 0.0922±0.0000 | 0.6602±0.0000 | 0.6377±0.0000 | 0.1177±0.0000 / 0.5054±0.0000 |
| mlp/decoded/marginal | 0.0951±0.0000 / 0.0330±0.0000 | 0.9976±0.0000 | — | 0.0754±0.0000 | 0.5830±0.0000 | 0.7114±0.0000 | 0.1177±0.0000 / 0.5112±0.0000 |
| affine/exact/joint | 0.1282±0.0000 / 0.0105±0.0000 | 1.0144±0.0000 | — | 0.0747±0.0000 | 0.5947±0.0000 | 0.6924±0.0000 | 0.1406±0.0000 / 0.5596±0.0000 |
| affine/decoded/joint | 0.0964±0.0000 / 0.0177±0.0000 | 1.0314±0.0000 | — | 0.0569±0.0000 | 0.5947±0.0000 | 0.6924±0.0000 | 0.0767±0.0000 / 0.5610±0.0000 |
| mlp/exact/joint | 0.1042±0.0000 / 0.0101±0.0000 | 0.9050±0.0000 | — | 0.1339±0.0000 | 0.5947±0.0000 | 0.6484±0.0000 | 0.1177±0.0000 / 0.5054±0.0000 |
| mlp/decoded/joint | 0.0951±0.0000 / 0.0330±0.0000 | 1.0184±0.0000 | — | 0.0742±0.0000 | 0.5947±0.0000 | 0.6802±0.0000 | 0.1177±0.0000 / 0.5112±0.0000 |

### Paired differences versus historical affine/exact at its archived λ

Signs are raw new−historical differences (negative favors occupancy/spillover; positive favors reward/gap). Full-State mirror is nondirectional.

| Arm | Metric | Δ seed42 | Δ seed43 | Δ seed44 | Mean ± SD |
|---|---|---:|---:|---:|---:|
| affine/decoded/marginal | ladder_occupancy | -0.031738 | — | — | -0.0317±0.0000 |
| affine/decoded/marginal | ladder_mirror | +0.007202 | — | — | 0.0072±0.0000 |
| affine/decoded/marginal | moving_reward | +0.015137 | — | — | 0.0151±0.0000 |
| affine/decoded/marginal | moving_f1_changed | +0.015015 | — | — | 0.0150±0.0000 |
| affine/decoded/marginal | xor_eps02_reward | +0.010742 | — | — | 0.0107±0.0000 |
| affine/decoded/marginal | xor_v2_reward | +0.003418 | — | — | 0.0034±0.0000 |
| affine/decoded/marginal | full_occupancy | -0.063965 | — | — | -0.0640±0.0000 |
| affine/decoded/marginal | full_mirror | +0.001465 | — | — | 0.0015±0.0000 |
| mlp/exact/marginal | ladder_occupancy | -0.023926 | — | — | -0.0239±0.0000 |
| mlp/exact/marginal | ladder_mirror | -0.000366 | — | — | -0.0004±0.0000 |
| mlp/exact/marginal | moving_reward | +0.018433 | — | — | 0.0184±0.0000 |
| mlp/exact/marginal | moving_f1_changed | -0.039185 | — | — | -0.0392±0.0000 |
| mlp/exact/marginal | xor_eps02_reward | +0.019531 | — | — | 0.0195±0.0000 |
| mlp/exact/marginal | xor_v2_reward | +0.085938 | — | — | 0.0859±0.0000 |
| mlp/exact/marginal | full_occupancy | -0.022949 | — | — | -0.0229±0.0000 |
| mlp/exact/marginal | full_mirror | -0.054199 | — | — | -0.0542±0.0000 |
| mlp/decoded/marginal | ladder_occupancy | -0.033081 | — | — | -0.0331±0.0000 |
| mlp/decoded/marginal | ladder_mirror | +0.022461 | — | — | 0.0225±0.0000 |
| mlp/decoded/marginal | moving_reward | +0.037598 | — | — | 0.0376±0.0000 |
| mlp/decoded/marginal | moving_f1_changed | -0.055908 | — | — | -0.0559±0.0000 |
| mlp/decoded/marginal | xor_eps02_reward | -0.057617 | — | — | -0.0576±0.0000 |
| mlp/decoded/marginal | xor_v2_reward | +0.159668 | — | — | 0.1597±0.0000 |
| mlp/decoded/marginal | full_occupancy | -0.022949 | — | — | -0.0229±0.0000 |
| mlp/decoded/marginal | full_mirror | -0.048340 | — | — | -0.0483±0.0000 |
| affine/decoded/joint | moving_reward | +0.016968 | — | — | 0.0170±0.0000 |
| affine/decoded/joint | moving_f1_changed | -0.017822 | — | — | -0.0178±0.0000 |
| affine/decoded/joint | xor_eps02_reward | +0.000000 | — | — | 0.0000±0.0000 |
| affine/decoded/joint | xor_v2_reward | +0.000000 | — | — | 0.0000±0.0000 |
| mlp/exact/joint | moving_reward | -0.109375 | — | — | -0.1094±0.0000 |
| mlp/exact/joint | moving_f1_changed | +0.059204 | — | — | 0.0592±0.0000 |
| mlp/exact/joint | xor_eps02_reward | +0.000000 | — | — | 0.0000±0.0000 |
| mlp/exact/joint | xor_v2_reward | -0.043945 | — | — | -0.0439±0.0000 |
| mlp/decoded/joint | moving_reward | +0.004028 | — | — | 0.0040±0.0000 |
| mlp/decoded/joint | moving_f1_changed | -0.000488 | — | — | -0.0005±0.0000 |
| mlp/decoded/joint | xor_eps02_reward | +0.000000 | — | — | 0.0000±0.0000 |
| mlp/decoded/joint | xor_v2_reward | -0.012207 | — | — | -0.0122±0.0000 |

## All four arms reselected with the historical grid/rule on 16 episodes

| Encoder / source / coordinates | Ladder ↓ / ↓ | Moving reward ↑ | Moving gap ↑ | Moving F1 changed ↓ | XOR eps02 ↑ | XOR v2 ↑ | Full-State |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine/exact/marginal | 0.1058±0.0000 / 0.0105±0.0000 | 0.9600±0.0000 | — | 0.1313±0.0000 | 0.7104±0.0000 | 0.6064±0.0000 | 0.1406±0.0000 / 0.5596±0.0000 |
| affine/decoded/marginal | 0.0861±0.0000 / 0.0177±0.0000 | 0.9838±0.0000 | — | 0.1251±0.0000 | 0.7104±0.0000 | 0.6553±0.0000 | 0.0806±0.0000 / 0.5610±0.0000 |
| mlp/exact/marginal | 0.1023±0.0000 / 0.0101±0.0000 | 0.9038±0.0000 | — | 0.0507±0.0000 | 0.6602±0.0000 | 0.5366±0.0000 | 0.1177±0.0000 / 0.5054±0.0000 |
| mlp/decoded/marginal | 0.0951±0.0000 / 0.0330±0.0000 | 0.9492±0.0000 | — | 0.0924±0.0000 | 0.5830±0.0000 | 0.7832±0.0000 | 0.1177±0.0000 / 0.5112±0.0000 |
| affine/exact/joint | 0.1058±0.0000 / 0.0105±0.0000 | 0.9470±0.0000 | — | 0.1360±0.0000 | 0.6167±0.0000 | 0.6445±0.0000 | 0.1406±0.0000 / 0.5596±0.0000 |
| affine/decoded/joint | 0.0861±0.0000 / 0.0177±0.0000 | 0.9766±0.0000 | — | 0.1125±0.0000 | 0.5791±0.0000 | 0.6499±0.0000 | 0.0806±0.0000 / 0.5610±0.0000 |
| mlp/exact/joint | 0.1023±0.0000 / 0.0101±0.0000 | 0.9050±0.0000 | — | 0.1339±0.0000 | 0.7095±0.0000 | 0.6484±0.0000 | 0.1177±0.0000 / 0.5054±0.0000 |
| mlp/decoded/joint | 0.0951±0.0000 / 0.0330±0.0000 | 1.0157±0.0000 | — | 0.0642±0.0000 | 0.5640±0.0000 | 0.6406±0.0000 | 0.1177±0.0000 / 0.5112±0.0000 |

### Paired differences versus historical affine/exact at its archived λ

Signs are raw new−historical differences (negative favors occupancy/spillover; positive favors reward/gap). Full-State mirror is nondirectional.

| Arm | Metric | Δ seed42 | Δ seed43 | Δ seed44 | Mean ± SD |
|---|---|---:|---:|---:|---:|
| affine/exact/marginal | ladder_occupancy | -0.022339 | — | — | -0.0223±0.0000 |
| affine/exact/marginal | ladder_mirror | +0.000000 | — | — | 0.0000±0.0000 |
| affine/exact/marginal | moving_reward | +0.000000 | — | — | 0.0000±0.0000 |
| affine/exact/marginal | moving_f1_changed | +0.000000 | — | — | 0.0000±0.0000 |
| affine/exact/marginal | xor_eps02_reward | +0.069824 | — | — | 0.0698±0.0000 |
| affine/exact/marginal | xor_v2_reward | +0.054688 | — | — | 0.0547±0.0000 |
| affine/exact/marginal | full_occupancy | +0.000000 | — | — | 0.0000±0.0000 |
| affine/exact/marginal | full_mirror | +0.000000 | — | — | 0.0000±0.0000 |
| affine/decoded/marginal | ladder_occupancy | -0.042114 | — | — | -0.0421±0.0000 |
| affine/decoded/marginal | ladder_mirror | +0.007202 | — | — | 0.0072±0.0000 |
| affine/decoded/marginal | moving_reward | +0.023804 | — | — | 0.0238±0.0000 |
| affine/decoded/marginal | moving_f1_changed | -0.006226 | — | — | -0.0062±0.0000 |
| affine/decoded/marginal | xor_eps02_reward | +0.069824 | — | — | 0.0698±0.0000 |
| affine/decoded/marginal | xor_v2_reward | +0.103516 | — | — | 0.1035±0.0000 |
| affine/decoded/marginal | full_occupancy | -0.060059 | — | — | -0.0601±0.0000 |
| affine/decoded/marginal | full_mirror | +0.001465 | — | — | 0.0015±0.0000 |
| mlp/exact/marginal | ladder_occupancy | -0.025879 | — | — | -0.0259±0.0000 |
| mlp/exact/marginal | ladder_mirror | -0.000366 | — | — | -0.0004±0.0000 |
| mlp/exact/marginal | moving_reward | -0.056152 | — | — | -0.0562±0.0000 |
| mlp/exact/marginal | moving_f1_changed | -0.080688 | — | — | -0.0807±0.0000 |
| mlp/exact/marginal | xor_eps02_reward | +0.019531 | — | — | 0.0195±0.0000 |
| mlp/exact/marginal | xor_v2_reward | -0.015137 | — | — | -0.0151±0.0000 |
| mlp/exact/marginal | full_occupancy | -0.022949 | — | — | -0.0229±0.0000 |
| mlp/exact/marginal | full_mirror | -0.054199 | — | — | -0.0542±0.0000 |
| mlp/decoded/marginal | ladder_occupancy | -0.033081 | — | — | -0.0331±0.0000 |
| mlp/decoded/marginal | ladder_mirror | +0.022461 | — | — | 0.0225±0.0000 |
| mlp/decoded/marginal | moving_reward | -0.010742 | — | — | -0.0107±0.0000 |
| mlp/decoded/marginal | moving_f1_changed | -0.038940 | — | — | -0.0389±0.0000 |
| mlp/decoded/marginal | xor_eps02_reward | -0.057617 | — | — | -0.0576±0.0000 |
| mlp/decoded/marginal | xor_v2_reward | +0.231445 | — | — | 0.2314±0.0000 |
| mlp/decoded/marginal | full_occupancy | -0.022949 | — | — | -0.0229±0.0000 |
| mlp/decoded/marginal | full_mirror | -0.048340 | — | — | -0.0483±0.0000 |
| affine/exact/joint | moving_reward | -0.067383 | — | — | -0.0674±0.0000 |
| affine/exact/joint | moving_f1_changed | +0.061279 | — | — | 0.0613±0.0000 |
| affine/exact/joint | xor_eps02_reward | +0.021973 | — | — | 0.0220±0.0000 |
| affine/exact/joint | xor_v2_reward | -0.047852 | — | — | -0.0479±0.0000 |
| affine/decoded/joint | moving_reward | -0.037842 | — | — | -0.0378±0.0000 |
| affine/decoded/joint | moving_f1_changed | +0.037842 | — | — | 0.0378±0.0000 |
| affine/decoded/joint | xor_eps02_reward | -0.015625 | — | — | -0.0156±0.0000 |
| affine/decoded/joint | xor_v2_reward | -0.042480 | — | — | -0.0425±0.0000 |
| mlp/exact/joint | moving_reward | -0.109375 | — | — | -0.1094±0.0000 |
| mlp/exact/joint | moving_f1_changed | +0.059204 | — | — | 0.0592±0.0000 |
| mlp/exact/joint | xor_eps02_reward | +0.114746 | — | — | 0.1147±0.0000 |
| mlp/exact/joint | xor_v2_reward | -0.043945 | — | — | -0.0439±0.0000 |
| mlp/decoded/joint | moving_reward | +0.001343 | — | — | 0.0013±0.0000 |
| mlp/decoded/joint | moving_f1_changed | -0.010498 | — | — | -0.0105±0.0000 |
| mlp/decoded/joint | xor_eps02_reward | -0.030762 | — | — | -0.0308±0.0000 |
| mlp/decoded/joint | xor_v2_reward | -0.051758 | — | — | -0.0518±0.0000 |
