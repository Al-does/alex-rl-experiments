# Full-State Quotient belief steering

Occupancy is decision-time reward-state occupancy, including reset decisions.
Actions: Noop, Tilt X, Tilt Y. States: 1, 2, 3 (reward state).

Reference: N=60 nearest-grid, gamma=0.97, 800 iterations; near-optimal, not certified Bayes-optimal.
Raw affine decoded beliefs are not clipped. Point ties choose state 2.
Warmup is eight steps only for probe fitting/testing (baseline protocol); the banded four-layer context-32 encoder can depend on up to 128 observations.
Shadow forward uses the current executed cache and is discarded; earlier shifts affect subsequent-block caches only.
Selected strengths: {"point": 0.55, "mirror": 0.75}

Uncertainty: episode standard error within one frozen training seed

| Condition | Occupancy ± SE |
|---|---:|
| block_1/point | 0.106842 ± 0.002430 |
| block_2/point | 0.099426 ± 0.002743 |
| block_3/point | 0.086060 ± 0.002525 |
| block_4/point | 0.085602 ± 0.002066 |
| post_final_norm/point | 0.106323 ± 0.002059 |
| orthogonal/point | 0.313019 ± 0.002399 |
| block_1/mirror | 0.081848 ± 0.002217 |
| block_2/mirror | 0.081421 ± 0.002202 |
| block_3/mirror | 0.077240 ± 0.001994 |
| block_4/mirror | 0.083801 ± 0.001795 |
| post_final_norm/mirror | 0.084595 ± 0.001659 |
| orthogonal/mirror | 0.313019 ± 0.002399 |
| intact | 0.313019 ± 0.002399 |
| random | 0.162842 ± 0.001746 |

## Final pre-LayerNorm curves

| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |
|---:|---:|---:|---:|---:|
| 0 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.25 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.4 | 0.309113 ± 0.002268 | 0.311584 ± 0.002341 | 0.289948 | 0.311584 |
| 0.45 | 0.274628 ± 0.002515 | 0.293854 ± 0.002239 | 0.272522 | 0.311584 |
| 0.5 | 0.184906 ± 0.002340 | 0.241486 ± 0.002195 | 0.071381 | 0.183350 |
| 0.55 | 0.085602 ± 0.002066 | 0.214630 ± 0.002588 | 0.070862 | 0.076202 |
| 0.6 | 0.073639 ± 0.001682 | 0.202057 ± 0.002365 | 0.069427 | 0.076202 |
| 0.75 | 0.073639 ± 0.001682 | 0.083801 ± 0.001795 | 0.069122 | 0.069427 |
| 0.9 | 0.073639 ± 0.001682 | 0.073639 ± 0.001682 | 0.068787 | 0.069336 |
| 1 | 0.073639 ± 0.001682 | 0.073639 ± 0.001682 | 0.068726 | 0.069122 |
