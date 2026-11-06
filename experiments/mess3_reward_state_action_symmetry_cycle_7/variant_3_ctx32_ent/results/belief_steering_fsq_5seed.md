# Full-State Quotient belief steering

Occupancy is decision-time reward-state occupancy, including reset decisions.
Actions: Noop, Tilt X, Tilt Y. States: 1, 2, 3 (reward state).

Reference: N=60 nearest-grid, gamma=0.97, 800 iterations; near-optimal, not certified Bayes-optimal.
Raw affine decoded beliefs are not clipped. Point ties choose state 2.
Warmup is eight steps only for probe fitting/testing (baseline protocol); the banded four-layer context-32 encoder can depend on up to 128 observations.
Shadow forward uses the current executed cache and is discarded; earlier shifts affect subsequent-block caches only.
Selected strengths: {"42": {"point": 0.55, "mirror": 0.75}, "43": {"point": 0.5, "mirror": 0.6}, "44": {"point": 0.55, "mirror": 0.75}, "45": {"point": 0.5, "mirror": 0.75}, "46": {"point": 0.5, "mirror": 0.75}}

Uncertainty: standard error across five independent training-seed means, not episode SE

| Condition | Occupancy ± SE |
|---|---:|
| block_1/point | 0.119318 ± 0.005646 |
| block_2/point | 0.121674 ± 0.014947 |
| block_3/point | 0.125433 ± 0.017740 |
| block_4/point | 0.123120 ± 0.015884 |
| post_final_norm/point | 0.145770 ± 0.020384 |
| orthogonal/point | 0.313019 ± 0.000000 |
| block_1/mirror | 0.091620 ± 0.008778 |
| block_2/mirror | 0.093903 ± 0.009411 |
| block_3/mirror | 0.092706 ± 0.014678 |
| block_4/mirror | 0.101343 ± 0.015258 |
| post_final_norm/mirror | 0.113727 ± 0.019754 |
| orthogonal/mirror | 0.312927 ± 0.000071 |
| intact | 0.313019 ± 0.000000 |
| random | 0.162842 ± 0.000000 |

## Final pre-LayerNorm curves

| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |
|---:|---:|---:|---:|---:|
| 0 | 0.313019 ± 0.000000 | 0.313019 ± 0.000000 | 0.311584 | 0.311584 |
| 0.25 | 0.313007 ± 0.000012 | 0.313007 ± 0.000012 | 0.311584 | 0.311584 |
| 0.4 | 0.309052 ± 0.000980 | 0.312408 ± 0.000235 | 0.289948 | 0.311584 |
| 0.45 | 0.262714 ± 0.005943 | 0.302075 ± 0.002710 | 0.272522 | 0.311584 |
| 0.5 | 0.163556 ± 0.012747 | 0.235419 ± 0.005324 | 0.071381 | 0.183350 |
| 0.55 | 0.084430 ± 0.002717 | 0.206787 ± 0.003605 | 0.070862 | 0.076202 |
| 0.6 | 0.074353 ± 0.000314 | 0.191400 ± 0.008141 | 0.069427 | 0.076202 |
| 0.75 | 0.073639 ± 0.000000 | 0.084540 ± 0.003858 | 0.069122 | 0.069427 |
| 0.9 | 0.073639 ± 0.000000 | 0.074585 ± 0.000630 | 0.068787 | 0.069336 |
| 1 | 0.073639 ± 0.000000 | 0.073785 ± 0.000118 | 0.068726 | 0.069122 |
