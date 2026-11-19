# RockSample r7 marginal belief steering

## Interpretation: mixed intervention effects, not a clean belief flip

The requested strong, selective two-way steering demonstration is **not established**.
Good-rock pickup is strongly suppressed for six of seven primary-agent rocks,
but the decoder-null controls also suppress pickup, and other-rock sampling drops.
Bad-rock pickup increases substantially for primary-agent rocks 0 and 5 (and more
modestly 1 and 2), but fails for rocks 3, 4, and 6. Rock 0's Bad→Good increase
replicates across all three agents: 0.624, 0.624, and 0.938, versus intact rates
0.009, 0.009, and 0.014. These are causal effects of activation intervention,
not proof that an otherwise intact policy simply adopted the requested belief.

The tuning threshold was **not met at any grid point**. λ=2 is the predeclared
fallback endpoint, not a successful threshold crossing. It reduces pooled
Good-rock pickup to 0.1038 in tuning, still above intact Bad pickup 0.0089.
No confirmation outcome was used to change this choice.

The intended decoder sanity check also fails as an endpoint-matching claim:
mean post-shift target decoding is -0.477 for Good→Bad and 1.607 for Bad→Good
at the primary site, rather than 0 and 1. With separately fitted encoder and
decoder, λ=2 overshoots on these histories; it is not an exact decoder inverse.
These directly measured failures, plus disruption by decoder-null controls,
preclude interpreting absent pickup as selective causal belief steering.
The reason particular rocks remain resistant is not identified here; these
results do not distinguish nonlinear/off-manifold effects from navigation and
action-selection changes. Rock 2 remains frequently sampled when Good, while
rocks 4 and 6 essentially never become sampled when Bad in any replicate.

## Budgets and primary-site diagnostics

Each agent uses 256 complete own-policy fit episodes and 256 independent test
episodes. Tuning uses 128 episodes per rock/direction for each strength;
confirmation uses **1,024 episodes per rock/direction/condition/site/agent**.
The seven target-rock conditions each force just that rock's initial quality;
the other six are independent fair draws. Per-rock indices below are literal
implementation indices, consistent with the archived RockSample reports.
Checkpoints and canonical durability manifests are in ignored `artifacts/`;
all three final-update checkpoint byte counts and SHA-256 checks passed.
This analysis ran on this VM's CPU, without training or paid compute.

Primary agent, final block pre-LN. Diagnostics average equally across the seven
target-rock experiments. Other-rock pickup is the mean of the six nontarget
rock sampling probabilities, not conditioned on their initial quality. Decoded
means include only intervention-eligible valid decision steps (Bad→Good stops
at first pickup); intact/control means use the same eligibility convention.

| Direction | Condition | Other-rock pickup | Target checks/episode | Return | Decoded before | Decoded after |
|---|---|---:|---:|---:|---:|---:|
| Good→Bad | intact | 0.500 | 3.53 | 48.83 | 0.378 | 0.378 |
| Good→Bad | steered | 0.195 | 5.26 | 17.92 | 0.574 | -0.477 |
| Good→Bad | control | 0.261 | 4.13 | 5.48 | 0.522 | 0.522 |
| Bad→Good | intact | 0.499 | 3.15 | 39.00 | 0.139 | 0.139 |
| Bad→Good | steered | 0.224 | 5.77 | -39.81 | 0.210 | 1.607 |
| Bad→Good | control | 0.164 | 3.74 | -17.38 | 0.237 | 0.237 |

`results.json` retains each individual nontarget-rock rate, Wilson interval,
target check count, return and decoded before/after mean for every condition,
site and agent. Negative returns and broad nontarget suppression are reported,
not discarded. The optional 128-configuration joint intervention was not run.
`pickup_probability.png` shows the primary site's per-rock confirmation rates
and binomial 95% intervals for the primary agent.

## Protocol and complete tables

Frozen r7 update-29 sampled policies, independent fit/test streams; all valid decision rows including reset priors, no warmup exclusion. Separate affine least-squares decoder and encoder per site (SVD rcond 1e-6). Tune the smallest grid λ whose pooled Good→Bad pickup is below intact Bad pickup; if none qualifies, use the grid endpoint and explicitly report selection failure. Freeze across agents/sites. Matched environment/action RNG across conditions. Bad→Good stops after the first target-rock sample; Good→Bad continues every step. K/V propagation is preserved.

Selected λ = 2; selection criterion met: False.

| Tuning λ | Steered Good pickup | Intact Bad pickup |
|---|---:|---:|
| 0 | 0.9810 | 0.0089 |
| 0.25 | 0.8650 | 0.0089 |
| 0.5 | 0.6752 | 0.0089 |
| 0.75 | 0.5223 | 0.0089 |
| 1 | 0.3281 | 0.0089 |
| 1.5 | 0.1496 | 0.0089 |
| 2 | 0.1038 | 0.0089 |

## 20261002T191712Z-7f1dc1ef seed 3

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.979 [0.969, 0.987] | 0.010 [0.005, 0.018] | 0.044 [0.033, 0.058] | 0.009 [0.005, 0.017] | 0.624 [0.594, 0.653] | 0.009 [0.005, 0.017] |
| 1 | 0.989 [0.981, 0.994] | 0.024 [0.017, 0.036] | 0.115 [0.097, 0.136] | 0.021 [0.013, 0.031] | 0.156 [0.135, 0.180] | 0.012 [0.007, 0.020] |
| 2 | 0.966 [0.953, 0.975] | 0.661 [0.632, 0.689] | 0.438 [0.407, 0.468] | 0.021 [0.014, 0.032] | 0.228 [0.203, 0.254] | 0.042 [0.031, 0.056] |
| 3 | 0.977 [0.965, 0.984] | 0.015 [0.009, 0.024] | 0.000 [0.000, 0.004] | 0.007 [0.003, 0.014] | 0.010 [0.005, 0.018] | 0.002 [0.001, 0.007] |
| 4 | 0.984 [0.975, 0.990] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.004 [0.002, 0.010] | 0.001 [0.000, 0.006] | 0.000 [0.000, 0.004] |
| 5 | 0.977 [0.965, 0.984] | 0.000 [0.000, 0.004] | 0.083 [0.068, 0.102] | 0.018 [0.011, 0.028] | 0.827 [0.803, 0.849] | 0.006 [0.003, 0.013] |
| 6 | 0.989 [0.981, 0.994] | 0.000 [0.000, 0.004] | 0.215 [0.191, 0.241] | 0.009 [0.005, 0.017] | 0.002 [0.001, 0.007] | 0.027 [0.019, 0.039] |

| Site | Good→Bad pickup | Bad→Good pickup | Held-out per-rock R² |
|---|---:|---:|---|
| 0 | 0.412 | 0.325 | 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719 |
| 1 | 0.308 | 0.282 | 0.728, 0.643, 0.628, 0.802, 0.655, 0.753, 0.846 |
| 2 | 0.101 | 0.264 | 0.920, 0.935, 0.776, 0.934, 0.928, 0.746, 0.941 |
| 3 | 0.359 | 0.097 | 0.930, 0.943, 0.790, 0.945, 0.936, 0.770, 0.950 |

## 20261002T182024Z-ad0c3a34 seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.987 [0.978, 0.993] | 0.003 [0.001, 0.009] | 0.689 [0.660, 0.717] | 0.009 [0.005, 0.017] | 0.624 [0.594, 0.653] | 0.008 [0.004, 0.015] |
| 1 | 0.990 [0.982, 0.995] | 0.150 [0.130, 0.174] | 0.154 [0.133, 0.178] | 0.014 [0.008, 0.023] | 0.385 [0.355, 0.415] | 0.002 [0.001, 0.007] |
| 2 | 0.960 [0.946, 0.970] | 0.588 [0.557, 0.618] | 0.857 [0.835, 0.878] | 0.017 [0.010, 0.026] | 0.075 [0.061, 0.093] | 0.021 [0.013, 0.031] |
| 3 | 0.988 [0.980, 0.993] | 0.045 [0.034, 0.059] | 0.065 [0.052, 0.082] | 0.010 [0.005, 0.018] | 0.229 [0.205, 0.256] | 0.015 [0.009, 0.024] |
| 4 | 0.979 [0.969, 0.987] | 0.001 [0.000, 0.006] | 0.000 [0.000, 0.004] | 0.010 [0.005, 0.018] | 0.001 [0.000, 0.006] | 0.000 [0.000, 0.004] |
| 5 | 0.962 [0.948, 0.972] | 0.370 [0.341, 0.400] | 0.833 [0.809, 0.855] | 0.021 [0.014, 0.032] | 0.117 [0.099, 0.138] | 0.124 [0.105, 0.146] |
| 6 | 0.987 [0.978, 0.993] | 0.001 [0.000, 0.006] | 0.000 [0.000, 0.004] | 0.003 [0.001, 0.009] | 0.015 [0.009, 0.024] | 0.000 [0.000, 0.004] |

| Site | Good→Bad pickup | Bad→Good pickup | Held-out per-rock R² |
|---|---:|---:|---|
| 0 | 0.373 | 0.134 | 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539 |
| 1 | 0.234 | 0.226 | 0.753, 0.864, 0.794, 0.688, 0.730, 0.641, 0.735 |
| 2 | 0.165 | 0.207 | 0.916, 0.934, 0.834, 0.916, 0.940, 0.707, 0.938 |
| 3 | 0.346 | 0.111 | 0.923, 0.941, 0.844, 0.923, 0.943, 0.716, 0.946 |

## 20261002T191712Z-7f1dc1ef seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.977 [0.965, 0.984] | 0.009 [0.005, 0.017] | 0.632 [0.602, 0.661] | 0.014 [0.008, 0.023] | 0.938 [0.921, 0.951] | 0.013 [0.007, 0.022] |
| 1 | 0.987 [0.978, 0.993] | 0.063 [0.050, 0.080] | 0.487 [0.457, 0.518] | 0.013 [0.007, 0.022] | 0.162 [0.141, 0.186] | 0.030 [0.021, 0.043] |
| 2 | 0.965 [0.952, 0.974] | 0.725 [0.696, 0.751] | 0.861 [0.839, 0.881] | 0.016 [0.010, 0.025] | 0.067 [0.054, 0.084] | 0.021 [0.013, 0.031] |
| 3 | 0.977 [0.965, 0.984] | 0.200 [0.177, 0.226] | 0.121 [0.103, 0.143] | 0.012 [0.007, 0.020] | 0.019 [0.012, 0.029] | 0.013 [0.007, 0.022] |
| 4 | 0.965 [0.952, 0.974] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.010 [0.005, 0.018] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] |
| 5 | 0.966 [0.953, 0.975] | 0.027 [0.019, 0.039] | 0.634 [0.604, 0.663] | 0.019 [0.012, 0.029] | 0.623 [0.593, 0.652] | 0.121 [0.103, 0.143] |
| 6 | 0.988 [0.980, 0.993] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.015 [0.009, 0.024] | 0.002 [0.001, 0.007] | 0.001 [0.000, 0.006] |

| Site | Good→Bad pickup | Bad→Good pickup | Held-out per-rock R² |
|---|---:|---:|---|
| 0 | 0.354 | 0.197 | 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591 |
| 1 | 0.327 | 0.238 | 0.720, 0.678, 0.706, 0.806, 0.810, 0.623, 0.891 |
| 2 | 0.146 | 0.259 | 0.883, 0.948, 0.748, 0.908, 0.925, 0.729, 0.929 |
| 3 | 0.342 | 0.112 | 0.894, 0.955, 0.764, 0.922, 0.945, 0.751, 0.948 |

Site indices 0–2 are post-block pre-final-LN; site 3 is post-final-LN.
Intervals are marginal Wilson binomial 95% CIs, not training-seed uncertainty.
Random controls use a fixed per-rock decoder-null direction with the norm of the
counterfactual steering shift on their own current history (not a paired trajectory norm).
Raw decoded beliefs are not clipped; separate least-squares encoders are not decoder inverses,
so D(h′) need not equal the requested endpoint. Synthetic shifts can leave the policy manifold.
All initial rock qualities are forced only for behavioral conditioning; decoder targets remain
the ordinary fair-prior action/observation posterior. Bad→Good stops after first pickup;
Good→Bad continues at every decision step.
