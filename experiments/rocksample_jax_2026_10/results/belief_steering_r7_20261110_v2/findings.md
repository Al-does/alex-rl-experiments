# RockSample r7 endpoint-calibrated steering (v2)

## Conclusion

**Endpoint calibration succeeds numerically, but a selective two-way behavioral
belief flip is still not established.** Neither calibrated variant satisfies
the joint tuning criterion anywhere in the sweep, so both use the predeclared
λ=1 fallback. The calibrated encoder is the displayed variant because its
tuning target-contrast/selectivity score (-0.4334) exceeds the historical encoder
(-0.4466) and min-norm decoder (-0.9484), not because confirmation chose it.

At the primary site, calibrated-encoder decoded endpoints average 0.000000 and
1.000000; the maximum target endpoint errors are 3.62e-6 and 2.03e-6, and the
maximum change in any other decoded coordinate is 1.01e-6. However, nontarget
Good pickup falls from 0.9833 to 0.5810 for Good→Bad and from 0.9822 to 0.6476
for Bad→Good. Nontarget Bad pickup rises from 0.0086 to 0.0844 for Bad→Good;
mean return becomes -26.99. Decoder-coordinate isolation therefore does not
imply behavioral selectivity.

Min-norm decoder steering is much more behaviorally selective, but essentially
ineffective on the target: pooled Good pickup is 0.9629 versus intact 0.9834,
and Bad pickup is 0.0103 versus intact 0.0103, despite exact decoded endpoints.
This rules out endpoint miscalibration as the sole explanation of the failed
demonstration. It does not establish whether a different nonlinear or
on-manifold representation is causally used by the policy.

All three upper-plateau r7 agents, all three variants and all four sites were
confirmed with 1,024 episodes per rock/direction/condition on this VM's CPU.
No training or paid compute was used. The first results directory is untouched.
The JSON records the executed source hashes and implementation commit
`3a3b38a76321ed8109ceeccb1d9d62688549311d`; only Markdown table headers changed
after execution, and the separate renderer hash records that reporting change.

## What the resistant-rock rollouts actually do

These observations concern the primary agent, calibrated encoder, λ=1, site 2.
Rates use all held-out episodes; illustrative episodes are fixed episodes 0–1,
not chosen for their outcomes. They identify behavioral bottlenecks, not an
unverified internal causal mechanism.

- **Rock 0:** Bad→Good still reaches the rock in 94.2% of episodes and checks it
  in 100%, but samples in only 44.6%; 49.6% visit without pickup. Episode 0
  repeatedly moves around `(1,0)` and `(2,0)` with 15 target checks and no target
  sample, whereas episode 1 does sample it: this is not just failure to arrive.
- **Rock 1:** Bad→Good visits in 73.8% and samples in 31.1%, while Good→Bad
  visits in 88.1% but still samples in 29.4%. Both first Bad→Good episodes
  reach and sample `(2,1)`, but full-episode rates show many visits without
  pickup and some failures to arrive; successful examples are not universal.
- **Rock 2:** Good→Bad still visits in 99.6% and samples in 93.8%; the first
  example follows the usual route through `(1,2)` and samples it despite a
  zero decoded target. Bad→Good visits in 76.5%, checks in 100% (7.34 target
  checks/episode), yet samples in only 2.8%; both first examples pass through
  `(1,2)` without sampling, then move onward to the eastern/northern rocks.
- **Rock 3:** Bad→Good visits `(2,2)` in only 14.5%, despite checking it in
  99.7%, so failure to approach is a major observed bottleneck. Episode 0
  instead repeatedly samples rock 2 at `(1,2)` 29 times; episode 1 moves
  around the western rocks and finishes sampling at `(0,1)`, off any rock.
- **Rock 4:** Bad→Good visits `(4,2)` in only 1.7% and samples in 0.7%, with
  79.4% checking the target at least once. Episode 0 instead samples rock 0
  at `(1,0)` 32 times, and episode 1 stays in the western grid: these are
  navigation/action-disruption failures, not evidence that the target was
  reached and deliberately rejected in most episodes.
- **Rock 5:** Bad→Good is comparatively effective (83.5% pickup, 100% visit);
  both first examples approach `(0,3)` and sample it. Good→Bad still samples
  in 14.2%, above the 10% threshold; 46.4% visit, and the first example
  passes through `(0,3)` without sampling before roaming eastward.
- **Rock 6:** Bad→Good visits `(3,4)` in only 12.7%, despite checking in
  98.7%, and samples in 7.7%. Episode 0 does reach and sample the target,
  but episode 1 instead samples rock 2 36 times and rock 5 12 times without
  reaching rock 6, illustrating broad action disruption rather than a
  consistent target-quality reversal.

## Measurement protocol

Frozen r7 update-29 sampled policies, 7 current-Good marginals, fair-prior action/observation Bayesian targets; independent own-policy fit/test streams, all valid rows including reset, no warmup exclusion. Separate affine least squares (rcond 1e-6). Calibrated encoder = solve(EW,E); min-norm decoder = pseudoinverse(W). For each calibrated variant choose the smallest tuning λ with pooled Good→Bad pickup ≤0.10 AND Bad→Good pickup ≥0.5, else λ=1. Original encoder retains historical fallback λ=2. Choose displayed variant on tuning only by (Bad→Good pickup minus Good→Bad pickup minus maximum absolute nontarget conditional-rate drift from intact across directions/qualities); ties follow encoder, calibrated encoder, min-norm order. Freeze across agents/sites. Matched environment/action RNG across conditions; Bad→Good stops at first target pickup, Good→Bad continues every decision step. Cached K/V propagation preserved.

Displayed variant: **calibrated_encoder**, chosen using tuning only.

Fit/test: 256/256 episodes per agent; tuning: 128; confirmation: 1,024 per
rock/direction/condition/variant/site (smoke uses 8). Fresh confirmation seed 91423.
Endpoints are measured at the intervened site, before downstream nonlinearities.
Sites 0–2 = post-block pre-final-LN; site 3 = post-final-LN.

## Tuning evidence

| Variant | λ | Good→Bad pickup | Bad→Good pickup |
|---|---:|---:|---:|
| encoder | 2.0 | 0.1038 | 0.2556 |
Selected encoder: λ=2.0; threshold met: False; tuning score=-0.4466.

| Variant | λ | Good→Bad pickup | Bad→Good pickup |
|---|---:|---:|---:|
| calibrated_encoder | 0.5 | 0.6607 | 0.1105 |
| calibrated_encoder | 1 | 0.2712 | 0.2422 |
| calibrated_encoder | 1.5 | 0.1451 | 0.3002 |
| calibrated_encoder | 2 | 0.1205 | 0.2277 |
| calibrated_encoder | 3 | 0.0692 | 0.1864 |
Selected calibrated_encoder: λ=1.0; threshold met: False; tuning score=-0.4334.

| Variant | λ | Good→Bad pickup | Bad→Good pickup |
|---|---:|---:|---:|
| min_norm_decoder | 0.5 | 0.9754 | 0.0089 |
| min_norm_decoder | 1 | 0.9531 | 0.0078 |
| min_norm_decoder | 1.5 | 0.9330 | 0.0145 |
| min_norm_decoder | 2 | 0.8772 | 0.0357 |
| min_norm_decoder | 3 | 0.6473 | 0.0480 |
Selected min_norm_decoder: λ=1.0; threshold met: False; tuning score=-0.9484.

## Primary-agent variant comparison (site 2)

| Variant | λ | G→B pickup | B→G pickup | G→B decoded after | B→G decoded after |
|---|---:|---:|---:|---:|---:|
| encoder | 2.0 | 0.1002 | 0.2543 | -0.477571 | 1.606886 |
| calibrated_encoder | 1.0 | 0.2886 | 0.2578 | 0.000000 | 1.000000 |
| min_norm_decoder | 1.0 | 0.9629 | 0.0103 | -0.000000 | 1.000000 |

## Primary-agent selectivity, return and endpoint sanity (site 2)

| Variant | Direction | Condition | Other Good | Other Bad | Return | Decoded before→after | Max endpoint error | Max other-coordinate change |
|---|---|---|---:|---:|---:|---|---:|---:|
| encoder | Good→Bad | intact | 0.9833 | 0.0097 | 49.27 | 0.35535→0.35535 | 1.51 | 0 |
| encoder | Good→Bad | steered | 0.3840 | 0.0032 | 17.96 | 0.57393→-0.47757 | 1.27 | 0.154 |
| encoder | Good→Bad | control | 0.4440 | 0.0746 | 5.20 | 0.52545→0.52545 | 1.57 | 8.34e-07 |
| encoder | Bad→Good | intact | 0.9822 | 0.0086 | 39.36 | 0.17783→0.17783 | 1.48 | 0 |
| encoder | Bad→Good | steered | 0.3846 | 0.0524 | -41.45 | 0.20964→1.60689 | 1.25 | 0.146 |
| encoder | Bad→Good | control | 0.2408 | 0.0844 | -17.52 | 0.23755→0.23755 | 1.51 | 7.75e-07 |
| calibrated_encoder | Good→Bad | intact | 0.9833 | 0.0097 | 49.27 | 0.35535→0.35535 | 1.51 | 0 |
| calibrated_encoder | Good→Bad | steered | 0.5810 | 0.0043 | 25.40 | 0.61419→0.00000 | 3.62e-06 | 1.01e-06 |
| calibrated_encoder | Good→Bad | control | 0.6307 | 0.0501 | 24.02 | 0.51995→0.51995 | 1.66 | 3.87e-07 |
| calibrated_encoder | Bad→Good | intact | 0.9822 | 0.0086 | 39.36 | 0.17783→0.17783 | 1.48 | 0 |
| calibrated_encoder | Bad→Good | steered | 0.6476 | 0.0844 | -26.99 | 0.16570→1.00000 | 2.03e-06 | 7.75e-07 |
| calibrated_encoder | Bad→Good | control | 0.4062 | 0.0526 | 11.49 | 0.18349→0.18349 | 1.36 | 4.17e-07 |
| min_norm_decoder | Good→Bad | intact | 0.9833 | 0.0097 | 49.27 | 0.35535→0.35535 | 1.51 | 0 |
| min_norm_decoder | Good→Bad | steered | 0.9821 | 0.0087 | 49.00 | 0.40192→-0.00000 | 4.96e-06 | 3.64e-07 |
| min_norm_decoder | Good→Bad | control | 0.9655 | 0.0125 | 46.94 | 0.42852→0.42852 | 1.53 | 4.44e-07 |
| min_norm_decoder | Bad→Good | intact | 0.9822 | 0.0086 | 39.36 | 0.17783→0.17783 | 1.48 | 0 |
| min_norm_decoder | Bad→Good | steered | 0.9746 | 0.0113 | 38.76 | 0.10320→1.00000 | 4.53e-06 | 3.65e-07 |
| min_norm_decoder | Bad→Good | control | 0.9486 | 0.0155 | 37.25 | 0.12691→0.12691 | 1.33 | 3.8e-07 |

## Best variant: 20261002T191712Z-7f1dc1ef seed 3

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.987 [0.978, 0.993] | 0.425 [0.395, 0.455] | 0.621 [0.591, 0.650] | 0.009 [0.005, 0.017] | 0.446 [0.416, 0.477] | 0.016 [0.010, 0.025] |
| 1 | 0.985 [0.976, 0.991] | 0.294 [0.267, 0.323] | 0.792 [0.766, 0.816] | 0.012 [0.007, 0.020] | 0.311 [0.283, 0.340] | 0.009 [0.005, 0.017] |
| 2 | 0.981 [0.971, 0.988] | 0.938 [0.921, 0.951] | 0.604 [0.573, 0.633] | 0.019 [0.012, 0.029] | 0.028 [0.020, 0.040] | 0.046 [0.035, 0.061] |
| 3 | 0.980 [0.970, 0.987] | 0.216 [0.192, 0.242] | 0.009 [0.005, 0.017] | 0.014 [0.008, 0.023] | 0.101 [0.084, 0.121] | 0.003 [0.001, 0.009] |
| 4 | 0.985 [0.976, 0.991] | 0.000 [0.000, 0.004] | 0.000 [0.000, 0.004] | 0.003 [0.001, 0.009] | 0.007 [0.003, 0.014] | 0.000 [0.000, 0.004] |
| 5 | 0.977 [0.965, 0.984] | 0.142 [0.122, 0.164] | 0.452 [0.422, 0.483] | 0.012 [0.007, 0.020] | 0.835 [0.811, 0.856] | 0.010 [0.005, 0.018] |
| 6 | 0.987 [0.978, 0.993] | 0.007 [0.003, 0.014] | 0.798 [0.772, 0.821] | 0.005 [0.002, 0.011] | 0.077 [0.062, 0.095] | 0.051 [0.039, 0.066] |

| Variant | Site | G→B pickup | B→G pickup | R² by rock |
|---|---|---:|---:|---|
| encoder | 0 | 0.409 | 0.327 | 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719 |
| encoder | 1 | 0.313 | 0.279 | 0.728, 0.643, 0.628, 0.802, 0.655, 0.753, 0.846 |
| encoder | 2 | 0.100 | 0.254 | 0.920, 0.935, 0.776, 0.934, 0.928, 0.746, 0.941 |
| encoder | 3 | 0.360 | 0.096 | 0.930, 0.943, 0.790, 0.945, 0.936, 0.770, 0.950 |
| calibrated_encoder | 0 | 0.489 | 0.352 | 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719 |
| calibrated_encoder | 1 | 0.568 | 0.236 | 0.728, 0.643, 0.628, 0.802, 0.655, 0.753, 0.846 |
| calibrated_encoder | 2 | 0.289 | 0.258 | 0.920, 0.935, 0.776, 0.934, 0.928, 0.746, 0.941 |
| calibrated_encoder | 3 | 0.739 | 0.067 | 0.930, 0.943, 0.790, 0.945, 0.936, 0.770, 0.950 |
| min_norm_decoder | 0 | 0.978 | 0.011 | 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719 |
| min_norm_decoder | 1 | 0.962 | 0.011 | 0.728, 0.643, 0.628, 0.802, 0.655, 0.753, 0.846 |
| min_norm_decoder | 2 | 0.963 | 0.010 | 0.920, 0.935, 0.776, 0.934, 0.928, 0.746, 0.941 |
| min_norm_decoder | 3 | 0.960 | 0.009 | 0.930, 0.943, 0.790, 0.945, 0.936, 0.770, 0.950 |

## Best variant: 20261002T182024Z-ad0c3a34 seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.989 [0.981, 0.994] | 0.322 [0.294, 0.352] | 0.959 [0.945, 0.970] | 0.007 [0.003, 0.014] | 0.700 [0.671, 0.727] | 0.010 [0.005, 0.018] |
| 1 | 0.991 [0.983, 0.995] | 0.761 [0.734, 0.786] | 0.694 [0.665, 0.722] | 0.009 [0.005, 0.017] | 0.163 [0.142, 0.187] | 0.008 [0.004, 0.015] |
| 2 | 0.963 [0.949, 0.973] | 0.937 [0.920, 0.950] | 0.940 [0.924, 0.953] | 0.019 [0.012, 0.029] | 0.041 [0.030, 0.055] | 0.022 [0.015, 0.033] |
| 3 | 0.981 [0.971, 0.988] | 0.290 [0.263, 0.319] | 0.338 [0.310, 0.367] | 0.014 [0.008, 0.023] | 0.302 [0.274, 0.331] | 0.022 [0.015, 0.033] |
| 4 | 0.987 [0.978, 0.993] | 0.002 [0.001, 0.007] | 0.025 [0.017, 0.037] | 0.009 [0.005, 0.017] | 0.009 [0.005, 0.017] | 0.006 [0.003, 0.013] |
| 5 | 0.976 [0.964, 0.983] | 0.668 [0.639, 0.696] | 0.927 [0.909, 0.941] | 0.013 [0.007, 0.022] | 0.084 [0.069, 0.103] | 0.105 [0.088, 0.126] |
| 6 | 0.988 [0.980, 0.993] | 0.005 [0.002, 0.011] | 0.004 [0.002, 0.010] | 0.004 [0.002, 0.010] | 0.120 [0.102, 0.141] | 0.000 [0.000, 0.004] |

| Variant | Site | G→B pickup | B→G pickup | R² by rock |
|---|---|---:|---:|---|
| encoder | 0 | 0.372 | 0.131 | 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539 |
| encoder | 1 | 0.247 | 0.232 | 0.753, 0.864, 0.794, 0.688, 0.730, 0.641, 0.735 |
| encoder | 2 | 0.170 | 0.200 | 0.916, 0.934, 0.834, 0.916, 0.940, 0.707, 0.938 |
| encoder | 3 | 0.356 | 0.127 | 0.923, 0.941, 0.844, 0.923, 0.943, 0.716, 0.946 |
| calibrated_encoder | 0 | 0.473 | 0.139 | 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539 |
| calibrated_encoder | 1 | 0.557 | 0.109 | 0.753, 0.864, 0.794, 0.688, 0.730, 0.641, 0.735 |
| calibrated_encoder | 2 | 0.426 | 0.203 | 0.916, 0.934, 0.834, 0.916, 0.940, 0.707, 0.938 |
| calibrated_encoder | 3 | 0.739 | 0.056 | 0.923, 0.941, 0.844, 0.923, 0.943, 0.716, 0.946 |
| min_norm_decoder | 0 | 0.969 | 0.012 | 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539 |
| min_norm_decoder | 1 | 0.947 | 0.014 | 0.753, 0.864, 0.794, 0.688, 0.730, 0.641, 0.735 |
| min_norm_decoder | 2 | 0.945 | 0.013 | 0.916, 0.934, 0.834, 0.916, 0.940, 0.707, 0.938 |
| min_norm_decoder | 3 | 0.943 | 0.010 | 0.923, 0.941, 0.844, 0.923, 0.943, 0.716, 0.946 |

## Best variant: 20261002T191712Z-7f1dc1ef seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.966 [0.953, 0.975] | 0.334 [0.306, 0.363] | 0.934 [0.917, 0.947] | 0.010 [0.005, 0.018] | 0.573 [0.543, 0.603] | 0.011 [0.006, 0.019] |
| 1 | 0.985 [0.976, 0.991] | 0.543 [0.512, 0.573] | 0.956 [0.942, 0.967] | 0.009 [0.005, 0.017] | 0.165 [0.144, 0.189] | 0.024 [0.017, 0.036] |
| 2 | 0.978 [0.967, 0.985] | 0.941 [0.925, 0.954] | 0.936 [0.919, 0.949] | 0.010 [0.005, 0.018] | 0.013 [0.007, 0.022] | 0.011 [0.006, 0.019] |
| 3 | 0.978 [0.967, 0.985] | 0.608 [0.578, 0.638] | 0.433 [0.403, 0.463] | 0.007 [0.003, 0.014] | 0.132 [0.112, 0.154] | 0.030 [0.021, 0.043] |
| 4 | 0.986 [0.977, 0.992] | 0.004 [0.002, 0.010] | 0.000 [0.000, 0.004] | 0.011 [0.006, 0.019] | 0.002 [0.001, 0.007] | 0.008 [0.004, 0.015] |
| 5 | 0.975 [0.963, 0.983] | 0.446 [0.416, 0.477] | 0.860 [0.838, 0.880] | 0.014 [0.008, 0.023] | 0.470 [0.439, 0.500] | 0.075 [0.061, 0.093] |
| 6 | 0.991 [0.983, 0.995] | 0.000 [0.000, 0.004] | 0.007 [0.003, 0.014] | 0.008 [0.004, 0.015] | 0.049 [0.037, 0.064] | 0.002 [0.001, 0.007] |

| Variant | Site | G→B pickup | B→G pickup | R² by rock |
|---|---|---:|---:|---|
| encoder | 0 | 0.356 | 0.191 | 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591 |
| encoder | 1 | 0.329 | 0.242 | 0.720, 0.678, 0.706, 0.806, 0.810, 0.623, 0.891 |
| encoder | 2 | 0.154 | 0.253 | 0.883, 0.948, 0.748, 0.908, 0.925, 0.729, 0.929 |
| encoder | 3 | 0.349 | 0.118 | 0.894, 0.955, 0.764, 0.922, 0.945, 0.751, 0.948 |
| calibrated_encoder | 0 | 0.471 | 0.164 | 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591 |
| calibrated_encoder | 1 | 0.522 | 0.139 | 0.720, 0.678, 0.706, 0.806, 0.810, 0.623, 0.891 |
| calibrated_encoder | 2 | 0.411 | 0.200 | 0.883, 0.948, 0.748, 0.908, 0.925, 0.729, 0.929 |
| calibrated_encoder | 3 | 0.735 | 0.050 | 0.894, 0.955, 0.764, 0.922, 0.945, 0.751, 0.948 |
| min_norm_decoder | 0 | 0.969 | 0.011 | 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591 |
| min_norm_decoder | 1 | 0.927 | 0.013 | 0.720, 0.678, 0.706, 0.806, 0.810, 0.623, 0.891 |
| min_norm_decoder | 2 | 0.951 | 0.010 | 0.883, 0.948, 0.748, 0.908, 0.925, 0.729, 0.929 |
| min_norm_decoder | 3 | 0.956 | 0.010 | 0.894, 0.955, 0.764, 0.922, 0.945, 0.751, 0.948 |

## Resistant-rock rollout evidence: primary agent, best variant, site 2

Full confirmation rates below; examples are fixed episodes 0–3, not selected for outcomes.
Raw position/action traces stay in ignored artifacts.

- Rock 0, Good→Bad: pickup 0.425, visit 0.636, checked 0.994, target checks 6.66/episode; visited without pickup 0.211.
  Episode 0: route [[0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4], [3, 3], [4, 3], [4, 2], [4, 3], [3, 3], [3, 4], [3, 3], [4, 3], [3, 3], [3, 2], [2, 2], [3, 2], [2, 2], [2, 1], [2, 2], [2, 1], [2, 2], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1]]; check counts by rock [6, 7, 10, 8, 8, 9, 11]; target pickup False; final action 6.
  Episode 1: route [[0, 2], [0, 3], [1, 3], [1, 2], [1, 3], [2, 3], [1, 3], [1, 4], [1, 3], [1, 4], [1, 3], [1, 4], [1, 3], [1, 4], [1, 3], [2, 3], [2, 2], [2, 3], [3, 3], [3, 4], [2, 4], [1, 4], [2, 4], [2, 3], [2, 4], [2, 3], [2, 2], [1, 2], [1, 3], [1, 2], [2, 2], [1, 2], [1, 3], [1, 2], [1, 1], [2, 1], [2, 2], [2, 1], [2, 2], [3, 2], [3, 3], [3, 4], [3, 3]]; check counts by rock [5, 3, 7, 9, 10, 6, 13]; target pickup False; final action 2.
- Rock 0, Bad→Good: pickup 0.446, visit 0.942, checked 1.000, target checks 11.50/episode; visited without pickup 0.496.
  Episode 0: route [[0, 2], [0, 1], [1, 1], [1, 0], [2, 0], [1, 0], [1, 1], [1, 2], [1, 1], [2, 1], [2, 0], [2, 1], [1, 1], [1, 0], [2, 0], [1, 0], [0, 0], [1, 0], [2, 0]]; check counts by rock [15, 10, 10, 9, 3, 8, 8]; target pickup False; final action 6.
  Episode 1: route [[0, 2], [0, 3], [0, 2], [1, 2], [1, 1], [1, 0], [0, 0], [1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [0, 0], [1, 0], [2, 0], [1, 0], [0, 0], [1, 0], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2], [2, 3]]; check counts by rock [15, 10, 7, 12, 6, 8, 6]; target pickup True; final action 2.
- Rock 1, Good→Bad: pickup 0.294, visit 0.881, checked 0.999, target checks 8.07/episode; visited without pickup 0.587.
  Episode 0: route [[0, 2], [1, 2], [2, 2], [2, 1], [2, 2], [3, 2], [4, 2], [4, 3], [3, 3], [3, 4], [4, 4], [4, 3], [3, 3], [3, 2], [3, 3], [3, 2], [3, 3], [3, 4], [2, 4], [2, 3], [1, 3], [2, 3], [2, 2], [2, 3], [2, 2], [2, 3], [2, 2], [2, 1], [3, 1], [4, 1]]; check counts by rock [6, 8, 13, 11, 9, 8, 13]; target pickup False; final action 6.
  Episode 1: route [[0, 2], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 2], [1, 2]]; check counts by rock [9, 7, 13, 6, 5, 20, 6]; target pickup False; final action 5.
- Rock 1, Bad→Good: pickup 0.311, visit 0.738, checked 1.000, target checks 5.58/episode; visited without pickup 0.428.
  Episode 0: route [[0, 2], [1, 2], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3], [3, 2], [3, 3], [3, 2], [3, 3], [3, 2], [3, 1], [2, 1], [2, 2], [2, 1], [2, 2], [2, 3], [3, 3], [4, 3]]; check counts by rock [6, 9, 3, 7, 3, 3, 8]; target pickup True; final action 6.
  Episode 1: route [[0, 2], [0, 3], [0, 2], [1, 2], [1, 1], [1, 0], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3], [3, 4], [4, 4], [4, 3], [4, 2], [4, 1]]; check counts by rock [4, 4, 2, 5, 2, 1, 1]; target pickup True; final action 2.
- Rock 2, Good→Bad: pickup 0.938, visit 0.996, checked 0.994, target checks 3.14/episode; visited without pickup 0.059.
  Episode 0: route [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [3, 4], [4, 4], [4, 3], [4, 2], [4, 3]]; check counts by rock [3, 3, 3, 6, 3, 3, 8]; target pickup True; final action 2.
  Episode 1: route [[0, 2], [0, 3], [0, 2], [1, 2], [1, 1], [2, 1], [2, 0], [1, 0], [2, 0], [2, 1], [2, 2], [3, 2], [3, 3], [3, 4], [4, 4], [4, 3], [4, 2]]; check counts by rock [3, 3, 4, 6, 4, 5, 3]; target pickup False; final action 2.
- Rock 2, Bad→Good: pickup 0.028, visit 0.765, checked 1.000, target checks 7.34/episode; visited without pickup 0.736.
  Episode 0: route [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [3, 4], [3, 3], [4, 3], [4, 2], [4, 1]]; check counts by rock [3, 2, 6, 4, 3, 4, 3]; target pickup False; final action 2.
  Episode 1: route [[0, 2], [0, 3], [0, 2], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [3, 3], [4, 3], [4, 2], [4, 3], [3, 3], [2, 3], [1, 3], [2, 3], [3, 3], [2, 3], [3, 3], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 3], [4, 3]]; check counts by rock [5, 5, 6, 10, 7, 9, 10]; target pickup False; final action 2.
- Rock 3, Good→Bad: pickup 0.216, visit 0.837, checked 0.993, target checks 7.35/episode; visited without pickup 0.621.
  Episode 0: route [[0, 2], [1, 2], [2, 2], [2, 1], [2, 2], [2, 3], [3, 3], [4, 3], [3, 3], [3, 4], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]; check counts by rock [4, 5, 7, 8, 9, 8, 11]; target pickup False; final action 2.
  Episode 1: route [[0, 2], [0, 3], [1, 3], [1, 2], [1, 3], [1, 2], [1, 3], [1, 2], [1, 3], [1, 2], [1, 3], [1, 2], [1, 1], [2, 1], [2, 2], [1, 2], [0, 2], [0, 3], [0, 4], [0, 3], [0, 2], [1, 2], [0, 2], [0, 3], [0, 4], [0, 3], [1, 3], [0, 3], [0, 4], [0, 3], [0, 4], [0, 3], [0, 2], [0, 3], [0, 4], [0, 3], [1, 3], [1, 4], [1, 3], [1, 2], [2, 2], [2, 3], [2, 2], [2, 1]]; check counts by rock [7, 5, 9, 6, 5, 11, 12]; target pickup False; final action 0.
- Rock 3, Bad→Good: pickup 0.101, visit 0.145, checked 0.997, target checks 4.83/episode; visited without pickup 0.044.
  Episode 0: route [[0, 2], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]; check counts by rock [4, 4, 4, 7, 2, 4, 5]; target pickup False; final action 4.
  Episode 1: route [[0, 2], [0, 3], [1, 3], [1, 2], [1, 1], [1, 0], [1, 1], [0, 1], [0, 2], [0, 1]]; check counts by rock [3, 4, 4, 6, 2, 4, 4]; target pickup False; final action 4.
- Rock 4, Bad→Good: pickup 0.007, visit 0.017, checked 0.794, target checks 1.33/episode; visited without pickup 0.010.
  Episode 0: route [[0, 2], [1, 2], [1, 1], [1, 0]]; check counts by rock [1, 1, 2, 2, 1, 1, 1]; target pickup False; final action 4.
  Episode 1: route [[0, 2], [0, 3], [0, 2], [1, 2], [1, 1], [0, 1]]; check counts by rock [0, 1, 1, 0, 1, 1, 0]; target pickup False; final action 4.
- Rock 5, Good→Bad: pickup 0.142, visit 0.464, checked 0.957, target checks 3.00/episode; visited without pickup 0.322.
  Episode 0: route [[0, 2], [0, 3], [1, 3], [1, 2], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4], [3, 4], [4, 4], [4, 3], [3, 3], [3, 4], [3, 3], [3, 4], [3, 3], [2, 3], [2, 2], [3, 2], [3, 3], [3, 4], [3, 3], [3, 2], [3, 3], [3, 2], [4, 2], [4, 3], [4, 2], [3, 2], [3, 1], [2, 1], [3, 1], [3, 2], [3, 1], [3, 2], [3, 1], [3, 0], [4, 0]]; check counts by rock [5, 6, 7, 7, 5, 7, 14]; target pickup False; final action 2.
  Episode 1: route [[0, 2], [1, 2], [1, 1], [1, 0], [1, 1], [2, 1], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 3], [4, 2]]; check counts by rock [3, 4, 2, 3, 3, 3, 2]; target pickup False; final action 2.
- Rock 6, Bad→Good: pickup 0.077, visit 0.127, checked 0.987, target checks 3.78/episode; visited without pickup 0.050.
  Episode 0: route [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [3, 4], [3, 3], [3, 4], [3, 3], [4, 3], [4, 2], [4, 1]]; check counts by rock [4, 1, 0, 2, 4, 2, 5]; target pickup True; final action 2.
  Episode 1: route [[0, 2], [0, 3], [0, 2], [1, 2], [1, 3], [1, 2], [0, 2], [0, 3]]; check counts by rock [3, 2, 0, 2, 1, 3, 3]; target pickup False; final action 4.

Nontarget conditional rates pool the six nontarget rocks by initial quality.
Wilson binomial 95% intervals do not represent training-seed uncertainty.
Calibrated maps are solved in float64 and executed in float32; identity errors,
condition numbers and row norms are recorded in results.json. Rank deficiency raises.
Controls are decoder-null random directions with the counterfactual shift norm on
their own current history, not a copied norm from a diverged steering trajectory.
Intact episodes are reused across variants/sites because they are identical.
Steering can still be off-manifold; endpoint calibration alone does not ensure selectivity.
