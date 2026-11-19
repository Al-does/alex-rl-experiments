# RockSample r7 site-0 held-out replication (v4)

Site 0 (after the first transformer block, before final LayerNorm) was selected after looking at v3 confirmation data. V4 is the held-out test of that choice, not a predeclared v3 site selection. Freeze the v3 nuisance-controlled encoder, λ=1, fit seed 41101, test seed 51203 and decoder-null seed 81517. Refit the same 256-episode maps, with 256 independent decoder-test episodes. No tuning or selection on v4 data. Confirmation seed 271829 is new; 1,024 episodes per rock/direction/condition for each of the same three r7 update-29 agents. Matched environment/action RNG, sampled stochastic actions, cached K/V propagation and the matched-norm decoder-orthogonal random controls are unchanged. Bad→Good stops at first target pickup; Good→Bad continues every decision. Smoke uses 8 fit/test/confirmation episodes and the primary agent only.

Nuisance design: 25-cell rover-position one-hot (24 reference-coded columns), 100-step decision-index one-hot (99 columns; reset reference), previous-action one-hot (12 columns plus reset reference), current sensor-symbol one-hot (2 columns plus NONE reference), and 7 already-sampled indicators computed from preceding executed actions and observed positions. Total 144 nuisance columns, plus 7 beliefs and an intercept. No reward, hidden quality, or future event enters the regression. No additional nuisance features; fixed layout is constant. One-hot reference coding avoids dummy/intercept redundancy. Counterfactual sensor-pair replay skipped to keep the final CPU iteration focused.

Pickup entries are probability [Wilson binomial 95% CI]. Rock labels are
implementation indices; intervals do not represent training-seed uncertainty.

## 20261002T191712Z-7f1dc1ef seed 3

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.983 [0.974, 0.990] | 0.115 [0.097, 0.136] | 0.982 [0.972, 0.989] | 0.011 [0.006, 0.019] | 0.573 [0.543, 0.603] | 0.018 [0.011, 0.028] |
| 1 | 0.988 [0.980, 0.993] | 0.232 [0.208, 0.259] | 0.991 [0.983, 0.995] | 0.016 [0.010, 0.025] | 0.413 [0.383, 0.444] | 0.021 [0.014, 0.032] |
| 2 | 0.976 [0.964, 0.983] | 0.122 [0.103, 0.144] | 0.938 [0.921, 0.951] | 0.016 [0.010, 0.025] | 0.933 [0.916, 0.946] | 0.019 [0.012, 0.029] |
| 3 | 0.974 [0.962, 0.982] | 0.213 [0.189, 0.239] | 0.945 [0.930, 0.958] | 0.013 [0.007, 0.022] | 0.705 [0.676, 0.732] | 0.010 [0.005, 0.018] |
| 4 | 0.978 [0.967, 0.985] | 0.027 [0.019, 0.039] | 0.963 [0.949, 0.973] | 0.007 [0.003, 0.014] | 0.925 [0.907, 0.939] | 0.029 [0.021, 0.042] |
| 5 | 0.979 [0.969, 0.987] | 0.003 [0.001, 0.009] | 0.645 [0.615, 0.673] | 0.009 [0.005, 0.017] | 0.983 [0.974, 0.990] | 0.007 [0.003, 0.014] |
| 6 | 0.986 [0.977, 0.992] | 0.304 [0.276, 0.333] | 0.950 [0.935, 0.962] | 0.004 [0.002, 0.010] | 0.270 [0.243, 0.298] | 0.003 [0.001, 0.009] |

| Direction | Condition | Target pickup | Other Good | Other Bad | Mean return |
|---|---|---:|---:|---:|---:|
| Good→Bad | intact | 0.9806 | 0.9813 | 0.0111 | 48.42 |
| Good→Bad | steered | 0.1452 | 0.9633 | 0.0100 | 39.08 |
| Good→Bad | control | 0.9163 | 0.9723 | 0.0121 | 47.23 |
| Bad→Good | intact | 0.0106 | 0.9837 | 0.0113 | 38.66 |
| Bad→Good | steered | 0.6860 | 0.9072 | 0.0327 | 23.36 |
| Bad→Good | control | 0.0152 | 0.9569 | 0.0145 | 35.70 |

Held-out decoder R² by rock: 0.594, 0.524, 0.512, 0.465, 0.555, 0.594, 0.719.
Maximum steered endpoint error: 9.54e-07; maximum nontarget decoded-coordinate change: 3.87e-07.

## 20261002T182024Z-ad0c3a34 seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.991 [0.983, 0.995] | 0.032 [0.023, 0.045] | 0.987 [0.978, 0.993] | 0.014 [0.008, 0.023] | 0.965 [0.952, 0.974] | 0.013 [0.007, 0.022] |
| 1 | 0.990 [0.982, 0.995] | 0.392 [0.362, 0.422] | 0.991 [0.983, 0.995] | 0.009 [0.005, 0.017] | 0.371 [0.342, 0.401] | 0.018 [0.011, 0.028] |
| 2 | 0.977 [0.965, 0.984] | 0.054 [0.041, 0.069] | 0.926 [0.908, 0.940] | 0.008 [0.004, 0.015] | 0.293 [0.266, 0.322] | 0.014 [0.008, 0.023] |
| 3 | 0.981 [0.971, 0.988] | 0.144 [0.123, 0.166] | 0.982 [0.972, 0.989] | 0.016 [0.010, 0.025] | 0.381 [0.352, 0.411] | 0.012 [0.007, 0.020] |
| 4 | 0.987 [0.978, 0.993] | 0.243 [0.218, 0.270] | 0.957 [0.943, 0.968] | 0.015 [0.009, 0.024] | 0.091 [0.075, 0.110] | 0.014 [0.008, 0.023] |
| 5 | 0.965 [0.952, 0.974] | 0.023 [0.016, 0.035] | 0.911 [0.892, 0.927] | 0.014 [0.008, 0.023] | 0.771 [0.744, 0.795] | 0.013 [0.007, 0.022] |
| 6 | 0.992 [0.985, 0.996] | 0.119 [0.101, 0.140] | 0.970 [0.957, 0.979] | 0.006 [0.003, 0.013] | 0.297 [0.270, 0.326] | 0.001 [0.000, 0.006] |

| Direction | Condition | Target pickup | Other Good | Other Bad | Mean return |
|---|---|---:|---:|---:|---:|
| Good→Bad | intact | 0.9834 | 0.9842 | 0.0122 | 48.37 |
| Good→Bad | steered | 0.1438 | 0.9444 | 0.0087 | 38.38 |
| Good→Bad | control | 0.9607 | 0.9766 | 0.0120 | 47.74 |
| Bad→Good | intact | 0.0114 | 0.9837 | 0.0121 | 38.50 |
| Bad→Good | steered | 0.4526 | 0.8089 | 0.0404 | 12.22 |
| Bad→Good | control | 0.0119 | 0.9697 | 0.0111 | 37.54 |

Held-out decoder R² by rock: 0.565, 0.520, 0.619, 0.570, 0.617, 0.645, 0.539.
Maximum steered endpoint error: 9.24e-07; maximum nontarget decoded-coordinate change: 4.17e-07.

## 20261002T191712Z-7f1dc1ef seed 2

| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.972 [0.960, 0.980] | 0.017 [0.010, 0.026] | 0.985 [0.976, 0.991] | 0.013 [0.007, 0.022] | 0.961 [0.947, 0.971] | 0.048 [0.036, 0.063] |
| 1 | 0.985 [0.976, 0.991] | 0.217 [0.193, 0.243] | 0.980 [0.970, 0.987] | 0.021 [0.013, 0.031] | 0.351 [0.322, 0.380] | 0.022 [0.015, 0.033] |
| 2 | 0.966 [0.953, 0.975] | 0.025 [0.017, 0.037] | 0.962 [0.948, 0.972] | 0.011 [0.006, 0.019] | 0.622 [0.592, 0.651] | 0.016 [0.010, 0.025] |
| 3 | 0.983 [0.974, 0.990] | 0.386 [0.356, 0.416] | 0.971 [0.958, 0.979] | 0.009 [0.005, 0.017] | 0.795 [0.769, 0.819] | 0.042 [0.031, 0.056] |
| 4 | 0.982 [0.972, 0.989] | 0.140 [0.120, 0.162] | 0.436 [0.405, 0.466] | 0.009 [0.005, 0.017] | 0.552 [0.521, 0.582] | 0.008 [0.004, 0.015] |
| 5 | 0.974 [0.962, 0.982] | 0.001 [0.000, 0.006] | 0.970 [0.957, 0.979] | 0.010 [0.005, 0.018] | 0.933 [0.916, 0.946] | 0.014 [0.008, 0.023] |
| 6 | 0.985 [0.976, 0.991] | 0.004 [0.002, 0.010] | 0.508 [0.477, 0.538] | 0.010 [0.005, 0.018] | 0.613 [0.583, 0.643] | 0.008 [0.004, 0.015] |

| Direction | Condition | Target pickup | Other Good | Other Bad | Mean return |
|---|---|---:|---:|---:|---:|
| Good→Bad | intact | 0.9782 | 0.9793 | 0.0142 | 48.31 |
| Good→Bad | steered | 0.1127 | 0.9051 | 0.0103 | 36.87 |
| Good→Bad | control | 0.8302 | 0.9132 | 0.0147 | 42.49 |
| Bad→Good | intact | 0.0116 | 0.9796 | 0.0132 | 38.48 |
| Bad→Good | steered | 0.6895 | 0.8447 | 0.0553 | 16.08 |
| Bad→Good | control | 0.0225 | 0.8953 | 0.0150 | 33.01 |

Held-out decoder R² by rock: 0.609, 0.525, 0.618, 0.513, 0.621, 0.644, 0.591.
Maximum steered endpoint error: 1.21e-06; maximum nontarget decoded-coordinate change: 4.77e-07.

## Illustrative paths

Rock 0, episode 0 for both directions was fixed before v4 execution, not
selected for a successful outcome. Intact/steered pairs use identical RNG
and initial qualities. Paths show pre-action positions through termination
or the 100-step horizon; stationary steps are collapsed. Sample markers
include repeated pickups and empty-cell attempts; they are not quality labels.

Nontarget pickup pools the six other rocks conditioned on initial quality.
Controls match the counterfactual shift norm on their own current history,
not the norm of a diverged steered trajectory. Endpoints are measured at
the intervened site, before downstream nonlinearities. Raw arrays stay in artifacts.

The fixed primary-agent Good→Bad pair samples rock 0 intact but avoids it steered
(45 versus 63 decisions). The Bad→Good pair does not pick rock 0 in either
condition; steering reaches the target cell but continues to the 100-step horizon.
It is a retained failure example, not an illustration selected to demonstrate success.
Figures: `pickup_probability.png` and `example_paths.png`.

## Held-out replication conclusion

The primary-agent site-0 pattern replicates on the new confirmation seed:
Good→Bad/Bad→Good pooled pickup is 0.145/0.686, compared with v3's 0.141/0.702.
Corresponding v4 intact rates are 0.981/0.011 and matched-null control rates are
0.916/0.015. Primary-agent rocks 4 and 5 meet both numerical target thresholds
(Good→Bad ≤0.10 and Bad→Good ≥0.5); rock 0 is partially suppressed (0.115),
rather than meeting the Good→Bad point-estimate threshold as in v3. Rock 5's
strong two-way target effects replicate across all three agents: Good→Bad
0.003/0.023/0.001 and Bad→Good 0.983/0.771/0.933.

Selectivity is improved relative to v3's final-block primary site, not perfect:
primary-agent nontarget Good pickup is 0.963/0.907 versus intact 0.981/0.984,
and nontarget Bad pickup is 0.010/0.033 versus 0.011/0.011. Mean return falls
from 48.42/38.66 to 39.08/23.36. Across the two replicate agents, pooled
Good→Bad pickup is 0.144/0.113 and Bad→Good is 0.453/0.690; nontarget Good
pickup is 0.944/0.905 and 0.809/0.845 respectively. Thus the site-0 directional
effect replicates, with agent-dependent strength and residual nontarget disruption,
not a uniformly selective seven-rock flip. No agent's pooled Good→Bad rate
meets 0.10 at the frozen λ=1. This is a held-out test of a post-hoc site choice,
not proof of an internal Bayesian model or nuisance-free causal coordinates.

Post-run verification found the refitted decoder, bias, nuisance-controlled
encoder and null-direction arrays bitwise identical to the saved v3 arrays for
every agent. Execution source SHA-256 hashes match the current source files;
v1–v3 historical result directories are unchanged.
