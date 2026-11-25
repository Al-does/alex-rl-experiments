# Critic-aware Belief-Quotient Ladder results

Three seeds (42, 44, 46), frozen final checkpoints, and affine probe error
`1 − R²` throughout the matched-history comparisons. Each donor selects the
actions and observations replayed through all five trunks. See the [methods](../README.md)
and [checkpoint manifest](provenance.json).

The interior-window self-replays also reproduce the earlier per-seed outputs.
The largest absolute differences are $6.4\times10^{-7}$ across 90 belief
scores and $2.0\times10^{-8}$ across 225 nondegenerate optimal-value scores,
including both SAC Q trunks on SAC histories.

## Reward-State Quotient

The reported SAC actor δ error of 0.3029 was a transcription error. The same
checkpoints give 0.1380 with the interior window and 0.1303 with the archived
full-window protocol. The interior-window actor errors are 0.13682, 0.14988, and
0.12730 by seed. The actor's mean δ error is slightly lower than either of its
critics' on every donor's histories.

| History donor | REINFORCE | PPO | SAC actor | SAC Q1 | SAC Q2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| REINFORCE | 0.2776 | 0.2945 | 0.1195 | 0.1397 | 0.1487 |
| PPO | 0.2840 | 0.2966 | 0.1251 | 0.1377 | 0.1465 |
| SAC | 0.2927 | 0.3028 | 0.1380 | 0.1480 | 0.1513 |

In the same rows, the SAC Q trunks have the lowest errors on ρ and the
approximate optimal value. The nonlinear value residual accounts for
0.31–0.32% of optimal-value variance. The Q trunks decode that small residual
best, with errors of 0.037–0.041 across histories. Corresponding error ranges
are 0.049–0.057 for the SAC actor, 0.072–0.078 for PPO, and 0.126–0.139 for
REINFORCE. These comparisons describe linear accessibility in frozen trunks.

## Trivial Quotient

On REINFORCE and PPO histories, the value-relevant ρ probe errors are 0.1371
for REINFORCE, 0.0694 for the SAC actor, and 0.0044–0.0050 for the three
critic-bearing trunks. The linear value target gives the same ordering.

On SAC histories, seed 46 gives the REINFORCE trunk a ρ error of 270.64 and a
δ error of 47.17. The fit window has only Tilt X decisions. Four Tilt Y
decisions appear in the test window. The 466 of 21,440 held-out rows with a
recent non–Tilt X input contribute 99.97% of the ρ squared error and 99.59%
of the δ squared error. This is an out-of-support transfer result. The
[diagnostic](trivial_seed46_history_support.json) leaves all probe scores and
rows intact.

## Full-State Quotient

The SAC Q trunks have the smallest ρ and approximate optimal-value errors on
all three donors. All three policy trunks have lower δ error than both SAC Q
trunks on every donor. On SAC histories, for example, ρ errors are 0.0362 for
the actor and 0.0026 for each Q trunk, while δ errors are 0.0599 for the actor
and 0.0753 and 0.0759 for Q1 and Q2.

The shared PPO trunk has the lowest error on the nonlinear optimal-value
residual on all three donors (0.101–0.116). Both SAC Q trunks also improve
on the SAC actor for this target. That residual contributes 0.34–0.35% of
total optimal-value variance. The $A^*=Q^*-V^*$ probe measures squared
advantage prediction error. Control regret would require comparing the true
value of a decision from the probe with that of the reference policy.

## SAC policy value and target-network lag

The [Monte Carlo comparison](sac_soft_value_per_seed.csv) evaluates the
frozen stochastic SAC policy halfway through the 1,024-step episode, with
the learned entropy coefficient and the original terminal boundary. The
minimum of the online critic heads strongly underpredicts absolute value.
Subtracting the policy-weighted Q at each history yields mean held-out
$R^2$ of 0.701 in Reward-State and 0.718 in Full-State. Trivial's estimated
Monte Carlo noise variance for these action differences exceeds the estimated
conditional target variance in every seed.

| Condition | Mean scaled soft V bias | Policy-relative Q $R^2$ (seeds 42, 44, 46) |
| --- | ---: | --- |
| Trivial Quotient | −0.698 | −0.852, −0.298, −0.301 |
| Reward-State Quotient | −0.337 | 0.621, 0.700, 0.782 |
| Full-State Quotient | −0.314 | 0.640, 0.701, 0.814 |

**A scalar Bellman model closely tracks the training-time Q levels.**
The archived critics started with zero Q heads and ran 6,464 updates with
Polyak rate 0.005 and discount 0.99. In the scalar model, the target moves a
fraction $\tau(1-\gamma)=0.00005$ of the way toward its stationary value per
update. Starting at zero, 6,464 updates cover 27.6% of that gap for a constant
reward. Driving the scalar model with each run's actual logged mean rewards
and entropies reproduces the logged mean policy-Q trajectories to 0.0018–0.0028
RMSE, using the same scaled units as the heads. This is consistent with slow
target propagation. The scalar model uses no observations and makes no
prediction about learned action rankings or causal effects of changing
$\tau$. Per-seed values are in the [target-lag table](sac_target_lag_per_seed.csv).
