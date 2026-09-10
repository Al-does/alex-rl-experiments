# Strata context-64 entropy-4x selected-checkpoint analysis

## Design

This post-hoc analysis restores only initialization, penultimate, and final RLModule checkpoints for each run. It does not retrain models or select layers/checkpoints using held-out probe performance.

Initialization is a separately restored and sampled checkpoint baseline, not a matched-history `initialization_features` comparison: on-policy histories differ across checkpoints.

Each checkpoint uses 20,000 process-weighted learned-policy samples for fit and 20,000 independent samples for test, eight environments, a 64-step per-episode warmup, and whole-episode grouped fitting/bootstrap. The representation is each transformer's current-position block residual before final layer normalization.

## Task performance

| run | initialization | penultimate | final |
| --- | ---: | ---: | ---: |
| Token guess | 0.5816 | 0.7628 | 0.7592 |
| Reward both | 0.3351 | 0.7617 | 0.7715 |
| Reward factor 1 | 0.3322 | 0.7749 | 0.7794 |

These are archived held-out policy means from the original checkpoint reports. Fresh deterministic re-collection is recorded separately in `metrics.json`.

## Fixed final-layer geometry

| run / factor | init R² | penultimate R² | final R² | final log-NTP R² | final probe − log-NTP R² (95% episode bootstrap) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Token guess | 0.706 | 0.976 | 0.980 | 0.953 | 0.027 [0.025, 0.028] |
| Reward both / factor 1 | 0.266 | 0.567 | 0.629 | 0.440 | 0.189 [0.149, 0.224] |
| Reward both / factor 2 | 0.265 | 0.609 | 0.671 | 0.646 | 0.025 [-0.097, 0.192] |
| Reward factor 1 / factor 1 | 0.266 | 0.716 | 0.783 | 0.398 | 0.385 [0.374, 0.397] |
| Reward factor 1 / factor 2 | 0.265 | -0.003 | -0.002 | 0.568 | -0.571 [-0.575, -0.567] |

All layers, controls, covariance-matched Gaussian and observation-matched feature nulls, permuted-label nulls, emission-null-direction contrasts, and activation geometry are in `metrics.json` and `metrics.csv`.

## Findings

- All three archived policy scores rise substantially from initialization; penultimate and final scores are close, so the requested late-checkpoint comparison does not hinge on a large task-performance swing.
- Token-guess final-layer accessibility reaches 0.980 R² and exceeds the log-NTP control by 0.027 [0.025, 0.028]. Its Strata emission-null-direction R² is 0.969.
- Reward-both final-layer probes decode both factors (factor 1 0.629 R²; factor 2 0.671 R²). Factor 1 robustly exceeds log-NTP; factor 2's overall paired difference is 0.025 with an interval spanning zero.
- Reward-factor-1 is selective at the final layer: factor 1 reaches 0.783 R², while factor 2 is -0.002 R² despite a 0.568 log-NTP baseline. This is an accessibility association with the rewarded factor, not evidence of causal use.

## Timing and controls

- Targets are read from `info["belief_current"]` before the current action.
- Token guess uses the delay-one arrival belief for the pending hidden token; the completed action is scored against `event.raw_token_before`.
- Two-factor beliefs already include the preceding executed action through the prior edge-belief update, never reward information.
- Reward both pays factor indices 0 and 1; reward factor 1 pays factor index 0 only. Report labels `factor_1` and `factor_2` are one-indexed.
- The token-guess NTP control is recovered from the filtered source belief and predicts the pending hidden token; the two-factor NTP control projects the current factor belief through the Strata emission matrix.
- First-episode lengths are randomized, and recurrent histories continue through resets before the 64-step warmup filter is applied.
- Fit/test streams are independent: token spawn keys 800/801 and two-factor spawn keys 700/701 under run seed 42.

## Limitations

- Linear decodability is descriptive accessibility, not evidence of causal use and not a unique identification of the network's representation.
- Only one training seed (42) is available; bootstrap intervals quantify held-out episode variation, not training-seed uncertainty.
- On-policy state/history occupancy changes across checkpoints, so initialization-to-trained differences combine representation and occupancy changes.
- The exact training harness revision differs from the current analysis checkout. Source hashes and both revisions are recorded in `provenance.json`; the checked diff adds analysis/checkpoint utilities while the verified rollout ordering and environment timing are unchanged.
- No confidence intervals are inferred for archived task-performance means. Only explicitly computed whole-episode bootstrap intervals are shown.
- Five stochastic null repetitions are a diagnostic battery, not a precise estimate of a null distribution.
