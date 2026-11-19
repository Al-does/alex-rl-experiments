# Twin steering: Reward-State Quotient

## Results

Reward-state occupancy per step at block 4 and λ=1. Each pair is the old
regression write → the twin write on the same training seed and episode
draws. Parentheses show percent of actions changed from the intact shadow.

| Seed | Intact / uniform random | ρ flip | ρ point | δ mirror | δ point |
|---|---|---|---|---|---|
| 42 | .4398 / .1057 | .1158 → .0816 (32.64% → 31.47%) | .0870 → .0842 (66.78% → 72.34%) | .4332 → .1933 (0.86% → 30.75%) | .4299 → .1973 (1.30% → 30.42%) |
| 43 | .4032 / .1057 | .0778 → .0713 (58.43% → 61.60%) | .0651 → .0649 (72.87% → 72.83%) | .4009 → .2720 (0.36% → 20.84%) | .4037 → .2841 (0.56% → 20.16%) |
| 44 | .4311 / .1057 | .0912 → .0748 (57.22% → 58.32%) | .0748 → .0751 (76.50% → 76.61%) | .4397 → .4222 (0.45% → 4.55%) | .4432 → .4224 (0.57% → 4.60%) |

**Verdict:** The twin write suppresses occupancy more under the full-strength
ρ flip (.0949 → .0759 on average; all three seeds), but the old regression
write is more selective. Its δ mirror barely changes behavior, whereas the
twin write drives mean occupancy down from .4246 to .2958 and changes 18.71%
of actions (old: 0.56%). The δ point gives the same pattern. With λ selected
on tuning episodes, the ρ flip improvement shrinks to .0959 → .0895 and the
ρ point is slightly worse on average (.0834 → .0875). In particular,
seed 44's selected twin ρ point just misses the random-action level on
confirmation (.1068 versus .1057). Full-strength norm-matched nulls give
occupancies between .4032 and .4486; they do not explain the suppression.

At least 96.5% of sampled goals in every seed/target have a training pair
within a Euclidean distance of 0.1 in source–target belief space. Yet the
held-out displacement fit explains only 0.567–0.666 of residual displacement
energy. A separate readout sees an average absolute ρ change of
0.034–0.041 on rows where the true twin ρ changes by less than 0.02. The
fit is not coordinate-specific: lower ρ-flip occupancy alone is not
evidence of a better belief write. These are three paired seeds, not a new
population estimate for the 15-seed campaign.

## Protocol

The comparison uses the final Reward-State Quotient transformer checkpoints
(training seeds 42–44), greedy closed-loop action selection, and block 4
before final normalization. We use the paper's affine decoder and target
construction, the same per-seed environment draws, and the same tuning
grid and selection rule. The archive data come from PR #239, `45517b13`.
The table comparison at strength 1 matches the manuscript; the JSON also
records the tuning-selected comparison.

Twin prefixes preserve the actual executed actions. The token evidence is
either exchanged between A and B or sampled from alternative state-emission
rows in 64-step chunks. These are deliberately constructed evidence
proposals, not draws from the intact policy or samples of counterfactual
hidden states. We refilter each entire prefix under the original HMM. At a
paired timestep we fit `g(h, b_twin)` to `h_twin - h` and
`g(h_twin, b)` to `h - h_twin` jointly by affine least squares
(`rcond=1e-6`). The intervention adds `λ g(h, b*)` at each step, with
no SVD projection.

The held-out diagnostic predicts new twin displacement in activation space
and measures the movement along `rho = P(state 3)` and
`delta = P(state 1) - P(state 2)` using a separate affine readout fitted
on original and twin fitting rows. Preserved-coordinate error uses rows
with an exact posterior change below 0.02 in that coordinate. Goal support
is a nearest-neighbor check in concatenated (source, target) belief
coordinates, not a density estimate or a test of support in activation
space. An approximate neighbor cannot be taken as evidence that an exact
point target occurs under the policy.

Reproduce from the experiment repo with the historical refs available
locally (the runner reads the archived per-seed reports directly from Git)
and B2 credentials configured:

```bash
git fetch origin devin/1794864678-eps02-belief-steering devin/1794864773-xor-moving-eps02
uv run --group jax pytest -q tests/test_hmm_twin_steering.py experiments/algorithm_generality_probes_eps02_2026_11/test_steering.py
uv run --group jax python -m experiments.belief_quotient_twin_steering_2026_11.run --cohort ladder --workers 3
```

The runner refuses to overwrite results. Use `--smoke --seeds 42` for
a reduced-budget check writing to ignored artifacts.
