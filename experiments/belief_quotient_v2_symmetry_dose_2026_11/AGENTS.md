# belief_quotient_v2_symmetry_dose_2026_11

Graded symmetry breaking between the Reward-State and Full-State Quotients, for Reviewer 1's
within-class question (W2/Q1) and symmetry-perturbation request (W3(i)). Analysis lives in
`../within_class_drivers_2026_11`.

Every leaf is the PR #219 transformer recipe (`belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11`:
v2 baseline transitions and effect size 2.5, 2B steps, T=1.5 in rollout and PPO likelihoods,
constant entropy 0, epsilon 0, critic-free) with one environment knob changed:

- `lam_{p100,p075,p050,p000,m050,m100}`: state 2's tilt directions are (lam, -lam) for
  (Tilt X, Tilt Y). `lam_p100` is exactly the Reward-State Quotient and `lam_m100` exactly the
  Full-State Quotient (tested); the others interpolate.
- `kappa_{050,100}`: Reward-State tilts (lam=1), emissions P(C | state 1) = .85 - kappa and
  P(C | state 2) = .85 + kappa with each state's 3:1 A/B split kept. Transitions and reward stay
  exactly lumpable; only the observation channel breaks the state-1/state-2 symmetry.

Seeds are runtime (`--seed`); the launch used 43-48 for the two endpoints and 43-45 elsewhere.

```bash
uv run pytest -q tests/test_v2_symmetry_dose.py
uv run --group jax rl-harness experiments.belief_quotient_v2_symmetry_dose_2026_11.lam_p050.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
