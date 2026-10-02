# RockSample belief probing

Requires the exact filter in [RL-Harness PR 81](https://github.com/Al-does/RL-Harness/pull/81)
and the existing `jax` dependency group. This is offline analysis; it does not train.

```bash
uv run --group jax pytest -q tests/test_rocksample_belief_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --group jax python \
  -m experiments.rocksample_jax_2026_10.belief_probe \
  --download --output experiments/rocksample_jax_2026_10/results/NEW_RESULT
uv run --group jax python -m experiments.rocksample_jax_2026_10.report_beliefs \
  experiments/rocksample_jax_2026_10/results/NEW_RESULT
```

Downloads use the provisioned B2 variables and verify byte counts and SHA-256
against the canonical durability manifests before restoring trusted training
pickles. Checkpoints remain under ignored `artifacts/belief_checkpoints`.
To reuse downloads, omit `--download`. Both analysis and rendering refuse
existing output directories. `--seeds 0 --smoke` runs 16 episodes per
distribution, skips the expensive joint null battery, and should write to an
ignored artifacts directory. The complete evaluation uses 512 episodes per
distribution and seed; no raw rollout/activation arrays are saved.

Both targets are decision-time posteriors over **current** rock qualities:
M contains seven marginals; J contains 128 configurations indexed by Good bits.
Only executed actions, emitted symbols and observed pre-action position enter
the filter. Sample deterministically clears a rock's Good probability.
Privileged truth is used only for calibration, and rewards only for behavior.
Exit/cap ends each trajectory; no post-exit activation/target row is scored.

The final and reconstructed initialization encoders replay the same complete
histories. The exact initialization follows the recorded sweep and PPO key
splits; a test compares its parameters with `ppo.init` for all four seeds.
Final-policy histories are the primary paired-init distribution. Own-init
histories and identical forced-check histories are additional controls.
The three distributions are kept separate in each seed report.

Half the episodes fit probes; half evaluate them. SVD cutoff selection uses
training-episode cross-validation only. The affine predictions are never
clipped or normalized before scoring. Reports include MSE, target variance,
1−R², per-rock unrestricted/relevant/informed scores, episode-bootstrap
uncertainty, nuisance/predictive baselines and joint-posterior null controls.
Constant coordinates have undefined R², never an invented zero error score.

Forced checks supply observations even for rocks a policy neglects. Probes
are refitted on this separate distribution: accessibility there is not proof
of causal use or generalization by a probe fitted on on-policy histories.
Raw coordinate traces and decoded before/after-check changes accompany the
metrics. All row weights are uniform; variable-length episodes thus contribute
different numbers of probe rows, while return averages weight episodes equally.

The archived run has only the final checkpoint, at 30,408,704 environment
steps. Initialization-to-final connecting lines are guides; the separate
29-point reward curves do not imply intermediate belief measurements.

Outputs include source/checkpoint hashes, compact seed JSONs, figures, a CSV,
and a generated methods/metrics report. Linear accessibility does not establish
a unique internal Bayesian model or causal use of any rock coordinate.
