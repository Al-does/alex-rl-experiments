# Covariance loss replication

Each condition runs training seeds 42 and 43 with covariance weights 0.01 and
0.1. The loss is the sum of squared off-diagonal sample covariances divided
by feature width, calculated over the decision positions in each optimization
minibatch. Belief-Quotient Ladder uses the post-final-LayerNorm embedding,
Reward Relevance uses the pre-final-LayerNorm residual, and RockSample uses
the decision-time embedding fed to the PPO heads.

The Ladder and Reward Relevance use the eps02, temperature-1.5 recipes from
PR #234 and save all 102 checkpoints through 2,012,217,344 steps. Their
original training-run probes are reported at every checkpoint. Additional
Ladder scalar probes and frozen-history controls follow the paper's grouped
re-probe schedule. Final-checkpoint steering uses the Ladder protocol from
`45517b13` and the Joint Reward moving-reward protocol from `3504bfb7`.

RockSample uses r7 `d128_kl0.1` with the original 4-way key split and key index
0 for each requested root seed. Training carries the full state through
update 29 (30,408,704 steps) to update 316 (331,350,016 steps). Both are
resumable checkpoints. The r7 and continuation schedules drive the probes,
and the held-out twin suite from `372b3f88` runs at both endpoints on both
seeds. The reports identify cells with absent fit coverage or singular
clamp calibration.

From an activated GPU environment with B2 configured, run one queue per seed.

```sh
python -m experiments.paper_covariance_2026_11.seed_queue --seed 42
python -m experiments.paper_covariance_2026_11.seed_queue --seed 43
```

Run an individual leaf locally with `rl-harness` and
`--smoke --hardware cpu --no-upload-artifacts`. A full run writes compact
curves, probes, steering metrics and provenance under the leaf's `results/`.
The B2 durability index links checkpoints and full analysis payloads.
The shared loss is available at RL-Harness revision
`886493f621ee97c9e5a1ce09c67554e299a31deb` (PR Al-does/RL-Harness#91).
