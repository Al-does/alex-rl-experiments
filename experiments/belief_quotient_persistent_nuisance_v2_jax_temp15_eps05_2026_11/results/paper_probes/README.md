# PR #219 manuscript probe follow-up

Post-hoc analysis of all 24 completed seed-42 agents from
[PR #219](https://github.com/Al-does/alex-rl-experiments/pull/219); no retraining.
The generating analysis revision is recorded in every arm's JSON, with both
training/analysis harness revisions, model/environment metadata, checkpoint
SHA-256 values, and library/device details. The reported source is preserved
even when later commits change documentation or metadata wording.

## Protocol and outputs

- Initialization plus all 101 trained checkpoints per arm: 2,448 reports,
  ending at 2,012,217,344 environment steps. No best-checkpoint selection.
- Fresh independent greedy fit/test streams: 60,000/80,000 scored decisions,
  complete 1,024-step episode histories, masks at warmup 64 and 128.
- Post-final-layer-normalization embeddings; affine ridge 1e-6, NumPy float64
  SVD; raw held-out normalized error MSE / target variance = 1 − R².
- Full belief, reward-state marginal ρ = b₃, and distinction δ = b₁ − b₂;
  the exact quotient reward belief is also evaluated in Reward-State Quotient.
- Nine control checkpoints per arm: 20K/20K rows, warmup 128, frozen networks
  with separately refitted probes; targets remain the unaltered history beliefs.
- Reward-State A/B sensitivity at every checkpoint: 64 sampled-behavior
  episodes, T=1.5 and the arm's training epsilon, factual actions/C tokens fixed,
  warmup 128. A separate final paired closed-loop greedy intervention uses all
  1,024 decisions per episode. Pointwise 95% intervals bootstrap 2,000 episode
  resamples; they are not across-training-seed confidence intervals.

Run the sibling `deep-beliefs/figures/plot_pr219_ladder_probes.py` to create
20 native-size vector PDFs, a combined 20-page PDF, a final-checkpoint CSV,
and a source/output hash manifest. Full protocol and scientific caveats live in
`deep-beliefs/figures/pr219_ladder_probes_method.md`; the paper's historical
cycle-7 text/results are not overwritten.

## Findings and limitations

- The exact quotient belief **equals the full-filter reward marginal** in v2.
  The historical target-separation claim is not testable in this environment.
- Reward-State Quotient's six Transformer/GRU endpoints decode ρ with error
  0.000061–0.00321, versus 0.107–0.339 for δ. Full-State Quotient's δ errors
  are 0.0244–0.0389. These are descriptive single-network comparisons.
- Trivial Quotient's Transformer/GRU endpoints retain δ accessibility
  (error 0.141–0.228); the result is not exclusive/minimal quotient learning.
- All six Reward-State Transformer/GRU agents retain small nonzero A/B
  sensitivity: greedy disagreement is 0.0698%–1.87% on sampled-behavior
  fixed-input comparisons. The separate closed-loop greedy interventions
  also change some choices. All eight paired-reward 95% intervals include
  zero, which does not establish reward equivalence or exact policy invariance.
- Memoryless full-belief errors are 0.813–0.934. Low A/B sensitivity must be
  interpreted alongside behavioral performance, not treated as sufficient
  evidence of successful abstraction or causal belief use.
- Preserve negative R² and control outliers: the epsilon-0.05 Full-State
  Transformer at step 39,845,888 has random-input error 426.197 and shuffle
  error 19.5938. Raw affine predictions are unconstrained; these control
  generalization failures are not clipped or used to claim accessibility.
- The raw outside-simplex diagnostic uses a strict coordinate < 0 test after
  sum-to-one plane projection. Memoryless models' negative coordinates are
  all smaller than 2.12e-10 in magnitude; their nonzero strict counts reflect
  roundoff-scale boundary signs, not materially negative probability mass.

## Durability

`run_manifest.json` → `remote_artifacts` → the canonical B2 durability manifest
indexes ignored final geometry and checkpoint-report restart caches. Original
weights remain in the source training runs' canonical B2 archives. No weights,
raw histories, or artifact trees belong in Git. `verification.json` records the
independent metric/geometry/provenance audit and SHA-256 readback of every B2
archive entry before teardown of this session's GPU instance.
