# Within-greedy-action-class belief decoding

Offline response to Reviewer 1 W2 / Question 1: distinguish continuous belief
accessibility inside a decision region from decoding the decision region alone.
This study does not train policies or test causal use. `deep-beliefs` is read-only.

## Frozen baseline

- PPO: `mess3_token_guess_cycle_2-ppo-seed45-2p5m`, all nine checkpoints,
  initialization through 2,515,796 sampled tokens; paper checkpoint 662,029.
- CE: `mess3_token_guess_cycle_2-supervised_ce-seed45-2p5m-ppomatch`, all nine
  checkpoints, initialization through 2,543,125; paper checkpoint 626,197.
- Exact source revisions, resolved recipes, B2 manifest locations, checkpoint
  hashes, and runtime revisions are in `results/passive_within_class.json`.
- Passive MESS3, alpha 0.85, one-step-delayed token guessing, reset without a
  populated token; action correctness does not affect the process. Target is
  the exact predictive state belief at decision time, not the filtered source
  posterior or a reward-conditioned target. States are 1, 2, 3; tokens A, B, C.
- Four transformer blocks, 64-dimensional post-final-LayerNorm representation,
  learned absolute positions, strict last-10-token input, one head of width 8,
  ReLU MLP width 256. PPO has gamma/lambda zero and a correctness reward; CE
  uses only next-token cross-entropy. Training seed 45, nominal 2.5M budget,
  learning rate 1e-4, batch 32768, minibatch 4096, six epochs. No new training.

## Execution

From the experiment repository, CPU only:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python -m experiments.within_class_belief_probes_2026_11.gate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python -m experiments.within_class_belief_probes_2026_11.analyze
uv run pytest -q tests/test_within_class_belief_probes.py
git diff --check
```

The gate downloads only native module checkpoint files, validates byte counts
and SHA-256 against the run's canonical durability manifest, and restores with
`analysis.checkpoints.load_module_only`. It reuses the original probe collector
with independent deterministic streams, greedy actions, 16 environments, 64-step
warmup, 60,000 fit rows and 80,000 test rows. It aborts on a global R² discrepancy
greater than 1e-6. Activations, logits, actions, exact beliefs, episode IDs,
episode steps, environment IDs and one-hot token windows are saved per split;
the head weights and bias are saved separately. Checkpoints and arrays remain
in ignored `artifacts/`. Remove or move study-local caches before changing the
protocol: caches are for exact reruns, not automatic invalidation on edits.

## Analysis conventions

Both Bayes classes (`argmax` of the exact belief) and each checkpoint's own
greedy actions use original, unperturbed rows. Global probes and class-refitted
probes have an unpenalized intercept and ridge 1e-6. All scores use held-out
test rows. Class scores centre the denominator on that test class's mean;
pooled scores sum within-class SSE and SST, not arithmetic means of R². Missing
fit classes produce undefined predictions and explicit coverage counts; constant
targets have undefined R². Raw predictions are never clipped or renormalized.

The head-blind space is the 62-dimensional null space of the two independent
logit-difference weight rows. This removes action-relevant linear head
directions while preserving common logit shifts. It is not an intervention.

History controls reconstruct complete original episodes from their original
seeds and verify reconstructed windows exactly against saved windows. They
preserve reset rows and original targets/classes. One control shuffles non-reset
tokens within each full episode; the other samples non-reset tokens with
replacement from the split's empirical token pool. Both recompute activations
on the resulting strict-10 windows and refit within classes. Labels are permuted
within fit class for a separate refitted control. Each control uses one fixed
draw, not a permutation significance test; history randomization is a distribution
shift, not a null required to score zero. The window baseline uses all 30 one-hot
features with the same affine fit. Class indicators use fit-class mean targets.

Boundary scores reuse the Bayes-class-refitted probes and centre the test
denominator within each class and margin bin. Bins are [0,.05), [.05,.2),
[.2,.5), [.5,1]. Total and per-class counts/low-count flags accompany scores.
Scalar probes are separately class-refitted for the Bayes margin and the split
between the two non-winning coordinates, ordered by coordinate index.

Initialization and both paper checkpoints have 1,000 episode-cluster bootstrap
resamples for the requested pooled full-representation, head-blind, logits and
scalar scores (also the window baseline). Probes stay fixed; class means and
denominators are recomputed in each resample using cluster sufficient statistics.
Intervals do not include probe fitting or training-seed uncertainty. The two
initial checkpoints are the same initialized network, not independent replicates.

## T4: Belief-Quotient Ladder

`ladder.py` adds the three canonical quotient conditions, seeds 42–56. It
explicitly selects the historical `cycle_6`-prefixed runs in the cycle-7 code
folder; newer `cycle_7`-prefixed reruns are not substitutes. Source checkpoint
hashes, byte counts, original experiment/library revisions and curve hashes are
retained in the output. The recovery notes are actually in the unchanged
cycle-6 folder (`B2_AGENTS.md`), not the cycle-7 folder.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.within_class_belief_probes_2026_11.ladder gate --workers 4
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.within_class_belief_probes_2026_11.ladder analyze --workers 4 --stages init final
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.within_class_belief_probes_2026_11.ladder analyze --workers 4 --stages iteration_4
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 uv run python -m experiments.within_class_belief_probes_2026_11.ladder_report
```

The gate tests initialization, iteration 4 and final for seed 42 in every
condition against the archived **full-belief** checkpoint curves. These use
streams 300/301, unlike the later scalar-component campaign's 510/511. All T4
collection retains the original full-belief streams, 16 environments, 60k/80k
fit/test rows, warmup 64 and ridge 1e-6. `collect_history_data` preserves complete
episode prefixes; exact beliefs are independently replay-checked to 1e-10.
The environment has current-token/previous-action observations, zero delay,
reward-state occupancy rewards, and action-dependent controlled transitions.
The original model is the cached/banded four-layer transformer with a 32-frame
per-layer attention window (128-frame effective receptive field), not the
passive study's strict-window encoder. The REINFORCE policies were trained to
10M steps with entropy annealed 0.03 to zero, not retrained here.

All classes are the source policy's actual greedy actions. In particular,
initialized Trivial Quotient policies need not be constant. Every analyzed
checkpoint has within-class belief, rho and delta probes from full activations,
the head-blind space, and logits. Controls use the original cycle-7
`shuffled_histories` and `empirical_random_histories` functions and their seed
scheme: complete token/previous-action pairs, shuffle reset rows fixed, and
random inputs from separate empirical reset/non-reset pools. These are frozen
input replays, not closed-loop interventions; original classes and beliefs stay
fixed. Label permutations are restricted to fitting rows within class.

For the primary final comparison, the seed's initialized module replays the
final policy's complete fit/test histories from reset, retaining exactly the
same scored rows, targets and **final-policy** classes. Its own head defines
its head-blind and logits features. The same controls are also evaluated through
this initialized encoder. Own-policy initialization is separately reported as
a secondary comparison with different visitation and greedy classes.

`ladder_report.py` takes equal-weight means over the 15 training seeds and
10,000 whole-seed bootstrap draws. Final-minus-matched-initialization intervals
are paired by training seed. It adds the ladder table to the passive results
README without changing the passive result JSON or PNG. Raw complete-prefix
histories, matched initialization activations, module files and checkpoint
reports remain in ignored `artifacts/ladder/`.

## PR202 temperature-1.5 ladder: reproduction stopped

Fetched origin/main and fast-forwarded this branch from `ea6f0cb2` to
`2fb56dce` to reuse the new study code. Only
`belief_quotient_history_jax_lumped_temp15_2026_11` was selected; the
reward-factors study was not analyzed. Source run manifests record experiment
revision `37c3895435f81d3ae9fd8baa9028f4c38fea122b`.

`temp15.py gate` restores parameter NPZ files with the source continuation
loader after verifying SHA-256 and byte count against the canonical B2
durability manifest. It uses the study's own `_probe_rollout`: greedy fresh
1,024-step episodes, 64-step warmup, 60,000 fit / 80,000 test rows, post-final-LN
embedding, ridge 1e-6, `split(fold_in(PRNGKey(42), 2))` fit/test keys. Complete
observation/action histories remain under ignored `artifacts/temp15/`.

The first checkpoint (Trivial Quotient / GRU128 initialization) failed the
1e-6 global-R² gate: archived 0.980304509038, reproduced 0.980342474993,
difference +3.797e-5. Checkpoint SHA-256 and bytes matched; archived and local
JAX versions are both 0.11.1. The cause is unresolved. As requested, no further
checkpoints, conditional metrics, controls, matched-init replays or bootstrap
intervals were computed. `results/ladder_temp15_gate.json` records this failure;
no completed within-class result or figure is claimed.

Reproduce on CPU:
```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.within_class_belief_probes_2026_11.temp15 gate
```

### Authorized 1e-3 gate and CPU precision diagnosis

The initial 1e-6 attempt above is superseded by the user's authorization of
absolute tolerance **1e-3 on both global R² and archived global 1−R²**.
This allows cross-device float32/matmul differences for the protocol check;
all proposed primary comparisons would be CPU-only. GPU TF32 is a possible
explanation, not an established cause.

`temp15 diagnosis` reruns fresh histories at default and highest JAX matmul
precision. For Trivial Quotient / GRU128 init, both CPU runs give R²
0.980342474993, action fractions [0.244, 0.70895, 0.04705] and mean probe reward
0.431812495. Archived values are R² 0.980304509038, action fractions
[0.2483125, 0.7074875, 0.0442] and mean reward 0.433512509. Archived per-row
actions were not available in the compact report; these are aggregate checks.

With the relaxed tolerance, init passes but the same arm's final checkpoint
(4,004,511,744 actual steps) fails: archived R² 0.984582334736, CPU R²
0.986766985071, difference +0.002184650335. The archived global 1−R² is
0.015417665036 versus CPU 0.013233014734, also outside tolerance. Both have
greedy action fractions [0, 1, 0] (only Tilt X). Checkpoint hashes and bytes
match. Analysis stopped; 22 other gates and all conditional analyses remain
unattempted. `results/ladder_temp15_gate.json` records the tolerance, rationale,
both score checks and policy fractions; `ladder_temp15_precision_diagnosis.json`
records the precision comparison. No tolerance was widened further.

### Latest protocol check: max(0.01, five bootstrap SEs)

The preceding stopping rules are superseded by the user's authorization to
record every init/final check, stopping conditional analysis only for a gap
above max(0.01, 5 × CPU global-R² episode-bootstrap SE). This is a
cross-device protocol check, not a precision-equivalence certificate. Both
global R² and archived global 1−R² use this limit. All 24 checkpoints were
downloaded with manifest SHA-256/size verification and checked on CPU.

The final Trivial Quotient / GRU128 same-row refits give R² 0.986766984913
(float64) and 0.986738441853 (float32). The float64-centered design condition
number is 4,395,979; ridge augmentation reduces it to 193,939. A 10,000-resample
episode bootstrap gives global-R² SE 0.000137691 and interval
[0.986485315, 0.987027860], conditional on one training seed and fixed fit data.
The float32 solve uses float32 centering, response, ridge augmentation,
coefficients and prediction; scoring is float64 in both cases.

Fixed Tilt X actions remove policy-dependent branching, but identical keys
are not a guarantee of bitwise-identical cross-device histories: categorical
sampling also uses floating-point log/Gumbel arithmetic. The environment uses
Threefry-driven keys and samples observations/states through JAX categorical;
there is no separate device-independent integer categorical sampler. Archived
raw histories were not available for direct comparison, so neither history
identity nor TF32 as the source of this gap is verified.

All 24 checks are now recorded in `results/ladder_temp15_gate.json`. Four
exceed the latest hard-stop limit: all three GRU92 initializations, and
Trivial Quotient / transformer final. Therefore the full conditional analysis
has not run. `temp15_report.py` appends the complete check table, diagnosis
and one-class flags to the results README without replacing older sections.
`results/ladder_temp15_final_diagnosis.json` contains same-row fits, condition
numbers and bootstrap details. The bootstrap resamples 84 complete test-episode
clusters (last scored episode truncated), holding the fitted probe fixed;
protocol checks use 2,000 resamples and the requested diagnosis uses 10,000.

### Completed CPU conditional analysis (latest authorization)

The earlier hard stops are superseded: all 12 arms are analyzed on CPU from
manifest-verified checkpoint bytes, with own-policy init, final, and a saved
checkpoint near 2B steps. Final-policy histories are also replayed through each
initialized encoder, retaining the final targets and greedy classes. GRU carry
starts at zero per episode and continues through all 1,024 steps including
warmup, without resets at the training recipe's 32-step chunk boundaries.

The three GRU92 init checkpoints and Trivial Quotient / transformer final remain
annotated **archived global R² not reproduced on CPU**. These discrepancies do
not block the internally CPU-computed matched comparisons. GRU92 init bytes and
parameter arrays are identical across variants; archived and CPU aggregate
actions and rewards agree, and logits are not near-tied. See the separate
diagnosis JSON and completed results section for exact numbers.

Run `temp15_analyze analyze`, `temp15_analyze intermediate`, then
`temp15_results` as modules under this study with `CUDA_VISIBLE_DEVICES=''`,
`JAX_PLATFORMS=cpu`, and `OPENBLAS_NUM_THREADS=1`. Outputs are
`results/ladder_temp15_within_class.{json,png}` and the appended results README
section; checkpoints, arrays and logs remain in ignored `artifacts/temp15/`.
The JSON records checkpoint SHA-256/bytes, source and harness revisions, source
file hashes, runtime versions and CPU devices. Paired intervals use 10,000
whole-test-episode resamples with probes fixed, conditional on one training seed
(42); they do not measure training-seed uncertainty. One-class policies have
within-class R² equal to global R². Accessibility is not causal use or proof of
Bayesian computation.

### Persistent-nuisance entropy-0.005 analysis (PR #215)

Only `variant_{1,2,3}_transformer_ent0p005`, seed 42, are analyzed. Source and
compact run records were found on
`origin/devin/1794419502-persistent-nuisance-egreedy`, not `origin/main`, after
fetching origin. Selected arm directories and their small shared recipe sources
were imported with `git archive`, without merging/rebasing the stacked branch.
No entropy-0.01 arms or other study results were imported. The upstream base JAX
collector differs locally only by epsilon support; its greedy temperature-1,
epsilon-0 execution is equivalent. No training API is called.

This process has three latent states: `belief_targets` returns the entire
filtered three-vector, not an extra-factor belief or a marginal. The persistent
nuisance is the slowly varying nonreward-state identity in the modified sticky
transition/emission process. Accordingly delta=b1−b2 and rho=b3 still apply.
Delta is the within-nonreward contrast (action-irrelevant in Reward-State
Quotient, potentially action-relevant in Full-State Quotient); rho is reward-state
probability. Neither coordinate is required by the Trivial action partition.
Classes are measured greedy actions, not the underlying latent-state partition.

Run `nuisance` then `nuisance_report` as modules under this study, with
`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
The adapter uses manifest-verified NPZ restoration and the source study's greedy
post-final-LayerNorm collector, 60K/80K rows, 64-step warmup, 1,024-step episodes,
and ridge 1e−6. It analyzes init, the saved checkpoint nearest 1B, and final.
Matched initialization replays final histories from empty transformer caches per
episode, including warmup, retaining final targets/classes. Controls and paired
10,000-draw episode bootstraps follow temp15; archive checks use 2,000 draws and
the same max(0.01,5×SE) annotation threshold, never a hard stop.

Compact outputs are `results/ladder_nuisance_ent0p005_within_class.{json,png}`
and an appended results README section. Raw arrays/checkpoints/logs stay under
ignored `artifacts/nuisance_ent0p005*`. JSON records imported-source revision,
checkpoint hashes/bytes, source/harness revisions, source hashes, runtime and CPU
device information. Own-policy init is secondary. Bootstrap intervals condition
on one training seed and fixed fit probes. One-class within-class R² equals
global R²; affine accessibility is not causal use or Bayesian computation.

### Persistent-nuisance v2 temperature 1.5 (PR #219)

Imported selected transformer arm records and recipe/environment sources with
`git archive` from PR branch `devin/1794431156-v2-temp15-eps-greedy` at
`58a26c8c6c2bc1fda82845db4727a6c7397842a4`, without merging/rebasing.
Only `variant_{1,2,3}_transformer` from the no-epsilon and epsilon-0.05 studies
are analyzed. Existing shared recipe files from earlier imports are not
overwritten; restoration uses the archived per-run resolved process/model, not
the training entrypoint. The operative probe/models/env files match upstream;
the PR's only base collector changes add non-greedy training epsilon mixing.
Greedy argmax and probe key splitting are unchanged.

The v2 process remains three-state; the source `belief_targets` returns the full
filtered belief, so delta=b1−b2 and rho=b3 are unchanged. New environment constants
are baseline transition rows (.995505,.002495,.002), (.002495,.995505,.002),
(.005,.005,.99), emission rows (.1125,.0375,.85), (.0375,.1125,.85), (0,0,1),
and tilt strength 2.5. Initialization is uniform; reward is pre-transition state-3
occupancy. The three task partitions and coordinate interpretations remain as
above. Training is seed 42, entropy 0, T=1.5, epsilon 0 or 0.05, 2B target;
probes are greedy with neither temperature sampling nor epsilon exploration.

Run `nuisance_v2` then `nuisance_v2_report` as modules under this study, CPU-only
with the environment settings above. They reuse the PR215 analysis with separate
raw roots per epsilon study, avoiding same-arm checkpoint/cache collisions.
All six arms have init, nearest-to-1B intermediate and final analyses, matched
initialization replay, controls and paired 10,000-draw episode bootstrap intervals.
Archive checks use the same annotation-only max(0.01,5×SE) rule, never a hard stop.
Outputs are `results/ladder_nuisance_v2_temp15_within_class.{json,png}` and an
appended results README section; raw artifacts/logs remain ignored under
`artifacts/nuisance_v2_temp15*`. Checkpoint SHA-256/bytes and source/runtime
provenance are recorded. All uncertainty is conditional on one training seed and
fixed fit probes; one-class within-class R² is global R², not new conditional
evidence. These are descriptive accessibility comparisons, not causal claims.
