# Causal decoding: seed-42 CPU pilot

**This is a feasibility pilot, not a three-seed replication or a demonstrated
steering improvement.** All four requested tests, plus Full-State as a positive
control, ran against the pinned final temperature-1.5 REINFORCE checkpoints.
No training, paid machines, existing study edits, or checkpoint commits occurred.
The [generated tables](tables.md) contain the complete compact comparison.

## What the pilot says

- **Better held-out decoding did not make the default gradient writes better.**
  Band concatenation improved Reward-State belief R² from .849 to .900, and the
  MLP improved Joint Reward from .833 to .885 and v2 from .909 to .940. Neither
  default gradient write beat Eq. (4) across the tasks. The gradient optimizer
  had only 20 Adam steps in this pilot; failure to move actions is not evidence
  that a sufficiently optimized nonlinear write cannot work.
- **DAS captured replay effects, not a selective transferable belief edit.**
  Rank 8 won independent selection in all five agents/tasks. Block-4 held-out
  joint-policy replay fidelity was .972 for Reward-State, .797 for moving,
  .825 for paper XOR and .869 for v2 XOR, versus 1.0 for the full block-4
  reference patch. The earlier block-2 diagnostic was weaker or more limited
  by the latest-only reference patch; all ranks at both blocks are saved.
  Reward-State DAS changed 16.55% of mirror-control actions versus 1.17% for
  Eq. (4). Its rho-flip occupancy was .1665 versus .0869. This repeats the
  twin study's warning: suppressing occupancy is not sufficient evidence of
  coordinate specificity.
- **v2 DAS is a lead, not a finding.** Its .7021 XOR reward exceeded the paired
  Eq. (4) factored .5518 and joint .6924, and was near the paired exact-filter
  map .7129. It failed paper XOR (.5508 versus intact .5947). Two confirmation
  episodes are far too few to rank these interventions. In particular, paper
  XOR's pilot exact-filter reference was below intact (.5366 versus .5947),
  illustrating the variance rather than invalidating the action map.
- **Policy-implied inverses are not accurate Bayes-belief recoveries.** Mean
  absolute probability errors were .180 for Reward-State, .121/.151 for the
  two eps02 factors, .115/.124 for v2, and .106 for Full-State. Reports also
  save rho and signed-delta MAE. These are distances from a particular
  uniform-initialized inverse of a fitted map, not proof that the agent fails
  to track Bayes belief: the forward map can be misspecified and its inverse
  nonidentifiable. Forward-map held-out calibration was **not** measured in
  these pilot reports, so the gap cannot settle that question.
- **The local non-target Jacobian constraint trades reward for specificity.**
  Moving reward's projected Eq. (4) write earned 1.0879, with factor-1 reward
  .4429 and factor-1 action changes .0811. Paired factored Eq. (4) earned
  1.1816, .4731 and .1680; paired joint Eq. (4) earned 1.2217, .4683 and .0762.
  Thus projection did not dominate the joint baseline. XOR targets both
  factors, so there is no non-target Jacobian: its `null` direction is the
  factored Eq. (4) direction, and any difference there is **only retuning
  lambda**, not a decoder improvement.

## Protocol and implementation

The new `decoders.py` and `causal.py` live in the sibling ladder decoding
folder. The shared runner restores/checks the historical checkpoint hash and
process arrays, and reuses historical filters, action tables, QMDP tables,
target anchors, residual hooks, and JAX environment/model code. There is no
retraining. Latest block-4 residuals are before final normalization.

Pilot fit/test trajectories use independent seeds 42/44; DAS rank selection
uses 43. Each split has **two full 1,024-step episodes**, discarding 8 ladder
or 64 factor warmup steps for fitting. Closed-loop tuning and confirmation use
the historical random streams (1042/2042 for ladder; archived seeds for
transfer), but only two episodes each. JAX matmul precision is highest; new
least squares/Jacobian pseudoinverses use rcond 1e-6. Historical factor Eq. (4)
solver behavior is retained. Eq. (4) point strengths are copied from the
archive; other reward writes use the historical grid and smallest maximizing
strength. Ladder uses the first strength at/below the paired random-action
tuning occupancy, otherwise the smallest minimizing strength. Delta mirror
is fixed at 1. Norm-matched nulls use the factored Eq. (4) norm, decoder-null
direction and archived strength; they are not separately optimized controls.

Default combinations actually piloted:

| Name | Decoder / representation | Write |
|---|---|---|
| policy | KL-fit exact-belief-to-logits map; simplex inverse of observed policy | gradient, using inverse as current belief |
| das | rank 1/2/4/8 orthonormal interchange basis; affine belief matching for donors | bank donor coordinates in selected causal basis |
| band | affine probe over latest and lags 1/4/8 residuals | gradient over latest residual; past positions held fixed |
| mlp | one hidden layer, tanh, width 64, standardized inputs | gradient |
| null | original affine Eq. (4) decoder | Eq. (4) direction projected off non-target marginal log-policy Jacobian |

DAS fits 300 Adam steps through QR. Moving training targets factor-2 marginal
policy KL plus beta times factor-1 KL to intact. XOR trains on factor-1,
factor-2 and both-factor action-fixed mirrors, using joint-policy KL; ladder
uses its A/B symmetry, relabeling Full-State actions too. Block-2 training
patches only the latest position of a 129-position residual window and
recomputes the frozen suffix. Held-out fidelity is to the full replay policy;
its latest-position reference patch need not have fidelity 1 at block 2.
Closed-loop DAS uses the block-4 basis selected by held-out-selection joint
KL, not the held-out test score. Beta is 1 for moving and inactive when
there is no preserved factor. The earlier-block results are **offline only**.

Donor matching uses a subsampled fitting-activation bank and its decoded
beliefs. Moving donors must preserve the decoded other factor to .05 in every
probability coordinate; unsupported rows abstain. The targeted coordinates
use the nearest supported bank belief. This makes the *donor coordinates*
real, not a guarantee of an exact point target or reachable full history after
subspace replacement. `--write donor|gradient|null` exposes alternative
combinations; the full cross product has **not** been piloted. Gradient writes
optimize decoded-target MSE plus beta non-target policy KL plus .01 mean-square
activation displacement. Null-space constraints are first-order, not exact
finite-write invariance guarantees.

## Comparison to previous findings

The historical numbers are context, **not paired with this reduced pilot**:

- Reward-State seed 42: full-strength Eq. (4) → twin rho occupancy
  .1158 → .0816, mirror actions .0086 → .3075. The new pilot's DAS also
  has substantial mirror spillover; it has not improved that tradeoff.
- v2 XOR seed 42: old → twin factored .5845 → .5676; joint .5478 → .5557.
  The new DAS pilot exceeds these archive values numerically, but its two
  confirmation episodes and reduced decoder fit prohibit a fair ranking.
- eps02 XOR seed 42: intact .4807, factored old .4841, joint old/twin .4807.
  The pilot intact .5947 alone shows why cross-budget comparisons mislead.
- The replay study's three-seed moving means favored joint reachable Eq. (4)
  (1.043 reward, .754 gap closed, .391 factor-1 reward, .059 factor-1 changes).
  There is no moving-reward twin result in the supplied twin findings.

See the original [ladder replay](../belief_quotient_replay_edit_2026_11/findings.md),
[factor replay](../reward_factors_replay_edit_2026_11/findings.md), and the twin
branch `origin/devin/1795127017-twin-steering` findings. No new three-seed
conclusion should be inferred from this stage.

## Commands and budget

One job per `(test, seed, method)`; no launch/provisioning is performed:

```sh
uv run --group jax python -m experiments.reward_factors_causal_decoding_2026_11.run \
  --test TEST --seed SEED --method METHOD --pilot --workers 1
# TEST: ladder, ladder_full, moving, xor_eps02, xor_v2
# SEED: 42, 43, 44; only 42 was run here.
# METHOD: eq4, eq4_joint, policy, das, band, mlp, null, norm_null, intact, map
# map is a transfer reference only; eq4_joint duplicates eq4 on ladder.
```

For GPU boxes use `--group jax-cuda`, omit `--pilot`, and pin this branch plus
the harness revision when launching. The runner does not force CPU, so JAX
uses the installed backend. Add `--beta 0.1`, `--beta 1`, or `--beta 10` for a
beta sweep; filenames distinguish nondefault beta and write combinations.
Each DAS job sweeps k=1,2,4,8 at blocks 4 and 2 internally. Rank is selected
offline before closed-loop evaluation. Use tuning/selection data, not
confirmation reward, to select beta or a write family. Existing outputs
refuse overwriting.

For example, these emit independent farmable commands (do not execute them
on this CPU host as one unbounded batch):

```sh
for test in ladder ladder_full moving xor_eps02 xor_v2; do
  for seed in 42 43 44; do
    for method in policy das band mlp null; do
      for beta in 0.1 1 10; do
        printf '%s\n' "uv run --group jax-cuda python -m experiments.reward_factors_causal_decoding_2026_11.run --test $test --seed $seed --method $method --beta $beta --workers 1"
      done
    done
  done
done
```

Full mode uses 24 fitting episodes, 12 held-out episodes, 16/32 ladder
tuning/confirmation episodes and 64/256 transfer episodes. Historical Eq. (4)
readouts use 12,000 ladder rows or the archived 60,000 transfer rows; new
decoders use the independent fit trajectories. Gradient optimization grows
from 20 to 60 steps. One final-job pilot pass consumed about **1.10 allocated
core-hours**, excluding discarded preliminary pilots; policy inversion took
roughly 110–266 seconds/job on two allocated cores and dominated cost.

**Planning estimate: 300–800 core-hours** for three seeds, all five tests,
default decoder/write pairings, three beta settings, historical lambda grids
and offline k sweep. This is a rough CPU extrapolation, not a benchmark:
the transfer decision budget grows by about 39x and gradient iterations by
3x; vectorization and compilation overhead make scaling uncertain. Running
every decoder with all three write operators can approach 1,000–2,400
core-hours under the same conservative extrapolation. GPU hours were not
measured. Start by checking v2 DAS and the specificity controls with larger
episode budgets before paying for that full cross product.

Checks: `uv run --group jax pytest -q tests/test_causal_decoding.py
tests/test_belief_quotient_replay_edit.py tests/test_reward_factors_replay_edit.py`
printed `14 passed in 11.90s`. The tables regenerate with
`uv run --group jax python -m experiments.reward_factors_causal_decoding_2026_11.summarize`.
The result-hash manifest is in `results/pilot_manifest.json`. UI/visual
verification is not applicable to this numerical CLI study.
