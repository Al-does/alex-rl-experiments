# Generalized belief-state extraction

Post-hoc study of the final manuscript agents, seeds 42–44. No agent training,
provider provisioning, or instance rental occurs here. See `~/ghmm_post.md`
Sections 3.2–3.3 for the predictive recurrent-state objective.

## Method and boundaries

`data.py` discovers the unique completed nonsmoke run per seed from committed
recipes/manifests, selects its final step, downloads its native NPZ, and verifies
both size and SHA-256 against the canonical B2 durability manifest before loading
with the existing model loader. Source run/revisions and manifest hash accompany
every compact fit report. Existing cached checkpoints are reverified too.

Data are independent complete 1,024-position episodes, including reset, under
the actual stochastic policy `softmax(logits / 1.5)` with no epsilon mixture.
The existing cached transformer rollout is reused; no model is reimplemented.
Policy distributions are reconstructed from that rollout's final-normalized
embedding and the checkpoint's existing policy head. Histories use integer
`previous_action * n_tokens + token` (9 or 81 symbols). At reset the environment
has an all-zero previous-action block: the integer action slot is zero, and
position zero is explicitly treated as reset (no transition in HMM filters).
There is no fabricated executed action before reset.

Truth-free fitting (`core.py`) receives only integer histories and predictive
distributions. It never receives beliefs, environment matrices, rewards,
quotients, or probe targets. Learned initial states and recurrent updates are
differentiated with `lax.scan`; all positions have equal weight. Updates are
symbol-specific affine, GRU, tanh RNN, or action-conditioned HMM Bayes filters
with row-softmax transitions/emissions. HMM state has `k+1` probabilities and
stores the first `k` coordinates. `k=0` is a constant predictor in every class.
Heads have one tanh hidden layer. All starts, rates, and head widths are selected
by validation KL; validation runs every 1,000 updates and at the final update,
retaining each candidate's best parameters independently.
Nonfinite candidates are recorded; an entirely failed grid fails closed.
Minibatches contain entire episodes, sampled with replacement from training.
By default, training traverses each sampled episode in consecutive 256-position
TBPTT windows, carrying state (with stopped gradients) through all 1,024 positions
before sampling new episodes. Each window is one Adam update; there are still
20,000 updates per candidate. The batch increases from 32 to 128 episodes, so
each update still uses 32,768 positions and total training exposure is unchanged.
A compiled outer scan executes up to four updates per dispatch. Validation
boundaries preserve both carry and optimizer state. Initial-state gradients are
learned at reset, not at artificial window boundaries. Validation, test KL,
state extraction, and geometry always use complete episodes from reset.
Use `--chunk-length 0` for full-episode BPTT; positive chunk lengths must divide
the episode length. Recurrence scans are not unrolled; tiny recurrent matrix
products use explicit multiply/reduce operations rather than small GEMMs.
For each class/dimension/width, all initializations × learning rates train in a
single vmapped, jitted step with independent Adam moments, gradient clipping,
minibatch RNG streams, and validation-best snapshots. Widths run sequentially.
The six default candidates share shapes; the data and updates stay on-device.
An unroll-16 affine k=6 GPU probe compiled in 5.83 seconds but did not finish its
first update within five minutes. Removing unroll completed compilation in
2.59 seconds and the first four-update block in 0.024 seconds on Box 6's 4090.

The positive control must condition on the *current* action, which is not in
`h_t`. Regressing the randomly action-conditioned target onto `g(h_t)` alone
would introduce irreducible label variation. We instead use the **same pipeline**
with one next-token softmax head per current action and average KL equally over
actions. `g` still receives only past token/action symbols. The stored dataset
includes both the requested executed-action predictive distribution and all
action-conditional distributions. This stronger, explicitly action-conditioned
control predicts tokens, not actions; the output tensor is `(action, token)`.
Beliefs and control targets are recomputed from complete token/action histories
on CPU in NumPy float64, using the source recipe's matrices. Priors, transition
rows, emission rows, posteriors, and final predictive rows are normalized. This
avoids GPU reduced-precision matrix-product drift without weakening the fit's
distribution validation. Fit arrays are transferred to JAX's usual float32.

After selection, `compare.py` alone opens truth arrays. It reports affine and
32-unit tanh-MLP held-out R² in both directions for full beliefs, the Reward-State
Quotient, and both Reward Relevance factor marginals. All least-squares fits use
`jax.default_matmul_precision("highest")`. Small MLP probes use training histories
only, 300 Adam steps (configurable), without test selection. Constant-target R²
is JSON null. The Reward-State Quotient comparison is its exact two-state filter
driven by A/B-merged observations; in other ladder conditions this is a
counterfactual comparison target, not an assertion of exact lumpability.

Commutation is mean squared Euclidean discrepancy between `M U(g,x)` and
`Bayes(Mg,x)` over nonreset updates, compared with a constant train-mean decoder.
Affine outputs are not silently projected to the simplex: invalid-simplex and
nonfinite rates are explicit. HMM matrices are compared jointly up to an
**exhaustive** shared state permutation for matching state counts (including all
9! permutations for Reward Relevance). Two-state ladder filters are additionally
compared after merging learned A/B emission columns with the exact Reward-State
Quotient matrices from manuscript Appendix “Exact Reward-State quotient POMDP”.
Matrix agreement is descriptive, not an identifiability theorem with an MLP head.
Undefined, near-constant, or nonfinite R² values are JSON null with a reason.
Probe exceptions are recorded per comparison and do not discard the fitted model
or stop the remaining fits. The JSON writer also replaces any residual nonfinite
metric with null and records its path; the summarizer accepts missing probes.

The elbow is the smallest **swept** `k` whose test KL is within an absolute
0.001 nats of the best swept test KL in its agent/class (configurable tolerance).
This is a descriptive test-set elbow, not independently validated dimension
selection. A missing low-KL fit is an optimization limitation, not a lower bound.

## Runs

CPU setup: `uv sync --group dev --group jax`. GPU setup on user-provisioned boxes:
`uv sync --group dev --group jax-cuda`, then use the activated `.venv/bin/python`.
Do not set `JAX_PLATFORMS=cpu` on GPU boxes. Both dependencies and interpreter are
locked in the existing repository configuration; no configuration changes needed.

The default sweep is 2,048/128/256 train/validation/test episodes, `k=0..6`
(ladder) or `0..10` (Reward Relevance), all four classes, widths 32/128, learning
rates 0.001/0.003, three initializations, 20,000 updates per candidate, validation
every 1,000 updates, minibatches of 128 episodes per candidate, 256-position
carried-state TBPTT windows.
Every grid/count is CLI-configurable; see `run --help`.
In particular, use `--updates`, `--val-interval`, and `--chunk-length` to change
the fit budget and gradient horizon.

```sh
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.run \
  --condition variant_2 --seeds 42 --classes hmm --ks 2 --widths 32 \
  --lrs .003 --inits 1 --updates 200 --batch 4 --rollout-batch 2 \
  --train-episodes 4 --val-episodes 2 --test-episodes 2 --probe-updates 200 \
  --smoke --no-upload --run-name my-smoke
```

For Reward Relevance substitute module
`experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11.analysis_gbs.run`
and `--condition reward_both` or `--condition reward_factor_1`.

All datasets and fitted parameters are ignored `analysis_gbs/artifacts/<run>/`.
Compact per-fit JSON, summaries, and terminal manifests go under
`analysis_gbs/results/<run>/` (or ignored `.smoke/<run>/`). Existing run names are
never overwritten. With configured B2 credentials the existing harness uploader
uploads the artifact tree and writes canonical/legacy durability indexes. Only
compact results are published to Git. Candidate summaries are atomically written
immediately after each width group finishes. Selected-fit metrics are atomically
written before geometry probing and then augmented with the probe results. Thus
completed fits remain on disk if a later fit or comparison fails.

There is **no custom publication or teardown path**. The provisioner's
`run_remote.sh` invokes the launch command in its activated venv, captures its
exit code, and on success calls `devops.vast.self_destruct`. The experiment leaves
an uncommitted `results/<run>/run_manifest.json` with terminal `status`, a
`remote_artifacts.status` of `completed`, and the uploader's sibling
`remote_artifacts.json`. These satisfy `pending_run_manifests` and
`required_durability_completed` in required durability mode. The harness checks
all pending manifests, publishes to the launch
branch, and destroys only after successful publication and durability checks.
A failed command leaves the box up unless teardown-on-error was explicitly
enabled. Partial JSON remains available on disk; partial artifacts are not
claimed durable until the uploader has completed. The provider watchdog remains
a cost backstop. Do not use `--smoke` or `--no-upload` for paid campaign commands:
they deliberately cannot satisfy required durability.

## Six-box plan

On each already-provisioned box, check out the launch branch and run GPU setup
above. Ensure `VAST_PUBLISH_BRANCH` (or the provisioner's launch-branch setting),
Vast instance identity, `GH_TOKEN`, `VAST_API_KEY`, and `B2_*` are available as
required by the existing publisher. `provision up --branch` supplies the launch
branch setting. Use a unique campaign prefix. These are the exact `--run` strings
(the setup command is included so JAX/CUDA dependencies are installed):

```sh
uv sync --group jax-cuda && python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 1 --run-prefix gbs-v2
uv sync --group jax-cuda && python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 2 --run-prefix gbs-v2
uv sync --group jax-cuda && python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 3 --run-prefix gbs-v2
uv sync --group jax-cuda && python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 4 --run-prefix gbs-v2
uv sync --group jax-cuda && python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 5 --run-prefix gbs-v2
uv sync --group jax-cuda && python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 6 --run-prefix gbs-v2
```

Boxes 1–3 run policy then control for Trivial, Reward-State, and Full-State
Quotient respectively; boxes 4–5 run Joint Reward and Marginal Reward policy.
Box 6 runs both Reward Relevance controls sequentially with width 128 only,
returning successfully to the harness after both succeed. Each uses seeds 42–44.
No process rents a box. This is the user's revised six-box allocation.

Expected optimizer budgets (validation/probing excluded):

| Boxes | Independent fits | Updates per fit | Candidate updates | Vmapped step calls |
|---|---:|---:|---:|---:|
| 1, 2, or 3 | 2,016 | 20,000 | 40,320,000 | 6,720,000 |
| 4 or 5 | 1,584 | 20,000 | 31,680,000 | 5,280,000 |
| 6 | 1,584 | 20,000 | 31,680,000 | 5,280,000 |

Each vmapped update advances six independent fits. TBPTT batches up to four such
updates per compiled call. Step timing uses one extra discarded block per width
group, not included in those optimization budgets.

4090 repair measurements and eight real 500-update GPU smoke fits are in
[`results/gpu_repair_4090_rolled/profile.json`](results/gpu_repair_4090_rolled/profile.json).
All class/k timing probes compiled in under six seconds. At 20,000 updates,
conservative training/validation/compile projections are 13.34 hours for each
ladder policy+control box, 11.35 hours for each Reward Relevance policy box, and
13.08 hours for both width-128 Reward Relevance controls. Applying width-128
timings to both widths and the largest measured k cost to every dimension is
conservative, not a complete campaign measurement. A planning allowance of 20%
plus one hour for collection/probes/upload gives 17.0, 14.6, and 16.7 hours
respectively. Full-size probe and publication overhead was not measured.

Reproduce one class/dimension timing probe (policy and control) with a hard
five-minute timeout; repeat for each class and `ladder2`, `ladder6`, `rr10`:

```sh
timeout -k 10s 300s python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.profile \
  --baseline-ref 2322c49 --modes after_chunk --cases ladder6 --classes affine
```

For a CPU launch-command smoke, append overrides after `--`:

```sh
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch \
  --box 2 --run-prefix my-launch-smoke -- --smoke --no-upload --seeds 42 \
  --classes hmm --ks 2 --widths 32 --lrs .001 .003 --inits 3 \
  --updates 200 --val-interval 50 --batch 2 --rollout-batch 2 \
  --train-episodes 4 --val-episodes 2 --test-episodes 2 --probe-updates 20
```

```sh
uv run --group jax pytest -q -s tests/test_gbs_extraction.py
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.benchmark
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.summarize --output findings.csv
```

The summarizer merges completed results across both studies, refusing duplicate
agent/class/k entries rather than silently mixing campaigns. Use `--inputs` to
select a specific campaign's run directories. It never reads large artifacts.
