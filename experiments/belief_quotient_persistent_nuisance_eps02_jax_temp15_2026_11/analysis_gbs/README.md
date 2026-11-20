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
quotients, or probe targets. Learned initial states and the entire episode are
differentiated with `lax.scan`; all positions have equal weight. Updates are
symbol-specific affine, GRU, tanh RNN, or action-conditioned HMM Bayes filters
with row-softmax transitions/emissions. HMM state has `k+1` probabilities and
stores the first `k` coordinates. `k=0` is a constant predictor in every class.
Heads have one tanh hidden layer. All starts, rates, and head widths are selected
by validation KL; the best iterate at ten equally spaced checks is retained.
Nonfinite candidates are recorded; an entirely failed grid fails closed.
Minibatches contain entire episodes, sampled with replacement from training.
Initializations/rates run sequentially, not vmapped: all arrays, scans, gradients,
and optimizer updates are nevertheless JIT-compiled and device-resident.

The positive control must condition on the *current* action, which is not in
`h_t`. Regressing the randomly action-conditioned target onto `g(h_t)` alone
would introduce irreducible label variation. We instead use the **same pipeline**
with one next-token softmax head per current action and average KL equally over
actions. `g` still receives only past token/action symbols. The stored dataset
includes both the requested executed-action predictive distribution and all
action-conditional distributions. This stronger, explicitly action-conditioned
control predicts tokens, not actions; the output tensor is `(action, token)`.

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
rates 0.001/0.003, three initializations, 2,000 updates, minibatches of 32 episodes.
Every grid/count is CLI-configurable; see `run --help`.

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
compact results are published to Git. `--publish-and-destroy` invokes the existing
Vast publisher/destructor **only after success and verified upload**. Failures
leave the box intact for recovery; the provider watchdog remains a cost backstop.

## Six-box plan

On each already-provisioned box, check out the launch branch and run GPU setup
above. Ensure `VAST_PUBLISH_BRANCH` (or the provisioner's launch-branch setting),
Vast instance identity, `GH_TOKEN`, `VAST_API_KEY`, and `B2_*` are available as
required by the existing publisher. Use a unique campaign prefix:

```sh
.venv/bin/python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 1 --run-prefix gbs-v1 --publish-and-destroy
.venv/bin/python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 2 --run-prefix gbs-v1 --publish-and-destroy
.venv/bin/python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 3 --run-prefix gbs-v1 --publish-and-destroy
.venv/bin/python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 4 --run-prefix gbs-v1 --publish-and-destroy
.venv/bin/python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 5 --run-prefix gbs-v1 --publish-and-destroy
.venv/bin/python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.launch --box 6 --run-prefix gbs-v1 --publish-and-destroy
```

Boxes 1–3 run Trivial, Reward-State, and Full-State Quotient; boxes 4–5 run Joint
Reward and Marginal Reward; box 6 runs all five positive controls sequentially,
publishing/destructing only after the fifth succeeds. Each uses seeds 42–44.
No process rents a box. The control box has substantially more work than the
other boxes: this intentionally follows the requested six-box allocation.

```sh
uv run --group jax pytest -q -s tests/test_gbs_extraction.py
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.benchmark
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.summarize --output findings.csv
```

The summarizer merges completed results across both studies, refusing duplicate
agent/class/k entries rather than silently mixing campaigns. Use `--inputs` to
select a specific campaign's run directories. It never reads large artifacts.
