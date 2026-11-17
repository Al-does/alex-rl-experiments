# PPO and auto-alpha SAC: eps02 ladder probes

All 30 frozen final actors (step 2,012,217,344; seeds 42/44/46/49/52), plus
their saved initializations, were analyzed locally on the eight CPU cores.
The PPO Reward-State Quotient smoke preceded the full run. All 60 checkpoint
references passed SHA-256 and byte-size checks against their canonical B2
durability manifests. No training, paid compute or steering was run.

## Three-line interpretation

1. **PPO reproduces the REINFORCE-like pattern:** Reward-State Quotient rho is highly accessible, delta much less so; within-class rho improves and delta declines from matched initialization in all five seeds.
2. **SAC auto-alpha supports preferential rho accessibility, but not the same selective loss:** Reward-State Quotient delta remains more accessible than in REINFORCE/PPO and within-class delta improves over initialization in all five seeds.
3. **The exact quotient is easier to decode than full belief for both algorithms**, but this is not independent evidence of a Bayesian implementation or causal use; five-seed, policy-dependent comparisons do not establish equivalence across algorithms.

## Reward-State Quotient summary

Global columns are normalized error **1-R² (lower is better)**. Within-class
columns are **R², matched initialization → trained (higher is better)**.

| Algorithm | b error | rho error | delta error | exact c[B] error | Within-class rho | Within-class delta |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE, all 15 | 0.1019 | 0.0173 | 0.2405 | 0.0158 | 0.9021 → 0.9684 | 0.8748 → 0.7977 |
| REINFORCE, same 5 | 0.1024 | 0.0166 | 0.2441 | 0.0150 | 0.9025 → 0.9695 | 0.8770 → 0.7954 |
| PPO, same 5 | 0.1080 | 0.0036 | 0.2820 | 0.0026 | 0.8988 → 0.9921 | 0.8741 → 0.7749 |
| SAC auto-alpha, same 5 | 0.0522 | 0.0115 | 0.1194 | 0.0117 | 0.8870 → 0.9866 | 0.8682 → 0.8969 |

The rho/delta separation survives the head-blind control: trained within-class
rho/delta R² are 0.9919/0.7743 for PPO and 0.9861/0.8961 for SAC; logit-only
values are 0.8847/0.0383 and 0.8161/0.0272. Thus the full embedding results are
not exhausted by the two immediate logit-difference directions.

See **[tables.md](tables.md)** for the complete side-by-side means across all
three conditions and per-seed global, coarse and within-class b/rho/delta
scores. **[summary.json](summary.json)** and **[per_seed.csv](per_seed.csv)**
also include head-blind/logit controls and all 15 REINFORCE reference seeds.
Compact agent JSONs contain per-class counts, raw global errors, checkpoint
hashes/B2 keys, source hashes, repository revisions and wall times.

## Coverage and limits

- No global support-constraint replacements were needed (zero raw global
  errors above one). The adapter preserves the paper's selective constraint
  procedure and raw values; within-class scores are not retrospectively clipped.
- All Reward-State Quotient and Full-State Quotient within-class evaluations
  cover every held-out row. Most Trivial Quotient policies use one action, so
  their scores are descriptive global accessibility, not action-region evidence.
- SAC Trivial Quotient seed 44 has fit action counts `[0,60000,0]` and test
  counts `[0,79999,1]`: the unsupported held-out class row is explicitly excluded
  from within-class scores only. Seed 46 has fit `[0,59996,4]` and test
  `[0,79989,11]`, making its rare-class fit particularly small.
- SAC Trivial Quotient seed 44 has unstable matched-init within-class b R²
  -8.9796 (delta -39.4200), strongly distorting its five-seed initialization
  mean. A supplementary [diagnostic](trivial_seed44_diagnostic.json) associates
  97.98% of its b SSE with 1,623 held-out rows containing Tilt Y in a recent
  history window; the same criterion identifies 926 fit rows. This is an
  association, not a proven root cause or a reason to replace the primary score.
- Full-State Quotient merged-token filtering is **not an exact quotient**.
  Each target's error uses its own variance denominator. PPO's merged target
  being easier than full b is not evidence that Full-State control is lumpable.
- Greedy histories differ across trained policies/algorithms; matched-init
  replay fixes histories only within an agent. Raw global delta is not
  inaccessible: the result concerns relative linear accessibility and changes
  from initialization, not total information erasure.
- A/B behavioral mixing, frozen shuffle/random-input controls, within-class
  perturbation/permutation controls and checkpoint curves were intentionally
  skipped. Existing REINFORCE results were reused, not rerun.

## Reproduction and checks

Commands, immutable source revisions and vendored-function provenance are in
the parent README. Thirty final self-replays matched the collected embeddings
exactly (maximum absolute error 0). The full run used CPU affinities 0–7;
individual agent wall times were at most 41.09 seconds (summed agent time
915.17 seconds, not parallel job elapsed time). The optional rare-history
diagnostic is reproducible with `python -m
experiments.algorithm_generality_probes_eps02_2026_11.diagnose_trivial`.

- Adapter/report plus harness probe tests: `29 passed in 6.71s`.
- Existing PPO/auto-alpha SAC recipe tests: `26 passed in 8.14s`.
- Vendored source AST audit: `12 functions exactly match source ASTs`.
