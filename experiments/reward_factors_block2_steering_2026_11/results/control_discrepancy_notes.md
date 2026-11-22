# B4 XOR controls: same-host triangulation and counterfactuals

**The exact low-level cause remains unexplained.** The three original differences
are reproducibly environment-sensitive, not evidence of a layer advantage or of
a different encoder-training source. No historical result was replaced.

| Seed-42 control | Saved encoder / local CPU λ, reward | Fresh causal / rental CPU λ, reward |
|---|---:|---:|
| eps02 marginal | 1.00, .4442138671875 | .85, .45989990234375 |
| v2 marginal | .90, .555572509765625 | .90, .5528564453125 |
| v2 joint | .95, .564239501953125 | .95, .5645751953125 |

## Verified execution paths

`control_audit.py` invokes the unchanged saved encoder runner, the current B2-study
encoder runner at site 3, and the current causal runner at site 3. Each runs both
XOR cohorts and marginal/joint affine-exact/Eq.(4) controls: **12 full-budget
controls per host**, with 16 selection and 32×1,024 confirmation episodes.

All three implementations match **exactly within each host**, including the
complete λ-selection reward grids, confirmation rewards, factor-1 rewards and
action-change fractions. The local Intel host reproduces the saved-encoder
values; the rental AMD EPYC 7282 reproduces the fresh-causal values. This also
reproduces the matching eps02 joint control in both environments. Checkpoint
hashes agree. Captured exact-belief fitting arrays are byte-identical across
hosts; the 60,000-row activation fitting arrays are not. Fitted float32 decoder
and encoder parameters differ: maximum absolute weight differences are
2.42e-8–5.77e-8 and 4.77e-7–9.54e-7 respectively; target anchors agree.

The producer being instrumented is the historical `transfer.prepare` path,
reached through each runner's actual affine-exact/Eq.(4) preparation. The
counterfactual `control_swap.py` replaces the returned marginal/joint maps in
`run.prepare`; `run.worker` then consumes those supplied decoder/encoder maps
through its original Eq.(4) branch, selection loop and closed-loop rollout.
Keys, shapes and dtypes are checked before substitution. Thus these comparisons
exercise the real control path, not a separate implementation of the write.

## Attempts to disprove fitting-only / environment explanations

An additional **24 full-budget controls** export and transplant native maps in
both directions, or repeat the fixed-cloud-map experiment locally with an AVX2
cap or a separate Python-3.14 environment. Total: **48 full-budget controls**.

| Rollout environment / fitted maps | eps02 marginal λ, reward | v2 marginal λ, reward | v2 joint λ, reward |
|---|---:|---:|---:|
| Local / local | 1.00, .444214 | .90, .555573 | .95, .564240 |
| Rental / rental | .85, .459900 | .90, .552856 | .95, .564575 |
| Local / rental | 1.00, .441956 | .90, .552856 | .95, .564575 |
| Rental / local | .85, .459900 | .90, .553284 | .95, .564575 |
| Local AVX2 cap / rental | 1.00, .441956 | .90, .552856 | .95, .564575 |
| Local Python 3.14 / rental | 1.00, .441956 | .90, .552856 | .95, .564575 |

Fitting differences alone are **insufficient**: identical supplied maps still
give different eps02 results on the two hosts. The v2 joint transplant agrees
in one direction but not in reverse. The AVX2 cap and Python-version switch do
not resolve this. Matching the displayed v2 rewards is not necessarily equality
of every tuning or action-change value; the JSON retains those differences.

Both environments use JAX/jaxlib 0.11.1 and NumPy 2.5.1. The default local Python
is 3.13.18; the rental uses 3.14.10, including the active seed campaign. Both
have x64 disabled, default matmul precision, Threefry PRNG and simplified JAXPR
constants disabled, with no default `XLA_FLAGS`. Float32 hashes of Q/reward
tables match (this comparison does not assert float64 table identity).

The evidence rules out a distinct Eq.(4) formula, exact-versus-decoded fitting
source, selected site, checkpoint, episode budget, or selector implementation
as the explanation. It supports **host-dependent numerical sensitivity in
fitting and rollout**, but does not identify the responsible primitive/kernel
or the first divergent action. Do not claim a particular CPU ISA, Python ABI,
or argmax tie mechanism was established. The saved-versus-fresh B4 provenance
and these discrepancies remain a limitation of the layer comparison.

`control_discrepancy_audit.json` preserves all six counterfactual experiments,
full tuning grids, parameter hashes, exact same-host checks and input hashes.
`control_audit_local_seed42.json` and the rental-published
`control_audit_cloud_seed42.json` preserve the independent three-runner audits.
Raw profiles, model checkpoints and portable maps remain ignored artifacts.
