# Eight arms, five seeds — what the factoring campaign actually shows

**Campaign** `20260731T190000Z-eight-arms-five-seeds` · 8 arms × 5 seeds (42–46) ×
2.5M env steps · clipped PPO, `gamma = 0`, `lambda = 0`, `delay = 1`,
`d_model = 64`, 4 layers, context 10 · salvaged from `origin/results` commits
`644f7d3` + `3f4b589`.

Machine-readable companions in this directory:

| file | contents |
| --- | --- |
| `campaign_summary.json` | original per-seed salvage; **raw MSE only — see Correction 1** |
| `normalized_summary.json` | per-arm normalized error per target, ceiling fractions, collapse flags |
| `metric_calibration.json` | what the study's metrics report for hand-built idealized representations |

Regenerate the latter two from the per-seed run summaries with:

```bash
uv run python -m experiments.mess3_feedback_factoring_cycle_1.campaign_analysis
```

---

## Executive summary

All 40 runs completed the full budget and every probe beats its shuffled-label
permutation null at `p = 0.001`. The campaign is sound. But **two of the
reported headline numbers are artifacts, three of the six preregistered
predictions fail, and one of those failures is a design flaw rather than a
result about the agent.**

What survives is a cleaner and more interesting finding than the one the study
set out to test:

> **A `gamma = 0` RL agent learns the minimal recursively-closed statistic
> sufficient for its reward, and the amount of world structure it retains is set
> by how much that statistic fails to close under its own dynamics.** Where the
> reward statistic and the world belief coincide, belief decodability improves
> and reaches R² ≈ 0.98–0.998. Where the world belief is strictly richer, the
> agent keeps the reward statistic and **actively destroys** the rest — including
> a factorization that is exact, lossless, and announced in the observation.

Combined with the sibling study `mess3_feedback_cycle_1`, the sign of the change
in belief decodability is predicted by that single structural fact in all seven
independent cases. That also means the sibling study's headline claim
(R² ≈ 0.998, "the network learns the Bayesian belief") **cannot** be read as
evidence of world modelling: in its design the world belief and the reward
statistic are the same object up to an invertible linear map.

---

## 1. What the metrics mean

Every checkpoint fits one held-out affine ridge probe per target on
post-final-LayerNorm activations from greedy rollouts, with disjoint train and
test seed streams (60k fit / 80k test, 1000 bootstrap resamples).

The reported quantity throughout this document is

```text
normalized error = global_mse_ratio = mse / target_variance = 1 - R²
```

so **0 is exact decoding, 1 is no better than predicting the target's mean**, and
above 1 is worse than the mean. `action_awareness_ratio` (AAR) is the `joint`
normalized error divided by the `blind` normalized error.

The seven targets form an information ladder. `joint` is the exact
guess-conditioned 9-state predictive belief; `blind` marginalizes over guesses
under a uniform law; `marginal` is the stacked single-HMM filter; `composite` and
`composite_blind` aggregate onto `s = m + phi`, the 3-state belief that fixes the
reward; `factor_m` and `factor_phi` are the two factor predictive vectors whose
outer product is the factored representation of Shai et al.

**`composite` is the reward-sufficient statistic.** The reward scores only the
composite sub-token `x`, so under `gamma = 0` nothing beyond the composite
predictive distribution can pay off directly.

---

## 2. Correction 1 — raw cross-arm MSE is invalid, and the ranking inverts

Target variance differs about threefold across the `epsilon` sweep, because
`epsilon` changes how much of the 9-state belief is identifiable at all. Raw MSE
therefore compares arms on different scales. Normalizing reverses the ordering
of the two most-discussed arms:

| arm | raw MSE (final) | target variance | **normalized error** | raw rank | true rank |
| --- | ---: | ---: | ---: | :-: | :-: |
| `no_feedback` | 0.000817 | 0.039947 | **0.0205** | 3 | 1 |
| `factoring_impossible` | 0.000278 | 0.007982 | **0.0350** | 1 | 2 |
| `factoring_impossible_blind` | 0.002900 | 0.007054 | **0.4112** | 4 | 4 |
| `factoring_free` | 0.009459 | 0.048265 | **0.1960** | 6 | 3 |
| `factoring_costly` | 0.006412 | 0.015049 | **0.4258** | 5 | 6 |
| `factoring_cheap` | 0.017978 | 0.042259 | **0.4253** | 7 | 5 |
| `factoring_free_blind` | 0.024178 | 0.044008 | **0.5506** | 8 | 7 |
| `deterministic_feedback` | 0.051950 | 0.064469 | **0.8058** | 9 | 8 |

The prior write-up concluded that `factoring_costly` had lower probe error than
`factoring_free` (0.0064 vs 0.0095). Normalized, `factoring_free` is more than
twice as good (0.196 vs 0.426). Any cross-arm statement built on raw MSE in that
report should be discarded.

---

## 3. Correction 2 — `action_awareness_ratio` cannot test Prediction 1

Prediction 1 asked for AAR below one in the sighted arms. It fails in three of
four. That failure carries **no information about action awareness**, because the
metric is not neutral between representations.

At `epsilon = 0` the joint belief is *exactly* the outer product of its factor
marginals — `joint_product_mse = 6.31e-34`, i.e. numerically zero. The `joint`
target is therefore a **bilinear** function of the `factor_m`/`factor_phi`
targets, and no affine probe can recover a product from a representation that
stores the two factors separately. (The identity is exact only at `epsilon = 0`;
`joint_product_mse` rises to 7.5e-4, 6.9e-3 and 9.2e-3 across the sweep, so the
penalty is worst precisely in the arm the study cares most about.)

`campaign_analysis.calibrate` measures the consequence directly. It fits the
study's own probe from hand-built idealized representations, no network
involved:

**`factoring_free` (`kappa = 0.7`, `epsilon = 0`), exact-filter Bayes accuracy 0.6252**

| idealized representation | dims | `joint` error | **AAR** |
| --- | :-: | ---: | ---: |
| exact joint belief | 9 | 0.0000 | **0.000** |
| composite + both factors | 9 | 0.1072 | 0.740 |
| composite + register | 6 | 0.3369 | 0.440 |
| composite only (reward-sufficient) | 3 | 0.8394 | 0.859 |
| **perfectly factored `[b_m, b_phi]`** | 6 | 0.2733 | **1.580** |
| — *trained agent, final* | 64 | **0.1960** | **1.365** |

**A perfectly factored representation — precisely what Prediction 4 predicts —
scores AAR = 1.58.** Every other idealized representation scores below one. So
Predictions 1 and 4 are mutually incompatible at `epsilon = 0`: satisfying one
forces the other to fail. The study cannot confirm both, and AAR above one is
the expected signature of the structure the study was built to detect.

Two further readings from the same table. The agent's `joint` error (0.196) is
**better** than a pure product representation (0.273), so it is not purely
factored — it retains register information beyond the two marginals. And its AAR
(1.365) sits between composite-only (0.859) and perfectly factored (1.580),
closer to the latter than any other bar.

AAR is also unstable when both targets decode poorly (a ratio of two numbers
near 1 is dominated by noise) and undefined when `blind` is degenerate, which is
why `deterministic_feedback` reports `null` for all five seeds. That is expected
behaviour at `kappa = 1`, not a bug.

---

## 4. The central finding — reward-sufficient compression

**In every arm where the composite is not sufficient for the joint, `composite`
is the only one of six targets whose decodability improves over training.
Everything else degrades.**

Change in normalized error from initialization to 2.5M steps (negative =
improved, mean over 5 seeds):

| arm | `joint` | **`composite`** | `factor_m` | `factor_phi` | `blind` | `marginal` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `factoring_free` (ε=0) | +0.116 | **−0.064** | +0.045 | **+0.245** | +0.061 | +0.066 |
| `factoring_cheap` (ε=0.3) | +0.158 | **−0.058** | +0.143 | +0.261 | +0.130 | +0.123 |
| `factoring_costly` (ε=0.85) | +0.023 | **−0.068** | +0.180 | +0.120 | +0.205 | +0.049 |
| `factoring_impossible` (ε=1) | −0.173 | −0.070 | +0.013 | +0.002 | +0.038 | −0.019 |
| `no_feedback` (κ=0) | −0.082 | −0.082 | −0.082 | degen | −0.082 | −0.082 |
| `deterministic_feedback` (κ=1) | −0.027 | −0.126 | −0.001 | −0.001 | degen | −0.022 |

The composite is decoded at **R² = 0.93–0.98 in every arm, essentially
independent of `epsilon`** (normalized error 0.064, 0.066, 0.036, 0.035, 0.021,
0.043 across the six sighted arms). The agent always builds the thing the reward
pays for.

### Training destroys freely available structure

The sharpest single number in the campaign. In `factoring_free`, where
`epsilon = 0` means the register sub-token reports `phi` *exactly* and the
factorization is provably lossless:

| checkpoint | `factor_phi` error | R² | register entropy |
| --- | ---: | ---: | ---: |
| **0 (untrained)** | **0.017 ± 0.015** | **0.983** | 0.297 ± 0.240 nats |
| 2,504,905 (final) | **0.262 ± 0.014** | **0.738** | 0.288 ± 0.030 nats |

An **untrained** network decodes the register factor better than a fully trained
one, in all five seeds, with no overlap between the two distributions. This is
not a variance artifact: the metric is already normalized by the target's
variance, and the register's own entropy is unchanged in the mean (0.297 → 0.288
nats). The wide spread at initialization is expected — an untrained policy's
guess distribution varies by seed, so the register it induces does too — but it
does not affect the conclusion, since the *worst* initial seed (0.039) still
decodes far better than the *best* final one (0.238).

The information is in the observation, exact, and free — and `gamma = 0` pays
nothing for it, so training discards it. This is active compression, not a
failure to learn.

### `deterministic_feedback` is the cleanest instance, not a broken arm

Its `joint` R² of 0.19 looks like a failure. The calibration says otherwise:

| | agent | idealized composite-only |
| --- | ---: | ---: |
| `joint` normalized error (κ=1) | **0.806** | **0.796** |

The agent is representationally **indistinguishable from storing nothing but the
reward-sufficient statistic**. The mechanism is exact: at `kappa = 1` each guess
rotates the register deterministically, so the composite belief is *self-
propagating* — it shifts cyclically by the previous guess, and no register
information is needed to maintain it. Its `composite` error is 0.043 (R² = 0.96)
while `factor_m` and `factor_phi` sit at 1.001 (fully undecodable) and register
entropy is exactly 0.

### Why "recursively closed" is the right qualifier

The graded evidence separates *reward-sufficient* from *recursively closed*:

- At `kappa = 1` the composite closes under its own dynamics, so the agent
  stores the composite alone (0.806 ≈ the 0.796 composite-only bar).
- At `epsilon < 1` the composite does **not** close — updating it requires
  register information — so the agent retains some register structure. Its 0.196
  beats the 0.273 product bar and vastly beats the 0.839 composite-only bar, but
  is far worse than its own 0.080 at initialization.
- In the sibling study the 3-state belief is both reward-sufficient and
  self-closing, and is retained almost perfectly (0.0023, R² = 0.998).

---

## 5. Cross-study comparison — the sibling supplies the missing control

`mess3_feedback_cycle_1` asks the same question with a different intervention
(`U_a = (1 - eta) T + eta R_a`, `eta = 0.10`) on a **3-state** MESS3 process, and
reports R² = 0.998 with a 17.6× improvement over training. That looks like a
contradiction. It is not.

Its probe target is the belief over 3 hidden states. The scored-token predictive
is `p = b @ E`, and the MESS3 emission at `alpha = 0.85` is invertible and
extremely well conditioned:

```text
E = [[0.85  0.075 0.075]      det  = 0.600625
     [0.075 0.85  0.075]      cond = 1.2903
     [0.075 0.075 0.85 ]]     rank = 3
```

So `b = p @ inv(E)`: **the world belief and the reward-sufficient statistic are
the same object up to an invertible linear map**, and an affine probe cannot tell
them apart. There is no gap to compress, and near-perfect decoding is exactly
what reward-sufficient learning predicts.

### The law

| study / arm | world belief | init | final | change |
| --- | --- | ---: | ---: | ---: |
| **sibling** (η=0.10, 3-state) | 3 — **equals** reward statistic | 0.0412 | 0.0023 | **17.6× better** |
| `factoring_impossible` (ε=1) | ~3 — `phi` unidentifiable | 0.2080 | 0.0350 | 5.9× better |
| `no_feedback` (κ=0) | 3 — `phi` frozen | 0.1020 | 0.0205 | 5.0× better |
| `deterministic_feedback` (κ=1) | 3 — composite self-closing | 0.8325 | 0.8058 | 1.0× (flat) |
| `factoring_costly` (ε=0.85) | 9 — weakly identifiable | 0.4025 | 0.4258 | 1.1× worse |
| `factoring_cheap` (ε=0.3) | 9 — mostly identifiable | 0.2676 | 0.4253 | 1.6× worse |
| **`factoring_free`** (ε=0) | 9 — **all identifiable** | 0.0800 | 0.1960 | **2.5× worse** |

Feedback itself is **not** the discriminating variable — the sibling study has
feedback and improves the most. What matters is whether the world belief exceeds
the reward statistic.

`no_feedback` rules out the obvious confounds: it shares the factoring study's
9-symbol observation, 12-dimensional input, `d_model = 64` and 2.5M-step budget,
yet improves 5×. So the degradation in the low-`epsilon` arms is not input
crowding, probe capacity, or budget.

### Consequence for the sibling study's claims

Its headline — "normalized MSE ~0.2% of target variance, R² ≈ 0.998, the network
learns a representation nearly linearly decodable into the exact predictive
Bayesian belief" — is numerically correct but **cannot discriminate between
"learns a world model" and "learns the reward-sufficient statistic,"** because
its design makes those identical. The factoring study can discriminate them, and
the answer is the latter.

Its counterfactual result survives with narrower scope: cosine 0.951 ± 0.021
between the decoded belief shift and the exact delay-one Bayes update under
action substitution is genuine action-conditioned modelling. But in that design
modelling one's own influence is *required* for reward, so it is consistent with
reward-sufficiency rather than evidence against it.

The study that looked clean and confirmatory turns out to be the uninformative
one; the study that looked messy and full of failed predictions is where the
science is.

---

## 6. Prediction-by-prediction verdict

### Prediction 1 — sighted arms drive AAR below one, blind arms stay above ❌ **untestable**

| arm | sighted | AAR init | AAR final | below 1? |
| --- | :-: | ---: | ---: | :-: |
| `factoring_free` | yes | 0.941 ± 0.131 | 1.365 ± 0.122 | no |
| `factoring_cheap` | yes | 1.520 ± 0.230 | 1.402 ± 0.117 | no |
| `factoring_costly` | yes | 2.480 ± 1.054 | 1.166 ± 0.071 | no |
| `factoring_impossible` | yes | 28.60 ± 39.86 | **0.790 ± 0.172** | **yes** |
| `no_feedback` | yes | 1.000 | 1.000 | exactly 1 by construction |
| `deterministic_feedback` | yes | null | null | undefined (`blind` degenerate) |
| `factoring_free_blind` | **no** | 2.673 ± 2.028 | 1.726 ± 0.221 | no ✓ |
| `factoring_impossible_blind` | **no** | 55.02 ± 34.20 | 3.725 ± 0.546 | no ✓ |

The blind half behaves as designed. The sighted half fails, but see
[Correction 2](#3-correction-2--action_awareness_ratio-cannot-test-prediction-1):
a perfectly factored representation scores 1.580, so the prediction is
unreachable at `epsilon = 0` by the hypothesized structure. **Do not report this
as a negative result about action awareness.**

### Prediction 2 — hiding the guess costs little at ε=0, a lot at ε=1 ⚠️ **half confirmed**

| pair | acc sighted | acc blind | acc drop | AAR sighted | AAR blind | AAR ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ε = 0.0 | 0.6096 ± 0.0080 | 0.5025 ± 0.0185 | **0.1071** | 1.365 | 1.726 | **1.26×** |
| ε = 1.0 | 0.6039 ± 0.0073 | 0.5045 ± 0.0123 | **0.0994** | 0.790 | 3.725 | **4.71×** |

The representational half is confirmed: blinding costs 4.7× more in AAR at
`epsilon = 1` than at `epsilon = 0`. The accuracy half is **falsified** — the
cost is essentially identical (0.107 vs 0.099).

**The prediction rested on a timing error.** With `delay = 1` the agent observes
the token from the *previous* step while guessing the *current* one. The register
report it can see describes `phi` at `t-1`, but the guess must anticipate
`phi_t = phi_{t-1} + a_{t-1}`. **The report is always one step stale, so it never
substitutes for knowing your own guess** — it only bounds how far back you must
remember. The README's rationale ("the register announces itself") should be
corrected: the register announces its *past* value.

### Prediction 3 — task success tracks each arm's own ceiling ✅ **confirmed**

| arm | ceiling | accuracy | % of ceiling | excl. collapses |
| --- | ---: | ---: | ---: | ---: |
| `factoring_free` | 0.6214 | 0.6096 ± 0.0080 | 98.1% | 98.1% |
| `factoring_cheap` | 0.6141 | 0.6037 ± 0.0033 | 98.3% | 98.3% |
| `factoring_costly` | 0.6114 | 0.6058 ± 0.0049 | **99.1%** | 99.1% |
| `factoring_impossible` | 0.6114 | 0.6039 ± 0.0073 | 98.8% | 98.8% |
| `no_feedback` | 0.6903 | 0.6670 ± 0.0317 | 96.6% | **98.7%** |
| `deterministic_feedback` | 0.6899 | 0.6514 ± 0.0714 | 94.4% | **99.0%** |
| `factoring_free_blind` | 0.6214 | 0.5025 ± 0.0185 | **80.9%** | 80.9% |
| `factoring_impossible_blind` | 0.6114 | 0.5045 ± 0.0123 | **82.5%** | 82.5% |

Strong. Every sighted arm reaches 98–99% of its own ceiling once the collapsed
seeds are excluded; the blind arms reach only 81–83%.

This also corrects a second claim in the prior report. "`no_feedback` is the
strongest task performer" is true in absolute terms but trivial: `kappa ∈ {0, 1}`
both have ceiling 0.690 (the passive optimum) while `kappa = 0.7` has 0.611–0.621.
It is an **easier task, not a better learner.**

### Prediction 4 — `factor_subspace_overlap` falls in `factoring_free` ❌ **falsified, metric suspect**

| arm | overlap init | overlap final | change |
| --- | ---: | ---: | ---: |
| `factoring_free` | 0.238 ± 0.180 | 0.491 ± 0.011 | **+0.253** |
| `factoring_cheap` | 0.238 ± 0.204 | 0.492 ± 0.076 | +0.254 |
| `factoring_costly` | 0.405 ± 0.112 | 0.610 ± 0.097 | +0.205 |
| `factoring_impossible` | 0.500 ± 0.001 | 0.600 ± 0.103 | +0.100 |
| `deterministic_feedback` | 0.516 ± 0.028 | 0.508 ± 0.017 | −0.008 |
| `factoring_free_blind` | 0.231 ± 0.132 | 0.521 ± 0.029 | +0.289 |

Overlap *rises* where it was predicted to fall. **But do not trust this metric
yet.** It sits at ≈0.5 in nearly every arm and checkpoint, and 0.5 is exactly
what two rank-2 read-outs produce when they share one direction and are
orthogonal in the other. For two random rank-2 subspaces in `d = 64` the expected
overlap is `2·2/64/2 ≈ 0.03`. That suspicious constancy needs a null baseline —
fit two probes to independent random targets from the same activations and
measure their overlap — before the prediction is adjudicated either way.

### Prediction 5 — factor targets degrade as ε rises ✅ **confirmed, clean ladder**

Final normalized error per target:

| arm | ε | `joint` | `blind` | `marginal` | `composite` | `comp_blind` | `factor_m` | `factor_phi` |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `factoring_free` | 0.00 | 0.196 | 0.144 | 0.142 | **0.064** | 0.110 | 0.134 | 0.262 |
| `factoring_cheap` | 0.30 | 0.425 | 0.306 | 0.326 | **0.066** | 0.106 | 0.442 | 0.535 |
| `factoring_costly` | 0.85 | 0.426 | 0.366 | 0.315 | **0.036** | 0.052 | 0.850 | 0.846 |
| `factoring_impossible` | 1.00 | 0.035 | 0.048 | 0.051 | **0.035** | 0.048 | 1.006 | 1.006 |
| `no_feedback` | 1.00 | 0.021 | 0.021 | 0.021 | **0.021** | 0.021 | 0.021 | degen |
| `deterministic_feedback` | 1.00 | 0.806 | degen | 0.049 | **0.043** | degen | 1.001 | 1.002 |
| `factoring_free_blind` | 0.00 | 0.551 | 0.323 | 0.313 | 0.482 | 0.270 | 0.325 | 0.706 |
| `factoring_impossible_blind` | 1.00 | 0.411 | 0.113 | 0.100 | 0.411 | 0.113 | degen | degen |

`factor_m` / `factor_phi` degrade monotonically in `epsilon`: 0.134/0.262 →
0.442/0.535 → 0.850/0.846 → 1.006/1.006 (fully undecodable). Register entropy
tracks it: 0.288 → 0.441 → 0.972 → 1.099 nats, with `log 3 = 1.0986` as the
ceiling. Confirmed.

### Prediction 6 — `marginal_belief_mse` stays above zero wherever κ>0 ✅ **confirmed**

| arm | κ | raw | normalized |
| --- | :-: | ---: | ---: |
| `no_feedback` | 0.0 | 0.000000 | **0.0000** |
| `factoring_free` | 0.7 | 0.015119 | 0.3133 |
| `factoring_cheap` | 0.7 | 0.016385 | 0.3877 |
| `factoring_impossible` | 0.7 | 0.003942 | 0.4938 |
| `factoring_free_blind` | 0.7 | 0.021138 | 0.4803 |
| `factoring_costly` | 0.7 | 0.007923 | 0.5265 |
| `factoring_impossible_blind` | 0.7 | 0.005276 | 0.7479 |
| `deterministic_feedback` | 1.0 | 0.057972 | 0.8992 |

Exactly zero at `kappa = 0` and 31–90% of target variance everywhere else,
matching the network-free single-HMM finding. The closed loop is not one stacked,
renormalized HMM.

---

## 7. The blind arms have a clean causal chain

Hiding the previous guess does not degrade accuracy directly — it destroys the
agent's ability to form the reward-sufficient statistic, and accuracy follows:

| arm | `composite` error | `composite` R² | accuracy | % of ceiling |
| --- | ---: | ---: | ---: | ---: |
| `factoring_free` | 0.064 | 0.937 | 0.6096 | 98.1% |
| `factoring_free_blind` | **0.482** | **0.518** | 0.5025 | 80.9% |
| `factoring_impossible` | 0.035 | 0.965 | 0.6039 | 98.8% |
| `factoring_impossible_blind` | **0.411** | **0.589** | 0.5045 | 82.5% |

`factoring_free_blind` is also the only arm where `composite` decodability
*degrades* over training (+0.142) — the blind agent cannot build it at all. Note
too that the blind agents' representations match the `blind` and `marginal`
targets far better than the `joint` one (0.323 / 0.313 vs 0.551 at ε=0; 0.113 /
0.100 vs 0.411 at ε=1). That is a strong internal-validity check: agents denied
action information develop representations matching an action-blind observer.

---

## 8. Data quality

Two runs end below their own peak by more than 2 percentage points. They are PPO
instabilities, not analysis errors, and each one single-handedly drives its arm's
reported standard deviation:

| arm | seed | peak | final | drop | trajectory tail |
| --- | :-: | ---: | ---: | ---: | --- |
| `no_feedback` | 44 | 0.681 | 0.611 | 0.070 | … 0.679 → **0.499** → 0.611 |
| `deterministic_feedback` | 46 | 0.683 | **0.524** | 0.159 | … 0.681 → 0.683 → **0.524** |

A third run, `factoring_impossible` seed 43, dips mid-training
(0.605 → **0.365** → 0.474 → 0.544 → 0.595) but recovers to within 0.010 of its
peak, so it is not flagged and needs no exclusion.

Excluding the two terminal collapses, `no_feedback` is 0.681 (98.7% of ceiling,
not 96.6%) and `deterministic_feedback` is 0.683 (99.0%, not 94.4%). Their
reported spreads (± 0.032 and ± 0.071) are **entirely** driven by one seed each.
`normalized_summary.json` records both the raw and collapse-excluded aggregates,
plus full trajectories, under `collapsed_runs`.

Recommendation for future campaigns: keep the best-checkpoint metrics alongside
the final-checkpoint metrics, or report medians.

---

## 9. Threats to validity considered

| threat | status |
| --- | --- |
| Probe capacity too small for a 9-dim target | **ruled out** — initialization decodes it at 0.080; capacity is fixed while error grows |
| 9-symbol observation crowds `d_model = 64` | **ruled out** — `no_feedback` shares the observation and improves 5× |
| Budget too short | **ruled out** — degradation is monotone from 330k steps onward, not a plateau |
| `factor_phi` target variance collapsing | **ruled out** — register entropy is flat (0.297 → 0.288 nats) and the metric is variance-normalized |
| Probes fitting noise | **ruled out** — all arms `p = 0.001` against the held-out shuffled-label null |
| Affine probe penalizes factored representations | **real, quantified** — see Correction 2; the reason Prediction 1 is untestable |
| `factor_subspace_overlap` lacks a null | **open** — the ≈0.5 constancy is unexplained; Prediction 4 is not safely adjudicated |
| Exact-filter reference vs context-10 ceiling | **minor** — calibration Bayes accuracy (0.6252) slightly exceeds the recorded ceiling (0.6214), as expected for unbounded memory |

---

## 10. Follow-up recommendation

### The decisive experiment: change the reward, not the generator

The finding is a statement about what the *reward* makes sufficient. So the next
cycle should keep the generator, architecture, budget and seed plan **exactly**
as they are and change only the reward: **score the agent on the full 9-symbol
token `(x, rho)` instead of the composite `x` alone.**

The reward-sufficient statistic then *is* the 9-state joint belief, the
factorization becomes worth maintaining, and the Factored World Hypothesis
becomes testable in RL for the first time. Because everything else is held
fixed, the two hypotheses make opposite quantitative predictions on the same
generator.

**If reward-sufficiency drives representation** (this campaign's finding), then
at `epsilon = 0` with full-token reward:

| quantity | current (composite reward) | predicted (full-token reward) |
| --- | ---: | ---: |
| `joint` normalized error | 0.196 | **0.01 – 0.03** |
| `factor_phi` normalized error | 0.262 (init 0.017) | **< 0.05** |
| `factor_m` normalized error | 0.134 | **< 0.05** |
| direction over training | degrades 2.5× | **improves** |
| AAR | 1.365 | **below the 1.58 factored bar** |

The reference values are `no_feedback` (0.021) and the sibling study (0.0023),
both cases where the reward statistic equals the world belief.

**If factoring is an architectural inductive bias** (the reading of
arXiv:2602.02385), these should be largely unmoved by the reward change.

### Supporting arms, in priority order

1. **`factoring_free_fulltoken`** (κ=0.7, ε=0, full-token reward) — the decisive
   arm. Pair it with **`factoring_impossible_fulltoken`** (ε=1) where `rho` is
   pure noise, so full-token reward adds an unpredictable component and the
   prediction is *no* improvement. That pair separates "richer reward" from
   "richer *learnable* reward."
2. **Supervised next-token control** — train the identical transformer on
   next-token cross-entropy over `(x, rho)` on the same generator, no RL. This
   separates "RL objective" from "architecture" and reproduces the paper's own
   setting, giving a direct bridge to their results.
3. **`gamma > 0` arm** on the existing composite reward. Under `gamma = 0` the
   agent has no reason to *steer*; with discounting, register control becomes
   instrumentally valuable, which is a second independent route to making the
   register worth representing.

### Methodological fixes to land first

1. **Add a null baseline for `factor_subspace_overlap`** — fit two probes to
   independent random targets from the same activations and report the overlap
   null. Until then Prediction 4 is unadjudicated.
2. **Report a bilinear or quadratic probe alongside the affine one** for the
   `joint` target. The affine probe structurally cannot decode a product; a
   probe with interaction terms would separate "information absent" from
   "information present but factored."
3. **Retire `action_awareness_ratio` as a primary metric**, or always publish it
   next to the calibration bars in `metric_calibration.json`. Alone it is
   uninterpretable and, at `epsilon = 0`, actively misleading.
4. **Correct the README's Prediction 2 rationale** to acknowledge the one-step
   staleness of the register report under `delay = 1`.
5. **Keep best-checkpoint metrics** so single-seed terminal collapses stop
   driving arm-level conclusions.

### Promotion candidate

`campaign_analysis.calibrate` is the reusable idea here: calibrating a probe
metric by fitting it from hand-built idealized representations is not specific to
this generator, and it caught a defect that five seeds and 40 runs did not. It
should move to `rl-harness` under `analysis/probes/` once a second study uses it.
