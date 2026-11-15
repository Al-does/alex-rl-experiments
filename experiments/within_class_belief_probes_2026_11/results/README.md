# Passive within-class belief decoding

R² denominators are centred within test class; pooled R² = 1 − pooled SSE / pooled within-class SST.

| Objective | Steps | Global | Within Bayes class | Head-blind | Logits | Window | Margin | Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo init | 0 | 0.971377 | 0.917668 | 0.917543 | 0.660929 | 0.898158 | 0.922064 | 0.941704 |
| ppo paper | 662029 | 0.996487 | 0.987677 | 0.987417 | 0.772757 | 0.898158 | 0.987723 | 0.983116 |
| ppo final | 2515796 | 0.993555 | 0.978611 | 0.977647 | 0.680873 | 0.898158 | 0.979293 | 0.977732 |
| supervised_ce init | 0 | 0.971377 | 0.917668 | 0.917543 | 0.660929 | 0.898158 | 0.922064 | 0.941704 |
| supervised_ce paper | 626197 | 0.996308 | 0.985741 | 0.985502 | 0.904696 | 0.898158 | 0.985396 | 0.987364 |
| supervised_ce final | 2543125 | 0.973742 | 0.924611 | 0.923991 | 0.371142 | 0.898158 | 0.930268 | 0.911804 |

One training seed; episode-bootstrap intervals condition on frozen probes and are not training-seed uncertainty. Input controls preserve original targets/classes, refit probes separately, and alter histories rather than activations. All decoding is affine accessibility, not causal use. The 62-dimensional head-blind basis removes logit-difference directions but retains any common-logit-shift direction.

Margin bins use half-open boundaries, with the final bin including 0.5. Flags mark fewer than 50 total rows; per-class counts are included. Undefined scores are null, not zero. See JSON protocol and provenance; raw arrays/checkpoints are ignored under artifacts/.

## Belief-Quotient Ladder: within-greedy-action-class decoding

Means across 15 training seeds; brackets are 95% percentile intervals from 10,000 whole-seed bootstrap draws. Full JSON includes per-seed/per-class scores, all analyzed stages, and paired trained-minus-initialized-encoder contrasts.

| Condition | Representation/history | Full h | Head-blind | Logits | Delta | Rho | History shuffle | Random input | Label permutation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trivial Quotient | Final trained | 0.9660 [0.9643, 0.9676] | 0.9659 [0.9642, 0.9674] | 0.8935 [0.8579, 0.9239] | 0.9564 [0.9544, 0.9584] | 0.9712 [0.9697, 0.9726] | -0.0012 [-0.0014, -0.0010] | -0.0018 [-0.0020, -0.0016] | -0.0017 [-0.0041, 0.0008] |
| Trivial Quotient | Init encoder, final histories/classes | 0.9665 [0.9649, 0.9683] | 0.9665 [0.9648, 0.9682] | 0.8187 [0.7606, 0.8712] | 0.9577 [0.9557, 0.9598] | 0.9713 [0.9699, 0.9729] | -0.0010 [-0.0011, -0.0008] | -0.0017 [-0.0019, -0.0016] | -0.0015 [-0.0039, 0.0010] |
| Trivial Quotient | Own-policy init | 0.9578 [0.9533, 0.9622] | 0.9578 [0.9532, 0.9621] | 0.6892 [0.5775, 0.7870] | 0.9530 [0.9500, 0.9557] | 0.9672 [0.9640, 0.9703] | -0.0014 [-0.0021, -0.0006] | -0.0297 [-0.0839, -0.0023] | -0.0023 [-0.0049, 0.0002] |
| Reward-State Quotient | Final trained | 0.9666 [0.9653, 0.9680] | 0.9665 [0.9652, 0.9679] | 0.7008 [0.5972, 0.7943] | 0.9691 [0.9678, 0.9705] | 0.2552 [0.1926, 0.3637] | -0.0017 [-0.0020, -0.0013] | -0.0024 [-0.0027, -0.0021] | 0.0001 [-0.0050, 0.0050] |
| Reward-State Quotient | Init encoder, final histories/classes | 0.9674 [0.9661, 0.9688] | 0.9674 [0.9660, 0.9688] | 0.7974 [0.7201, 0.8617] | 0.9699 [0.9686, 0.9713] | 0.2655 [0.1997, 0.3757] | -0.0017 [-0.0021, -0.0013] | -0.0028 [-0.0030, -0.0025] | 0.0003 [-0.0046, 0.0050] |
| Reward-State Quotient | Own-policy init | 0.9520 [0.9294, 0.9664] | 0.9522 [0.9301, 0.9663] | 0.7039 [0.5881, 0.8033] | 0.9573 [0.9524, 0.9623] | 0.9326 [0.8468, 0.9782] | -0.0016 [-0.0020, -0.0012] | -0.0027 [-0.0035, -0.0021] | -0.0050 [-0.0127, 0.0002] |
| Full-State Quotient | Final trained | 0.9045 [0.9017, 0.9072] | 0.9042 [0.9014, 0.9070] | 0.5858 [0.5097, 0.6565] | 0.9174 [0.9148, 0.9200] | 0.7879 [0.7829, 0.7924] | -0.0031 [-0.0033, -0.0028] | -0.0034 [-0.0038, -0.0031] | -0.0082 [-0.0137, -0.0033] |
| Full-State Quotient | Init encoder, final histories/classes | 0.9038 [0.9011, 0.9066] | 0.9035 [0.9008, 0.9063] | 0.6398 [0.5118, 0.7499] | 0.9170 [0.9144, 0.9198] | 0.7850 [0.7811, 0.7892] | -0.0029 [-0.0031, -0.0027] | -0.0035 [-0.0038, -0.0032] | -0.0083 [-0.0138, -0.0033] |
| Full-State Quotient | Own-policy init | 0.9369 [0.9244, 0.9450] | 0.9377 [0.9267, 0.9449] | 0.6861 [0.5806, 0.7755] | 0.9581 [0.9555, 0.9605] | 0.8593 [0.7968, 0.9056] | -0.0017 [-0.0023, -0.0012] | -0.0029 [-0.0035, -0.0024] | -0.0105 [-0.0252, -0.0019] |

Primary paired full-h belief R² contrast (final trained minus init encoder on final histories):

| Condition | Mean difference | 95% seed-bootstrap interval |
|---|---:|---:|
| Trivial Quotient | -0.000570 | [-0.001925, +0.000841] |
| Reward-State Quotient | -0.000818 | [-0.001981, +0.000321] |
| Full-State Quotient | +0.000686 | [-0.002940, +0.003906] |

Primary comparison is final trained versus the seed's initialized encoder replaying exactly the final policy's histories, targets and greedy classes. Initialized head-blind/logit features use the initialized head. Own-policy initialization changes visitation and classes and is secondary. Trivial Quotient's trained policy may be constant, but initialized policies need not be; actual class counts are retained. Controls refit within the original classes against original beliefs. The empirical random control uses separate reset/non-reset pools; history shuffling preserves reset rows and complete token/previous-action pairs.

These descriptive, single-campaign affine-accessibility results do not establish causal use, belief computation, or information absence. Intervals quantify between-training-seed uncertainty, not individual timestep independence.

## PR202 temperature-1.5 ladder: failed reproduction gate

Analysis stopped at the first checkpoint, before conditional metrics. The
remaining 23 required init/final gates were not attempted.

| Condition / architecture | Stage | Archived global R² | CPU reproduction | Difference | Status |
|---|---|---:|---:|---:|---|
| Trivial Quotient / GRU128 | Init | 0.980304509038 | 0.980342474993 | +0.000037965955 | FAIL (tolerance 1e-6) |

Checkpoint bytes and SHA-256 match the canonical durability manifest. Both
archived and local JAX versions are 0.11.1; the discrepancy's cause is not
established. See `ladder_temp15_gate.json` and the study README. No temp15
within-class table, paired confidence intervals or figure were generated.

### Follow-up: authorized tolerance 1e-3

The prior 1e-6 gate is superseded by a user-authorized absolute 1e-3 tolerance
on both global R² and global 1−R², allowing cross-device float32/matmul
differences. Default and highest CPU matmul precision give identical init
R² (0.980342474993); this does not establish GPU TF32 as the cause.

| Condition / architecture | Stage | Archived R² | CPU R² | Difference | Status |
|---|---|---:|---:|---:|---|
| Trivial Quotient / GRU128 | Init | 0.980304509038 | 0.980342474993 | +0.000037965955 | PASS |
| Trivial Quotient / GRU128 | Final | 0.984582334736 | 0.986766985071 | +0.002184650335 | FAIL |

Final global 1−R² also fails (archived 0.015417665036 versus CPU
0.013233014734). Final greedy actions occupy one class, Tilt X, in both
reports. Checkpoint hashes and byte counts match the canonical manifests.
Stopped before the remaining 22 gates or any conditional analysis. No
temp15 within-class result or figure is claimed.

## PR202: completed 24 recorded protocol checks

Supersedes the earlier hard-stop rules: checks are recorded for every init/final arm; hard-stop threshold is max(0.01, 5 × CPU episode-bootstrap SE). No seed uncertainty is estimated: all observations condition on training seed 42 and fixed fit probes. Cross-device numerical execution is the motivation, not a verified attribution to TF32.

| Condition / architecture | Stage | Archived R² | CPU R² | Delta | Bootstrap SE | Hard-stop limit | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Trivial Quotient / gru128 | Init | 0.980304509 | 0.980342475 | +0.000037966 | 0.0001308 | 0.0100000 | OK |
| Trivial Quotient / gru128 | Final | 0.984582335 | 0.986766985 | +0.002184650 | 0.0001358 | 0.0100000 | OK |
| Trivial Quotient / gru92 | Init | 0.817106440 | 0.891244536 | +0.074138096 | 0.0003682 | 0.0100000 | HARD STOP |
| Trivial Quotient / gru92 | Final | 0.971698298 | 0.981194539 | +0.009496241 | 0.0001856 | 0.0100000 | OK |
| Trivial Quotient / memoryless | Init | 0.833613820 | 0.833613823 | +0.000000002 | 0.0023853 | 0.0119267 | OK |
| Trivial Quotient / memoryless | Final | 0.581374501 | 0.581374509 | +0.000000008 | 0.0025881 | 0.0129403 | OK |
| Trivial Quotient / transformer | Init | 0.911044254 | 0.910905306 | -0.000138948 | 0.0010334 | 0.0100000 | OK |
| Trivial Quotient / transformer | Final | 0.877119815 | 0.887734096 | +0.010614281 | 0.0013931 | 0.0100000 | HARD STOP |
| Reward-State Quotient / gru128 | Init | 0.986679853 | 0.986758389 | +0.000078536 | 0.0000798 | 0.0100000 | OK |
| Reward-State Quotient / gru128 | Final | 0.998388988 | 0.998390607 | +0.000001619 | 0.0000133 | 0.0100000 | OK |
| Reward-State Quotient / gru92 | Init | 0.817106440 | 0.891244536 | +0.074138096 | 0.0003682 | 0.0100000 | HARD STOP |
| Reward-State Quotient / gru92 | Final | 0.931351593 | 0.931368878 | +0.000017285 | 0.0004796 | 0.0100000 | OK |
| Reward-State Quotient / memoryless | Init | 0.833613820 | 0.833613823 | +0.000000002 | 0.0023853 | 0.0119267 | OK |
| Reward-State Quotient / memoryless | Final | 0.859070137 | 0.859070137 | -0.000000000 | 0.0008087 | 0.0100000 | OK |
| Reward-State Quotient / transformer | Init | 0.959947914 | 0.959973359 | +0.000025446 | 0.0005513 | 0.0100000 | OK |
| Reward-State Quotient / transformer | Final | 0.991781765 | 0.991779713 | -0.000002052 | 0.0000834 | 0.0100000 | OK |
| Full-State Quotient / gru128 | Init | 0.984493130 | 0.984614518 | +0.000121388 | 0.0000969 | 0.0100000 | OK |
| Full-State Quotient / gru128 | Final | 0.996556236 | 0.996556899 | +0.000000663 | 0.0000365 | 0.0100000 | OK |
| Full-State Quotient / gru92 | Init | 0.832860707 | 0.897228149 | +0.064367442 | 0.0003192 | 0.0100000 | HARD STOP |
| Full-State Quotient / gru92 | Final | 0.990939267 | 0.990939765 | +0.000000498 | 0.0000987 | 0.0100000 | OK |
| Full-State Quotient / memoryless | Init | 0.833613820 | 0.833613823 | +0.000000002 | 0.0023853 | 0.0119267 | OK |
| Full-State Quotient / memoryless | Final | 0.856482344 | 0.856482337 | -0.000000007 | 0.0009052 | 0.0100000 | OK |
| Full-State Quotient / transformer | Init | 0.961085184 | 0.961091725 | +0.000006541 | 0.0007375 | 0.0100000 | OK |
| Full-State Quotient / transformer | Final | 0.996682152 | 0.996681007 | -0.000001145 | 0.0000503 | 0.0100000 | OK |

### Trivial Quotient / GRU128 final diagnosis

Same CPU rows, ridge 1e-6: float64 R² 0.986766984913; float32 R² 0.986738441853. Float64-centered unregularized design condition number 4.39598e+06; ridge-augmented condition number 193939. Float32 centering changes these to 232396 and 148986.

10,000-resample held-fixed-probe episode-bootstrap global-R² SE: 0.000137691; 95% interval [0.9864853153763764, 0.9870278604379619]. Conditions on one training seed, the CPU histories and fixed fit probe.

Identical fixed actions remove policy branching, but categorical sampling uses floating-point log/Gumbel comparisons. Threefry keys/bits are intended to be portable; JAX/XLA floating-point sampling and environment execution are not guaranteed bitwise identical across CPU/GPU. No archived raw histories are available here to verify identity.

### One-class policies (both fit and test)

- Trivial Quotient / gru128, final: Tilt X
- Trivial Quotient / gru92, init: Tilt Y
- Trivial Quotient / gru92, final: Tilt X
- Trivial Quotient / memoryless, init: Noop
- Trivial Quotient / memoryless, final: Tilt X
- Trivial Quotient / transformer, final: Tilt X
- Reward-State Quotient / gru92, init: Tilt Y
- Reward-State Quotient / memoryless, init: Noop
- Full-State Quotient / gru92, init: Tilt Y
- Full-State Quotient / memoryless, init: Noop

For these cases, pooled within-class R² equals global R² for the same fitted probe: there is no action-class conditioning benefit or within-region-versus-between-region distinction. Other arms are not one-class on these rows.

All 24 checks completed; four exceed the user-authorized hard-stop threshold. Conditional metrics, matched-init replays, controls, paired differences and the requested within-class figure were not computed. Checkpoint bytes and SHA-256 were verified before restoration. Both R² and archived global 1−R² deltas are retained in the JSON.

## PR202 temperature-1.5: completed CPU within-class analysis

Supersedes the earlier stopping decisions. All 12 arms are included, including all initialized encoders replayed on final-policy histories. Four archived-global-R² discrepancies remain annotations, not blockers. **One training seed (42) only.** Intervals are paired test-episode bootstrap intervals with fixed fit probes, not training-seed uncertainty.

| Condition / architecture | Full h final | Matched init | Paired difference [95% CI] | Head-blind | Logits | Delta | Rho | Shuffle | Random | Labels | One-class final | Archive annotation |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Trivial Quotient / gru128 | 0.9868 | 0.9901 | -0.0033 [-0.0035, -0.0032] | 0.9868 | 0.7391 | 0.9875 | 0.9866 | -0.0023 | -0.0052 | 0.0138 | Yes (Tilt X) | — |
| Trivial Quotient / gru92 | 0.9812 | 0.9421 | +0.0391 [+0.0386, +0.0396] | 0.9805 | 0.7533 | 0.9719 | 0.9836 | -0.0012 | -0.0041 | 0.0136 | Yes (Tilt X) | init: archived global R² not reproduced on CPU |
| Trivial Quotient / memoryless | 0.5814 | 0.5814 | -0.0000 [-0.0000, +0.0000] | 0.5814 | 0.5814 | 0.7449 | 0.5381 | -0.0001 | -0.0002 | 0.0062 | Yes (Tilt X) | — |
| Trivial Quotient / transformer | 0.8877 | 0.8437 | +0.0440 [+0.0409, +0.0473] | 0.8866 | 0.5982 | 0.9151 | 0.8805 | 0.0014 | -0.0041 | 0.0129 | Yes (Tilt X) | final: archived global R² not reproduced on CPU |
| Reward-State Quotient / gru128 | 0.9996 | 0.9948 | +0.0049 [+0.0048, +0.0049] | 0.9996 | 0.4333 | 0.9996 | 0.9997 | -0.0043 | -0.0045 | -0.0061 | No | — |
| Reward-State Quotient / gru92 | 0.9547 | 0.8289 | +0.1258 [+0.1240, +0.1277] | 0.9546 | 0.2240 | 0.9444 | 0.9994 | -0.0039 | -0.0025 | -0.0053 | No | init: archived global R² not reproduced on CPU |
| Reward-State Quotient / memoryless | 0.7699 | 0.7699 | +0.0000 [-0.0000, +0.0000] | 0.7699 | 0.7699 | 0.9782 | -0.0005 | -0.0002 | -0.0002 | 0.0016 | No | — |
| Reward-State Quotient / transformer | 0.9904 | 0.9831 | +0.0073 [+0.0070, +0.0076] | 0.9903 | 0.4419 | 0.9957 | 0.9667 | -0.0016 | -0.0022 | -0.0083 | No | — |
| Full-State Quotient / gru128 | 0.9895 | 0.9949 | -0.0054 [-0.0056, -0.0052] | 0.9894 | 0.5229 | 0.9604 | 0.9968 | -0.0074 | -0.0060 | 0.0024 | No | — |
| Full-State Quotient / gru92 | 0.9791 | 0.9039 | +0.0753 [+0.0734, +0.0772] | 0.9791 | 0.6391 | 0.8868 | 0.9962 | -0.0060 | -0.0052 | 0.0005 | No | init: archived global R² not reproduced on CPU |
| Full-State Quotient / memoryless | -0.0006 | -0.0006 | +0.0000 [+0.0000, +0.0000] | -0.0006 | -0.0006 | -0.0004 | -0.0006 | -0.0006 | -0.0007 | -0.0006 | No | — |
| Full-State Quotient / transformer | 0.9867 | 0.9091 | +0.0777 [+0.0761, +0.0792] | 0.9865 | 0.3860 | 0.9898 | 0.9863 | -0.0034 | -0.0036 | 0.0206 | No | — |

### Reading the comparisons

Unannotated arms: interpret trained-minus-matched-init differences as CPU affine-accessibility differences on exactly matched histories and greedy classes; the paired intervals are conditional on seed 42.

**Separate annotated block:** GRU92 initialization in all three quotient conditions, and Trivial Quotient / transformer final: **archived global R² not reproduced on CPU**. Their CPU metrics and matched-init comparisons are retained in full, without assuming archived CPU/GPU equivalence.

All four Trivial Quotient final policies are single-class (Tilt X), as are all GRU92 initializations (Tilt Y) and all memoryless initializations (Noop). For these checkpoints pooled within-class R² equals global R² for the same probe: no evidence beyond action-class conditioning is obtained by relabeling it as within-class.

GRU92 initialization shares identical NPZ bytes and parameter arrays across all three variants. Archived/CPU actions are all Tilt Y. Rewards agree exactly: 0.012125 for Trivial and Reward-State Quotients, 0.107637502 for Full-State Quotient. Median/minimum held-out top-two logit gaps are 0.138522/0.031633, 0.138522/0.031633 and 0.154293/0.031634 respectively—not numerical ties.

GRU matched-init replay starts with zeros in every layer for each independent episode, consumes the full final observation/action stream including warmup, and carries hidden state through all 1024 steps without resetting at the 32-step training chunk boundaries. Transformer caches likewise start empty per episode. Memoryless models receive the same stream but architecturally read only its current token.

Own-policy transformer init has one excluded test row in Trivial Quotient and two in Full-State Quotient because the corresponding greedy class lacks two fit rows. All final and matched-init comparisons have complete prediction coverage. Counts and exclusions are retained per metric in JSON. Rounded near-zero memoryless contrasts are numerical-scale differences, not exact-zero uncertainty.

The JSON includes own-policy initialization as a secondary comparison, all per-class/scalar/control scores, intermediate checkpoints near 2B, paired intervals for every representation and control, checkpoint provenance and hashes. No seed averaging is performed. These are descriptive accessibility measurements, not causal use or proof of belief computation.

## Persistent-nuisance entropy 0.005: CPU within-class analysis

Three seed-42 transformer arms only, constant entropy 0.005, temperature 1.0, epsilon 0; no training. All checkpoints were verified against canonical durability-manifest SHA-256 and byte counts before restoration. Archive comparisons are annotations, not blockers, with threshold max(0.01, 5 × CPU episode-bootstrap SE) for both global R² and archived 1−R².

**Target:** the study's full three-state filtered belief `(b1,b2,b3)`, not a nuisance-augmented latent product or a marginal. The persistent nuisance comes from the transition/emission structure. Scalar probes retain `delta=b1−b2` and `rho=b3`. In Trivial Quotient both coordinates are action-unnecessary; in Reward-State Quotient rho tracks the reward-state partition and delta is the within-nonreward-state contrast; in Full-State Quotient both contrasts can be action-relevant. These task partitions are distinct from the measured policy's greedy-action classes.

### Archive protocol checks

| Quotient | Stage | Step | Archived global R² | CPU global R² | Δ | Annotation |
|---|---|---:|---:|---:|---:|---|
| Trivial Quotient | init | 0 | 0.879093534 | 0.879585155 | +0.000491621 | — |
| Trivial Quotient | intermediate | 996147200 | 0.961235877 | 0.969212790 | +0.007976914 | — |
| Trivial Quotient | final | 2012217344 | 0.949427587 | 0.959925589 | +0.010498002 | archived global R² not reproduced on CPU |
| Reward-State Quotient | init | 0 | 0.894730368 | 0.895023451 | +0.000293083 | — |
| Reward-State Quotient | intermediate | 996147200 | 0.912300950 | 0.912401138 | +0.000100188 | — |
| Reward-State Quotient | final | 2012217344 | 0.894127933 | 0.894384207 | +0.000256274 | — |
| Full-State Quotient | init | 0 | 0.904347289 | 0.904462226 | +0.000114937 | — |
| Full-State Quotient | intermediate | 996147200 | 0.986087217 | 0.986088977 | +0.000001760 | — |
| Full-State Quotient | final | 2012217344 | 0.980856114 | 0.980855696 | -0.000000418 | — |

### Final versus matched initialization

All columns are pooled class-centred within-class-refitted held-out R². Other than matched init and its paired contrast, columns describe the final trained encoder. Intervals resample whole test episodes 10,000 times with probes held fixed; **one training seed only**, not training-seed uncertainty.

| Quotient | Final h | Matched init | Difference [95% CI] | Head-blind | Logits | Delta | Rho | Shuffle | Random | Labels | One-class final | Archive annotation |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Trivial Quotient | 0.9599 | 0.8539 | +0.1060 [+0.1037, +0.1083] | 0.9597 | 0.5573 | 0.9679 | 0.9586 | 0.0007 | -0.0031 | 0.0025 | Yes | final: archived global R² not reproduced on CPU |
| Reward-State Quotient | 0.8815 | 0.8831 | -0.0015 [-0.0045, +0.0013] | 0.8810 | 0.3447 | 0.8582 | 0.9948 | -0.0041 | -0.0077 | 0.0043 | No | — |
| Full-State Quotient | 0.9709 | 0.8007 | +0.1702 [+0.1673, +0.1732] | 0.9707 | 0.8544 | 0.9726 | 0.9684 | -0.0031 | -0.0046 | 0.0038 | No | — |

### Own-policy initialization (secondary)

| Quotient | Full h | Head-blind | Logits | Delta | Rho |
|---|---:|---:|---:|---:|---:|
| Trivial Quotient | 0.8702 | 0.8676 | 0.3846 | 0.8915 | 0.8467 |
| Reward-State Quotient | 0.8817 | 0.8819 | 0.3333 | 0.8842 | 0.8580 |
| Full-State Quotient | 0.8794 | 0.8794 | 0.3629 | 0.8872 | 0.7871 |

**Separate archive-discrepancy block:** Trivial Quotient final: archived global R² not reproduced on CPU

**One-class checkpoint flags:** Trivial Quotient intermediate; Trivial Quotient final. For a one-class checkpoint, pooled within-class R² equals global R² for the same fitted probe, not additional action-conditioned evidence.

**Excluded predictions:** Reward-State Quotient init: 1 rows; Full-State Quotient init: 1 rows. Counts and per-class metrics are explicit in JSON.

Matched init consumes exactly the final policy's token/previous-action stream from an empty transformer cache per independent episode, including all 64 warmup steps. It retains final beliefs, greedy classes and scored rows. History shuffle preserves reset and complete token/action pairs; random input samples from separate reset/nonreset empirical pair pools; label permutation changes fit labels within class only. These are frozen-input controls, not activation interventions. Each uses one fixed random draw.

The head-blind probe removes the two action-logit-difference directions (62 dimensions remain), retaining common logit shifts. Each encoder uses its own policy head. Probe fit is float64 centered ridge least squares (ridge 1e−6). Last scored episodes are truncated to the 60K/80K budgets. Bootstrap uncertainty conditions on these CPU histories, fixed fit probes and seed 42; there is no across-training-seed inference. Accessibility differences do not establish causal use or Bayesian computation.

Provenance: records imported with `git archive` from `origin/devin/1794419502-persistent-nuisance-egreedy` at `a3b887b7188f8e08155a6ee03dea16575e6cab39`, not merged/rebased. Only entropy-0.005 arm records and small shared recipe sources were brought in. The local greedy collector is equivalent for temperature 1/epsilon 0; upstream's only base collector change adds epsilon support. JSON includes source/runtime/checkpoint hashes, archived and CPU scores, all per-class/scalar/control scores, matched replay, and paired intervals.

## Persistent-nuisance v2: temperature 1.5 with training epsilon 0 / 0.05

Six seed-42 transformer policies only, target 2B steps, entropy 0, training temperature 1.5, training exploration epsilon 0 or 0.05. All probes are greedy argmax: both temperature and epsilon are ignored. No training, paid compute, commits or pushes.

The source target remains the full filtered **three-state** belief, not a marginal or additional nuisance factor. Delta=b1−b2 and rho=b3 remain valid: delta contrasts the nonreward states; rho is reward-state probability. Both are action-unnecessary in Trivial Quotient; delta is the within-nonreward contrast in Reward-State Quotient; both may matter for Full-State actions. Classes are the policy's own greedy actions, not the latent partition.

The v2 environment uses baseline transition rows (.995505,.002495,.002), (.002495,.995505,.002), (.005,.005,.99); emission rows (.1125,.0375,.85), (.0375,.1125,.85), (0,0,1); uniform initial belief; reward on pre-transition state 3; tilt strength 2.5. Inputs remain token/previous-action one-hots. The shared probe, models, env/filter implementation and training dispatch match the local base files; PR219's collector changes affect non-greedy training sampling/likelihoods only. Greedy key splitting and argmax are unchanged.

### Archive checks (annotations only)

Threshold max(0.01,5×CPU episode-bootstrap SE), checked for both global R² and archived 1−R². All checkpoint bytes and SHA-256 hashes are verified against canonical durability manifests before restoration. No discrepancy blocks analysis.

| Quotient | Training epsilon | Stage | Step | Archived R² | CPU R² | Δ | Annotation |
|---|---:|---|---:|---:|---:|---:|---|
| Trivial Quotient | 0.00 | init | 0 | 0.827315388 | 0.827572811 | +0.000257423 | — |
| Trivial Quotient | 0.00 | intermediate | 996147200 | 0.971086351 | 0.973116104 | +0.002029753 | — |
| Trivial Quotient | 0.00 | final | 2012217344 | 0.954113418 | 0.959597688 | +0.005484270 | — |
| Reward-State Quotient | 0.00 | init | 0 | 0.858691820 | 0.858711980 | +0.000020160 | — |
| Reward-State Quotient | 0.00 | intermediate | 996147200 | 0.964591536 | 0.964625784 | +0.000034248 | — |
| Reward-State Quotient | 0.00 | final | 2012217344 | 0.956315582 | 0.956358951 | +0.000043369 | — |
| Full-State Quotient | 0.00 | init | 0 | 0.859583007 | 0.859847843 | +0.000264836 | — |
| Full-State Quotient | 0.00 | intermediate | 996147200 | 0.983278555 | 0.983253622 | -0.000024933 | — |
| Full-State Quotient | 0.00 | final | 2012217344 | 0.984925872 | 0.984924132 | -0.000001740 | — |
| Trivial Quotient | 0.05 | init | 0 | 0.827275063 | 0.827572811 | +0.000297747 | — |
| Trivial Quotient | 0.05 | intermediate | 996147200 | 0.963016076 | 0.970056273 | +0.007040197 | — |
| Trivial Quotient | 0.05 | final | 2012217344 | 0.958142356 | 0.967502691 | +0.009360335 | — |
| Reward-State Quotient | 0.05 | init | 0 | 0.858615484 | 0.858711980 | +0.000096495 | — |
| Reward-State Quotient | 0.05 | intermediate | 996147200 | 0.965845125 | 0.965885694 | +0.000040569 | — |
| Reward-State Quotient | 0.05 | final | 2012217344 | 0.966188284 | 0.966209535 | +0.000021250 | — |
| Full-State Quotient | 0.05 | init | 0 | 0.859587133 | 0.859847843 | +0.000260710 | — |
| Full-State Quotient | 0.05 | intermediate | 996147200 | 0.977617437 | 0.977616986 | -0.000000451 | — |
| Full-State Quotient | 0.05 | final | 2012217344 | 0.977384771 | 0.977385251 | +0.000000480 | — |

### Final versus matched initialization

Pooled within-greedy-class held-out R², class-specific refits and class-centred test denominators. Other than matched init and its paired contrast, columns describe the final trained encoder. Intervals use 10,000 paired episode resamples with probes fixed: **one training seed only**.

| Quotient | Epsilon | Final h | Matched init | Difference [95% CI] | Head-blind | Logits | Delta | Rho | Shuffle | Random | Labels | One-class final | Archive annotation |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Trivial Quotient | 0.00 | 0.9596 | 0.9009 | +0.0587 [+0.0497, +0.0717] | 0.9588 | 0.4168 | 0.8546 | 0.9807 | 0.0258 | -0.0097 | -0.0111 | Yes | — |
| Reward-State Quotient | 0.00 | 0.9173 | 0.9152 | +0.0021 [-0.0025, +0.0073] | 0.9167 | 0.3854 | 0.8728 | 0.9966 | 0.0072 | -0.0083 | 0.0013 | No | — |
| Full-State Quotient | 0.00 | 0.9697 | 0.9175 | +0.0522 [+0.0494, +0.0551] | 0.9694 | 0.8812 | 0.9239 | 0.9892 | 0.0128 | -0.0124 | -0.0115 | No | — |
| Trivial Quotient | 0.05 | 0.9675 | 0.9009 | +0.0666 [+0.0577, +0.0797] | 0.9662 | 0.7658 | 0.8671 | 0.9877 | 0.0277 | -0.0175 | -0.0065 | Yes | — |
| Reward-State Quotient | 0.05 | 0.9384 | 0.9181 | +0.0203 [+0.0174, +0.0236] | 0.9383 | 0.2866 | 0.9021 | 0.9943 | 0.0081 | -0.0086 | -0.0005 | No | — |
| Full-State Quotient | 0.05 | 0.9540 | 0.9143 | +0.0397 [+0.0364, +0.0431] | 0.9534 | 0.5707 | 0.8975 | 0.9778 | 0.0045 | -0.0102 | 0.0091 | No | — |

### Own-policy initialization (secondary)

| Quotient | Epsilon | Full h | Head-blind | Logits | Delta | Rho |
|---|---:|---:|---:|---:|---:|---:|
| Trivial Quotient | 0.00 | 0.8331 | 0.8326 | 0.2204 | 0.8544 | 0.6931 |
| Reward-State Quotient | 0.00 | 0.8646 | 0.8642 | 0.2323 | 0.8755 | 0.6341 |
| Full-State Quotient | 0.00 | 0.8671 | 0.8667 | 0.2709 | 0.8890 | 0.6005 |
| Trivial Quotient | 0.05 | 0.8331 | 0.8326 | 0.2204 | 0.8544 | 0.6931 |
| Reward-State Quotient | 0.05 | 0.8646 | 0.8642 | 0.2323 | 0.8755 | 0.6341 |
| Full-State Quotient | 0.05 | 0.8671 | 0.8667 | 0.2709 | 0.8890 | 0.6005 |

**Separate archive-discrepancy block:** None exceeded the annotation threshold.

**One-class stages:** Trivial Quotient, epsilon 0.00, intermediate; Trivial Quotient, epsilon 0.00, final; Trivial Quotient, epsilon 0.05, intermediate; Trivial Quotient, epsilon 0.05, final. Within-class R² equals global R² for a one-class policy; this is not additional action-conditioned evidence.

**Excluded predictions:** None across saved stages. Full counts and per-class scores are retained in JSON.

### Final-policy action counts

Counts are [Noop, Tilt X, Tilt Y], across the scored fit/test rows.

- Trivial Quotient, epsilon 0.00: fit [0, 60000, 0]; test [0, 80000, 0].
- Reward-State Quotient, epsilon 0.00: fit [36007, 23993, 0]; test [47543, 32457, 0].
- Full-State Quotient, epsilon 0.00: fit [36062, 11292, 12646]; test [48636, 14732, 16632].
- Trivial Quotient, epsilon 0.05: fit [0, 60000, 0]; test [0, 80000, 0].
- Reward-State Quotient, epsilon 0.05: fit [37390, 22610, 0]; test [49567, 30433, 0].
- Full-State Quotient, epsilon 0.05: fit [36205, 12330, 11465]; test [48737, 16372, 14891].

Matched initialization replays exact final-policy token/previous-action streams with empty transformer caches per episode, including warmup; final beliefs, classes and scored rows stay fixed. History shuffle preserves resets and whole input pairs; random input samples separate reset/nonreset empirical pair pools; fit labels are permuted within class. Original targets/classes are retained; each control uses one fixed draw.

Protocol: post-final-LayerNorm 64-D embeddings, 60K fit / 80K test rows, 64-step warmup, 1024-step episodes, ridge 1e−6, original split(fold_in(PRNGKey(42),2)) streams. Head-blinding removes two action-logit-difference directions (62 remain), retaining common logit shifts; each encoder uses its own head. Episode bootstrap recomputes class means but holds probes fixed, conditioning on the CPU histories and one seed. Last scored episodes are truncated to the row budgets. No training-seed or between-epsilon causal inference is supported. Accessibility does not establish causal use or Bayesian computation.

Source: PR219 branch `devin/1794431156-v2-temp15-eps-greedy` at `58a26c8c6c2bc1fda82845db4727a6c7397842a4`. Imported selected transformer records and recipe/environment sources using git archive, no merge/rebase. The JSON includes hashes/bytes and source/runtime provenance, init/intermediate/final per-class/scalar/control scores, matched-init metrics and paired intervals. Checkpoint caches are separated by study to avoid collisions between epsilon settings.

## PR219 checkpoint trajectories: own histories versus fixed final histories

All six seed-42 transformer runs have 102 durable checkpoints. The full per-run inventory (step, B2 key, bytes, SHA-256 and selection flag) is in the trajectory JSON. Twelve checkpoints per run are analyzed using a prespecified hybrid of early geometric and late even index spacing, including init, saved midpoint and final. No decoding values were used to choose the sampling grid. Peaks refer only to sampled checkpoints; an unsampled peak or drop cannot be ruled out.

### B2 inventory

Each prefix below contains `metadata/durability_manifest.json`; checkpoint keys, hashes and byte counts come from that canonical manifest. All selected checkpoint files are verified before restoration.

| Quotient | Epsilon | Saved / analyzed | B2 prefix |
|---|---:|---:|---|
| Trivial Quotient | 0.00 | 102 / 12 | `s3://slop-bucket/experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/variant_1_transformer/20261111T210842Z-052519d0/` |
| Reward-State Quotient | 0.00 | 102 / 12 | `s3://slop-bucket/experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/variant_2_transformer/20261111T210836Z-f0ecb162/` |
| Full-State Quotient | 0.00 | 102 / 12 | `s3://slop-bucket/experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/variant_3_transformer/20261111T210841Z-f5aebec9/` |
| Trivial Quotient | 0.05 | 102 / 12 | `s3://slop-bucket/experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11/variant_1_transformer/20261111T210836Z-35ca5c85/` |
| Reward-State Quotient | 0.05 | 102 / 12 | `s3://slop-bucket/experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11/variant_2_transformer/20261111T210838Z-f38d4c4c/` |
| Full-State Quotient | 0.05 | 102 / 12 | `s3://slop-bucket/experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11/variant_3_transformer/20261111T211013Z-047a55e1/` |

Every run has the following saved steps (shared inventory; each run's corresponding B2 objects are enumerated separately in JSON):

```text
0, 19922944, 39845888, 59768832, 79691776, 99614720, 119537664, 139460608, 159383552, 179306496, 199229440, 219152384, 239075328, 258998272, 278921216, 298844160, 318767104, 338690048, 358612992, 378535936, 398458880, 418381824, 438304768, 458227712, 478150656, 498073600, 517996544, 537919488, 557842432, 577765376, 597688320, 617611264, 637534208, 657457152, 677380096, 697303040, 717225984, 737148928, 757071872, 776994816, 796917760, 816840704, 836763648, 856686592, 876609536, 896532480, 916455424, 936378368, 956301312, 976224256, 996147200, 1016070144, 1035993088, 1055916032, 1075838976, 1095761920, 1115684864, 1135607808, 1155530752, 1175453696, 1195376640, 1215299584, 1235222528, 1255145472, 1275068416, 1294991360, 1314914304, 1334837248, 1354760192, 1374683136, 1394606080, 1414529024, 1434451968, 1454374912, 1474297856, 1494220800, 1514143744, 1534066688, 1553989632, 1573912576, 1593835520, 1613758464, 1633681408, 1653604352, 1673527296, 1693450240, 1713373184, 1733296128, 1753219072, 1773142016, 1793064960, 1812987904, 1832910848, 1852833792, 1872756736, 1892679680, 1912602624, 1932525568, 1952448512, 1972371456, 1992294400, 2012217344
```

### Sampled trajectories

R² values below are pooled class-centred within-greedy-class full-h belief scores. Own histories use each checkpoint's policy streams/classes; fixed histories use final-policy streams/targets/classes and refit the probe for every encoder. Reward is the realized greedy reward per scored own-policy step, not a reward earned by an encoder on fixed actions. JSON also includes global R², head-blind, logits, delta/rho, class counts, one-class flags and pointwise episode-bootstrap intervals for all scores and reward.

| Quotient | Epsilon | Steps | Own-history R² | Fixed-history R² | Greedy reward | One-class own |
|---|---:|---:|---:|---:|---:|---|
| Trivial Quotient | 0.00 | 0 | 0.833117 | 0.900899 | 0.087362 | No |
| Trivial Quotient | 0.00 | 19922944 | 0.922380 | 0.922380 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 39845888 | 0.925220 | 0.925220 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 99614720 | 0.939596 | 0.939596 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 219152384 | 0.970401 | 0.970401 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 498073600 | 0.966967 | 0.966967 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 796917760 | 0.974811 | 0.974811 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 996147200 | 0.973116 | 0.973116 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 1115684864 | 0.969468 | 0.969468 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 1414529024 | 0.964401 | 0.964401 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 1713373184 | 0.958737 | 0.958737 | 0.965163 | Yes |
| Trivial Quotient | 0.00 | 2012217344 | 0.959598 | 0.959598 | 0.965163 | Yes |
| Reward-State Quotient | 0.00 | 0 | 0.864559 | 0.915188 | 0.051488 | No |
| Reward-State Quotient | 0.00 | 19922944 | 0.897409 | 0.926689 | 0.172725 | Yes |
| Reward-State Quotient | 0.00 | 39845888 | 0.906467 | 0.932369 | 0.172725 | Yes |
| Reward-State Quotient | 0.00 | 99614720 | 0.939250 | 0.942844 | 0.172725 | Yes |
| Reward-State Quotient | 0.00 | 219152384 | 0.940165 | 0.938273 | 0.172725 | Yes |
| Reward-State Quotient | 0.00 | 498073600 | 0.939672 | 0.939759 | 0.499250 | No |
| Reward-State Quotient | 0.00 | 796917760 | 0.935131 | 0.935045 | 0.501238 | No |
| Reward-State Quotient | 0.00 | 996147200 | 0.934345 | 0.934195 | 0.501238 | No |
| Reward-State Quotient | 0.00 | 1115684864 | 0.930879 | 0.930659 | 0.501238 | No |
| Reward-State Quotient | 0.00 | 1414529024 | 0.930014 | 0.930014 | 0.501113 | No |
| Reward-State Quotient | 0.00 | 1713373184 | 0.931615 | 0.931615 | 0.501113 | No |
| Reward-State Quotient | 0.00 | 2012217344 | 0.917327 | 0.917327 | 0.501113 | No |
| Full-State Quotient | 0.00 | 0 | 0.867052 | 0.917478 | 0.070662 | No |
| Full-State Quotient | 0.00 | 19922944 | 0.805209 | 0.927627 | 0.038863 | No |
| Full-State Quotient | 0.00 | 39845888 | 0.645191 | 0.930724 | 0.086675 | No |
| Full-State Quotient | 0.00 | 99614720 | 0.878554 | 0.947304 | 0.130275 | No |
| Full-State Quotient | 0.00 | 219152384 | 0.956675 | 0.962517 | 0.429112 | No |
| Full-State Quotient | 0.00 | 498073600 | 0.958770 | 0.965077 | 0.463313 | No |
| Full-State Quotient | 0.00 | 796917760 | 0.964280 | 0.967384 | 0.458025 | No |
| Full-State Quotient | 0.00 | 996147200 | 0.963038 | 0.967798 | 0.455213 | No |
| Full-State Quotient | 0.00 | 1115684864 | 0.966168 | 0.969590 | 0.449313 | No |
| Full-State Quotient | 0.00 | 1414529024 | 0.969655 | 0.971206 | 0.454275 | No |
| Full-State Quotient | 0.00 | 1713373184 | 0.971982 | 0.971938 | 0.463575 | No |
| Full-State Quotient | 0.00 | 2012217344 | 0.969721 | 0.969721 | 0.465250 | No |
| Trivial Quotient | 0.05 | 0 | 0.833117 | 0.900899 | 0.087362 | No |
| Trivial Quotient | 0.05 | 19922944 | 0.924894 | 0.924894 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 39845888 | 0.930224 | 0.930224 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 99614720 | 0.944117 | 0.944117 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 219152384 | 0.973949 | 0.973949 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 498073600 | 0.971550 | 0.971550 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 796917760 | 0.970413 | 0.970413 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 996147200 | 0.970056 | 0.970056 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 1115684864 | 0.969839 | 0.969839 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 1414529024 | 0.969015 | 0.969015 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 1713373184 | 0.968076 | 0.968076 | 0.965163 | Yes |
| Trivial Quotient | 0.05 | 2012217344 | 0.967503 | 0.967503 | 0.965163 | Yes |
| Reward-State Quotient | 0.05 | 0 | 0.864559 | 0.918074 | 0.051488 | No |
| Reward-State Quotient | 0.05 | 19922944 | 0.900588 | 0.929910 | 0.172725 | Yes |
| Reward-State Quotient | 0.05 | 39845888 | 0.909995 | 0.935779 | 0.172725 | Yes |
| Reward-State Quotient | 0.05 | 99614720 | 0.947633 | 0.940714 | 0.172725 | Yes |
| Reward-State Quotient | 0.05 | 219152384 | 0.941447 | 0.938514 | 0.172725 | Yes |
| Reward-State Quotient | 0.05 | 498073600 | 0.928982 | 0.932313 | 0.490137 | No |
| Reward-State Quotient | 0.05 | 796917760 | 0.936920 | 0.940114 | 0.497375 | No |
| Reward-State Quotient | 0.05 | 996147200 | 0.938905 | 0.939867 | 0.503663 | No |
| Reward-State Quotient | 0.05 | 1115684864 | 0.938909 | 0.939092 | 0.505463 | No |
| Reward-State Quotient | 0.05 | 1414529024 | 0.938961 | 0.938960 | 0.505463 | No |
| Reward-State Quotient | 0.05 | 1713373184 | 0.936624 | 0.936002 | 0.508438 | No |
| Reward-State Quotient | 0.05 | 2012217344 | 0.938354 | 0.938354 | 0.505463 | No |
| Full-State Quotient | 0.05 | 0 | 0.867052 | 0.914324 | 0.070662 | No |
| Full-State Quotient | 0.05 | 19922944 | 0.860679 | 0.923244 | 0.090650 | No |
| Full-State Quotient | 0.05 | 39845888 | 0.884475 | 0.930459 | 0.036150 | Yes |
| Full-State Quotient | 0.05 | 99614720 | 0.926966 | 0.958601 | 0.130912 | No |
| Full-State Quotient | 0.05 | 219152384 | 0.959082 | 0.964592 | 0.377400 | No |
| Full-State Quotient | 0.05 | 498073600 | 0.958153 | 0.964563 | 0.424200 | No |
| Full-State Quotient | 0.05 | 796917760 | 0.957325 | 0.959753 | 0.446788 | No |
| Full-State Quotient | 0.05 | 996147200 | 0.955557 | 0.958031 | 0.446988 | No |
| Full-State Quotient | 0.05 | 1115684864 | 0.956625 | 0.959249 | 0.457475 | No |
| Full-State Quotient | 0.05 | 1414529024 | 0.956722 | 0.956369 | 0.461163 | No |
| Full-State Quotient | 0.05 | 1713373184 | 0.955528 | 0.954879 | 0.468250 | No |
| Full-State Quotient | 0.05 | 2012217344 | 0.954003 | 0.954003 | 0.470987 | No |

### Fixed-history peak minus final

Exploratory paired intervals condition on the observed selected peak and fitted probes; they are **not corrected for choosing the maximum**, do not cover unsampled checkpoints, and are not training-seed intervals. A final-checkpoint peak has exactly zero paired difference by construction.

| Quotient | Epsilon | Peak checkpoint steps | Peak − final | Paired 95% CI |
|---|---:|---:|---:|---|
| Trivial Quotient | 0.00 | 796917760 | +0.015214 | [+0.013129, +0.017664] |
| Reward-State Quotient | 0.00 | 99614720 | +0.025518 | [+0.021456, +0.029656] |
| Full-State Quotient | 0.00 | 1713373184 | +0.002217 | [+0.001761, +0.002677] |
| Trivial Quotient | 0.05 | 219152384 | +0.006446 | [+0.003911, +0.009626] |
| Reward-State Quotient | 0.05 | 99614720 | +0.002359 | [+0.000181, +0.004520] |
| Full-State Quotient | 0.05 | 219152384 | +0.010589 | [+0.008892, +0.012212] |

Descriptively, 6/6 sampled fixed-history trajectories peak before final. This is not a claim of monotonic decline, statistical confirmation after maximum selection, or loss of causally used belief information. Own-history changes additionally reflect visitation and action-class changes; reward is a separate measure.

**One-class own-history points:** Trivial Quotient, epsilon 0.00, step 19922944; Trivial Quotient, epsilon 0.00, step 39845888; Trivial Quotient, epsilon 0.00, step 99614720; Trivial Quotient, epsilon 0.00, step 219152384; Trivial Quotient, epsilon 0.00, step 498073600; Trivial Quotient, epsilon 0.00, step 796917760; Trivial Quotient, epsilon 0.00, step 996147200; Trivial Quotient, epsilon 0.00, step 1115684864; Trivial Quotient, epsilon 0.00, step 1414529024; Trivial Quotient, epsilon 0.00, step 1713373184; Trivial Quotient, epsilon 0.00, step 2012217344; Reward-State Quotient, epsilon 0.00, step 19922944; Reward-State Quotient, epsilon 0.00, step 39845888; Reward-State Quotient, epsilon 0.00, step 99614720; Reward-State Quotient, epsilon 0.00, step 219152384; Trivial Quotient, epsilon 0.05, step 19922944; Trivial Quotient, epsilon 0.05, step 39845888; Trivial Quotient, epsilon 0.05, step 99614720; Trivial Quotient, epsilon 0.05, step 219152384; Trivial Quotient, epsilon 0.05, step 498073600; Trivial Quotient, epsilon 0.05, step 796917760; Trivial Quotient, epsilon 0.05, step 996147200; Trivial Quotient, epsilon 0.05, step 1115684864; Trivial Quotient, epsilon 0.05, step 1414529024; Trivial Quotient, epsilon 0.05, step 1713373184; Trivial Quotient, epsilon 0.05, step 2012217344; Reward-State Quotient, epsilon 0.05, step 19922944; Reward-State Quotient, epsilon 0.05, step 39845888; Reward-State Quotient, epsilon 0.05, step 99614720; Reward-State Quotient, epsilon 0.05, step 219152384; Full-State Quotient, epsilon 0.05, step 39845888.

A one-class within-class score equals the corresponding global affine score. All fixed-history points inherit their final policy's classes, so both Trivial fixed-history tracks are one-class throughout. Class changes make the own-history denominator vary; fixed-history tracks keep that denominator and visitation fixed.

**Excluded predictions:** None.

Uncertainty: 2000 pointwise whole-test-episode bootstrap resamples; 10000 paired peak-minus-final resamples; probes held fixed and class means recomputed. The final scored episode is truncated to the 80K-row budget. One training seed only; no fit-data, seed-to-seed, simultaneous-curve or selected-peak correction. The observation/action streams and post-warmup scoring follow the original probe protocol. Affine accessibility does not establish causal use or Bayesian computation.

Reproduce with CPU-only environment variables and `uv run python -m experiments.within_class_belief_probes_2026_11.trajectory`, then `uv run python -m experiments.within_class_belief_probes_2026_11.trajectory_report`. Flags `--points`, `--bootstrap`, and `--peak-bootstrap` set the sampling budget and bootstrap counts. Caches are parameterized by points/bootstrap under each study/arm's ignored artifact directory. No new training or paid compute.

## PR219 own-policy-only checkpoint trajectory (primary follow-up)

**Primary result: each encoder is scored only on its own greedy-policy trajectories.** These curves do not use final-policy fixed histories. Each checkpoint has its own belief distribution and action partition; compare these as distribution-specific affine accessibility, not causal information use.

2000 pointwise draws; 10000 unpaired contrast draws with independently resampled episodes, class means recomputed and probes fixed. Underlying checkpoint rollouts reuse PRNG stream keys: independent resampling is the requested convention, not a claim that these common-random-number samples are strictly independent.

Peak maximizes observed own-history full-h within-class R2 among 12 checkpoints; contrasts condition on selected peak, no selection/multiplicity correction; one training seed only.

R2=1-sum(q_k*SSE_k/n_k)/sum(q_k*SST_k/n_k). q is final test class proportions, restricted and renormalized to shared peak/final support. Zero-presence bootstrap replicates are invalid, not silently renormalized. This controls class proportions, not within-class visitation or target variance.

Controls retain original own-policy targets/classes and use one fixed perturbation draw. Input controls are distribution shifts, not required-to-be-zero nulls. Init−peak is signed init minus peak, so a negative value indicates a rise.

### Trivial, epsilon 0.0

| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.833117 [0.825889, 0.839119] | 0.832589 | 0.220412 | 0.854448 | 0.693100 | 0.087362 | 2 |
| 19922944 | 0.922380 [0.910120, 0.931247] | 0.920884 | 0.639724 | 0.845679 | 0.937833 | 0.965163 | 1 |
| 39845888 | 0.925220 [0.913697, 0.933576] | 0.923883 | 0.648252 | 0.847894 | 0.940799 | 0.965163 | 1 |
| 99614720 | 0.939596 [0.930811, 0.946173] | 0.938184 | 0.676792 | 0.854405 | 0.956760 | 0.965163 | 1 |
| 219152384 | 0.970401 [0.965545, 0.974350] | 0.969928 | 0.747787 | 0.872305 | 0.990165 | 0.965163 | 1 |
| 498073600 | 0.966967 [0.960834, 0.972136] | 0.966470 | 0.745108 | 0.848148 | 0.990907 | 0.965163 | 1 |
| 796917760 | 0.974811 [0.970025, 0.978890] | 0.974649 | 0.624595 | 0.874051 | 0.995112 | 0.965163 | 1 |
| 996147200 | 0.973116 [0.968309, 0.977069] | 0.973067 | 0.514719 | 0.869167 | 0.994059 | 0.965163 | 1 |
| 1115684864 | 0.969468 [0.964425, 0.973578] | 0.969541 | 0.503438 | 0.863165 | 0.990885 | 0.965163 | 1 |
| 1414529024 | 0.964401 [0.958868, 0.968878] | 0.964463 | 0.516271 | 0.857603 | 0.985919 | 0.965163 | 1 |
| 1713373184 | 0.958737 [0.952772, 0.963635] | 0.958006 | 0.520524 | 0.855772 | 0.979482 | 0.965163 | 1 |
| 2012217344 | 0.959598 [0.953714, 0.964431] | 0.958762 | 0.416803 | 0.854629 | 0.980746 | 0.965163 | 1 |

Peak checkpoint: `checkpoints/step_0796917760.npz`.

| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |
|---|---:|---:|---|
| Peak − final | 0.974811 | 0.959598 | +0.015214 [+0.008465, +0.022137] |
| Init − peak | 0.833117 | 0.974811 | -0.141694 [-0.150143, -0.133606] |
| Peak − final, final class proportions on shared support | 0.974811 | 0.959598 | +0.015214 [+0.008465, +0.022137] |

Shared classes: [1]; standardized weights: [0.0, 1.0, 0.0]. Final mass covered: 1.000000; peak mass covered: 1.000000.

| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |
|---:|---:|---:|---|
| 1 | 0.974811 | 0.959598 | +0.015214 [+0.008465, +0.022137] |

| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |
|---|---|---|---|
| init / 0 | 0.034169 [-0.002036, 0.067714] | -0.029880 [-0.061698, -0.010287] | -0.028826 [-0.060747, -0.009711] |
| peak / 796917760 | 0.019486 [-0.007773, 0.039434] | -0.015052 [-0.028804, -0.003837] | -0.012761 [-0.019487, -0.009566] |
| final / 2012217344 | 0.025828 [-0.001907, 0.047207] | -0.009705 [-0.021695, 0.000525] | -0.011051 [-0.017842, -0.007725] |

### Reward-State, epsilon 0.0

| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.864559 [0.858358, 0.869660] | 0.864196 | 0.232276 | 0.875484 | 0.634079 | 0.051488 | 2 |
| 19922944 | 0.897409 [0.894414, 0.900122] | 0.897187 | 0.281735 | 0.919183 | 0.802121 | 0.172725 | 1 |
| 39845888 | 0.906467 [0.903599, 0.909061] | 0.906373 | 0.243454 | 0.925092 | 0.824964 | 0.172725 | 1 |
| 99614720 | 0.939250 [0.936797, 0.941494] | 0.938821 | 0.312174 | 0.938874 | 0.940893 | 0.172725 | 1 |
| 219152384 | 0.940165 [0.937263, 0.942867] | 0.939895 | 0.255928 | 0.928277 | 0.992187 | 0.172725 | 1 |
| 498073600 | 0.939672 [0.936307, 0.942700] | 0.939399 | 0.326322 | 0.908640 | 0.995528 | 0.499250 | 2 |
| 796917760 | 0.935131 [0.931505, 0.938432] | 0.935081 | 0.346882 | 0.899308 | 0.998548 | 0.501238 | 2 |
| 996147200 | 0.934345 [0.930652, 0.937818] | 0.934366 | 0.348111 | 0.898660 | 0.997557 | 0.501238 | 2 |
| 1115684864 | 0.930879 [0.926998, 0.934502] | 0.930849 | 0.337076 | 0.893403 | 0.997264 | 0.501238 | 2 |
| 1414529024 | 0.930014 [0.926271, 0.933586] | 0.928376 | 0.351423 | 0.892193 | 0.997391 | 0.501113 | 2 |
| 1713373184 | 0.931615 [0.927818, 0.935360] | 0.929246 | 0.382478 | 0.894484 | 0.997765 | 0.501113 | 2 |
| 2012217344 | 0.917327 [0.912456, 0.921890] | 0.916750 | 0.385431 | 0.872847 | 0.996568 | 0.501113 | 2 |

Peak checkpoint: `checkpoints/step_0219152384.npz`.

| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |
|---|---:|---:|---|
| Peak − final | 0.940165 | 0.917327 | +0.022838 [+0.017525, +0.028194] |
| Init − peak | 0.864559 | 0.940165 | -0.075606 [-0.082523, -0.069404] |
| Peak − final, final class proportions on shared support | 0.940165 | 0.878384 | +0.061781 [+0.054681, +0.069249] |

**Full-final-class-mix pooled contrast is undefined:** peak has no samples for final classes [0]. The numeric standardized contrast below/above is restricted to shared support; it must not be interpreted as covering the entire final policy distribution.

Shared classes: [1]; standardized weights: [0.0, 1.0, 0.0]. Final mass covered: 0.405712; peak mass covered: 1.000000.

| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |
|---:|---:|---:|---|
| 1 | 0.940165 | 0.878384 | +0.061781 [+0.054681, +0.069249] |

| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |
|---|---|---|---|
| init / 0 | 0.050405 [0.015456, 0.081178] | -0.028907 [-0.059162, -0.012567] | -0.026072 [-0.051790, -0.015866] |
| peak / 219152384 | 0.006543 [-0.001369, 0.012134] | -0.009734 [-0.016064, -0.006285] | -0.007156 [-0.012422, -0.005909] |
| final / 2012217344 | 0.007233 [-0.006027, 0.016767] | -0.008323 [-0.019169, -0.004098] | 0.001280 [-0.009921, 0.005640] |

### Full-State, epsilon 0.0

| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.867052 [0.861614, 0.871858] | 0.866748 | 0.270876 | 0.889033 | 0.600466 | 0.070662 | 2 |
| 19922944 | 0.805209 [0.789391, 0.817688] | 0.803974 | 0.123460 | 0.822783 | 0.549508 | 0.038863 | 2 |
| 39845888 | 0.645191 [0.369308, 0.861938] | 0.436392 | 0.161201 | 0.650677 | 0.583207 | 0.086675 | 3 |
| 99614720 | 0.878554 [0.871542, 0.884515] | 0.877501 | 0.410261 | 0.877495 | 0.882206 | 0.130275 | 3 |
| 219152384 | 0.956675 [0.953945, 0.959102] | 0.956468 | 0.760412 | 0.909649 | 0.980927 | 0.429112 | 3 |
| 498073600 | 0.958770 [0.955124, 0.962204] | 0.958283 | 0.832945 | 0.898931 | 0.986512 | 0.463313 | 3 |
| 796917760 | 0.964280 [0.961157, 0.967182] | 0.964107 | 0.884988 | 0.910159 | 0.990121 | 0.458025 | 3 |
| 996147200 | 0.963038 [0.959404, 0.966325] | 0.962630 | 0.874497 | 0.904791 | 0.991018 | 0.455213 | 3 |
| 1115684864 | 0.966168 [0.962713, 0.969295] | 0.965733 | 0.882342 | 0.913713 | 0.992020 | 0.449313 | 3 |
| 1414529024 | 0.969655 [0.966511, 0.972463] | 0.969566 | 0.890579 | 0.917354 | 0.992629 | 0.454275 | 3 |
| 1713373184 | 0.971982 [0.968970, 0.974628] | 0.971721 | 0.887506 | 0.925981 | 0.991730 | 0.463575 | 3 |
| 2012217344 | 0.969721 [0.966708, 0.972367] | 0.969416 | 0.881228 | 0.923861 | 0.989194 | 0.465250 | 3 |

Peak checkpoint: `checkpoints/step_1713373184.npz`.

| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |
|---|---:|---:|---|
| Peak − final | 0.971982 | 0.969721 | +0.002260 [-0.001776, +0.006227] |
| Init − peak | 0.867052 | 0.971982 | -0.104930 [-0.111124, -0.099253] |
| Peak − final, final class proportions on shared support | 0.972272 | 0.969721 | +0.002551 [-0.000991, +0.006022] |

Shared classes: [0, 1, 2]; standardized weights: [0.60795, 0.18415, 0.2079]. Final mass covered: 1.000000; peak mass covered: 1.000000.

| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |
|---:|---:|---:|---|
| 0 | 0.988344 | 0.985881 | +0.002463 [+0.000708, +0.004230] |
| 1 | 0.893589 | 0.880972 | +0.012616 [-0.004970, +0.030406] |
| 2 | 0.902906 | 0.904095 | -0.001189 [-0.015434, +0.012575] |

| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |
|---|---|---|---|
| init / 0 | 0.024845 [0.007703, 0.042269] | -0.018077 [-0.033374, -0.009755] | -0.015739 [-0.028648, -0.011157] |
| peak / 1713373184 | 0.013819 [0.000565, 0.025556] | -0.011073 [-0.018081, -0.007768] | -0.008749 [-0.014431, -0.006573] |
| final / 2012217344 | 0.012788 [-0.000446, 0.025638] | -0.012393 [-0.019486, -0.009002] | -0.011545 [-0.017599, -0.009573] |

### Trivial, epsilon 0.05

| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.833117 [0.825889, 0.839119] | 0.832589 | 0.220412 | 0.854448 | 0.693100 | 0.087362 | 2 |
| 19922944 | 0.924894 [0.913045, 0.933446] | 0.923526 | 0.644509 | 0.847437 | 0.940500 | 0.965163 | 1 |
| 39845888 | 0.930224 [0.919574, 0.938070] | 0.929064 | 0.658096 | 0.850809 | 0.946224 | 0.965163 | 1 |
| 99614720 | 0.944117 [0.936322, 0.950176] | 0.942807 | 0.627713 | 0.859053 | 0.961255 | 0.965163 | 1 |
| 219152384 | 0.973949 [0.968619, 0.978161] | 0.973779 | 0.764861 | 0.872862 | 0.994316 | 0.965163 | 1 |
| 498073600 | 0.971550 [0.966314, 0.975611] | 0.971413 | 0.788682 | 0.870823 | 0.991844 | 0.965163 | 1 |
| 796917760 | 0.970413 [0.964830, 0.974682] | 0.970006 | 0.782672 | 0.869217 | 0.990802 | 0.965163 | 1 |
| 996147200 | 0.970056 [0.964250, 0.974444] | 0.969432 | 0.778896 | 0.868653 | 0.990487 | 0.965163 | 1 |
| 1115684864 | 0.969839 [0.963933, 0.974280] | 0.969095 | 0.777806 | 0.868221 | 0.990313 | 0.965163 | 1 |
| 1414529024 | 0.969015 [0.962883, 0.973633] | 0.968009 | 0.772891 | 0.867672 | 0.989433 | 0.965163 | 1 |
| 1713373184 | 0.968076 [0.961718, 0.972815] | 0.966866 | 0.768891 | 0.867191 | 0.988402 | 0.965163 | 1 |
| 2012217344 | 0.967503 [0.961061, 0.972318] | 0.966225 | 0.765818 | 0.867105 | 0.987730 | 0.965163 | 1 |

Peak checkpoint: `checkpoints/step_0219152384.npz`.

| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |
|---|---:|---:|---|
| Peak − final | 0.973949 | 0.967503 | +0.006446 [-0.000435, +0.013848] |
| Init − peak | 0.833117 | 0.973949 | -0.140832 [-0.149283, -0.132717] |
| Peak − final, final class proportions on shared support | 0.973949 | 0.967503 | +0.006446 [-0.000435, +0.013848] |

Shared classes: [1]; standardized weights: [0.0, 1.0, 0.0]. Final mass covered: 1.000000; peak mass covered: 1.000000.

| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |
|---:|---:|---:|---|
| 1 | 0.973949 | 0.967503 | +0.006446 [-0.000435, +0.013848] |

| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |
|---|---|---|---|
| init / 0 | 0.034169 [-0.002036, 0.067714] | -0.029880 [-0.061698, -0.010287] | -0.028826 [-0.060747, -0.009711] |
| peak / 219152384 | 0.022426 [-0.010983, 0.052625] | -0.015530 [-0.030098, -0.002922] | -0.010223 [-0.016830, -0.006803] |
| final / 2012217344 | 0.027682 [-0.008650, 0.060074] | -0.017464 [-0.034341, -0.003599] | -0.006488 [-0.013071, -0.003602] |

### Reward-State, epsilon 0.05

| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.864559 [0.858358, 0.869660] | 0.864196 | 0.232276 | 0.875484 | 0.634079 | 0.051488 | 2 |
| 19922944 | 0.900588 [0.897564, 0.903319] | 0.900479 | 0.324689 | 0.919755 | 0.816712 | 0.172725 | 1 |
| 39845888 | 0.909995 [0.907167, 0.912582] | 0.909924 | 0.384560 | 0.925245 | 0.843260 | 0.172725 | 1 |
| 99614720 | 0.947633 [0.945219, 0.949851] | 0.947595 | 0.449948 | 0.938549 | 0.987387 | 0.172725 | 1 |
| 219152384 | 0.941447 [0.938704, 0.943939] | 0.940111 | 0.394748 | 0.930374 | 0.989902 | 0.172725 | 1 |
| 498073600 | 0.928982 [0.925292, 0.932350] | 0.928337 | 0.359181 | 0.894118 | 0.996131 | 0.490137 | 2 |
| 796917760 | 0.936920 [0.933283, 0.940146] | 0.936635 | 0.320918 | 0.902315 | 0.997492 | 0.497375 | 2 |
| 996147200 | 0.938905 [0.935302, 0.941996] | 0.938716 | 0.328901 | 0.902195 | 0.997254 | 0.503663 | 2 |
| 1115684864 | 0.938909 [0.935119, 0.942160] | 0.938784 | 0.344226 | 0.901373 | 0.996874 | 0.505463 | 2 |
| 1414529024 | 0.938961 [0.935123, 0.942351] | 0.938805 | 0.311312 | 0.901324 | 0.997029 | 0.505463 | 2 |
| 1713373184 | 0.936624 [0.932969, 0.939936] | 0.936260 | 0.288744 | 0.896814 | 0.994672 | 0.508438 | 2 |
| 2012217344 | 0.938354 [0.934611, 0.941614] | 0.938323 | 0.286622 | 0.902098 | 0.994294 | 0.505463 | 2 |

Peak checkpoint: `checkpoints/step_0099614720.npz`.

| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |
|---|---:|---:|---|
| Peak − final | 0.947633 | 0.938354 | +0.009279 [+0.005016, +0.013631] |
| Init − peak | 0.864559 | 0.947633 | -0.083074 [-0.089759, -0.077138] |
| Peak − final, final class proportions on shared support | 0.947633 | 0.908531 | +0.039102 [+0.033547, +0.045002] |

**Full-final-class-mix pooled contrast is undefined:** peak has no samples for final classes [0]. The numeric standardized contrast below/above is restricted to shared support; it must not be interpreted as covering the entire final policy distribution.

Shared classes: [1]; standardized weights: [0.0, 1.0, 0.0]. Final mass covered: 0.380412; peak mass covered: 1.000000.

| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |
|---:|---:|---:|---|
| 1 | 0.947633 | 0.908531 | +0.039102 [+0.033547, +0.045002] |

| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |
|---|---|---|---|
| init / 0 | 0.050405 [0.015456, 0.081178] | -0.028907 [-0.059162, -0.012567] | -0.026072 [-0.051790, -0.015866] |
| peak / 99614720 | 0.006790 [-0.001660, 0.012702] | -0.010845 [-0.016934, -0.007195] | -0.008026 [-0.013311, -0.006749] |
| final / 2012217344 | 0.008060 [-0.006512, 0.020368] | -0.008550 [-0.015616, -0.004925] | -0.000472 [-0.007529, 0.001102] |

### Full-State, epsilon 0.05

| Steps | Within-class R² [95% CI] | Head-blind | Logits | Delta | Rho | Reward | Test classes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.867052 [0.861614, 0.871858] | 0.866748 | 0.270876 | 0.889033 | 0.600466 | 0.070662 | 2 |
| 19922944 | 0.860679 [0.855315, 0.865592] | 0.860260 | 0.116108 | 0.879953 | 0.657759 | 0.090650 | 2 |
| 39845888 | 0.884475 [0.878262, 0.889335] | 0.882804 | 0.348756 | 0.899907 | 0.665909 | 0.036150 | 1 |
| 99614720 | 0.926966 [0.924058, 0.929577] | 0.925543 | 0.661828 | 0.928549 | 0.923218 | 0.130912 | 3 |
| 219152384 | 0.959082 [0.956436, 0.961400] | 0.958344 | 0.805404 | 0.926547 | 0.981200 | 0.377400 | 3 |
| 498073600 | 0.958153 [0.955244, 0.960755] | 0.956505 | 0.800258 | 0.904458 | 0.984145 | 0.424200 | 3 |
| 796917760 | 0.957325 [0.954361, 0.960193] | 0.955395 | 0.700177 | 0.901666 | 0.981957 | 0.446788 | 3 |
| 996147200 | 0.955557 [0.952652, 0.958359] | 0.954390 | 0.689120 | 0.900062 | 0.980212 | 0.446988 | 3 |
| 1115684864 | 0.956625 [0.953289, 0.959705] | 0.955434 | 0.680299 | 0.901190 | 0.980926 | 0.457475 | 3 |
| 1414529024 | 0.956722 [0.953713, 0.959394] | 0.955479 | 0.662259 | 0.900883 | 0.980054 | 0.461163 | 3 |
| 1713373184 | 0.955528 [0.952419, 0.958442] | 0.954616 | 0.621109 | 0.900153 | 0.978590 | 0.468250 | 3 |
| 2012217344 | 0.954003 [0.950386, 0.957292] | 0.953369 | 0.570734 | 0.897462 | 0.977844 | 0.470987 | 3 |

Peak checkpoint: `checkpoints/step_0219152384.npz`.

| Contrast | Left R² | Right R² | Difference [unpaired 95% CI] |
|---|---:|---:|---|
| Peak − final | 0.959082 | 0.954003 | +0.005079 [+0.000974, +0.009456] |
| Init − peak | 0.867052 | 0.959082 | -0.092030 [-0.098063, -0.086458] |
| Peak − final, final class proportions on shared support | 0.969343 | 0.954003 | +0.015340 [+0.011922, +0.018865] |

Shared classes: [0, 1, 2]; standardized weights: [0.6092125, 0.20465, 0.1861375]. Final mass covered: 1.000000; peak mass covered: 1.000000.

| Shared action | Peak R² | Final R² | Difference [unpaired 95% CI] |
|---:|---:|---:|---|
| 0 | 0.985118 | 0.972059 | +0.013060 [+0.010900, +0.015307] |
| 1 | 0.928900 | 0.883309 | +0.045591 [+0.034451, +0.057424] |
| 2 | 0.933608 | 0.855637 | +0.077970 [+0.066147, +0.090257] |

| Control stage / steps | Shuffle R² [CI] | Random input R² [CI] | Label permutation R² [CI] |
|---|---|---|---|
| init / 0 | 0.024845 [0.007703, 0.042269] | -0.018077 [-0.033374, -0.009755] | -0.015739 [-0.028648, -0.011157] |
| peak / 219152384 | 0.011032 [-0.002524, 0.023194] | -0.016822 [-0.024370, -0.011875] | -0.000378 [-0.005169, 0.001677] |
| final / 2012217344 | 0.004524 [-0.006650, 0.014045] | -0.010177 [-0.017283, -0.007285] | 0.009148 [0.002806, 0.011004] |

One-class points are marked with crosses in the figure; their class-centred full-h scores equal global affine scores. Per-class point estimates/intervals, all global scores, action counts, head-blind/logits/scalar/reward intervals and checkpoint provenance are in JSON. Class reweighting does not fix within-class visitation, representation scaling effects, or training-seed uncertainty.

Reproduce: with the documented CPU-only/B2 environment, run `uv run python -m experiments.within_class_belief_probes_2026_11.own_trajectory` then `uv run python -m experiments.within_class_belief_probes_2026_11.own_trajectory_report`. Own rollouts are reused; missing predictions/controls are cached under each run's ignored `own_trajectory/` directory. No training or paid compute.
