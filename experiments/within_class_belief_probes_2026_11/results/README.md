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
