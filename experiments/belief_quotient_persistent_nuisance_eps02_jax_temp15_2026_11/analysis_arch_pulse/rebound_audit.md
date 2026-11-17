# Reward-State Quotient GRU128 ρ-flip rebound audit

## Conclusion

No pulse implementation bug was found. The rebound is reproducible dynamics
of the affine-probed, perturbed recurrent state, not an indexing error or a
few-episode artifact. This is not evidence that the rebound is a valid Bayesian
belief trajectory. In particular, the affine decoder and write are unconstrained
and need not remain on the probability simplex or natural hidden-state manifold.

The original six reports, `results/summary.{json,md}`, and the paper's pulse
PDF/plot script were checked by SHA-256 before/after and are unchanged.
No manuscript changes, training, paid compute, commits or pushes.

## Fresh reproduction and outputs

Same final seed-42 checkpoint: run `20261116T211927Z-b181bccd`,
`step_2012217344.npz`, SHA-256
`366c60e7ed8b46e3c4dc054b2c4316b188dd8cb6b73c52590ba195ea1a317fe9`.
Bytes/SHA-256 were reverified against the canonical B2 manifest before loading.
Same original protocol: greedy fit/test seeds 42/43, 14 × 1024-step episodes,
12,000 rows after eight-step exclusion, affine rcond 1e-6, pulse seed 3042,
32 episodes, pulse times 256/512/768, full write once, 129 lags, fixed-input
replay. Probes were refitted rather than copied; held-out scores match exactly.

Two fresh original-path reruns reproduce the saved numerical curves exactly:

```text
absolute bitwise_equal: True max_abs_difference: 0.0
signed bitwise_equal: True max_abs_difference: 0.0
```

The comparison covers all 129 lags of the requested coordinate; the full rerun's
four pulse/site conditions, signed/absolute and measured/filter curves also
have maximum difference 0.0. Whole JSON file bytes differ because these are new
reports with different metadata. The diagnostic reduction of the per-episode
arrays uses a different float32 summation order: its maximum absolute-mean
difference is 2.384185791015625e-7, not a new trajectory discrepancy.

Final outputs (relative to this directory):

- `diagnose_rebound.py`: independent diagnostic/control driver.
- `results/variant_2_gru128_seed42_rebound_rerun_v2.json`: fresh original-path
  full pulse result, preserving the existing report.
- `results/rebound_audit_gru128_reward_state_v2.json`: final distribution,
  timing, controls, checkpoint provenance and preserved-output hashes.
- `../artifacts/arch_pulse/rebound_audit_per_episode_v2.npz`: complete signed
  differences, axes `(pulse time, lag, episode)` = `(3,129,32)`, plus controls
  and projected cell-update components. Original episode IDs are array indices
  0–31, shared across pulse times. The raw data stays ignored.

Initial exploratory outputs without `_v2` are also retained, not overwritten.
Use `_v2`: it adds an exact-matched-top-state lower-layer control and correctly
distinguishes leave-one-episode-out from leave-one-pulse/episode-pair-out.

Reproduction from repository root (use new output names if these already exist):

```bash
uv run python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse.diagnose_rebound \
  --output NEW_AUDIT.json --rerun-output NEW_RERUN.json \
  --raw-output experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/artifacts/arch_pulse/NEW_RAW.npz
```

## Breakdown by pulse time

Mean absolute decoded ρ difference, 32 episodes per row:

| Pulse time | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | .635003 | .166956 | .295780 | .428172 | .513110 | .536304 | .526744 | .471947 | .408848 |
| 512 | .606223 | .177849 | .281978 | .422855 | .501097 | .518574 | .503507 | .472079 | .431926 |
| 768 | .638292 | .175358 | .301347 | .430545 | .483974 | .479480 | .420896 | .383972 | .344096 |

Raw signed mean (`written - intact`):

| Pulse time | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | -.006299 | -.069804 | -.062857 | -.208399 | -.295639 | -.318937 | -.316818 | -.287295 | -.261624 |
| 512 | .002237 | -.049071 | -.054666 | -.142549 | -.204495 | -.227602 | -.222856 | -.204301 | -.179998 |
| 768 | .113793 | -.037482 | -.002039 | -.050779 | -.111665 | -.145106 | -.166551 | -.154816 | -.145555 |

The rebound occurs in every pulse-time group. At +4 versus +1, 24/32, 27/32
and 27/32 pulse/episode pairs increase. Averaging each episode over its three
pulse times, **all 32 episodes** increase. Removing any one entire episode
leaves a +4-minus-+1 absolute-mean increase between .313707 and .335900.

## Distribution and signed/absolute comparison

96 pulse/episode pairs; these are 32 independent episodes, not 96 independent
episodes. P10/median/P90 are quantiles of absolute per-pair differences:

| Lag | Mean absolute | Signed mean | Pulse-sign-aligned mean | P10 | Median | P90 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | .626506 | .036577 | .626506 | .234012 | .706339 | .890025 | 1.005645 |
| 1 | .173387 | -.052119 | .123417 | .024443 | .143849 | .341341 | .452192 |
| 2 | .293035 | -.039854 | .257313 | .072100 | .303703 | .460673 | .574809 |
| 3 | .427191 | -.133909 | .385818 | .081243 | .481553 | .735619 | .778222 |
| 4 | .499394 | -.203933 | .459320 | .070842 | .503775 | .946060 | 1.052633 |
| 5 | .511453 | -.230548 | .467406 | .077987 | .475240 | 1.019235 | 1.187480 |
| 6 | .483716 | -.235408 | .441422 | .072700 | .421443 | 1.006542 | 1.194950 |
| 7 | .442666 | -.215471 | .405278 | .065324 | .378427 | .943574 | 1.135852 |
| 8 | .394956 | -.195726 | .362369 | .055565 | .310486 | .867922 | 1.049138 |

Pulse-sign alignment multiplies each pair's entire signed trace by the sign of
its own lag-zero effect before averaging. Its rebound shows that cancellation
between positive and negative pulses is not the explanation. Only 13.5% of
pairs at +1 and 17.7% at +5 oppose their initial pulse direction.
The largest five pairs contribute 10.8% of absolute change at +4 and 11.7% at
+5; the largest five of 32 episodes, averaged across times, contribute 26.4%
and 27.5%. The median and broad episode participation rule out a handful of
outliers driving the rebound. Values exceeding one reflect the unconstrained
affine decoder, not differences between two valid probability vectors.

## Nearby comparisons

Absolute means at lags 0–8:

| Agent/condition/site, ρ flip | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU128 Reward-State Quotient layer 1 | .524308 | .381879 | .285971 | .227251 | .207429 | .196963 | .178357 | .169328 | .164960 |
| GRU92 Reward-State Quotient layer 4 | .643188 | .582771 | .432077 | .294385 | .240143 | .215599 | .204498 | .192652 | .181563 |
| GRU128 Full-State Quotient layer 2 | .609642 | .498276 | .439917 | .399118 | .376876 | .353332 | .329524 | .306400 | .285742 |

None shows this early top-site dip/rebound. These were freshly refitted/replayed;
their independent per-episode reductions match saved absolute curves within
5.96e-8 (GRU92) and 8.94e-8 (Full-State GRU128).

## Bug audit and mechanism

1. **Carry and indexing.** `recurrent.step` appends the modified output after
   each cell, feeds it to the next layer, and returns that stack as the next
   carry. `snapshots` saves pre-decision carry; `branch` starts with observation
   `t0` and writes only at lag zero. Intact replay decodes match recorded intact
   readouts at every lag exactly (maximum difference 0). Independent explicit
   cell-by-cell replay agrees with the real branch within 4.53e-6 over all
   times/episodes/lags, and tests compare every layer's actual carry.
2. **Input alignment.** Both branches consume the same observation array,
   including the preceding intact-run action; there is no branch environment
   step or branch-selected action in replay. For a top-layer write, the actual
   layer-1 hidden states are identical across branches throughout (difference
   exactly 0): no top-to-bottom feedback is accidentally introduced.
3. **Probe/target.** The same forward and reverse affine maps are fitted at each
   site with the correct top-site coefficients. Independent `rho_target` agrees
   with the driver exactly on all 96 pulse inputs; target ρ is `1-rho` with
   maximum error 0. No clipping exists in the reference protocol. One decoded
   pulse ρ is slightly negative (range [-.0062734,.9493865]); the nonreward-mass
   fallback is never used. Clipping source ρ as an explicit alternative changes
   +5 from .511453 to .511413, leaving the rebound. It is not its cause.
4. **Actual recurrent mechanism.** For a GRU cell `h' = z*h + (1-z)*n`, the
   written-minus-intact difference is exactly decomposed as
   `zb*(hb-hi) + (zb-zi)*(hi-ni) + (1-zb)*(nb-ni)` before affine ρ projection.
   At +1 the pulse-sign-aligned projected terms are +.435388, +.008849,
   **−.320821**: the candidate term cancels much of the carried perturbation.
   At +5 they are +.260732, −.005132, **+.211806**: the candidate term now
   reinforces it. The per-pair component reconstruction residual is below
   9.85e-6; this is observed on these actual trajectories, not just a possible
   code path.

### Causal controls on the lower/top-state mismatch hypothesis

All controls use the original saved future inputs, same 96 windows, and one
write only. These are diagnostic counterfactuals, not replacements for the
paper protocol. Absolute means:

| Control | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Actual top-only write | .626506 | .173388 | .293035 | .427191 | .499394 | .511453 |
| Erase altered top carry after pulse | .626506 | 0 | 0 | 0 | 0 | 0 |
| Keep **exact same** top pulse state; additionally flip layer 1 | .626506 | .624690 | .664157 | .624918 | .581079 | .546318 |
| Coordinated shadow-target writes to both layers | .627459 | .464916 | .361918 | .290699 | .253816 | .233354 |
| Hold lower-layer drive at its intact pulse value | .626506 | .167804 | .286269 | .412437 | .482925 | .493774 |
| Zero lower-layer drive after pulse | .626506 | .558606 | .490978 | .380802 | .279710 | .192532 |
| Clip source ρ before computing flip | .626441 | .173390 | .292993 | .427146 | .499351 | .511413 |

The same-top control has maximum initial top-state difference **0.0** from
normal, isolating the effect of changing the lower-layer drive/carry. It
removes the sharp +1 collapse and shifts the peak to +2. Erasing the altered
top carry eliminates all future differences. Thus top-state persistence and
its interaction with the unaltered lower drive are causally involved.

Holding lower drive constant still produces the dip and rebound: changes in
future tokens or time-varying lower input are **not necessary**. Zeroing lower
drive removes it, but is an off-distribution control. The precise supported
explanation is nonlinear recurrent relaxation of a top-only perturbed state
under intact lower-layer drive, including the measured candidate-term
cancellation/reinforcement. Neither the coordinated-write control nor this
audit proves hidden states are on-manifold or establishes Bayesian recursion.

## Verification

```text
39 passed in 21.49s
Protected original reports, summaries, PDF and plot-script hashes: UNCHANGED
Diagnostic source SHA matches: True
```

Tests cover independent GRU cell equations, all-layer replay/carry parity,
matched-top control initialization, carry erasure, gate-term reconstruction,
unclipped/clipped target handling including out-of-simplex inputs, and
per-episode distribution/leave-one-episode-out calculations, in addition to
the existing pulse and eps02 tests.
