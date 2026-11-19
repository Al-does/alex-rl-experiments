# Twin steering: XOR transfer

## Results

Mean XOR reward per decision at block 4. Each pair is the old regression
write → the twin write at each method's tuning-selected λ. In the table,
"null" is the twin's norm-matched random direction at its selected λ.

### Effective v2 environment

Uniform random actions earn .1815 and the exact-belief QMDP map earns .6189.

| Seed | Intact | Factored old → twin | Twin λ / null | Joint old → twin | Twin λ / null |
|---|---:|---:|---:|---:|---:|
| 42 | .4959 | .5845 → .5676 | .9 / .4969 | .5478 → .5557 | .8 / .4959 |
| 43 | .4972 | .5895 → .5875 | .9 / .4931 | .5666 → .5699 | 1.25 / .4905 |
| 44 | .4917 | .5815 → .5729 | .85 / .4912 | .5732 → .5745 | 1 / .4934 |

Both twin writes improve on intact and the matched null in all three
agents. The old factored write is stronger on every seed (mean .5851
versus .5760; 72.8% versus 65.5% of the intact-to-exact gap closed). The twin
joint write narrowly beats the old joint write in all three (mean .5667
versus .5626; 57.8% versus 54.5% of the gap). The **old factored write**
remains the strongest overall in this cohort.

### Paper emission-ε=0.02 environment

Uniform random actions earn .1815 and the exact-belief QMDP map earns .5626.

| Seed | Intact | Factored old → twin | Twin λ / null | Joint old → twin | Twin λ / null |
|---|---:|---:|---:|---:|---:|
| 42 | .4807 | .4841 → .4807 | 0 / .4807 | .4807 → .4807 | 0 / .4807 |
| 43 | .4880 | .5071 → .4880 | 0 / .4880 | .4880 → .4880 | 0 / .4880 |
| 44 | .4784 | .4784 → .4784 | 0 / .4784 | .5037 → .4915 | 1.5 / .4774 |

The twin method does **not** rescue paper XOR. Tuning selects λ=0 in five
of six seed/target combinations. Its one nonzero result beats intact and
its null but not the corresponding old joint write. The old method closes
9.9% and 10.0% of the gap for factored and joint, respectively; the twin
closes 0.0% and 5.2%. These are descriptive paired comparisons on just
three agents per cohort; the small positive joint difference in v2 is
not evidence of a reliable population advantage.

### Held-out diagnostics

The twin write predicts a substantial fraction of paired residual
displacement energy in both environments (v2 .647–.718; eps02
.654–.693). But plausible anchor targets are less well represented in the
eps02 training pairs: 55.4–64.1% of factored goals and 37.2–45.5% of joint
goals have a source–target neighbor within 0.1, versus 72.6–76.0% and
71.1–73.8% in v2. An independent readout also detects absolute changes
of up to .112 on eps02 coordinates whose true change is less than .02
(up to .094 in v2). Support and specificity are limitations of this fit;
these data do not isolate the cause of the weaker transfer.

## Protocol

The comparison covers two cohorts of final Joint Reward transformers
(training seeds 42–44 each): persistent-nuisance v2, no exploration, in
which state 3 always emits C, and the paper's emission-ε=0.02 cohort.
The historical block-4 baseline is the discounted (γ=0.99) finite-horizon
QMDP map, with separate factored and joint affine writes. An XOR reward of 1
is scored when exactly one factor occupies state 3 before the transition.
Exact-belief QMDP is a reference, not an optimality bound.

The twin run retains the same decoders, action-anchor targets, QMDP map,
candidates for λ, and smallest-best-λ rule. Both methods are compared on
the same environment draws using 64 tuning and 256 confirmation episodes
of 1,024 decisions each. The old values are read from the archived
`rewards.xor` reports at `3504bfb7`. It is not a moving-reward run.

The [ladder protocol](../belief_quotient_twin_steering_2026_11/findings.md#protocol)
describes the common paired fit, support and specificity checks. Here we
generate token proposals for factor 1 alone, factor 2 alone, or both,
always preserving both executed-action histories. The single-factor
proposals leave the other factor's tokens intact. The old factored and
joint targets choose the coordinates supplied to `g(h,b*)`. We fit on
60,000 intact rows plus their twins; the new specificity diagnostic uses
12,000 base rows plus their independently generated twins, with the
original warmup of 64 decisions.

```bash
uv run --group jax python -m experiments.belief_quotient_twin_steering_2026_11.run --cohort v2_xor eps02_xor --workers 6
```

The v2 seed-44 training recipe and run manifest were not on main; that
run read them from the pinned archive after adding the metadata fallback
to the runner. The first eight runs used the original runner with their
main-branch manifests. The numerical implementation did not change.
