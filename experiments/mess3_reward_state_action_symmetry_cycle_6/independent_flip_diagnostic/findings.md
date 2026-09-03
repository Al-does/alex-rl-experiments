# Variant-2 independent-flip findings

## Checkpoint audit

The final Cycle-6 checkpoints are not misidentified. Their training returns
are 282.8--283.2 per 1,024-step episode, which matches the analytic
always-positive baseline of 282.98. Their greedy action frequency is exactly
`[noop, positive, negative] = [0, 1, 0]`.

Seed 44 briefly learned a non-degenerate Cycle-6 policy at
`checkpoint_000001`: its held-out reward is 0.318 and its action frequencies
are approximately `[0.298, 0.702, 0]`. Independent flips change 8.25% of its
paired greedy actions, but its mean closed-loop reward change is only
$8.75\times10^{-5}$. This transient checkpoint is therefore not purely
coarse-invariant. It is the only strong two-action Cycle-6 checkpoint among
the battery's existing checkpoint probes.

The main comparison therefore uses the final Cycle-5 Variant-2 checkpoints,
which share the same HMM and action task and retain the intended two-action
policy across seeds 42--44.

## Main result

The converged two-action policies behave **much more like policies using
$c_t$ than policies using the exact-history value $s_t$**:

- factual action frequencies are approximately `[0.33, 0.67, 0]`;
- independent flips move policy probabilities by only 0.0043 mean total
  variation;
- paired greedy actions agree on 99.997% of rows;
- all eight closed-loop randomizations preserve reward exactly for every seed.

The hidden representation is not uniformly a pure coarse-filter state. The
coarse decoder has about 49 times lower factual MSE than the fine decoder, and
counterfactual coarse MSE remains stable while fine MSE rises 25%. However,
the frozen fine decoder tracks 55% of the exact $s_t$ shift variance in seed
42, 8% in seed 43, and none in seed 44.

The best-supported interpretation is therefore:

1. $c_t$ is the dominant, cleanly accessible coordinate;
2. some learned states retain seed-dependent fine-history information;
3. the converged action rule is almost completely invariant to that
   information and is consequently coarse-filter-like.

This also shows why “does the agent use $s_t$ or $c_t$?” needs separate
representation and policy answers. A hidden state can retain some $s_t$
information while the action head relies almost entirely on the $c_t$-like
coordinate.

## Protocol

Each main-cohort checkpoint used 30,000 factual probe-training rows, 40,000
held-out factual rows, eight independent flip replicas, and 40,000 closed-loop
rows per replica. Paired replay held the factual previous-action sequence
fixed and recomputed the exact fine filter for every randomized token history.
