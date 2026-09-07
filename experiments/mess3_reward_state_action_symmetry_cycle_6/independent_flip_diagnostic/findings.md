# Cycle-6 Variant-2 independent-flip findings

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

## Result

Across the final checkpoints for seeds 42--44, the coarse target is more
stable under independent flips, but the hidden state retains measurable
fine-history information. The frozen $s_t$ decoder tracks 18--32% of the
counterfactual shift variance, fine-target MSE rises 7--12%, and coarse-target
MSE changes by less than 2%.

These final checkpoints cannot answer the policy-use question because all
three greedily choose the positive action everywhere.

The transient seed-44 `checkpoint_000001` does use two actions, with held-out
reward 0.318 and action frequencies near `[0.298, 0.702, 0]`. Independent
flips change 8.25% of its paired greedy actions, while its mean closed-loop
reward change is only $8.75\times10^{-5}$. Its policy is therefore not purely
coarse-invariant, although reward is robust to the intervention.

## Protocol

Each checkpoint used 30,000 factual probe-training rows, 40,000 held-out
factual rows, eight independent flip replicas, and 40,000 closed-loop rows per
replica. Paired replay held the factual previous-action sequence fixed and
recomputed the exact fine filter for every randomized token history.
