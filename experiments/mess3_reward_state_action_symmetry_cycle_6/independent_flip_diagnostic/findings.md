# Cycle-6 Variant-2 independent-flip findings

## Result

The learned representation is **not a purely coarse-filter-invariant state**.
It is better described as a mixed representation in which the coarse target
$c_t$ is more linearly accessible, while some exact-history information about
$s_t$ remains.

Across final checkpoints for seeds 42--44:

- independent token-0/1 flips changed the exact $s_t$ target by about 0.0342
  RMS while leaving $c_t$ exactly unchanged;
- the frozen $s_t$ decoder tracked 18--32% of the counterfactual shift
  variance ($R^2$, mean 0.23);
- fine-target MSE increased 7--12% under the counterfactual histories;
- coarse-target MSE changed by less than 2% and slightly improved in all three
  seeds.

Thus a strictly coarse representation is rejected by the measurable decoded
$s_t$ response, but the relative stability and lower MSE of the coarse decoder
still support $c_t$ as the cleaner dominant coordinate.

## Policy limitation

All three final policies greedily selected the positive action on 100% of
factual and randomized steps. Greedy-action agreement and closed-loop reward
were therefore exactly unchanged. The intervention moved action probabilities
by less than 0.001 mean total variation, but this does not establish whether a
non-degenerate policy would causally use the retained fine information.

## Protocol

Each reported checkpoint used 30,000 factual probe-training rows, 40,000
held-out factual rows, eight independent flip replicas, and 40,000 closed-loop
rows per replica. Paired replay held the factual previous-action sequence fixed
and recomputed the exact fine filter for every randomized token history.
