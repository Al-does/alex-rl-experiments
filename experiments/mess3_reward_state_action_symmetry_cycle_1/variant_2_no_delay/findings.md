# Variant 2 no-delay intervention

Run `20260728T183456Z-17b402c4` used seed 42 for 1,022,738 environment
steps on one RTX 4090.

## Diagnosis

The action IDs are `0=noop`, `1=positive`, and `2=negative`. The original
`delay=1` policy therefore collapsed to always positive, not always negative.

For variant 2, positive beats noop for the next-state occupancy objective until
the current reward-state belief exceeds 0.7541. With a one-step observation
delay, every current belief has already passed through a controlled transition,
and its reward-state probability is at most 0.3324. The intended noop decision
region is therefore unreachable. Entropy cannot solve this information
constraint.

Setting `delay=0` exposes the current token. Under the learned state
distribution, token evidence can put the posterior inside the noop region.

## Result

At the final checkpoint, the greedy action fractions were:

- hidden state 0: 4.7% noop, 95.3% positive, 0% negative
- hidden state 1: 4.8% noop, 95.2% positive, 0% negative
- hidden state 2: 67.8% noop, 32.2% positive, 0% negative

The policy chose noop for 76.7% of observations whose exact belief was
state-2-dominant and positive for every state-0- or state-1-dominant
observation. Greedy agreement with the exact-belief one-step oracle reached
96.3%, and greedy reward-state occupancy reached 25.97%.

The sampled policy showed the same ordering but remained softer: noop was
chosen about 4.5% in states 0/1 and 47.8% in state 2, with 87.6% one-step-oracle
agreement.

The conditioned policy emerged by 527,868 steps and strengthened by 1,022,738
steps, so the original 5–10M-step budget is unnecessary for this intervention.
