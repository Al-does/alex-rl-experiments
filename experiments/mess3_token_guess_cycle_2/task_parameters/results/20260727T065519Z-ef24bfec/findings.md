# Choosing the MESS3 operating point

## Candidates

`range` is what the belief-probe metric can move through; `ESS` is how many independent samples a 30,000-step probe rollout actually contains, given how slowly the chain mixes.

| point | α | p | τ | R² floor | R² range | acc range | ESS | probe ±95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cycle_1` | 0.85 | 0.900 | 7 | 0.9671 | 0.0329 | 0.0151 | 2324 | 0.0006 |
| `candidate_a` | 0.70 | 0.990 | 67 | 0.8967 | 0.1033 | 0.1337 | 198 | 0.0037 |
| `candidate_b` | 0.60 | 0.995 | 133 | 0.8782 | 0.1218 | 0.1362 | 104 | 0.0061 |
| `candidate_c` | 0.70 | 0.995 | 133 | 0.9061 | 0.0939 | 0.1447 | 104 | 0.0046 |

## Context length

Best belief R² available to a model that can see only the last k observations. No objective can beat this.

| point | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|
| `cycle_1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `candidate_a` | 0.9418 | 0.9969 | 1.0000 | 1.0000 |
| `candidate_b` | 0.7623 | 0.9431 | 0.9965 | 1.0000 |
| `candidate_c` | 0.9232 | 0.9951 | 1.0000 | 1.0000 |

`CausalTransformerEncoder` applies its causal band at every layer, so the receptive field is `n_layers * context_len`. The study's `context_len=64` over three layers reaches 192 observations, roughly six times what the belief needs. Compute in both the learner and the cached rollout path scales with that reach.

| point | smallest sufficient context_len | receptive field |
|---|---:|---:|
| `cycle_1` | 4 | 12 |
| `candidate_a` | 8 | 24 |
| `candidate_b` | 16 | 48 |
| `candidate_c` | 8 | 24 |

Measured on this repository's model: dropping `context_len` from 64 to 16 makes a learner step 3.7x faster and a cached rollout step 3.4x faster, for an end-to-end PPO speed-up of 1.6x once environment stepping, which does not change, is included.

## Supervised ceiling

Training the study architecture on next-token prediction to the Bayes cross-entropy, then probing it:

| point | 1.5k steps | 3k | 4.5k | 6k |
|---|---:|---:|---:|---:|
| `cycle_1` | 0.9567 | 0.9532 | 0.9434 | 0.9318 |
| `candidate_c` | 0.9487 | 0.9478 | 0.9439 | 0.9367 |

Belief-probe R² falls with continued training at both points while cross-entropy stays at the Bayes floor. This is supervised training, so the decline cycle 1 saw over 20M PPO steps is not caused by reinforcement learning. It is optimiser-driven drift in a representation the task no longer constrains, which makes learning rate, optimiser, and training duration larger influences on the headline metric than most of the differences between arms.
