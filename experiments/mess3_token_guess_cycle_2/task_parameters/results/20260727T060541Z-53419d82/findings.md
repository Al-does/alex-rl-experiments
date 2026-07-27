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

The study's context length of 64 is sufficient at every candidate, so it does not need sweeping. The belief converges much faster than the hidden state does, because each observation is informative enough to wash out the prior well before the state decorrelates.

## Supervised ceiling

Training the study architecture on next-token prediction to the Bayes cross-entropy, then probing it:

| point | 1.5k steps | 3k | 4.5k | 6k |
|---|---:|---:|---:|---:|
| `cycle_1` | 0.9567 | 0.9532 | 0.9434 | 0.9318 |
| `candidate_c` | 0.9487 | 0.9478 | 0.9439 | 0.9367 |

Belief-probe R² falls with continued training at both points while cross-entropy stays at the Bayes floor. This is supervised training, so the decline cycle 1 saw over 20M PPO steps is not caused by reinforcement learning. It is optimiser-driven drift in a representation the task no longer constrains, which makes learning rate, optimiser, and training duration larger influences on the headline metric than most of the differences between arms.
