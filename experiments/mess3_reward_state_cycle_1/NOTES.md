# MESS3 reward-state cycle 1

This study trains PPO transformer agents to keep the controlled MESS3 process
in rewarding state 2. Continuous two-dimensional actions tilt the transition
law. The three conditions use occupancy reward alone, subtract transition
KL/4, or subtract `0.05 ||w||₂`.

Headline held-out results after 30 million agent steps (seed 42):

| Condition | Global R² | Fine R² | State-2 occupancy | Net reward/step |
|---|---:|---:|---:|---:|
| Occupancy only | 0.881 | 0.812 | 74.40% | 0.744 |
| Transition KL | 0.882 | 0.836 | 76.40% | 0.334 |
| Action norm | 0.900 | 0.858 | 76.39% | 0.438 |

The action-norm condition produced the strongest linearly decodable belief
geometry. Both penalized conditions reached about 76.4% rewarding-state
occupancy, versus 74.4% without a control cost. Their net rewards are not
directly comparable because they subtract different penalties. The archived
condition folders contain both the initial 10M runs and the fresh 30M runs.
