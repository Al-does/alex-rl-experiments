# MESS3 reward-state Kelly/IQN battery

## Question

For partially observed control of MESS3, how do a scalar PPO critic, an IQN
critic, predictive Kelly representation shaping, and their combination compare
at `gamma=0` and `gamma=0.99`?

## Controlled design

All eight conditions use seed 42, 30 million sampled environment steps, the
same three-layer transformer, and the continuous occupancy-control task. State
2 pays reward 1; states 0 and 1 pay 0. Each two-dimensional continuous action
tilts the HMM transition matrix, with components clipped to `[-5, 5]`.
Full runs use 2,048-item minibatches without CUDA-graph compilation so the
largest Kelly+IQN composition remains within a 24 GiB RTX 4090.

The four arms at each gamma are:

1. `PPO`: standard scalar value critic.
2. `IQN`: implicit-quantile value critic trained by quantile Huber loss.
3. `Kelly`: scalar critic plus an auxiliary three-token predictor and three
   sigmoid wager outputs. Cross-entropy trains the token logits. The wager
   corresponding to the predictor's selected token receives direct fair-odds
   Kelly log-growth loss.
4. `Kelly + IQN`: both predictive Kelly shaping and the IQN critic.

The Kelly objective is auxiliary: PPO still trains the continuous controller
on state-2 occupancy. Both auxiliary losses backpropagate through the shared
transformer.

## Evaluation

Checkpoints near 10, 20, and 30 million environment steps are evaluated on
held-out trajectories. The report records state-2 occupancy as a reward
percentage, greedy reward percentage, and global and token-branch-conditional
affine-probe R² against the exact Bayesian belief. Because actions alter the
transition matrix, the belief target is the action-aware predictive transducer
belief; every probe verifies it against environment diagnostics before
reporting.
