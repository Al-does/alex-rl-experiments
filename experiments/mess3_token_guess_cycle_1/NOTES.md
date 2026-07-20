# MESS3 token-guess cycle 1

This study trains transformer agents to predict the next token emitted by the
passive three-state MESS3 process. Each of the three possible tokens has a
matching discrete action, and the agent receives a reward of 1 for a correct
guess and 0 otherwise.

Four conditions were compared: reinforcement learning from token-guess reward
alone, the same reinforcement-learning objective with an auxiliary predictive
loss, reinforcement learning with policy entropy added to the reward stream,
and an implicit quantile network (IQN) distributional value critic. After
training, a held-out rank-2 affine probe mapped each transformer's hidden state
to the exact Bayesian belief simplex.

| Condition | Held-out R² | Greedy token accuracy |
|---|---:|---:|
| Reward only | 0.8552 | 0.6733 |
| Predictive loss | 0.9319 | 0.6734 |
| Max entropy | 0.8558 | 0.6733 |
| IQN value critic | 0.9760 | 0.6787 |

The predictive loss substantially improved the linear representation of the
Bayesian belief state without materially changing prediction accuracy. At the
tested entropy coefficient, max-entropy training produced essentially the same
belief representation and token accuracy as reward-only training. The IQN
critic produced the strongest belief representation and a small accuracy gain,
although this is a single-seed result.
