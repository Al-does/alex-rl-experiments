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
| IQN value critic, 20M steps | 0.9686 | 0.6847 |
| IQN value critic, γ=1.0 at 3M steps | 0.9829 | 0.6796 |
| IQN value critic, γ=0 at 3M steps | 0.9837 | 0.6817 |
| IQN differential average reward at 3M steps | 0.9657 | 0.6808 |

The predictive loss substantially improved the linear representation of the
Bayesian belief state without materially changing prediction accuracy. At the
tested entropy coefficient, max-entropy training produced essentially the same
belief representation and token accuracy as reward-only training. The IQN
critic produced the strongest belief representation and a small accuracy gain,
although this is a single-seed result.

The 20M-step IQN follow-up showed that reward had already plateaued by the first
retained checkpoint at 0.83M steps: held-out greedy accuracy stayed near 68–69%
through 20M steps. Belief-probe R² was 0.9902 at that first checkpoint and
fluctuated around 0.97 thereafter, ending at 0.9686. Longer training therefore
did not produce a continuing reward rise in this seed.

The 3M-step IQN run with γ=1.0 reached 67.96% greedy accuracy and R² 0.9829.
Its quantile spread grew to 30.75 and value explained variance was only 0.021,
consistent with the fact that undiscounted value is unbounded for this
positive-reward continuing process.

The controlled return-objective comparison favored the myopic γ=0 arm on both
greedy accuracy and belief R², although all policy differences were small. The
differential average-reward arm learned a reward-rate estimate of 0.674 and
restored healthy critic diagnostics: quantile spread 1.24 and value explained
variance 0.78, compared with spread 30.75 and explained variance 0.02 for the
naive γ=1 arm.
