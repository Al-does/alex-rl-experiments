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

## Reference bands (supersedes the interpretation above)

Neither headline number starts near zero, so the raw table cannot be read
directly. The `untrained_reference` arm and `baselines.py` measure the floors
on the identical rollout, environment, architecture, and probe:

| reference | belief R² | greedy accuracy |
|---|---:|---:|
| randomly initialised transformer, never trained | 0.8823 ± 0.0055 | 0.32–0.50 |
| rank-2 affine probe on the last token alone | 0.8046 | — |
| rank-2 affine probe on the last two tokens | 0.9305 | — |
| echo the last visible token | — | 0.6735 |
| exact Bayesian argmax | — | 0.6893 |

The whole accuracy axis is therefore 1.6 percentage points wide, and belief R²
below roughly 0.88 is what a random causal filter over the token window already
produces. Re-reading the arms against those bands:

- Reward-only and max-entropy (0.8552, 0.8558) sit *below* the untrained
  network, and their accuracy sits at the echo baseline. Neither arm has been
  shown to learn anything beyond guessing the last token it saw.
- Predictive loss (0.9319) matches the two-token probe almost exactly and also
  stays at the echo accuracy: it improves the readout, not the policy.
- The IQN arms are the only ones that clear both floors, reaching 33–71% of the
  accuracy range and exceeding the two-token probe on R².

Three design facts explain most of this. `NextTokenGuessTask.resolve_action`
ignores the action, so the process is passive and only the immediate reward
term of the GAE sum carries any action-dependent signal; at γ=0.99, λ=0.95 the
remaining terms inflate advantage variance by 7.7×, which costs roughly 8.7× in
sample efficiency. For the same reason the entropy bonus in `max_entropy` is a
function of the state alone and cancels out of the advantage, making that arm a
provable no-op at any coefficient while `entropy_coeff` stays at zero. And a
binary correctness reward only constrains the argmax cell of the predictive
distribution, so a reward-only optimum never requires a faithful belief.

`delay` is not a usable difficulty knob here: at `delay=0` the graded token is
already visible and echoing it scores 1.0000. Widening the two axes instead
needs a noisier emission channel and a stickier chain; at `p_stay=0.93`,
`alpha=0.65` the accuracy range grows from 0.016 to 0.053 and the two-token
probe floor drops from 0.93 to 0.65.
