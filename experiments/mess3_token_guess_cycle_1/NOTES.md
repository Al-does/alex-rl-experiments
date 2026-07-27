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
already visible and echoing it scores 1.0000.

## Operating point

`process_design` scores a candidate process as a measuring instrument before
anything is trained, and `operating_point_validation` then trains matched PPO
and IQN arms at two points with the recipe, architecture, budget, seeds, and
probe all held fixed.

| | shipped | proposed |
|---|---:|---:|
| `stay`, `alpha` | 0.90, 0.85 | 0.96, 0.55 |
| Bayes accuracy | 0.6893 | 0.4561 |
| echo-to-Bayes range | 0.0167 | 0.0568 |
| probe floor, last token | 0.805 | 0.235 |
| probe floor, last two tokens | 0.930 | 0.405 |
| probe floor, 8-token window | 0.964 | 0.802 |
| probe floor, argmax cell | 0.882 | 0.824 |
| probe floor, untrained network | 0.883 | 0.832 |
| usable band | **0.036** | **0.168** |

The shipped point puts its binding floor at 0.964, so the whole interpretable
range is 0.036 wide and the IQN-minus-PPO gap of 0.129 is 3.6 times wider than
it. That is an off-scale reading, not a large effect. At the proposed point the
three floors land within 0.03 of each other, which is what `stay` and `alpha`
were chosen to do, and the same gap is 0.81 of the band.

Three seeds, 2.5M steps, γ=0.99:

| point | arm | belief R² | clears floor | share of accuracy range |
|---|---|---:|:--:|---:|
| shipped | ppo | 0.8461 ± 0.0065 | no | −0% ± 0% |
| shipped | iqn | 0.9756 ± 0.0032 | yes | 47% ± 2% |
| proposed | ppo | 0.8281 ± 0.0548 | no | 52% ± 22% |
| proposed | iqn | 0.9646 ± 0.0056 | yes | 91% ± 1% |

The largest change is that the task becomes learnable. At the shipped point
plain PPO sat at the echo policy on all three seeds to within 0.2% of the
range, so its low belief R² was measuring an agent that never did the task. At
the proposed point it captures 52% of the range and IQN reaches 91%. Gradient
signal-to-noise for the beyond-echo improvement, `headroom / sqrt(A(1-A))`,
rises from 0.034 to 0.114.

Two honest costs. PPO's seed spread grows sharply (R² ±0.0065 to ±0.0548, and
52% ± 22% on accuracy) because it now sometimes learns and sometimes does not,
which is the γ=0.99 variance problem rather than a property of the process, so
this point needs more seeds or γ=0. And the mixed-state distribution stops
being the sparse Cantor-like fractal that makes the α=0.85 picture recognisable
and becomes a dense triangle; that is why the token-window floors collapse, but
it does change the figure.

The conclusion about reward-only PPO survives the change: it fails to clear the
untrained-network floor at both points. That is worth more than it looks,
because it means the original finding was not an artifact of a badly chosen
process.

Within-branch residual R² at depth two is the most discriminating readout and
is what future cycles should lead with. At the proposed point it separates
untrained 0.72, PPO 0.71 ± 0.09, and IQN 0.94 ± 0.01.
