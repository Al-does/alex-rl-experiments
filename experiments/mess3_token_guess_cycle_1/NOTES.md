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

Recommended: **`stay = 0.95`, `alpha = 0.75`** (`CANTOR` in
`operating_points.py`).

`process_design` scores a candidate process as a measuring instrument before
anything is trained. `operating_point_validation` and
`fractal_preserving_validation` then train matched PPO and IQN arms at four
points with recipe, architecture, budget, seeds, and probe all held fixed.

### Visible gaps and the global-R² floor are the same number

The mixed-state set is the attractor of an iterated function system with one
contractive map per token. Disjoint first-level images need a contraction ratio
below `1/sqrt(3)`, which puts at least two thirds of belief variance *between*
the branches — and between-branch variance is exactly what a probe on the last
token reads. So a sparse Cantor picture forces a last-token floor above 0.667:

| `alpha` (at `stay=0.95`) | 0.85 | 0.80 | 0.75 | 0.70 | 0.55 |
|---|---:|---:|---:|---:|---:|
| box dimension | 0.85 | 1.01 | 1.22 | 1.38 | 1.82 |
| last-token floor | 0.730 | 0.633 | 0.543 | 0.461 | 0.264 |

`stay` carries no such cost. At fixed `alpha` it moves the accuracy axis while
leaving the geometry alone, which is most of the free lunch here.

### Four points, three seeds, 2.5M steps, γ=0.99

| point | `stay`/`alpha` | floor | band | arm | belief R² | within-branch R² | share of accuracy range |
|---|---|---:|---:|---|---:|---:|---:|
| shipped | 0.90 / 0.85 | 0.964 | 0.036 | untrained | 0.883 | −0.717 | — |
| | | | | ppo | 0.8461 ± 0.0065 | −1.252 ± 0.091 | −0% ± 0% |
| | | | | iqn | 0.9756 ± 0.0032 | +0.642 ± 0.048 | 47% ± 2% |
| cantor_sharp | 0.95 / 0.85 | 0.944 | 0.056 | untrained | 0.866 | −0.089 | — |
| | | | | ppo | 0.9138 ± 0.0356 | +0.300 ± 0.290 | 41% ± 31% |
| | | | | iqn | 0.9793 ± 0.0015 | +0.832 ± 0.011 | 90% ± 1% |
| **cantor** | **0.95 / 0.75** | 0.915 | 0.085 | untrained | 0.823 | +0.300 | — |
| | | | | ppo | 0.8990 ± 0.0188 | +0.601 ± 0.076 | 63% ± 13% |
| | | | | iqn | 0.9737 ± 0.0012 | +0.896 ± 0.005 | 92% ± 1% |
| proposed | 0.96 / 0.55 | 0.832 | 0.168 | untrained | 0.833 | +0.715 | — |
| | | | | ppo | 0.8281 ± 0.0548 | +0.708 ± 0.093 | 52% ± 22% |
| | | | | iqn | 0.9646 ± 0.0056 | +0.940 ± 0.010 | 91% ± 1% |

`cantor` is recommended because it is the only point where the three
conditions order cleanly with non-overlapping error bars on the metric that has
a floor of zero by construction: untrained 0.300 < PPO 0.601 ± 0.076 < IQN
0.896 ± 0.005. It also has the lowest IQN seed spread of any point, keeps
visible level-one gaps, and takes 2.4× the global-R² band and 4.1× the accuracy
resolution of the shipped point.

The largest single change at any of these points is that the task becomes
learnable. At the shipped point plain PPO sat at the echo policy on all three
seeds to within 0.2% of the range, so its low belief R² was measuring an agent
that never attempted the task. Gradient signal-to-noise for the beyond-echo
improvement, `headroom / sqrt(A(1-A))`, is 0.034 shipped and 0.121 at `cantor`.

### Report within-branch residual R², not global R²

Nearly all of a Cantor-structured belief's variance is which cluster you are
in, which the last token already gives away. `conditional_residual_r2` at depth
two discards exactly that term, so any depth-two token model scores zero there
by construction at *every* operating point. That is why the sparser points
still discriminate well on it while their global-R² bands look hopeless.

`proposed` remains the widest global-R² band, but its mixed-state set is a
dense triangle rather than a fractal, PPO's seed spread there is the worst of
the four, and on the within-branch metric it is the most compressed.

Reward-only PPO fails to clear the global-R² floor at all four points, so that
finding is not an artifact of a badly chosen process.
