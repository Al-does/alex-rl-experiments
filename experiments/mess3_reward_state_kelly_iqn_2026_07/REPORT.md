# MESS3 reward-state Kelly/IQN battery: findings

## Executive summary

This single-seed battery compares PPO, IQN, predictive Kelly, and Kelly+IQN
at `gamma=0` and `gamma=0.99` on continuous MESS3 occupancy control.

The main result is a sharp separation between control performance and belief
decodability:

- `gamma=0.99` produced strong control in every arm: final state-2 occupancy
  ranged from 74.41% to 78.78%.
- Three of four `gamma=0` arms performed below the exact 10% zero-action
  occupancy baseline. IQN was the exception at 31.51%, close to the
  numerically optimized 32.46% constant-action ceiling.
- Predictive Kelly produced highly linearly decodable Bayesian belief states
  even when control failed. At `gamma=0`, Kelly reached global/fine belief R²
  of 0.989/0.943 with only 6.44% occupancy; Kelly+IQN reached 0.980/0.911 with
  only 2.81% occupancy.

The data therefore do **not** support “a good belief representation is
sufficient for good control.” They support the narrower statement that the
Kelly auxiliary objective can make the action-aware Bayesian belief highly
linearly decodable independently of whether PPO receives useful temporal
credit for controlling future occupancy.

## Experimental design

All conditions used:

- seed 42 and 30 million sampled environment steps;
- the same 3-layer, 4-head transformer (`d_model=96`, context length 64);
- 32,768-step train batches and 2,048-item minibatches;
- a delayed MESS3 observation with two-dimensional continuous actions clipped
  to `[-5, 5]`;
- reward 1 when the pre-transition hidden state is state 2, otherwise 0;
- held-out probes near 10M, 20M, and 30M environment steps.

An action changes the transition matrix used to sample the *next* hidden
state, while the current reward is determined by the pre-transition state.
This timing matters: at `gamma=0`, a current action receives no direct
discounted credit for improving occupancy on the next step.

The predictive Kelly arms add next-visible-token cross-entropy and direct
fair-odds Kelly wager losses to the shared transformer. These are auxiliary
objectives; PPO still trains the continuous policy on occupancy reward.

Belief R² measures a held-out affine probe against the exact action-aware
predictive transducer belief:

- **global R²** fits one affine decoder over all samples;
- **fine R²** is the more demanding token-branch-conditional score.

These metrics establish linear decodability, not causal use of the decoded
belief by the policy.

## Final results

| arm | gamma | occupancy % | greedy occupancy % | global R² | fine R² |
|---|---:|---:|---:|---:|---:|
| PPO | 0 | 2.31 | 2.50 | 0.9148 | 0.2955 |
| PPO | 0.99 | 74.76 | 74.66 | 0.8526 | 0.7581 |
| IQN | 0 | 31.51 | 31.65 | 0.7923 | -0.6667 |
| IQN | 0.99 | 74.87 | 74.66 | 0.8523 | 0.7576 |
| Kelly | 0 | 6.44 | 6.19 | 0.9894 | 0.9431 |
| Kelly | 0.99 | 74.41 | 76.71 | 0.9228 | 0.8769 |
| Kelly+IQN | 0 | 2.81 | 1.45 | 0.9798 | 0.9105 |
| Kelly+IQN | 0.99 | **78.78** | **80.78** | 0.8653 | 0.7894 |

Across arms, mean occupancy was 10.77% at `gamma=0` and 75.70% at
`gamma=0.99`. The `gamma=0` mean is pulled upward by IQN; its median was only
4.63%.

Reference points for interpreting occupancy:

- exact zero-action stationary occupancy: **10.00%**;
- numerically optimized bounded constant-action occupancy: **32.46%**;
- best learned final policy in this battery: **78.78%**.

Thus PPO, Kelly, and Kelly+IQN at `gamma=0` are not merely weak compared with
their long-horizon counterparts; they are below passive dynamics. IQN
`gamma=0` is qualitatively different: it is 0.94 percentage points below the
constant-action ceiling, although still far below the history-dependent
policies learned at `gamma=0.99`.

## Finding 1: discounting dominates occupancy control

Changing `gamma` from 0 to 0.99 increased final occupancy by:

| arm | occupancy increase (percentage points) |
|---|---:|
| PPO | +72.45 |
| IQN | +43.35 |
| Kelly | +67.97 |
| Kelly+IQN | +75.97 |

All `gamma=0.99` arms had already reached 74% or better occupancy by the first
10M-step probe and remained stable through 30M steps. The simplest
interpretation is a temporal-credit effect: actions affect future state
occupancy, but `gamma=0` excludes that future reward from the action's return.

This interpretation should not be overstated as a general claim about
`gamma=0`. It is specific to this task's reward/action timing. The IQN result
also shows that optimization and auxiliary critic structure can still produce
a useful near-constant control policy without a strong fine-grained belief
geometry.

## Finding 2: belief decodability and control are dissociable

The two predictive Kelly `gamma=0` arms provide the clearest evidence:

| arm | occupancy at 10M → 20M → 30M | global R² at 30M | fine R² at 30M |
|---|---|---:|---:|
| Kelly | 16.44% → 6.47% → 6.44% | 0.9894 | 0.9431 |
| Kelly+IQN | 3.26% → 2.24% → 2.81% | 0.9798 | 0.9105 |

Their belief scores were already high at 10M steps and remained high while
occupancy stayed poor or deteriorated. This is not a transient mismatch caused
by one endpoint. It persists across all three checkpoints.

The mechanism is consistent with the objectives. Next-token prediction and
Kelly wagering require a predictive latent state and continue to train it when
`gamma=0`; they do not themselves provide the policy with the discounted
occupancy credit needed to select transitions that pay later.

Accordingly, “the network knows the belief” should be read precisely as “an
affine probe can recover the belief.” It does not imply that the policy head
uses the relevant directions, that the representation is causally necessary,
or that the control objective assigns useful gradients to belief-dependent
actions.

## Finding 3: global R² alone can be misleading

PPO `gamma=0` achieved global R² 0.915 but fine R² only 0.295. IQN `gamma=0`
had global R² 0.792 but negative fine R² (-0.667), meaning its conditional
affine probe was worse than the corresponding mean baseline.

These cases show why both metrics are needed. A strong global score can
coexist with poor local geometry within observation branches. The Kelly
`gamma=0` arms are more compelling representation results because both global
and fine scores are high.

## Finding 4: Kelly and IQN have different effects at `gamma=0.99`

At `gamma=0.99`:

- IQN alone was effectively tied with PPO in occupancy and belief R².
- Kelly improved global/fine R² from PPO's 0.853/0.758 to 0.923/0.877, but did
  not improve sampled occupancy.
- Kelly+IQN achieved the highest occupancy, 78.78% sampled and 80.78% greedy,
  about four percentage points above PPO. Its belief R² was better than PPO's
  but lower than Kelly alone's.

This is a useful warning against treating probe R² as the sole model-selection
criterion. Kelly alone had the best `gamma=0.99` belief geometry, whereas the
combined arm had the best control.

## Limitations

1. Every condition used only seed 42. Differences may include seed variance;
   the table supports hypotheses, not population-level estimates.
2. Probe R² is correlational. Causal interventions on latent belief directions
   are needed to establish whether the policy uses them.
3. The constant-action ceiling was obtained by numerical optimization over the
   bounded two-dimensional action, while the 10% passive baseline is exact.
4. Most successful `gamma=0.99` policies place nearly all evaluated actions on
   a control boundary. Results therefore describe this unconstrained
   occupancy-only regime and may change under action or transition-KL costs.
5. `gamma` changes the PPO return target but not the Kelly auxiliary targets.
   The design intentionally exposes this separation, but it also means the
   representation and control objectives receive different temporal signals.

## Recommended follow-ups

1. Repeat the eight arms over multiple seeds and report confidence intervals.
2. Add explicit zero-action and optimized constant-action controls to the
   generated comparison.
3. Sweep intermediate `gamma` values to locate where occupancy credit emerges.
4. Test whether ablating or intervening on probe-identified belief directions
   changes actions and occupancy.
5. Measure policy-head sensitivity to the latent belief directions instead of
   relying only on decoder accuracy.
6. Repeat with transition-KL or action-norm costs to test whether the
   Kelly+IQN occupancy advantage survives less saturated control.

## Artifacts

Compact per-arm outputs are stored under each condition's
`results/<run_id>/`. The generated aggregate comparison, checkpoint curves,
and source manifest are under `study_synthesis/results/`.
