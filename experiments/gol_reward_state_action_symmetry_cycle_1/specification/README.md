# Four-action controlled HMM — frozen specification v1.0

This specification freezes the four-action candidate accepted in the conversation. It supplies four environments: variants 2 and 3, each at half and quarter speed. Unequal S-to-M entry probabilities do not obstruct the exact variant-2 quotient. Both variants use four hidden states and the same four action IDs; the three-state variant-2 quotient is also supplied as a reference implementation.

The environment is ready to implement. Neural training, probe performance, and a certified Bayes-optimal reward benchmark are separate experimental results. The earlier control-performance numbers were numerical screening estimates, not guarantees or release acceptance thresholds.

## Simulated Bayesian-controller reward occupancy

The following achieved reward-occupancy estimates were supplied by the experiment owner on 2026-09-09. They used an **exact Bayesian belief filter with an approximate planner**, using discount **gamma = 0.99**.

| Setting | Variant 2 | Variant 3 |
|---|---:|---:|
| Half speed | approximately 37.62% | approximately 36.02% |
| Quarter speed | approximately 31.51% | approximately 29.62% |

These are simulated achieved occupancies, **not certified Bayes-optimal values and not upper bounds**. A better planner could improve them. They are reference-controller estimates, not neural-agent results. Since reward is `1[destination_state=E]`, the occupancy fraction equals mean reward (for example, 37.62% corresponds to approximately 0.3762 reward per step).

Keep these separate from the verified **full-information long-run occupancy upper bounds**: 42.481203% at half speed and 35.646688% at quarter speed, identical across variants. Those bounds assume access to the true hidden state at every decision; the simulated Bayesian controller instead uses the intended action/token history.

The planner implementation, simulation budgets/seeds, and uncertainty intervals were not supplied with these estimates. They have not been independently rerun here. The reference verifier checks the frozen model and full-information bounds; it does not reproduce or certify this simulation table. This documentation addition does not change the frozen v1.0 kernels or observation contract.

## Identifiers and timing

| Object | IDs in array order |
|---|---|
| Fine hidden states | 0=M1, 1=M2, 2=E, 3=S |
| Coarse hidden states, variant 2 only | 0=M, 1=E, 2=S |
| Actions, both variants | 0=a1, 1=a2, 2=aE, 3=aS |
| Emitted tokens | 0, 1 |

At decision time t, the hidden state is s_t and the exact belief is b_t. The agent chooses a_t using its previous actions and observed tokens. The environment then jointly draws the next state s_(t+1) and token x_(t+1), and returns reward r_(t+1)=1 if s_(t+1)=E, otherwise 0. The posterior b_(t+1) describes the destination state after observing that token.

Thus the data alignment is:

`belief_before -> action -> (destination_state, new_token, reward) -> belief_after`.

The actor's history at decision t is `(a_0,x_1),...,(a_(t-1),x_t)`. The environment supplies no initial token x_0. The first action uses the empty history and the known reset prior. An architecture may have its usual start-of-sequence marker; that is not an additional HMM emission.

**Rewards train the learner but are excluded from the actor's input history and from the Bayesian filter.** Do not feed previous reward, hidden-state labels, or reward-derived state indicators to the actor through any observation or auxiliary recurrent input. If a critic shares the probed representation, use the same observation contract. Observed reward would reveal whether the current state is E and would change the intended inference problem.

There are no terminal states, action costs, additional reward bonuses, or state-dependent time limits. This is a continuing task. Rollout/batch boundaries do not reset the environment or ground-truth filter. At an intentional episode reset, reset the environment, actor context, and filter together.

## Transition probabilities

Only M departures change with speed:

| Speed | Preferred M exit h | Other M exit l | Preferred M self-loop | Other M self-loop |
|---|---:|---:|---:|---:|
| Half | .200 | .050 | .800 | .950 |
| Quarter | .100 | .025 | .900 | .975 |

Here h is the probability of leaving the current M state for E under its preferred action; l is that probability under every other action. Half and quarter refer to M departure rates relative to the former .40/.10 construction; they do not slow E or S.

The complete fine-state transition table for variant 3 is:

| Source → destination | a1 | a2 | aE | aS |
|---|---:|---:|---:|---:|
| M1 → M1 | 1-h | 1-l | 1-l | 1-l |
| M1 → E | h | l | l | l |
| M2 → M2 | 1-l | 1-h | 1-l | 1-l |
| M2 → E | l | h | l | l |
| E → E | .300 | .300 | .490 | .465 |
| E → S | .700 | .700 | .510 | .535 |
| S → M1 | .145 | .145 | .170 | .030 |
| S → M2 | .400 | .400 | .020 | .070 |
| S → E | .300 | .300 | .490 | .465 |
| S → S | .155 | .155 | .320 | .435 |

Every unlisted transition has probability zero. Both optional self-loops remain. The smallest self-loop probability anywhere in the model is .155.

**Variant 2 changes only M2:** its exit-to-E probabilities become `(h,l,l,l)`, with complementary self-loops back to M2. M1, E, S, all emission labels, and the action space remain identical to variant 3. Keep a2 available in variant 2 for matching the architectures/action spaces, even though a full-belief controller does not need it for optimal control.

For fully observed long-run control, the verified preferred actions are `(a1,a2,aE,aS)` in variant 3 and `(a1,a1,aE,aS)` in variant 2. These statements concern a controller with state information at all subsequent times; they are not a rule for every uncertain posterior. The corresponding full-information occupancy upper bounds are 42.481203% at half speed and 35.646688% at quarter speed, the same for both variants.

In E, aE increases retention from .465 under aS to .490. In S, aS reduces combined M entry from .190 under aE to .100. It accepts a slightly smaller immediate chance of reward to avoid long nonrewarding M visits. The previous numerical planner used discount gamma=.99, where gamma weights rewards one step further in the future. Gamma is a training/planning setting, not a transition probability. A next-step-only objective, gamma=0 with this reward alignment, does not create the intended S-action incentive: aE has at least as much expected next reward as aS for every belief.

## Emissions: deterministic labels on stochastic transitions

| Token | Fine-state edges carrying that token |
|---|---|
| 0 | Every self-loop; S → M1 |
| 1 | M1 → E; M2 → E; E → S; S → M2; S → E |

There is no independent emission-noise draw, and these labels do not depend on the action. A transition is still stochastic. An observation need not identify its source or destination uniquely.

**Sample the token and destination jointly. Do not independently sample a destination from the transition matrix and a token from a marginal emission matrix.** Such independent sampling creates a different process.

The authoritative kernel is

\[
T[a,x,i,j]=P(X_{t+1}=x,S_{t+1}=j\mid S_t=i,A_t=a),
\]

where a is the action ID, x the token ID, i the source-state ID, and j the destination-state ID. Its fine-model shape is `(4,2,4,4)`; the exact quotient's shape is `(4,2,3,3)`. For every action and source, sum over token and destination to obtain one.

For any action, write q1 and q2 for its M1/M2 exit probabilities, e for its E self-loop probability (also its S-to-E probability), u for S-to-M1, v for S-to-M2, and z for S's self-loop. These quantities are all specified in the transition table. In fine-state order `(M1,M2,E,S)`, the token-specific matrices are

\[
T^{a,0}=\begin{pmatrix}
1-q1&0&0&0\\
0&1-q2&0&0\\
0&0&e&0\\
u&0&0&z
\end{pmatrix},\qquad
T^{a,1}=\begin{pmatrix}
0&0&q1&0\\
0&0&q2&0\\
0&0&0&1-e\\
0&v&e&0
\end{pmatrix}.
\]

Here the superscript a is the chosen action and the other superscript is the emitted token. The usual hidden-state transition matrix is their sum. The marginal emission matrix for that action is

\[
O^a=\begin{pmatrix}
1-q1&q1\\
1-q2&q2\\
e&1-e\\
e&1-e
\end{pmatrix}.
\]

Rows correspond to the current source state and columns to next token 0 or 1. This is an edge-emitting controlled HMM; O alone is insufficient to specify its observation/transition coupling.

## Reward and non-injectivity

The reward rule is `reward_by_destination=(0,0,1,0)` in fine order and `(0,1,0)` in coarse order. The next-reward prediction matrix G, with source states as rows and actions `(a1,a2,aE,aS)` as columns, is

\[
G_3=\begin{pmatrix}
h&l&l&l\\
l&h&l&l\\
.300&.300&.490&.465\\
.300&.300&.490&.465
\end{pmatrix}
\]

for variant 3. The subscript 3 means the experiment variant, not the number of rows. In variant 2 replace the M2 row by the M1 row; after quotienting, retain one M row. The vector of expected next rewards for all actions is bG, where b is the current row belief.

E and S have identical rows in G and in every marginal emission matrix. Thus shifting probability between E and S while keeping M probabilities fixed preserves every action-conditioned next-token and next-reward marginal. The common null direction is `(0,0,1,-1)` in the fine model and `(0,1,-1)` in the quotient.

The fine variant-3 G has rank 3, and the quotient's G has rank 2. Even combining all next-token and next-reward marginals plus the belief normalization leaves this direction unidentified. This claim concerns these marginals. It does not claim that joint token/reward forecasts, long-horizon value predictions, or arbitrary predictive representations are non-injective.

## Why unequal M entry is compatible with the quotient

Let C be the aggregation matrix

\[
C=\begin{pmatrix}1&0&0\\1&0&0\\0&1&0\\0&0&1\end{pmatrix}.
\]

It maps a fine row belief to `(b_M1+b_M2,b_E,b_S)`. In variant 2, for each of the four actions and each token,

\[
T_4^{a,x}C=CT_3^{a,x},
\]

where T4 is the fine joint kernel and T3 the quotient kernel. This equality is checked numerically for every action/token pair. Reward is also constant within the merged M block. The quotient therefore preserves the joint law of observed tokens and rewards for every action sequence, and its Bayesian belief is sufficient for Bayes-optimal variant-2 control.

The S-to-M arrows in the quotient are parallel labelled edges: under aS, emit 0 and enter M with probability .030, or emit 1 and enter M with probability .070. The hidden transition probability S-to-M is .100, but the two token possibilities must be retained. More generally their probabilities are u and v from the transition table.

Incoming branch frequencies can be unequal. Exact merging requires matching *outgoing joint token-and-aggregate-destination behavior* from M1 and M2 for every action; that condition holds here. A neural agent is allowed to compress the distinction in variant 2, not guaranteed to do so. Variant 3 intentionally breaks that condition under a1 and a2.

## Reset prior and filter

The default reset prior is the stationary hidden-state distribution under uniformly random independent actions. It is the same for both fine variants at a given speed because their action-averaged transition matrices coincide. It is not claimed to be stationary under a learned policy.

| Speed | M1 | M2 | E | S |
|---|---:|---:|---:|---:|
| Half | .227891803466 | .413925928744 | .195402408172 | .162779859618 |
| Quarter | .277609138938 | .504228844193 | .119015895820 | .099146121049 |

The JSON and Python compute/store full floating-point precision; the table is rounded. Obtain the coarse prior by adding the first two entries. Initialize the Bayesian filter to this prior, and sample the actual hidden state privately from it. Reward is not issued for the reset itself.

For a current belief b, chosen action a, and new token x, compute

\[
b' = \frac{bT^{a,x}}{bT^{a,x}\mathbf 1},
\]

where b' is the destination-state posterior and the vector of ones sums the unnormalized probabilities. The denominator is the probability of observing x given the action and history. **Do not additionally condition this filter on the observed reward.**

If you choose a different reset distribution, use that same distribution in the simulator, the Bayesian labels, and any finite-window baseline. The code accepts an override; changing the reset prior does not change the kernel, but it can change results for short episodes.

When probing the activation that selects a_t, target b_t. When probing after consuming x_(t+1), target b_(t+1). Privileged state labels and rewards belong in the evaluation/training-target records, not in the actor's observation history.

## Memory claim and remaining experimental checks

Under the fixed action aE, the quotient token-1 matrix is

\[
U=\begin{pmatrix}0&l&0\\0&0&.51\\.02&.49&0\end{pmatrix}.
\]

Its determinant is `.0102*l`, which is nonzero at both speeds. The vectors of continuation probabilities `1`, `U1`, and `U²1` are linearly independent. Three reachable prefix beliefs, for words 0, 1, and 00 starting from the stationary distribution of this fixed-action process, also have full rank. Multiplying them by any power of U preserves independence; they cannot all yield identical continuation laws after a suffix consisting of any finite number of ones. Consequently no finite observation Markov order suffices for this fixed-action process. The fine variant-3 model has the same observation law under fixed aE, where both M exit probabilities equal l, so it shares this witness.

This is a fixed-action infinite-order certificate. It does not establish that a trained policy requires a practically important 100-step memory, or that every additional observation monotonically increases state certainty.

Keep the previous-action/latest-token controller and probe baselines. With separate aE/aS actions, the previous action can carry information about the agent's earlier state estimate. For matched-history interventions, control for that input as well as the latest token and immediate prediction marginals when possible. Also retain untrained-network and shuffled-history controls. These are experiment measurements, not unresolved kernel definitions.

## Files and usage

`hmm_values.json` contains all numerical values for the four fine environments and both variant-2 quotients. Each record includes T, P, O, G, state/action/token orders, reset prior, and reward vector. Array axes are explicitly recorded. `four_action_hmm.py` is the reference generator, simulator, and exact filter. `verify_spec.py` validates the kernel and regenerates the JSON files; `verification.json` records ranks, quotient checks, full-state oracle bounds, and infinite-order witnesses.

Run from this directory with Python and NumPy installed:

```bash
python verify_spec.py
python four_action_hmm.py
```

Minimal integration:

```python
from four_action_hmm import build_model, ControlledHMM

model = build_model(variant=3, speed="half")
env = ControlledHMM(model, seed=123)
env.reset()
belief = model.reset_prior.copy()

action = 0  # Replace with the actor's choice from its action/token history.
token, reward = env.step(action)
belief = model.update(belief, action, token)

# The actor receives (action, token). Reward goes to the RL objective.
# belief and env.hidden_state_for_evaluation are evaluation/probe targets.

coarse_model = build_model(variant=2, speed="half", coarse=True)
```

The implementation was checked against explicit enumeration of hidden paths, not just against another call to the same filter. Additional tests verify joint sampling/reward timing, analytic posteriors, all emission labels, and commutation of filtering with the exact quotient. No parameter search or neural training is required to run these checks.
