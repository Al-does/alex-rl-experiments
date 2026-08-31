# All-checkpoint scalar-probe campaign

This campaign reads the existing cycle-4 and cycle-5 training artifacts from
B2. It does not retrain the policies. Each command recovers initialization
plus every saved RLlib checkpoint for seeds 42--46, probes them sequentially,
writes a PNG/PDF trajectory graph, and pushes compact results after each seed.

Run one command on each of six Vast boxes (one box per graph):

```bash
python -m experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue \
  --target symmetric_b2 --all-seeds

python -m experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue \
  --target antisymmetric_b0_minus_b1 --all-seeds

python -m experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue \
  --target coarse_b2 --all-seeds

python -m experiments.mess3_reward_state_action_symmetry_cycle_5.belief_symmetry_probes.seed_queue \
  --target symmetric_b2 --all-seeds

python -m experiments.mess3_reward_state_action_symmetry_cycle_5.belief_symmetry_probes.seed_queue \
  --target antisymmetric_b0_minus_b1 --all-seeds

python -m experiments.mess3_reward_state_action_symmetry_cycle_5.belief_symmetry_probes.seed_queue \
  --target coarse_b2 --all-seeds
```

The symmetric and antisymmetric commands cover variants 1, 2, and 3. The
coarse command intentionally covers only variant 2 and uses a separate
two-state HMM over `A={state 0,state 1}` and `B={state 2}`.

Outputs are written under:

```text
belief_symmetry_probes/variant_<n>/results/
belief_symmetry_probes/results/trajectory_campaign/<target>/
```

The graph shows held-out normalized affine-probe MSE against training
iteration. Iteration 0 is the separately saved untrained initialization;
RLlib's `checkpoint_000000` is training iteration 1.

The box must receive `B2_*` credentials and the launch branch must contain
this code. `--push-each` defaults to true. If a job fails, do not destroy the
box until its logs and compact partial `condition_summary.json` have been
inspected.
