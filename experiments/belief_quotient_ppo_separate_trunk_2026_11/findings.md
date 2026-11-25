# Separate-trunk PPO on the Belief-Quotient Ladder

**The Trivial Quotient result supports a role for the shared value-loss
gradient:** removing it from the actor left reward unchanged, but the actor's
final ρ error rose to **.4475**, versus **.0048** for the split critic,
**.0051** for standard shared PPO, and **.0041** for wide shared PPO. The actor
did not merely return to the REINFORCE level; it was worse in all three seeds.
**The Full-State Quotient prediction was directionally supported, not
replicated across seeds:** split actor δ error averaged **.0348**, versus
**.0435** in the critic, but seed 44 reversed the ordering at the endpoint.

## Final checkpoints

All numbers are per seed or arithmetic means. **Sampled** is reward per step
in the last stochastic training rollout; **Greedy** is reward per step from 84
complete held-out episodes at the final checkpoint. **Gap** is reference
minus Greedy, in reward per step: the Trivial exact optimum is **.941336** and
the Full-State feasible policy estimate is **.420969** (95% Monte Carlo
interval [.419385, .422552]; numerical Bayes upper bound **.421413**).
Negative gaps are possible under Monte Carlo noise. Critic rows repeat the
split actor policy's reward: there is no critic-driven policy. Probe columns
are held-out **1 − R²** (lower is better; errors above 1 mean negative R²).
Run IDs, reward SEs, and both
reference gaps are in [the CSV](results/final_by_seed.csv).

### Trivial Quotient

| Trunk | Seed | Sampled | Greedy | Gap | ρ | δ | V* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared | 42 | .941125 | .949358 | −.008023 | .004377 | .152689 | .957181 |
| Shared | 43 | .941359 | .931059 | .010276 | .005057 | .159645 | .962651 |
| Shared | 44 | .940577 | .947661 | −.006325 | .006010 | .200924 | .963281 |
| Shared | **Mean** | .941020 | .942693 | −.001357 | .005148 | .171086 | .961038 |
| Split actor | 42 | .941125 | .949358 | −.008023 | .445556 | .678259 | .987817 |
| Split actor | 43 | .941359 | .931059 | .010276 | .252068 | .487969 | .981608 |
| Split actor | 44 | .940575 | .947661 | −.006325 | .644863 | .847391 | .995563 |
| Split actor | **Mean** | .941020 | .942693 | −.001357 | .447496 | .671206 | .988329 |
| Split critic | 42 | .941125 | .949358 | −.008023 | .004215 | .170522 | .956610 |
| Split critic | 43 | .941359 | .931059 | .010276 | .005073 | .192868 | .961167 |
| Split critic | 44 | .940575 | .947661 | −.006325 | .004999 | .200394 | .960650 |
| Split critic | **Mean** | .941020 | .942693 | −.001357 | .004763 | .187928 | .959476 |
| Wide shared | 42 | .941125 | .949358 | −.008023 | .002851 | .098652 | .961298 |
| Wide shared | 43 | .941359 | .931059 | .010276 | .003133 | .094132 | .966422 |
| Wide shared | 44 | .940576 | .947661 | −.006325 | .006350 | .194118 | .963320 |
| Wide shared | **Mean** | .941020 | .942693 | −.001357 | .004111 | .128967 | .963680 |

### Full-State Quotient

| Trunk | Seed | Sampled | Greedy | Gap | ρ | δ | V* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wide shared | 42 | .413794 | .392334 | .028635 | .003599 | .033400 | .344476 |
| Wide shared | 43 | .415944 | .407506 | .013463 | .002843 | .039347 | .382786 |
| Wide shared | 44 | .414684 | .424398 | −.003429 | .003991 | .051222 | .343349 |
| Wide shared | **Mean** | .414807 | .408079 | .012890 | .003478 | .041323 | .356870 |
| Split actor | 42 | .413136 | .397786 | .023183 | .020107 | .031910 | .368864 |
| Split actor | 43 | .415938 | .400239 | .020730 | .028157 | .030725 | .398356 |
| Split actor | 44 | .414173 | .420689 | .000280 | .025682 | .041790 | .372409 |
| Split actor | **Mean** | .414415 | .406238 | .014731 | .024649 | .034808 | .379876 |
| Split critic | 42 | .413136 | .397786 | .023183 | .000991 | .044526 | .351780 |
| Split critic | 43 | .415938 | .400239 | .020730 | .001225 | .051831 | .383808 |
| Split critic | 44 | .414173 | .420689 | .000280 | .001089 | .034114 | .353797 |
| Split critic | **Mean** | .414415 | .406238 | .014731 | .001102 | .043490 | .363128 |
| Shared | 42 | .416334 | .389439 | .031530 | .003243 | .037706 | .350689 |
| Shared | 43 | .393905 | .383440 | .037529 | .005902 | .034079 | .355444 |
| Shared | 44 | .412207 | .424130 | −.003161 | .003717 | .037792 | .344389 |
| Shared | **Mean** | .407482 | .399003 | .021966 | .004287 | .036525 | .350174 |

## Learning trajectories and uncertainty

[**Learning curves (PDF)**](results/learning_curves.pdf) show all 102
checkpoints from initialization. Solid curves use the shared logarithmic
probe axis (left); dashed curves use complete-episode held-out greedy reward
(right). The dotted reference is the exact Trivial optimum or the numerical
Full-State upper bound; the Full-State dash-dot reference is the feasible
controller estimate. The step axis is logarithmic after the first batch
and linear down to zero. Bands are pointwise 2.5th–97.5th percentiles of all
27 three-seed resampled means: descriptive with n=3, not simultaneous
confidence bands. The seed-specific greedy episode SEs are .0063–.0074
(Trivial) and .0157–.0173 (Full-State); they are not training-seed uncertainty.

- **Trivial ρ:** The split actor first improved from .245 at initialization
  to .095 at 0.50B steps, then regressed to .166 at 1.00B, .406 at 1.79B,
  and .447 at 2.01B. Split critic ρ improved from .281 to .011 by 0.50B
  and .0048 at 2.01B. Both shared controls ended near the critic. All arms
  ended at the same greedy policy (always Tilt X) within each seed, and
  split greedy reward was already at its terminal value by 0.10B. Thus the
  late actor/critic divergence was not a failure to learn the task. Final
  paired actor-minus-critic differences (42/43/44) were **+.4413 / +.2470 /
  +.6399**, mean **+.4427**, seed SD **.1964**. The actor's late trajectory
  does not establish a plateau.
- **Full-State δ:** Split actor/critic means were .124/.125 at initialization,
  .0395/.0534 at 0.50B, .0360/.0456 at 1.00B, .0330/.0409 at 1.79B, and
  .0348/.0435 at 2.01B. Final paired actor-minus-critic differences were
  **−.0126 / −.0211 / +.0077**, mean **−.0087**, seed SD **.0148**. Both
  trunks improved; the actor led all three at 1.00B, but seed 44 favored the
  critic late. Split reward matched or exceeded standard shared on two of
  three greedy point estimates and was close on seed 44. Still, the
  split-policy greedy point estimates for seeds 42/43 were .023/.021 below
  the feasible reference, whereas 44 was within .0003. The reversal on the
  strongest split greedy seed and the seed/episode uncertainty preclude a
  general claim that the actor refines δ more. Treat the two lower-reward
  seed contrasts as descriptive, not confirmations under near-optimal play.

## Archived checks and interpretation

[Archived global probes](../algorithm_generality_probes_eps02_2026_11/results/per_seed.csv)
give the following REINFORCE errors (ρ / δ). The five-seed historical PPO
cohort is descriptive, not a paired baseline (only seeds 42 and 44 overlap).

| Reference | Trivial ρ / δ | Full-State ρ / δ |
|---|---:|---:|
| REINFORCE, 42 | .1112 / .2240 | .0235 / .0358 |
| REINFORCE, 43 | .0952 / .2135 | .0277 / .0341 |
| REINFORCE, 44 | .0968 / .3047 | .0280 / .0361 |
| REINFORCE, mean 42–44 | .1011 / .2474 | .0264 / .0353 |
| Archived shared PPO, five-seed mean | .0050 / .1679 | .0030 / .0327 |

Rerun shared PPO recovers the archived Trivial ρ result (.00515 vs .0050);
its Full-State δ mean is slightly worse (.0365 vs .0327). Its sampled
reward means were .941020/.407482 versus archived .941096/.413875,
with Full-State seed 43 the weakest new shared run. The archival study did
not supply this numerical V* probe. The Trivial split actor temporarily
passed through the REINFORCE ρ range, but ended at about 4.4× its three-seed
mean, while the isolated critic reproduced the ≈.005 prediction. The wide
control makes parameter count alone an unlikely explanation. This comparison
retains GAE while removing the actor's direct value gradient; it does not
hold realized advantages or training histories fixed. In Full-State the
actor's δ mean is close to the REINFORCE mean despite poorer δ in its critic
on two seeds. These are measurements of linear accessibility, not causal use.

## Controls, provenance, and durability

All 18 runs used seeds 42–44 and the archived PPO data/update budget:
101 × 19,922,944 environment steps (**2,012,217,344** per arm);
64 whole-episode minibatches × four epochs per rollout, Adam learning
rate 4.2 × 10⁻⁴, discount .99, GAE .95, ratio clip .2, and value coefficient .5.
Sampling and likelihood temperature were 1.5; entropy and exploration
ε were zero, with no annealing, KL penalty, value clipping, or gradient
clipping. Emission ε was .02. Standard shared/split/wide shared had
200,772/401,284/394,654 parameters. Wide was within 1.65% of split in count
but not FLOP-matched. All arms used the same held-out fit/test keys and
60k/80k rows after a 64-step warmup, with affine ridge 10⁻⁶. Across
arms, policy-conditioned histories can differ; within the split arm the
histories and targets are identical for both trunks. V* is a finite-horizon
discounted belief-grid approximation, not an exact optimality certificate.

The manifests record experiment `78f340890104899a7175ecde66b2bae7132f42ef`
and RL-Harness `a267508fdeb523e95b3ef3fc42119102540bbe38`.
The experiment worktree flag is clean on each box's first arm and dirty
thereafter, consistent with the preceding arms' untracked compact outputs;
every manifest records the same commits and the publication changed only
result files. All 18 compact manifests were published. We checked the
objects and byte sizes against each canonical B2 index, checked hashes of
the compact inputs and downloaded final checkpoints, and verified 102
checkpoint objects per run (1,926 indexed objects, 2,485,090,480 bytes total).

| Seed | Vast instance | Machine | Vast charge (USD) | Final provider status |
|---|---:|---:|---:|---|
| 42 | 62495879 | 36542 | 3.236 | Not found after self-teardown |
| 43 | 62496520 | 150418 | 2.771 | Not found after self-teardown |
| 44 | 62496653 | 153172 | 2.962 | Not found after self-teardown |
| **Total** | | | **8.969** | |

The three fresh on-demand RTX 5090s ran one sequential six-arm queue each.
Seed 43 took longer to provision (image load and bootstrap), but all queues
completed without relaunch. Charges are Vast instance charges, not B2
storage or retrieval costs. Rebuild the CSV and figure with
`uv run --no-sync python -m experiments.belief_quotient_ppo_separate_trunk_2026_11.report --output experiments/belief_quotient_ppo_separate_trunk_2026_11/results`.
