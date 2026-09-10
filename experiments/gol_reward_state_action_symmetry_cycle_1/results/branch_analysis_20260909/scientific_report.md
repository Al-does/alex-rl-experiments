# GOL branch analysis, 2026-09-09

Scope: archived scientific results from experiment checkout `ce4c868b72066bf661becc3785bc1277387f8ce6`, requested branch `devin/gol-reward-state-run-20260909`, study `gol_reward_state_action_symmetry_cycle_1`. The original report extracted saved data without fitting models, downloading/loading checkpoints, or generating rollouts. Decimal tables below are rounded for reading; accompanying JSON/CSV retain all saved numeric precision.

**Portability note:** local links and workflow prose were normalized when these completed results were integrated into this repository. The [extractor](../../../gol_results_report.py) now locates the repository from `__file__` and the library source from the installed `harness` package. No numeric results or plots were regenerated. Saved source hashes, revision records, and machine paths in the JSON remain original-generation provenance, not claims about the current checkout or normalized generator. A future extraction records its own generator hash and installed-library state.

## 1. Provenance and completed budgets

| Arm | Run ID | Status | Environment steps | Iterations | Saved probes |
|---|---|---|---:|---:|---:|
| v2 half, failed attempt | 20260909T084818Z-6603a47f | failed | 98,352 | 3 | 0 |
| v2 half | 20260909T104731Z-c4aff136 | completed | 2,524,368 | 77 | 9 |
| v2 quarter | 20260909T111613Z-484ad76e | completed | 2,524,368 | 77 | 9 |
| v3 half | 20260909T110146Z-8c31130d | completed | 2,524,368 | 77 | 9 |
| v3 quarter | 20260909T113037Z-a43b263e | completed | 2,524,368 | 77 | 9 |

All runs use seed 42, no resume, and a 2,500,000-step threshold. Completed arms cross it at a whole iteration, 100.97472% of the threshold. The failed run reached 3.93408%; its Tune error is actor/worker death, with OOM only a suggested cause, not a proven diagnosis. Its remote-artifact upload completed, but training did not. It is an earlier attempt, not an independent seed or part of a pooled trajectory. All four completed arms have iteration 0, 1, 2, 4, 8, 16, 32, 64, 77 reports; all 77 training-curve rows and final step counts agree. Every arm reports zero completed episodes.

**Recorded revision:** every run manifest records clean training harness `32a15242b01450472c522f0bac250e1a2f7e2a6d`. The archived [provenance record](provenance_verification.json) reports an exact local revision match but `local_library_clean: false`, with empty committed/relevant-code diffs. Its free-text execution field contains an obsolete clean/nonidentical-checkout description; that historical record is retained unchanged, and the structured fields must not be read as a certificate for unrecorded dirty files. Neither that local checkout state nor any later library revision is substituted for the recorded training revision.

Completed manifests record experiment `1db27ffc096aac88e9281528675f0e86273a6d0b`: v2 half is clean; v3 half and both quarter runs record dirty experiment worktrees. The archived report recorded no source difference from that commit for `analysis.py`, `shared.py`, `process.py`, and `design.py`; this does not assert equality with today's integrated source. Dirty manifests do not record a full patch, so exact source reproducibility of those three executions cannot be certified solely from commit hashes. Their resolved recipes and report metadata agree on variant/speed/probe design. The failed attempt records dirty `08c72dd4d098aa26cd6dd348482d896e1a6c556c`; the intervening shared-code change adds the continuing-runner metrics-cache fix. This is consistent with an infrastructure failure, not evidence establishing its cause.

Sources: each listed run's `run_manifest.json` (git/runtime/status fields, lines 28–42, 77–100 in completed manifests), `tune_summary.json` (steps lines 28–35, iteration/status lines 105–110), and `resolved_recipe.json:195–204`. Failed `run_manifest.json:13–45,79–103` and `tune_summary.json:5–6,28–35,105–110`. `run_inventory.csv` retains full relative paths. `provenance_verification.json` records the Git checks.

## 2. What the experiment actually tests

### Controlled edge-emitting process

The four hidden states, in target order, are **M1, M2, E, S**; actions are **a1, a2, aE, aS** (indices 0–3). A step jointly samples the next token and destination. Reward is **1 on arrival in E**, including E self-loops, not a reward for the state before acting. There are two visible tokens, 0 and 1.

The frozen kernel can be read as follows, where vectors run over the four actions:

- From M1: remain M1 and emit 0 with probability `1-q1`; enter E and emit 1 with probability `q1`.
- From M2: remain M2 and emit 0 with probability `1-q2`; enter E and emit 1 with probability `q2`.
- Half speed: `h=0.2, l=0.05`; quarter speed: `h=0.1, l=0.025`. `q1=(h,l,l,l)`. In v2, `q2=q1`; in v3, `q2=(l,h,l,l)`.
- From E: E/token-0 with `e=(0.3,0.3,0.49,0.465)`; S/token-1 with `1-e`.
- From S: M1/token-0 with `(0.145,0.145,0.170,0.030)`; M2/token-1 with `(0.400,0.400,0.020,0.070)`; E/token-1 with `e`; S/token-0 with `(0.155,0.155,0.320,0.435)`.

Thus quarter speed halves **only M escape probabilities**, not the entire Markov kernel. It lengthens M dwell times and changes occupancy; it is not a global rescaling of training time. Both variants have the same uniform-random-action prior at each speed. Reset samples that prior and emits no token/reward; actor inputs start at six zeros. Subsequently the network sees only latest token one-hot(2) and previous executed action one-hot(4). Belief/hidden state are disabled in observations, reward is not an input, and both policy and value heads consume the same observation-derived embedding. The exact diagnostic posterior is updated using executed action and token, **not reward**. Training reward still supplies PPO's optimization signal; that is not an observation leak.

The task is continuing: `episode_length=None`, `reset_emission=False`, delay zero, no environmental resets at training batch boundaries. PPO truncates batches and bootstraps; environment, filter, and model context continue. The custom runner discards RLlib's never-ending episode-metrics cache. There is no meaningful saved episodic-return statistic to substitute for per-step reward.

Sources: `../../process.py:12–33`, `../../shared.py:31–118,146–157,161–208`; harness `envs/gol/model.py:9–44,61–74`, `envs/gol/tasks/reward_state.py:47–65`, `envs/hmm/env.py:562–578,705–738,763–802,804–896`; `learners/models/base.py:34–68` and `learners/components/heads.py:11–35`. Harness paths and line numbers in this report refer to the recorded library revision, not a machine-local checkout or a promise of unchanged current source.

### Which contrasts matter, and for which predictions

Let `b=(b_M1,b_M2,b_E,b_S)` at decision time. The probe targets are fine b, coarse `(b_M1+b_M2,b_E,b_S)`, M1−M2, and E−S. Coarse versus fine *targets* must not be confused with `fine_mse_ratio`, a within-observable-branch error normalization.

- **Variant 2:** M1/M2 have an exact controlled token-labelled quotient. Aggregating them commutes with every action/token kernel; saved quotient error is 0. M1−M2 is not needed for any future token/reward law or control of this quotient process, although it is needed to reconstruct fine hidden-state beliefs. Fine-state transition destinations still distinguish the labels; quotient equivalence does not say those labelled states are identical.
- **Variant 3:** the same aggregation is not Markov-sufficient. Saved aggregation error is 0.15 at half speed, 0.075 at quarter speed. For a1 versus a2, immediate expected reward differs by `(h-l)*(b_M1-b_M2)`; the difference in token-1 probability is the same. Exact all-action reward or token prediction therefore needs M1−M2 information. This does not imply that approximate reward maximization must linearly reconstruct the whole posterior; actions aE/aS themselves have identical M1/M2 escape rates.
- **Both variants:** `(0,0,1,-1)` is a null direction of the separate one-step reward and token marginals for *every* action. Both pure E and pure S have `P(reward=1)=e_a` and `P(token=0)=e_a`. E−S is therefore not required for those separate marginal predictions.
- **Crucial distinction:** E−S is **not** generally null for the joint reward-token outcome or future-state law. For pure E, `P(reward=1,token=0)=e_a`; for pure S that probability is 0, and reward instead pairs with token 1. Their token-conditioned next-state distributions differ even if reward is not observed. E−S can matter for multi-step prediction and planning. Full-information long-run policies choose aE in E versus aS in S despite identical immediate reward marginals. It would be wrong to call E−S task-irrelevant, or the entire coarse belief unnecessary.

The one-step reward feature matrix has raw rank 2 in v2 and 3 in v3. On the belief simplex these encode respectively one and two free predictive coordinates: M-total, then M-total plus M1−M2. The all-action token features have the same affine information here. E−S cannot be recovered universally from either, but correlations on a particular policy's visited belief distribution can still give nonzero regression R². The saved `combined_marginals` feature is not a joint-outcome feature. A log-NTP transform was not fitted/saved; its scores cannot be inferred from NTP's scores because affine accessibility can change under nonlinear feature transformations.

Sources: `../../analysis.py:52–84`; `../../design.py:13–51`; harness `envs/gol/model.py:28–44`; each `resolved_recipe.json` → `analytic_design`.

### Encoder and specificity estimands

The encoder has width 96, three layers, four heads, RoPE, and **64 previous frames plus current per attention layer**. The mask uses lag `<=64`, cache length 65, and raw lookback `3*64=192`: up to **193 frames including current**, not strict 64 or even strict 192 total frames. `max_seq_len=32` is a training chunk size, not that receptive field. Probe features are post-final-LayerNorm embeddings, the same embeddings used by the linear policy/value heads.

All probes are separately fitted affine maps, ridge 1e-6, with independent fit/test environment seed streams. Controls estimate:

| Saved control | Estimand, not an intervention claim |
|---|---|
| `previous_action_latest_token` | Train-fitted target centroid for each of 8 observable action/token branches; unseen branches use the fit global mean. A history-free decoding baseline. |
| `next_token` | Decoding from privileged exact probabilities of each next token for all four actions (8 features), not from a learned NTP head. |
| `next_reward` | Decoding from privileged exact immediate expected rewards for all four actions. |
| `combined_marginals` | Concatenation of the preceding separate marginals, not their joint distribution. |
| `categorical_plus_marginals` | Joint affine feature set: branch indicators plus exact marginals. Not a conditional-information or causal-use test. |
| `single_permuted_training_labels` | One row-permutation of fit targets; fit on original activations, score on true test labels. Descriptive null, not a calibrated permutation test. |
| `shuffled_history_broken_alignment` | Independently permute complete action/token pairs within each full environment history, retaining reset; replay network; fit and score against the ORIGINAL time-indexed beliefs. Deliberately breaks alignment. |

The history shuffle **does not recompute correct beliefs**, does not preserve current-branch alignment, may create off-process sequences, and is not a causal ablation of memory or a null mathematically required to have R²=0. No separate alignment-preserving input-shuffle result exists. The initialization control is the actual saved zero-step network, not a freshly sampled surrogate.

Sources: harness `learners/components/transformer.py:106–149,171–225`, `learners/models/transformer.py:19–25,64–84,128–136`; `../../analysis.py:157–178,180–198,230–263,286–337`; `experiments/factored_representations_reproduction_PPO_2026_08/shared.py:74–86` and GOL `shared.py:98–107` for true initialization.

## 3. Task outcomes

Success here means **stochastic held-out probe-rollout mean reward per continuing step**, not a binary success rate, episodic return, or certified optimality measure.

| Arm | Initial → final reward | Myopic lookup initial → final | Uniform random, analytic | Best constant, analytic | Full-information upper, NOT Bayes | Final random-to-full-info gap closed |
|---|---:|---:|---:|---:|---:|---:|
| v2 half | 0.2298 → 0.3518 | 0.3314 → 0.3314 | 0.195402 | 0.260369 | 0.424812 | 68.17% |
| v2 quarter | 0.1449 → 0.2904 | 0.2493 → 0.1442 | 0.119016 | 0.174383 | 0.356467 | 72.18% |
| v3 half | 0.2135 → 0.3192 | 0.2070 → 0.2790 | 0.195402 | 0.260369 | 0.424812 | 53.96% |
| v3 quarter | 0.1276 → 0.1898 | 0.1308 → 0.1308 | 0.119016 | 0.174383 | 0.356467 | 29.81% |

Final model-based expected-reward means are respectively 0.3518325026292836, 0.29084584578942385, 0.31932613266567117, and 0.18640002084213456. They are conditional expectations on sampled decision histories, not new independent evaluations. All final empirical rewards exceed their best-constant analytic values; v3 quarter does so only modestly. No policy-return uncertainty interval was saved, so these differences are descriptive, not significance claims.

The upper reference exhaustively ranks deterministic full-state stationary policies by E occupancy: `(a1,a1,aE,aS)` for v2, `(a1,a2,aE,aS)` for v3. It has privileged state access and is **not a Bayes-optimal token/action-observing controller** or a certified attainable POMDP target. Random/constant values are analytic long-run occupancies, whereas learned/lookup values are finite sampled rollouts.

The memoryless controller is a myopic reward argmax fitted to the current checkpoint's learned-policy fit-history centroids, then separately rolled out. It is not an optimized long-run memory-one policy or a fixed baseline across checkpoints. In v2 quarter its final table starts with aE at reset and chooses aE after either aE/token branch, trapping the lookup at constant aE (saved evaluation action fraction 1.0); initialization's table uses a1 after token 0 and aE after token 1. Thus the lookup collapse is real for this estimand, but does not show that memoryless policies became worse or that the full gain over this final lookup is a benefit of memory.

Trajectories are nonmonotonic. v2 half reaches 0.3582 at 1,049,088 steps then ends 0.3518. v3 quarter reaches 0.2040 at 524,544 steps then ends 0.1898, with coarse R² also declining from 0.666417 there to 0.552880. These are descriptive saved-checkpoint maxima, not test-selected replacement final scores. All nine checkpoints, including initialization, are in `policy_trajectories.csv` and the PNGs.

Sources: `endpoint_policy_metrics.csv:2–9`, `policy_trajectories.csv`; original checkpoint reports `policy` / `previous_action_latest_token_controller` at lines 414–447. v2-quarter initial/final `action_table_by_branch` and fractions at lines 425–446; `../../analysis.py:201–227`; `../../design.py:20–38`.

## 4. Belief accessibility: initialization to final

| Arm | Target | R² initial → final | MSE initial → final |
|---|---|---:|---:|
| v2 half | fine | 0.243555 → 0.370999 | 0.0795432 → 0.0447944 |
| v2 half | coarse | 0.455097 → 0.508501 | 0.0653285 → 0.0491439 |
| v2 half | M1−M2 | -0.050606 → 0.065930 | 0.3065891 → 0.0818120 |
| v2 half | E−S | 0.020144 → 0.026972 | 0.2053284 → 0.2399036 |
| v2 quarter | fine | 0.147274 → 0.333712 | 0.0982083 → 0.0744350 |
| v2 quarter | coarse | 0.493736 → 0.571790 | 0.0521642 → 0.0590626 |
| v2 quarter | M1−M2 | -0.151331 → -0.095073 | 0.5219348 → 0.2587547 |
| v2 quarter | E−S | 0.014356 → 0.049623 | 0.1652249 → 0.3014244 |
| v3 half | fine | 0.248145 → 0.400362 | 0.0796871 → 0.0520505 |
| v3 half | coarse | 0.460499 → 0.588053 | 0.0636472 → 0.0473032 |
| v3 half | M1−M2 | -0.023807 → 0.033089 | 0.3163969 → 0.1473647 |
| v3 half | E−S | 0.006736 → 0.034800 | 0.1995325 → 0.2394792 |
| v3 quarter | fine | 0.195887 → 0.234907 | 0.0919638 → 0.0939357 |
| v3 quarter | coarse | 0.510755 → 0.552880 | 0.0464014 → 0.0565239 |
| v3 quarter | M1−M2 | -0.030941 → -0.123216 | 0.5004160 → 0.4522025 |
| v3 quarter | E−S | 0.009504 → 0.010278 | 0.1490655 → 0.2195629 |

### Final specificity controls, R²

NTP and next-reward-only scores agree with combined-marginal scores to the displayed precision; their separate full-precision values are retained in CSV/JSON. **Log-NTP is unavailable for every arm/target/checkpoint.** The final-layer probe column is repeated here to make comparisons local. Initial controls for every target are also fully recorded in `endpoint_probe_metrics.csv`.

| Arm | Target | Probe | Branch | NTP | Reward+token marginals | Branch+marginals | Shuffled fit labels | Input-history shuffle, broken alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2 half | fine | .370999 | .299395 | .443619 | .443619 | .461571 | -.004384 | -.024390 |
| v2 half | coarse | .508501 | .404213 | .592456 | .592456 | .607135 | .000976 | -.019644 |
| v2 half | M1−M2 | .065930 | .071557 | .172411 | .172411 | .188637 | -.020427 | -.062741 |
| v2 half | E−S | .026972 | .032118 | .008344 | .008344 | .044062 | -.013368 | -.006619 |
| v2 quarter | fine | .333712 | .307183 | .379459 | .379459 | .400582 | -.014581 | -.029371 |
| v2 quarter | coarse | .571790 | .492613 | .618078 | .618078 | .637697 | -.013996 | -.022104 |
| v2 quarter | M1−M2 | -.095073 | -.008247 | -.009461 | -.009461 | .001722 | -.017216 | -.055709 |
| v2 quarter | E−S | .049623 | .044485 | .003447 | .003447 | .054639 | -.011748 | -.008429 |
| v3 half | fine | .400362 | .281351 | .656588 | .656588 | .672856 | .002087 | -.036156 |
| v3 half | coarse | .588053 | .415023 | .653869 | .653869 | .670266 | .003581 | -.034392 |
| v3 half | M1−M2 | .033089 | .008346 | 1.000000 | 1.000000 | 1.000000 | .003933 | -.056598 |
| v3 half | E−S | .034800 | .042071 | .038851 | .038851 | .084382 | -.009615 | -.008281 |
| v3 quarter | fine | .234907 | .207594 | .776409 | .776409 | .777012 | -.021855 | -.034144 |
| v3 quarter | coarse | .552880 | .401273 | .710464 | .710464 | .711245 | -.031365 | -.022414 |
| v3 quarter | M1−M2 | -.123216 | .001797 | 1.000000 | 1.000000 | 1.000000 | -.011237 | -.054412 |
| v3 quarter | E−S | .010278 | .003593 | .010045 | .010045 | .012715 | -.014635 | -.004013 |

**Best interpretation:** reward learning and partial belief accessibility occur, but these runs do not demonstrate accurate full belief-state geometry. Coarse R² is already approximately 0.46–0.51 at initialization; training improves it modestly and nonmonotonically. Final fine/coarse probes beat a train-fitted branch baseline but remain below exact one-step-marginal decoding baselines. E−S decoding remains weak (final R² 0.010–0.050) and offers little reliable incremental benefit over the branch baseline; it is below branch+marginals in every arm. This is no convincing evidence for recovering the marginal-null contrast.

M1−M2 is not robustly decoded, including v3 where exact all-action NTP/reward predictions encode it essentially perfectly. Final v3-half probe R² is 0.033; v3-quarter is -0.123, worse than the global-mean predictor. The oracle control demonstrates the target's algebraic recoverability, not that the network has learned that prediction. Good task reward need not entail accurate affine reconstruction of all counterfactual beliefs; poor affine R² does not establish information absence or lack of causal use.

The quarter-speed comparison is suggestive, not replicated: v2 closes a similar or larger fraction of the random-to-full-information gap than at half speed, while v3 quarter fares much worse and regresses late. Increased M dwell time plus the need to distinguish M1/M2 is a plausible difficulty, but the data do not isolate a causal mechanism, prove memory insufficiency, or show global optimality. There is one training seed per arm.

Sources: original checkpoint reports' `probe_fits` at lines 448–512; `controls` at lines 4–354. Exact source paths accompany every row of `endpoint_probe_metrics.csv` and `all_probe_metrics.csv`.

## 5. Sampling, uncertainty, and comparison hazards

1. **Counts and bootstrap unit:** each checkpoint has 10,000 fit and 10,000 test decision samples in distinct streams, each pooled across 16 continuing environment trajectories: 625 retained samples/environment after 64 warmup steps, decision times 64–688 inclusive. It is not 10,000 per environment or 10,000 independent episodes. The lookup gets its own 10,000 samples/16 environments. Probe MSE intervals use 200 resamples of whole held-out trajectories with a fixed trained network and fixed fitted probe. They do not cover training-seed variation, probe refitting, uncertainty of controls or reward differences. No R² or return CI was saved.
2. **Finite context/warmup:** because raw lookback is 192, the first 128 of 625 retained observations/environment precede a full 193-frame encoder history: 2,048/10,000 = 20.48% of evaluated samples. This is a common finite-history component, not an indexing error. A 64-step warmup is not evidence of stationarity, especially after changing M dwell times. Later targets condition on the complete token/action history since reset while the encoder has finite context.
3. **Changing occupancy:** each checkpoint's fit/test histories are collected under that checkpoint's learned stochastic policy. These are not fixed histories shared across training, despite reusing named seed streams. Target variance, branch frequencies and regression conditioning change, including with speed and action distribution. For v2-half M1−M2, target variance falls 0.29182129953744806 → 0.08758661262325397, helping explain much of its MSE reduction. For v2-quarter E−S, variance rises 0.1676314488401541 → 0.31716292054681405 while MSE rises 0.16522486978076867 → 0.3014243748636497 even though R² rises. In v3-quarter coarse belief, variance rises 0.09484288519358246 → 0.12641767722771077: higher final R² accompanies worse absolute MSE. Neither raw MSE nor R² trajectories isolate representation learning from occupancy changes.
4. **Metric semantics:** MSE averages over samples and target coordinates; multivariate R² is `1-MSE/target_variance`, not an unweighted average of per-coordinate R². Fine and coarse MSE use different coordinate weights. `branch_baseline_mse` uses test-set branch centroids as a variance decomposition, not a fitted deployable controller. `fine_evaluation_mse` equals ordinary MSE here because every group is retained (`min_group_size=1`); it is not an extra fine-state target. Actual train-fitted branch prediction is reported separately.
5. **Schema/inventory audit:** all 36 individual reports equal the copies embedded in their condition summaries; no duplicate pooling was used. Step schedules, sample counts, target variances across controls and all 1,152 saved R²/MSE identities passed extraction assertions. The code's time-major collector and `selected % 16` trajectory IDs agree; 10,000 is divisible by 16. No inspected schema/sampling bug invalidates these descriptive per-arm summaries. Invalid comparisons would result from pooling the failed restart, interpreting missing episodic returns as zero, claiming independent-timestep/bootstrap seed uncertainty, using the wrong shuffle estimand, or treating occupancy-changing curves as fixed-distribution representation comparisons.
6. **Limits of saved data:** there is no saved log-NTP, joint reward-token probe, corrected-belief input shuffle, neural belief scatter, causal-use intervention or matched-policy probe dataset. No raw activations were downloaded to produce those. Marginal feature controls are privileged oracle decoders, not trained neural next-token heads. Negative R² may reflect generalization/conditioning as well as weak accessibility. Dirty experiment manifests and a locally different harness commit are provenance caveats, not observed numerical schema failures.

Sources: `../../analysis.py:123–154,192–198,230–263,294–337`; harness `analysis/rollouts.py:245–309,342–354`, `analysis/probes/linear.py:53–99,110–219`, `analysis/probes/resampling.py:12–64`.

## 6. Archived deliverables and reproduction limits

- `saved_results.json`: compact, full-precision saved metrics/metadata/design references for each run and checkpoint, training curves, missing-control annotations, source SHA-256 hashes and audit checks.
- `run_inventory.csv`: five distinct attempts, completion/budget/provenance.
- `all_probe_metrics.csv`: 1,152 saved metric rows plus 144 explicit unavailable log-NTP rows; `endpoint_probe_metrics.csv`: initialization/final subset, all targets and controls, MSE CIs and sampling counts.
- `policy_trajectories.csv` and `endpoint_policy_metrics.csv`: learned/lookup mean and expected rewards, every constant-action analytic reference, random/full-information references and action fractions.
- `task_success_from_initialization.png`, `geometry_r2_from_initialization.png`, `geometry_mse_variance_from_initialization.png`: all four completed arms, starting at the actual zero-step checkpoint. The final figure includes target variance, branch MSE and fixed-probe environment-bootstrap MSE bands. These are decodability curves, not latent-geometry scatter plots.
- `provenance_verification.json`: original library revision, dirty-state, and source-diff records, including the historical prose inconsistency qualified above.

The included [saved-data extractor](../../../gol_results_report.py) can be imported as `experiments.gol_results_report` in the project environment. Its `extract()` function validates and returns the archived metrics without writing reports, loading checkpoints, or fitting probes. The CLI writes its owned report directory; intentional regeneration must preserve the original records and identify the current generator and installed library honestly. Git-based source comparison requires the recorded commit objects in the installed library's source checkout.

Additional checkpoint access, independent training seeds, fixed-occupancy/mature-context evaluations, and missing specificity controls were not performed for this report. Any later checkpoint execution requires a separately reviewed compatibility decision and must retain the dirty-manifest provenance limitations, rather than treating an arbitrary installed library as the recorded training revision.
