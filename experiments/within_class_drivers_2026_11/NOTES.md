# Working notes (session da0bbcc8) — within-class belief drivers

Branch: devin/1794716878-within-class-drivers (based on PR #229 branch origin/devin/1794701080-v2-exploration-sweep).
Reviewer: deep-beliefs/response.tex, Reviewer 1 W2 + Q1 (within-class R2), W3(i) (perturb S1/S2 symmetry).
Env: persistent_nuisance_v2 (experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/environment.py),
recipe PR #219: transformer, T=1.5, 2B steps, eps 0, entropy 0, seed 42; ~21 min/run on a 4090.

Hypotheses
- H1 rise-then-decay (parent): early peaks are pre-competence; modest gated decline in Reward-State under entropy.
- H2 exploration pressure (PR #229): not supported.
- H3 (this session) decision-relevant information: training adds within-class resolution only along belief
  coordinates that change optimal Q-gaps / future optimal actions. delta=b1-b2, rho=b3.
  Existing seed-42 data (final vs matched init, full h): Trivial delta +.014/+.026, rho +.07;
  Reward-State delta -.012/+.020, rho +.027; Full-State delta +.110/+.087, rho +.028.
- Dose-response family: X=(+1, lam, -1), Y=(-1, -lam, -1); lam=1 Reward-State, lam=-1 Full-State.
  Emission asymmetry kappa: S1 C-prob .85-kappa, S2 .85+kappa.

## Facts established (do not re-derive)
- Training is CRITIC-FREE REINFORCE-in-PPO-shell (reinforce.py): no value head. Any training-added
  geometry is driven only by the policy gradient through the logits. Transformer has a 32-token banded window.
- Creds present in env: VAST_API_KEY, B2_*, GH_TOKEN. Launch: `rl-harness <module> --seed N` (B2 upload automatic), ~22 min/run on 4090.
- Reuse for probes: within_class_belief_probes_2026_11/{temp15.py restore/histories/rows, temp15_analyze.py replay/analyze_encoder, metrics.py}.
  No artifacts on this VM; checkpoints must be fetched from B2 (restore()).
- theory.py: lam=1 nuisance_value 0, Q-gap share needing delta .005; lam<1 X vs Noop only while lam>0 (Y dominated), Y useful for lam<0.
- Plan for GPU: study belief_quotient_v2_symmetry_dose_2026_11, arms lam doses + kappa (emission asymmetry), seeds 42/43/44, call run_study directly.
- Vast gotcha: provision search returns only the 64 cheapest offers (all GPUs), hiding 4090s. Workaround: own SDK search
  (limit=2000) then `provision up -n 1 --machine-id M`. Launch logs in /home/ubuntu/vast_launch/.
- Launch plan (10 boxes x 3 runs, branch devin/1794716878-within-class-drivers @5fcd526a): lam_p100 s43-45, s46-48; lam_m100 s43-45, s46-48;
  lam_p075, lam_p050, lam_p000, lam_m050, kappa_050, kappa_100 each s43-45.
- Instances (this session's only): see /home/ubuntu/vast_launch/box_*.log "created instance" lines. Launched ~04:50Z 2026-11-15, max-age 3h, self-destruct on.
- 10 boxes launched ~04:50Z (instances 60739696 60739711 60739721 60739733 60739739 60739755 60739773 60739785 60739846 60739936); results push to launch branch -> `git pull`.
- probe.py written (matched-init within-class probes + q_contrast target + logit-margin vs Q-margin + Q-margin terciles); raw per-run output in
  artifacts/probe/<run_id>/analysis.json. Run: `uv run --group jax python -m experiments.within_class_drivers_2026_11.probe --shard i --shards k`.
  Test on PR219 variant_2 seed 42 logging to /tmp/probe_test.log. theory.py rerun (adds lumped_filter_loss, 2048 episodes) -> /tmp/theory2.log; old copy /tmp/theory_v1.json.
- Hypotheses beyond H3: H4 logit margin tracks Q-margin within class (policy gradient shapes logit magnitudes); H5 gains concentrate near decision boundary (low Q-margin tercile);
  kappa arms dissociate filtering-relevance (delta needed to filter rho) from action-relevance.
- Still to do: pull results, run probe shards, report.py (figure + compact results/dose_probe.json), findings.md, tests for probe/theory helpers, PR (fetch template), verify boxes destroyed.
- 04:45Z: all 10 boxes training (GPU 100%; box lam_p075 60739721 was slow to boot, started ~04:45). Box logs: ssh vast-<id> /root/run.log (empty while running); status: `uv run --group devops python -m devops.vast.provision status`.
- probe.py VERIFIED on PR219 variant_2 seed 42 (1 min/run on CPU): reproduces belief .9173 vs init .9152, d(delta) -.0116 [-.019,-.004], d(rho) +.0267;
  NEW: q_contrast full_h .9973 vs init .9731 (+.024); logits->q_contrast .887 vs .649; logit-margin vs Q-margin Spearman .61 trained vs .16 init; policy agrees with belief-MDP greedy 94%;
  reward .509+-.013 (CPU) / .501 (GPU archived) vs theory optimum .5337/.5316 -> ~95%.
- analysis.json keys: trained/matched_init[feature][target], paired_differences[feature][target]{difference,ci95}, reward, test_class_counts, logit_margin_vs_q_margin_spearman, by_q_margin_tercile, global_trained.
- theory.json (2048 ep) DONE. Key: delta-blind loss (lumped filter) lam 1:0, .75:.0006, .5:.0010, 0:.0026, -.5:.095, -1:.243; kappa .05:.003, .1:.018 (optimum .53/.46/.39/.31/.36/.49; .52/.49).
  Optimal policy uses only Noop+Tilt X for lam>=0 and all kappa; 3 actions only at lam<0. So decision relevance of delta is a STEP between lam 0 and -.5, not gradual.
  quotient_policy (exact rho, Q at symmetric belief) is a worse delta-blind policy at lam=0 (.040) -> report lumped as the delta-blind agent; both are upper bounds on the best delta-blind loss.
- probe.py --trajectory added (14 checkpoints replayed on final-policy histories -> artifacts/probe/<run>/trajectory.json). PR219 RSQ seed42: delta .885 init -> .917 @100M (policy all Tilt X, reward .17) -> .873 @2B; rho .970 -> .997 by 160M and stays.
- report.py written+works (reads artifacts/probe/*/analysis.json + trajectory.json -> results/dose_probe.json, dose_trajectory.json, dose_table.md, fig_within_class_symmetry_dose.{pdf,png}). Uncommitted results regenerate; rerun after all probes.
- 05:08Z first 10 runs done (22 min each). Boxes push results only when all 3 seeds finish, so /home/ubuntu/vast_launch/sync.sh scp's finished result dirs into the checkout
  (untracked; DELETE these untracked dirs before `git pull` of the box-pushed copies). Probing with 4 shards: `probe --shard i --shards 4 --trajectory` logs /tmp/probe_shard_i.log (idempotent per run).
- All 10 first runs reach 2 or 3 classes; rewards ~93-100% of belief-MDP optimum (lam_p000 .285/.3075, lam_m050 .338/.361 are below the 95% gate).
- PRELIM (first seed per arm + lam=+-1 x3), gain = trained - matched-init within-class R2 of delta (b1-b2), full_h:
  lam 1: -.012(s42 PR219) -.143 -.081 | .75: -.189 | .5: -.025 | 0: +.028 | -.5: +.082 | -1: +.111(s42) +.145 +.097 | kappa .05: -.161 | kappa .1: +.024.
  rho gain +.02..+.06 everywhere; q_contrast R2 .96-.997 trained in all runs. Logit-margin vs Q-margin Spearman .57-.89 trained.
  => working story: training moves within-class geometry toward ACTION-VALUE-relevant coordinates: adds delta where it changes reward, COMPRESSES it below init where it does not
  (even though delta still predicts future tokens -> argues against predictive/excess-entropy account). Slight symmetry breaks (lam .75, kappa .05) track the approximate quotient.
- 05:38Z: 20/30 runs synced+probing (sync.sh now waits for durability_manifest.json; a run synced mid-upload lacks manifest["remote_artifacts"]). Interim user message sent 05:22Z with prelim table.
  Interim arm means of delta gain (n): lam1 -.079(3) | .75 -.189(1) | .5 -.030(2) | 0 +.025(2) | -.5 +.072(2) | -1 +.120(5) | k.05 -.084(2) | k.1 +.021(2). rho gain + in all but kappa .1 (~0).
  Trajectories: near-symmetric arms: delta rises to ~.91-.92 when rho resolves (~160-300M), then decays below init; lam<0 rises and plateaus. (replicates hypothesis 1 for RSQ, not FSQ)
  Reward gate caveat: CPU greedy reward SE ~.013 so 95% gate flags a few runs at .93-.95; report all-runs and gated.
- Remaining: third batch (~05:50Z) -> sync.sh -> probe shards -> then boxes push + self-destruct: verify `provision status` empty, delete untracked synced dirs, git pull, rerun report, findings.md, tests, PR.
- 05:58Z: 9 boxes self-destructed after pushing (27 runs on origin branch); box 60739721 (lam_p075) last. 29 run dirs synced locally (untracked), probes running (logs /tmp/probe_shardE_*.log).
- 06:25Z ALL 30 dose runs on origin launch branch (pulled; local synced copies deleted), all 10 Vast boxes `gone` (provision status: 0 running). Seeds complete: endpoints 43-48, others 43-45.
  Manifests: status completed, commit 5fcd526a, harness a267508f, B2 remote_artifacts completed for 30/30. git.dirty true for the 2nd/3rd run on each box (20 runs), false for the first (10).
- 32 probes done (30 + PR219 s42 RSQ/FSQ). FULL per-arm delta gain: lam1 -.050 [-.143,+.026] 2/7+ | .75 -.088 0/3 | .5 -.060 0/3 | 0 +.026 3/3 | -.5 +.064 3/3 | -1 +.117 7/7 | k.05 -.055 1/3 | k.1 +.023 3/3.
- KEY CAVEATS found on full data (must be in findings):
  (a) every run rises early to a common delta level ~.89-.93; arms differ in INIT baseline (FSQ init .77-.81 vs RSQ .86-.89, histories differ) and in whether delta later decays. Report peak and final-peak, not only gain.
  (b) q_contrast R2 == rho R2 to 3 dp in every run (within-class Q-contrast variance is dominated by b3) -> not independent evidence.
  (c) lumped-filter loss for lam .75/.5/0, kappa .05 (.0006-.0029) is below MC SE (~.0035-.005): indistinguishable from 0. Only kappa .1 (.018), lam -.5, -1 clearly > 0.
  (d) theory "next optimal action needs delta" (h=1 increment) is .14-.25 at lam .75/.5/0 where delta is compressed/barely gained -> future-action info does NOT predict retention; qgap variance share flat (.004-.014).
  (e) big RSQ compression is seed-heterogeneous (bimodal: ~0 or -.08..-.19).
- report.py extended (trivial rung lam=None, init/trained/peak/final-peak columns, loss SE, predictors Spearman, abs trajectories); probe.py family(lam) adds PR219 variant_1 seed 42 (running, /tmp/probe_trivial.log).
- 06:45Z report rerun on 33 probes (30 dose + PR219 s42 trivial/RSQ/FSQ). Baseline-free outcome final-peak of delta: trivial -.018(1) | lam1 -.088 [-.201,-.037] | .75 -.125 | .5 -.089 | k.05 -.092 | lam0 -.017 | k.1 -.018 | -.5 -.001 | -1 -.003. Peak .89-.93 in all.
  Arm-level Spearman (n=8): delta-blind loss vs gain .76, vs final-peak .69; next-optimal-action share -.29/-.19; q-gap share .57/.62 (range only .004-.014).
  Trivial s42: reward saturated by 20M, yet rho .913->.996 @300M ->.981; delta .841->.873->.855. archived action_fractions are stochastic-policy -> action-onset timing NOT reliable, make no claim.
- Remaining: findings.md, small tests, full fast suite, git diff --check, commit results, push, PR template, PR (no merge), final message.
