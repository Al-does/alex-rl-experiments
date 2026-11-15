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
