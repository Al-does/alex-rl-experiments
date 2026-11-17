# PPO and auto-alpha SAC steering generality

Forty final actors were evaluated locally on eight CPU cores: thirty ladder
actors (all three conditions) and ten Joint Reward actors, each algorithm on
seeds 42/44/46/49/52. Every final checkpoint at step 2,012,217,344 was verified
against its canonical durability manifest. Both protocols were smoke-tested
before their full campaigns. No training, paid compute, XOR, factor-2
suppression or separate Marginal Reward agent campaign was run.

## Three-line qualitative comparison

1. **Direction:** both algorithms reproduce the REINFORCE ladder dissociation:
   unit-strength delta steering leaves Reward-State control almost unchanged
   but disrupts Full-State control; rho steering disrupts both. Moving-reward
   steering improves both algorithms, with joint targets outperforming factored
   targets at block 4.
2. **Magnitude versus controls:** Full-State delta occupancy is .184–.197 and
   Reward-State rho-point occupancy .066–.077, versus intact .407–.446 and
   decoder-orthogonal controls close to intact. Moving joint steering yields
   1.032 (PPO) / 1.009 (SAC), versus intact .900/.908, matched null .901/.904,
   random .602 and REINFORCE 1.033 (same five) / 1.042 (all fifteen).
3. **Sites:** all five sites show cohort-mean moving-reward gains and rho
   disruption in Reward-State/Full-State agents, generally stronger later.
   The single-write ladder responses terminate at the same finite-cache
   cutoffs as REINFORCE: lags 96/64/32 after blocks 1/2/3.

## Ladder details

[tables.md](tables.md) compares all fifteen REINFORCE seeds, the same five
REINFORCE seeds, PPO and auto-alpha SAC, including every per-seed headline
result. The all-fifteen REINFORCE means reproduce the manuscript's
`tab:ladder-steering-final` to its displayed precision.

- Reward-State intact occupancy is .4395 (PPO) / .4464 (SAC). Delta point/mirror
  occupancy remains .4365/.4385 and .4490/.4476, with .2–.8% action changes.
  Rho point reduces occupancy to .0661/.0772, below the .1057 random reference,
  with 58.5%/81.2% action changes. Rho flip is somewhat less disruptive for SAC
  (.1060) than REINFORCE (.0866 over fifteen seeds).
- Full-State intact occupancy is .4100/.4071. Delta point/mirror reduce it to
  .1870/.1973 (PPO) and .1974/.1840 (SAC), close to REINFORCE .1857/.1919.
  Full-strength rho flip gives .1119/.1138, less disruption than REINFORCE
  .0851, while rho point gives .0739/.0760 versus REINFORCE .0595.
- At tuning-selected strengths after block 4, Reward-State rho point gives
  .0760/.0806 against orthogonal controls .4377/.4353. Full-State delta mirror
  gives .1973/.1872 against .4065/.4070. Controls use selected strengths;
  they must not be mistaken for controls at the headline full strength.
- Norm-matched amplification of delta is **not** inert: Reward-State point/
  mirror occupancy becomes .3220/.3507 (PPO) and .3733/.3710 (SAC), compared
  with REINFORCE .3252/.3367. Achieved confirmation shift norms are near the
  rho-flip norms for these conditions. Thus the result is a selective
  unit-strength dissociation, not proof that delta information is absent or
  incapable of affecting behavior under large writes.
- Trivial agents ignore unit delta. PPO also mostly ignores unit rho; SAC
  rho-flip occupancy falls to .5794, with its orthogonal control at .8631.
  This is variable across algorithms/seeds: REINFORCE's same-five rho-flip
  mean is .9350 but its all-fifteen mean is .6879. Large-write and generic
  perturbation caveats matter especially here.
- Every one of the thirty actors completed all six pulse conditions. All
  measured last-nonzero lags are exactly 96/64/32 after blocks 1/2/3 for both
  pulse modes. Pulse amplitudes and exact full/quotient-filter comparisons are
  retained; a short-lived response in this finite cache is not evidence of
  implementing the exact Bayesian filter.

## Moving reward details

The block-4 discounted-map comparison reproduces `tab:moving-steering` for
the fifteen REINFORCE agents:

| Cohort | Intact | Random | Matched null (joint) | Factored | Joint | Exact-belief map |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE, 15 | .9009 | .6021 | .8987 | .9985 | 1.0420 | 1.0919 |
| REINFORCE, same 5 | .8968 | .6021 | .9006 | .9924 | 1.0334 | 1.0919 |
| PPO, 5 | .9003 | .6021 | .9013 | 1.0017 | 1.0321 | 1.0919 |
| SAC auto-alpha, 5 | .9084 | .6021 | .9040 | .9619 | 1.0092 | 1.0919 |

Both factored and joint block-4 steering exceed intact and the joint matched
null in every one of the five PPO and five SAC seeds. Joint gains are .1318
for PPO and .1008 for SAC, versus .1366 for the same five REINFORCE agents.
PPO closely matches the five-seed reference; SAC transfers in the same
direction but less strongly. Both myopic and discounted maps, component and
phase rewards, and per-factor action agreement/change are retained in
`summary.json`, not only these headline discounted-map means.

All five intervention sites improve cohort means. Joint reward ranges from
1.0063–1.0360 for PPO and .9408–1.0125 for SAC; blocks 2–4/final norm usually
outperform block 1. The controller alone knows reward phase; the frozen
actor receives no new phase observation or reward input. Factored refers to
the source's `marginal` steering target, not Marginal Reward-trained agents.

## Limits and provenance

These are descriptive comparisons of five independently trained PPO/SAC
agents with archived REINFORCE results, not a significance test or an
optimal-control certificate. Arbitrary large activation writes, collateral
changes to factor 1, and policy-dependent collected histories limit causal
interpretation. The numerical belief-grid and exact-belief QMDP references
are references, not certified POMDP optima. Steering does not establish a
unique Bayesian implementation or information erasure.

`STEERING.md` specifies original budgets, grids and seeds. Source revisions
are ladder `45517b13a67d746048a002f9db52ea55a654b219` and moving reward
`3504bfb72455a411b725ef368a18ff892760555f`. Checkpoint/B2 keys and hashes,
training/analysis/harness revisions, executed source hashes and CPU devices
are recorded per actor. The analysis HEAD is the preceding Step 1 commit;
source hashes identify the uncommitted adapter contents actually used.
Post-run report-header/terminal-blank-line normalization is documented in
`STEERING.md` and explicitly reconstructed by the audit. Original numerical
helper ASTs and the complete ladder numerical body are checked against the
pinned sources. Raw reports, maps, checkpoints and logs remain ignored under
`artifacts/`; no REINFORCE actor was rerun.
