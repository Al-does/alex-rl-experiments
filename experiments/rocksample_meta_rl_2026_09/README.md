# RockSample meta-RL: learned policy-optimisation objective

PPO on the default RockSample[5,7] plateaus near +24.5 undiscounted return
(`rocksample_ppo_2026_09`), well below the ~45 achievable optimum. This study
meta-learns the *learning algorithm* instead of tuning PPO, in the lineage of
Learned Policy Optimisation / DPO (Lu et al. 2022, "Discovered Policy
Optimisation") and DiscoPOP (Chris Lu et al. 2024): parameterise the policy
objective, train many agents from scratch with candidate objectives, and
update the objective on their final performance.

## Method (`lpo_es_drift`)

Mirror learning writes PPO's surrogate as `r*A - F(r, A)` with drift
`F_ppo = relu((r - clip(r, 0.8, 1.2)) * A)`. The learned objective is

```text
F_theta(r, A) = relu(w_ppo * F_ppo(r, A) + g(x(r, A)) - g(x(1, A)))
loss          = -(r*A - F_theta) + 0.5 * value_loss - exp(log_ent) * entropy  (+ PPO's KL term)
x(r, A)       = [r-1, log r, (r-1)^2, (log r)^2] and the same times A
```

`g` is an 8→32→1 tanh MLP. Meta-parameters `theta = [g, w_ppo, log_ent]`
(323 numbers). The output layer of `g` starts at zero, `w_ppo = 1`,
`log_ent = log 0.05`: generation 0's centre is exactly the PPO recipe
(d64 transformer, gamma 0.99, lambda 0.95, lr 3e-4, 8,192-step batches),
i.e. the outer loop is kickstarted at PPO. `F >= 0` and `F(1, A) = 0` hold
for every theta.

Outer loop: antithetic OpenAI-ES with centred-rank shaping and Adam
(8 pairs, sigma 0.05, lr 0.02, 50 generations). Each candidate trains a fresh
agent for 3M environment steps; fitness is its mean undiscounted
`episode_return_mean` over the final 20% of iterations. Both members of an
antithetic pair share an inner seed (common random numbers). Each generation
also trains the unperturbed centre, so `center_return` tracks the learned
objective and generation 0's centre is the PPO baseline.

All candidates of a generation run concurrently as one Tune grid
(`gpus_per_trial = 0.25`, 4 env runners each).

## Outputs

`results/meta_progress.jsonl` (per-generation fitness, centre return,
`w_ppo`, entropy coefficient, update norm), `results/meta_state.json` (ES
mean + Adam state, used for `--resume-from`), `results/generations/gen_XXXX/`
(per-generation Tune summaries). Inner Tune trees stay in `artifacts/`; no
inner checkpoints are saved.

## Running

```bash
uv run pytest -q tests/test_rocksample_meta_rl_2026_09.py
uv run rl-harness experiments.rocksample_meta_rl_2026_09.lpo_es_drift.experiment --smoke --hardware cpu --no-upload-artifacts
# resume a meta-run
uv run rl-harness experiments.rocksample_meta_rl_2026_09.lpo_es_drift.experiment --resume-from <results>/meta_state.json
```

The smoke run does 2 generations of 2 antithetic pairs + centre at 2,048
inner steps each; its returns are a pipeline check only.

## Cost note

A full generation is 17 inner runs x 3M steps (~51M steps); 50 generations is
~2.5B steps. The budgets live in `lpo_es_drift/experiment.py` and are
provisional.
