---
name: vast-experiment-run
description: Launch a specified alex-rl-experiments training run on a user-defined number of Vast.ai GPU boxes and perform timed post-launch health checks. Use when the user has supplied an experiment, seed plan, and Vast box count and wants the run provisioned, triggered, and monitored.
---

# Vast experiment run

Use this skill only after the user has provided the experiment-specific plan.
This skill handles remote execution and early monitoring; it does not choose the
scientific recipe, training budget, seeds, or GPU fleet size.

## 1. Confirm the launch contract

Before renting, record these details from the user's instructions:

- importable experiment module and all required CLI arguments;
- exact git ref containing the experiment code, already committed and pushed;
- number of seeds and their values;
- number of Vast boxes;
- explicit seed-to-box mapping and the remote command for each box or batch;
- GPU requirements, budget constraints, Vast mode, regions, and maximum box age;
- whether B2 artifact upload, result push, and self-destruction are required.

Do not silently infer a seed strategy from the number of boxes. In particular,
`provision up -n N --run "..."` runs the same command on every box, so it is
unsafe for distinct seeds unless that command itself deterministically selects a
different assigned seed. If the user gave only a seed count and box count, ask
for a mapping or a safe remote batch command before renting.

Run the experiment's local smoke test before a real remote launch when it has
not already passed for the committed ref. Do not use a full research run as the
smoke test.

## 2. Delegate provisioning to `/vast-provisioning`

Read and follow the sibling `rl-harness` skill
`/rl-harness/.cursor/skills/vast-provisioning/SKILL.md` before running any
Vast command. Its safety and ownership rules are mandatory, including:

- run a dry run before renting;
- rent a fresh box for this session and never inspect or operate another
  agent's box;
- use only redacted `status` and `inspect` output, never raw instance metadata;
- set a finite `--max-age` as a cost backstop;
- destroy only this session's tracked instance IDs, never `destroy --all`
  without the user's explicit confirmation.

Run local provisioning from this experiment checkout through the harness's
`devops` group. The remote `--run` command executes inside the activated,
pre-synced environment, so do not prefix it with `uv run`.

For per-box seed assignments, provision boxes separately with their own
`--run` commands unless the confirmed remote command safely dispatches the
assigned seed. Pass the agreed experiment-repo ref with `--branch` or
`--commit` so the remote code matches the reviewed launch contract.
`devops.vast.provision` expands abbreviated commit SHAs to full 40-character
forms before bootstrap; short hashes in launch prompts are fine once the ref
exists locally and on the remote.

### Vast mode: on-demand only

Always pass `--mode ondemand`. **Never** use `--mode interruptible` or
preemptible boxes for experiment runs — interruptible instances get outbid,
lose SSH mid-run, and waste completed training when the fleet is torn down.
On-demand is required unless the user explicitly overrides this in writing.

### Remote queue commands: use module invocation

Experiment seed/arm queue scripts must be invoked as modules, not as file paths:

```bash
python -m experiments.<study>.seed_queue --condition <arm> --seeds 42 43 44 45 46
```

Do **not** use `python experiments/<study>/seed_queue.py …`. The script path
shadows the harness `analysis` package (`analysis.py` vs `analysis/`) and
raises `ModuleNotFoundError` on the box.

### GitHub token and result push without self-destruct

When runs should push compact results (`--push-each`, `publish_results`, etc.)
but boxes should stay up (no `--self-destruct`), still pass a GitHub write
token via `--github-token` / `GITHUB_TOKEN`. Bootstrap always configures git
user identity on the box; the token enables clone and push.

### Parallel multi-box provisioning

Provisioning several boxes concurrently from separate `provision up` processes
can race on `devops/vast/state.json` (last write wins, earlier instance IDs
lost). Until that is fixed in rl-harness, **provision sequentially** — one
`provision up` per box/arm — or use a single process that records every
instance before moving on. Do not launch eight parallel `provision up` shells
expecting all IDs to appear in `state.json`.

## 3. Trigger the confirmed experiment

1. Run the agreed `--dry-run` and review the candidate count, price, GPU type,
   and launch command against the launch contract.
2. Rent exactly the agreed number of boxes with the confirmed commands and
   options.
3. Record only this session's returned instance IDs, aliases, and seed
   assignments.
4. Confirm every box is `running` and ready before treating the experiment as
   launched. Inspect the detached `tmux` run or the safe container-log tail to
   confirm the command started.

## 4. Timed health checks

After the launch, wait five minutes, then inspect every box rented in this
session. Check the tracked-box status, redacted instance inspection, and safe
run/container-log tail. Verify that the box remains running, the train process
is alive or advancing, no bootstrap failure marker is reported, and no
traceback, out-of-memory event, repeated worker failure, or immediate
experiment failure is visible.

If an obvious operational or command error appears, diagnose it from the safe
logs and fix it when the correction is clear and within this repository or the
confirmed launch configuration. Commit and push any experiment-repo change
before relaunching it, then re-check the affected box. Flag the user instead
when the cause requires a scientific decision, credentials/account access,
unapproved `rl-harness` changes, ambiguous data/result interpretation, or a
cost/fleet change. Do not keep a known-failed box billing while waiting for a
decision: preserve useful evidence and destroy only the affected session-owned
box if safe to do so.

If the five-minute check is healthy, wait another 30 minutes and repeat the
same inspection and triage. At that second healthy check, report the instance
IDs, seed assignments, observed progress, and configured max-age, then finish.
“Finish” means the requested early health monitoring is complete; leave healthy
training boxes running unless the confirmed launch contract requested
self-destruction or the user asks to stop them.
