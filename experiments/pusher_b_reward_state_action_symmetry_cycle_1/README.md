# Pusher-B action-symmetry worker sizing

The full Pusher-B reward-state action-symmetry recipes default to at most 52 CPU
EnvRunners. The shared layout caps that preference to the schedulable CPU limit
minus one, leaving one CPU for the Ray driver, learner coordination, and system
work. Smoke runs keep using the local EnvRunner.

For the standard 262,144-step batch and 127-step episodes, 2,065 complete
episodes are required. The default layout is 52 runners with 40 vector
environments each, so a sampling round collects 2,080 complete episodes. This
is the layout promoted from the isolated throughput study.

## Evidence and limits

The matched benchmark ran on a Vast.ai RTX 5090 host with 272 visible CPUs, a
261.12-CPU cgroup quota, and 62 GB RAM. The 52 × 40 layout cut sampling time by
61% and iteration time by 52% relative to 16 × 130, while using 72.9% of host
RAM. It was chosen to fit an offer advertising about 64 effective CPUs, but the
64-CPU candidate never completed bootstrap; the successful benchmark used the
larger host described above. The layout therefore was not directly validated
on a 64-CPU machine.

CPU capacity and GPU model are separate host properties. Prefer offers with
enough CPU and RAM for sampling rather than assuming a larger GPU will improve
this CPU-bound workload.

## Testing another host

Start below the host's effective CPU limit and leave at least one CPU unused:

- 52 runners need at least 53 schedulable CPUs and substantial RAM.
- 32 runners is the measured lower-memory fallback and needs at least 33 CPUs.
- 16 runners remains a conservative option on smaller hosts.

The shared default automatically reduces 52 when fewer CPUs are available.
Explicit `HardwareProfile.num_env_runners` values can select another count.
When testing a new count, set vector environments per runner to
`ceil(2065 / workers)`. Counts that divide 2,080, such as 16, 20, 26, 32, 40,
52, 65, or 80, preserve the benchmark's 2,080-episode sampling round exactly.

Compare candidates sequentially on the same host with the same seed, budget,
model, and PPO settings. Inspect sampling time, total iteration time, early
returns, CPU quota, and peak RAM. Increase the worker count only while sampling
time improves without memory pressure, worker failures, or changed early
learning behavior.

The benchmark implementation and compact results are under `throughput/`.
