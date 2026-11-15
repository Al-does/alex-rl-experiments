"""Within-greedy-class probes for the v2 temp15 exploration sweep (Reward-State and Full-State transformers).

Reuses the PR219 protocol in ``within_class_belief_probes_2026_11.nuisance`` unchanged; only the arm
selection and the recipe validator differ, so new epsilon and constant-entropy settings are accepted.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_explore_2026_11.shared import ARMS
from experiments.within_class_belief_probes_2026_11.nuisance import analyze
from experiments.within_class_belief_probes_2026_11.temp15 import TOLERANCE_REASON

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
RAW = HERE / "artifacts"
OUTPUT = HERE / "results/within_class_exploration_sweep.json"
PR219 = {"belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11": (0., 0.),
         "belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11": (.05, 0.)}
SWEEP = "belief_quotient_persistent_nuisance_v2_jax_temp15_explore_2026_11"
VARIANTS = (2, 3)


def expected_knobs(study, leaf):
    if study in PR219:
        return PR219[study]
    if study == SWEEP:
        settings = ARMS[leaf.split("_transformer_")[1]]
        return settings["exploration_epsilon"], settings["entropy_coeff"]
    raise ValueError(f"unexpected study {study}")


def selected_runs():
    runs = []
    for study in [*PR219, SWEEP]:
        suffix = "" if study in PR219 else "_*"
        for variant in VARIANTS:
            runs += sorted(p for p in (EXPERIMENTS/study).glob(f"variant_{variant}_transformer{suffix}/results/*")
                           if (p/"run_manifest.json").exists())
    return runs


def validate_recipe(recipe, leaf):
    training, model, process = recipe["training"], recipe["model"], recipe["process"]
    epsilon, entropy = expected_knobs(recipe["study"], leaf)
    if (recipe["environment_revision"] != "persistent_nuisance_v2" or int(recipe["condition"].split("_")[1]) not in VARIANTS
            or training["exploration_epsilon"] != epsilon or training["entropy_schedule"] != [[0, entropy]]
            or training["temperature"] != 1.5 or training["total_env_steps"] != 2_000_000_000
            or recipe["seed"] != 42 or model["kind"] != "transformer" or model["n_actions"] != 3
            or np.asarray(process["transitions"]).shape != (3, 3, 3)
            or not np.allclose(process["emission"], [[.1125, .0375, .85], [.0375, .1125, .85], [0, 0, 1]])
            or not np.allclose(process["initial"], [1/3]*3)
            or not np.allclose(recipe["baseline_transition_matrix"], [[.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99]])
            or recipe["effect_size"] != 2.5):
        raise ValueError(f"{recipe['study']}/{leaf}: expected seed42 v2 three-state transformer, T1.5, 2B steps, "
                         f"epsilon {epsilon}, constant entropy {entropy}")


def analyze_run(run):
    leaf = run.parent.parent.name
    recipe = json.loads((run/"resolved_recipe.json").read_text())
    validate_recipe(recipe, leaf)
    raw = RAW/recipe["study"]
    analyze([run], raw_root=raw, validator=lambda r: validate_recipe(r, leaf))
    record = json.loads((raw/leaf/"analysis.json").read_text())
    epsilon, entropy = expected_knobs(recipe["study"], leaf)
    record.update(study=recipe["study"], variant=int(recipe["condition"].split("_")[1]), training_epsilon=epsilon,
                  training_entropy=entropy, environment_revision=recipe["environment_revision"],
                  source_run=str(run.relative_to(REPO)))
    return record


def main(only=None):
    runs = [r for r in selected_runs() if only is None or any(o in str(r) for o in only)]
    records = [analyze_run(run) for run in runs]
    if only is not None:
        print("Analyzed", len(records), "runs (subset; report not written)", flush=True)
        return
    assert len(records) == 2*(len(PR219)+len(ARMS)), len(records)
    protocol = json.loads((EXPERIMENTS/"within_class_belief_probes_2026_11/results/ladder_nuisance_ent0p005_within_class.json").read_text())["protocol"]
    protocol.update(training_temperature=1.5, environment_revision="persistent_nuisance_v2", greedy_probes=True,
                    training_epsilon=sorted({r["training_epsilon"] for r in records}),
                    training_entropy=sorted({r["training_entropy"] for r in records}),
                    emission=[[.1125, .0375, .85], [.0375, .1125, .85], [0, 0, 1]],
                    baseline_transition=[[.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99]],
                    effect_size=2.5, archive_rule=TOLERANCE_REASON,
                    collector_confirmation="Probes use greedy argmax rollouts; training temperature, epsilon and entropy bonus affect only training sampling/loss.")
    files = [*HERE.glob("*.py"), *EXPERIMENTS.joinpath("within_class_belief_probes_2026_11").glob("*.py"),
             *EXPERIMENTS.joinpath("belief_quotient_history_jax_2026_11").glob("*.py"),
             *[p for study in [*PR219, SWEEP] for p in (EXPERIMENTS/study).glob("*.py")]]
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    report = {"schema_version": 1, "protocol": protocol, "arms": records,
              "provenance": {"experiment_commit": git("rev-parse", "HEAD"),
                             "harness_commit": git("-C", str(REPO.parent/"rl-harness"), "rev-parse", "HEAD"),
                             "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
                             "devices": str(jax.devices()),
                             "source_sha256": {str(p.relative_to(EXPERIMENTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
