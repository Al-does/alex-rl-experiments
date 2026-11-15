"""PR219 v2 transformer CPU probes, reusing the verified PR215 analysis."""

import hashlib
import json
import platform
import subprocess

import jax
import numpy as np

from .nuisance import analyze
from .temp15 import ROOT, TOLERANCE_REASON

REF = "58a26c8c6c2bc1fda82845db4727a6c7397842a4"
STUDIES = ["belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11",
           "belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11"]
RAW = ROOT/"artifacts/nuisance_v2_temp15"
OUTPUT = ROOT/"results/ladder_nuisance_v2_temp15_within_class.json"


def selected_runs():
    return [run for study in STUDIES for run in sorted((ROOT.parent/study).glob("variant_*_transformer/results/*"))
            if (run/"run_manifest.json").exists()]


def validate_recipe(recipe):
    expected = .05 if recipe["study"] == STUDIES[1] else 0.
    training, model, process = recipe["training"], recipe["model"], recipe["process"]
    if (recipe["study"] not in STUDIES or recipe["environment_revision"] != "persistent_nuisance_v2"
            or training["exploration_epsilon"] != expected or training["temperature"] != 1.5
            or training["entropy_schedule"] != [[0, 0.]] or training["total_env_steps"] != 2_000_000_000
            or recipe["seed"] != 42 or model["kind"] != "transformer" or model["n_actions"] != 3
            or np.asarray(process["transitions"]).shape != (3, 3, 3)
            or not np.allclose(process["emission"], [[.1125, .0375, .85], [.0375, .1125, .85], [0, 0, 1]])
            or not np.allclose(process["initial"], [1/3]*3)
            or not np.allclose(recipe["baseline_transition_matrix"], [[.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99]])
            or recipe["effect_size"] != 2.5):
        raise ValueError("expected seed42 v2 three-state transformer, T1.5, entropy0, study-specific epsilon0/.05")


def main():
    runs = selected_runs()
    assert len(runs) == 6
    records = []
    for run in runs:
        recipe = json.loads((run/"resolved_recipe.json").read_text())
        validate_recipe(recipe)
        raw = RAW/recipe["study"]
        analyze([run], raw_root=raw, validator=validate_recipe)
        record = json.loads((raw/run.parent.parent.name/"analysis.json").read_text())
        record.update(study=recipe["study"], training_epsilon=recipe["training"]["exploration_epsilon"],
                      environment_revision=recipe["environment_revision"])
        records.append(record)
    protocol = json.loads((ROOT/"results/ladder_nuisance_ent0p005_within_class.json").read_text())["protocol"]
    protocol.update(training_entropy=0., training_temperature=1.5, training_epsilon=[0., .05],
                    environment_revision="persistent_nuisance_v2", greedy_probes=True,
                    emission=[[.1125, .0375, .85], [.0375, .1125, .85], [0, 0, 1]],
                    baseline_transition=[[.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99]],
                    effect_size=2.5, archive_rule=TOLERANCE_REASON,
                    collector_confirmation="PR219 changes only training sampling/likelihoods through exploration_epsilon; greedy argmax and key splitting unchanged; shared probe/models/env/training files unchanged. Local collector is equivalent for greedy probes ignoring temperature and epsilon.")
    files = [*ROOT.glob("*.py"), *ROOT.parent.joinpath("belief_quotient_history_jax_2026_11").glob("*.py")]
    for study in STUDIES:
        files.extend((ROOT.parent/study).glob("*.py"))
    files.append(ROOT.parent/"belief_quotient_persistent_nuisance_jax_temp15_2026_11/shared.py")
    report = {"schema_version": 1, "protocol": protocol, "arms": records,
        "provenance": {"source_branch": "devin/1794431156-v2-temp15-eps-greedy", "source_revision": REF,
            "import_method": "git archive from recorded PR219 SHA; only selected transformer arms and recipe/environment sources retained; no merge/rebase",
            "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_commit": subprocess.check_output(["git", "-C", str(ROOT.parents[2]/"rl-harness"), "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__, "devices": str(jax.devices()),
            "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
