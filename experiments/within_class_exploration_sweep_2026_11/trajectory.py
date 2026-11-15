"""Fixed-final-history checkpoint trajectories for the sweep, reusing the parent PR219 trajectory protocol.

Usage: ``python -m ...trajectory [--only SUBSTRING ...] [parent trajectory options]``. With ``--only`` the
report goes to the ignored artifacts tree; per-checkpoint points are cached, so the full run reuses them.
"""

import json
import sys

from experiments.within_class_belief_probes_2026_11 import trajectory as parent
from .analyze import HERE, RAW, expected_knobs, selected_runs, validate_recipe

OUTPUT = HERE/"results/within_class_exploration_sweep_trajectory.json"


def validated_runs(only=None):
    runs = [r for r in selected_runs() if not only or any(o in str(r) for o in only)]
    for run in runs:
        validate_recipe(json.loads((run/"resolved_recipe.json").read_text()), run.parent.parent.name)
    return runs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    only = []
    while "--only" in argv:
        i = argv.index("--only")
        only.append(argv[i+1])
        del argv[i:i+2]
    output = RAW/"trajectory_subset.json" if only else OUTPUT
    parent.selected_runs = lambda: validated_runs(only)
    parent.validate_recipe = lambda recipe: None
    parent.RAW, parent.OUTPUT = RAW, output
    sys.argv = [sys.argv[0], *argv]
    parent.main()
    report = json.loads(output.read_text())
    for arm in report["arms"]:
        epsilon, entropy = expected_knobs(arm["study"], arm["arm"])
        arm.update(training_epsilon=epsilon, training_entropy=entropy,
                   variant=int(arm["arm"].split("_")[1]))
    report["provenance"]["sweep_wrapper"] = "within_class_exploration_sweep_2026_11.trajectory (parent protocol unchanged)"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
