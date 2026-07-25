"""Record the task-intrinsic floors and ceilings for the token-guess metrics."""

from __future__ import annotations

from typing import Any

from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    BASE_MODEL_CONFIG,
    ENV_CONFIG,
)
from experiments.mess3_token_guess_cycle_2.metric_references import (
    compute_references,
    normalise,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext

FIT_STEPS = 60_000
TEST_STEPS = 30_000
SMOKE_FIT_STEPS = 2_000
SMOKE_TEST_STEPS = 1_000

# Best reported cycle-1 scores, used only to illustrate where the published
# numbers sit inside the range the metric can move through.
CYCLE_1_SCORES = {
    "comparison/reward_only": 0.8552,
    "comparison/predictive_loss": 0.9319,
    "comparison/max_entropy": 0.8558,
    "iqn_value": 0.9760,
    "kelly_cycle_2/correctness_iqn": 0.9857,
    "kelly_cycle_3/conditional_decoupled_kelly_iqn": 0.9824,
}
# Supervised next-token replication, final-LayerNorm probe, seed 0. See
# experiments/mess3_supervised/README.md.
SUPERVISED_CEILING = 0.99888


def _findings(references: dict[str, Any]) -> str:
    floor = references["belief_r2_floor"]
    context = references["belief_r2_floor_context"]
    low, high = references["belief_r2_probe_noise_95ci"]
    lines = [
        "# Token-guess metric reference points",
        "",
        "## Belief-probe R²",
        "",
        f"An affine probe reading the one-hot encoded last {context} observations,",
        "with no network and no training, already scores "
        f"R² = {floor:.4f}.",
        "The supervised next-token replication reaches "
        f"{SUPERVISED_CEILING:.4f}.",
        "Belief-probe R² therefore moves through a usable range of only "
        f"{SUPERVISED_CEILING - floor:.4f}.",
        "",
        "| observations visible to the probe | R² |",
        "|---:|---:|",
    ]
    for k, value in sorted(
        references["raw_token_window_r2"].items(), key=lambda item: int(item[0])
    ):
        lines.append(f"| {k} | {value:.4f} |")
    untrained = references.get("untrained_module")
    if untrained is not None:
        lines.extend(
            [
                "",
                "A randomly initialised copy of the study transformer scores "
                f"R² = {untrained['r_squared']:.4f} with greedy accuracy "
                f"{untrained['token_accuracy_greedy']:.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            "Bootstrap resampling of the probe's test set puts its own sampling "
            f"noise at [{low:.4f}, {high:.4f}].",
            "",
            "## Where the cycle-1 scores sit",
            "",
            "| condition | reported R² | fraction of the floor-to-ceiling range |",
            "|---|---:|---:|",
        ]
    )
    for condition, value in CYCLE_1_SCORES.items():
        fraction = normalise(value, floor=floor, ceiling=SUPERVISED_CEILING)
        lines.append(f"| `{condition}` | {value:.4f} | {fraction:+.1%} |")

    accuracy = references["bayes_accuracy_by_context"]
    lines.extend(
        [
            "",
            "## Greedy token accuracy",
            "",
            "| observations visible to an exact Bayesian filter | accuracy |",
            "|---:|---:|",
        ]
    )
    for k, value in sorted(accuracy.items(), key=lambda item: int(item[0])):
        lines.append(f"| {k} | {value:.4f} |")
    lines.extend(
        [
            "",
            "One observation reproduces the trivial repeat-the-previous-token "
            f"rule at {references['accuracy_floor_repeat_previous_token']:.4f}; "
            "the filter saturates at "
            f"{references['accuracy_ceiling_bayes']:.4f}. Greedy token accuracy "
            "therefore moves through a usable range of only "
            f"{references['accuracy_ceiling_bayes'] - references['accuracy_floor_repeat_previous_token']:.4f}.",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("the reference computation requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    references = compute_references(
        seed=context.seed,
        fit_steps=SMOKE_FIT_STEPS if context.smoke else FIT_STEPS,
        test_steps=SMOKE_TEST_STEPS if context.smoke else TEST_STEPS,
        env_config=ENV_CONFIG,
        model_config=BASE_MODEL_CONFIG,
    )
    references["supervised_ceiling"] = SUPERVISED_CEILING
    references["cycle_1_normalised"] = {
        condition: normalise(
            value,
            floor=references["belief_r2_floor"],
            ceiling=SUPERVISED_CEILING,
        )
        for condition, value in CYCLE_1_SCORES.items()
    }
    outputs.write_json("references.json", references)
    (context.results_dir / "findings.md").write_text(_findings(references))
    return references
