"""Sequential single-seed PPO arms, following the RockSample sweep recipe."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

from harness.artifacts import RunArtifacts, maybe_upload_run_artifacts
from harness.context import new_run_id

from experiments.mrrxor_grokking_jax_2026_11 import ppo
from experiments.mrrxor_grokking_jax_2026_11.analysis import analyze
from experiments.pusher_b_jax_2026_10.model import ModelSpec


@dataclass(frozen=True)
class Arm:
    name: str
    ppo: ppo.PPOConfig


@dataclass(frozen=True)
class SweepRecipe:
    arms: tuple[Arm, ...]
    model: ModelSpec = ModelSpec(vocab=3, context_length=32)

    def __post_init__(self):
        if not self.arms:
            raise ValueError("a sweep needs at least one arm")
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("duplicate arm names")
        if any(not name or name in (".", "..") or "/" in name or "\\" in name for name in names):
            raise ValueError("arm names must be single directory names")
        if any(arm.ppo.sequence_length != self.model.context_length for arm in self.arms):
            raise ValueError("model context must match arm sequence lengths")


def grid_recipe():
    return SweepRecipe(tuple(
        Arm(f"n{n}", ppo.PPOConfig(train_sequences=n))
        for n in (256, 1024, 4096, 16384)
    ))


def smoke_recipe():
    return SweepRecipe(tuple(
        Arm(f"n{n}", ppo.PPOConfig.smoke(train_sequences=n))
        for n in (16, 32)
    ), model=ModelSpec(vocab=3, context_length=32, d_model=16, n_layers=1, n_heads=2, d_mlp=32))


def summary_markdown(rows):
    lines = ["# MRRXOR PPO sweep", "",
             "Sorted descending by final held-out random-bit accuracy, then held-out expected reward.",
             "One context seed; descriptive comparisons, not confirmation of grokking.", "",
             "| Arm | Steps | Train random | Held-out random | Held-out reward | Train excess CE | Held-out excess CE | NTP-PC 1−R² |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['env_steps']} | {row['train_random_bit_accuracy']:.6f} | "
                     f"{row['heldout_random_bit_accuracy']:.6f} | {row['heldout_reward']:.6f} | "
                     f"{row['train_excess_ce_nats']:.6f} | {row['heldout_excess_ce_nats']:.6f} | "
                     f"{row['ntp_pc_probe_1_minus_r2']:.6f} |")
    return "\n".join(lines) + "\n"


def run(context, recipe):
    if context.seed is None:
        raise ValueError("sweep requires a resolved context seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", {"seed": context.seed, "model": asdict(recipe.model),
                                              "arms": [asdict(arm) for arm in recipe.arms]})
    rows = []
    for arm in recipe.arms:
        print(f"[sweep] {arm.name}, seed={context.seed}", flush=True)
        child = replace(context, results_dir=outputs.results_dir / arm.name,
                        artifacts_dir=outputs.results_dir / arm.name / "artifacts",
                        run_id=f"{context.run_id}-{arm.name}")
        trained = ppo.train(child, arm.ppo, recipe.model)
        report = analyze(child.artifacts_dir, child.results_dir / "analysis.json")
        maybe_upload_run_artifacts(child)
        final = report["checkpoints"][-1]
        metrics = final["metrics"]
        rows.append({"arm": arm.name, "seed": context.seed,
                     "train_sequences": arm.ppo.train_sequences, "weight_decay": arm.ppo.weight_decay,
                     "lr_schedule": [list(point) for point in arm.ppo.lr_schedule], "env_steps": trained["env_steps"],
                     "train_random_bit_accuracy": metrics["train"]["random_bit_accuracy"],
                     "heldout_random_bit_accuracy": metrics["heldout"]["random_bit_accuracy"],
                     "train_reward": metrics["train"]["sampled_policy_mean_reward"],
                     "heldout_reward": metrics["heldout"]["sampled_policy_mean_reward"],
                     "train_excess_ce_nats": metrics["train"]["excess_ce_nats"],
                     "heldout_excess_ce_nats": metrics["heldout"]["excess_ce_nats"],
                     "ntp_pc_index": report["pca"]["ntp_pc_index"],
                     "ntp_pc_probe_1_minus_r2": final["probe_1_minus_r2"][f"pc{report['pca']['ntp_pc_index']}"],
                     "analysis_path": str(child.results_dir / "analysis.json"),
                     "artifact_run_dir": str(child.artifacts_dir)})
        rows.sort(key=lambda r: (r["heldout_random_bit_accuracy"], r["heldout_reward"]), reverse=True)
        summary = {"seed": context.seed, "arms": rows}
        outputs.write_json("summary.json", summary)
        (outputs.results_dir / "summary.md").write_text(summary_markdown(rows))
    print(summary_markdown(rows), flush=True)
    return summary


STUDY_DIR = Path(__file__).resolve().parent
EXPERIMENT = "experiments.mrrxor_grokking_jax_2026_11.ppo_sweep.experiment"


def run_paths(run_id, *, smoke=False, publish_smoke=False):
    if not run_id or run_id in (".", "..") or Path(run_id).name != run_id:
        raise ValueError("run ID must be a single directory name")
    root = STUDY_DIR / (".smoke" if smoke and not publish_smoke else "runs") / run_id
    return root, root / "artifacts"


def main(argv=None):
    """Resolve study-local paths before the harness writes its first manifest."""
    from harness.cli import build_parser, main as harness_main

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args([EXPERIMENT, *argv])
    if args.results_dir is not None or args.artifacts_dir is not None:
        parser.error("sweep paths are managed inside the study; use --run-id instead")
    run_id = args.run_id or new_run_id()
    results, artifacts = run_paths(run_id, smoke=args.smoke, publish_smoke=args.publish_smoke)
    print(f"[sweep] run folder: {results}", flush=True)
    return harness_main([EXPERIMENT, *argv, "--run-id", run_id,
                         "--results-dir", str(results), "--artifacts-dir", str(artifacts)])


if __name__ == "__main__":
    raise SystemExit(main())
