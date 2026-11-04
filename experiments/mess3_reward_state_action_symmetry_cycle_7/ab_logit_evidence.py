"""Test whether A/B-induced policy changes track fine reward-state evidence."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.probes.controls import fit_grouped_affine, paired_comparison
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_analysis import (
    coarse_filter_spec,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)


EXPECTED_MODULE_SHA256 = (
    "6ff2b60203e298c7967cd16e2a0ad5676fbc22c288e02e2515a38a06a635cc5b"
)
EVALUATION_SEED = 20261103
WARMUP = 128
EPISODES_PER_SPLIT = 64


def condition(prior: np.ndarray, emission: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    weighted = prior * emission[:, tokens].T
    return weighted / weighted.sum(axis=1, keepdims=True)


@torch.inference_mode()
def collect_pairs(
    module,
    *,
    split: int,
    episodes: int,
    seed: int,
    batch_size: int = 16,
    horizon: int = 1024,
) -> dict[str, np.ndarray]:
    """Keep actions fixed to the baseline; refilter both symbol histories."""
    config = environment_config(2)
    config.update(
        randomize_first_episode_length=False,
        episode_length=horizon,
        diagnostics={"belief": True},
    )
    initial, coarse_emission, coarse_transitions = coarse_filter_spec()
    coarse_transitions = np.stack([coarse_transitions[a] for a in range(3)])
    rows: dict[str, list[np.ndarray]] = {}
    module.eval().to("cpu")
    for start in range(0, episodes, batch_size):
        ids = np.arange(start, min(start + batch_size, episodes))
        count = len(ids)
        envs = [HMMEnv(config) for _ in ids]
        try:
            resets = [
                env.reset(
                    seed=int(np.random.SeedSequence([seed, split, int(i), 0])
                             .generate_state(1)[0])
                )
                for env, i in zip(envs, ids)
            ]
            observations, infos = map(list, zip(*resets))
            draws = [
                np.random.default_rng(np.random.SeedSequence([seed, split, int(i), 1]))
                for i in ids
            ]
            emission = np.asarray(envs[0].model.emission_matrix, dtype=np.float64)
            transitions = np.stack(
                [envs[0].task.transition_matrix_for_action(a) for a in range(3)]
            )
            full = np.tile(envs[0].model.initial_distribution, (2 * count, 1))
            coarse = np.tile(initial, (count, 1))
            state = _initial_state(module, 2 * count, torch.device("cpu"))
            previous = np.full(count, -1, dtype=np.int64)
            for step in range(horizon):
                baseline = np.stack(observations)
                tokens = baseline[:, :3].argmax(axis=1)
                coins = np.array([rng.integers(2) for rng in draws])
                randomized_tokens = np.where(tokens == 2, 2, coins)
                randomized = baseline.copy()
                randomized[:, :3] = np.eye(3)[randomized_tokens]
                full = condition(full, emission, np.concatenate((tokens, randomized_tokens)))
                coarse = condition(coarse, coarse_emission, (tokens == 2).astype(int))
                np.testing.assert_allclose(
                    full[:count], np.stack([info["belief_current"] for info in infos]),
                    rtol=0, atol=1e-11,
                )
                inputs = torch.from_numpy(np.concatenate((baseline, randomized)))
                embeddings, state = module.encode_step(inputs, state)
                logits = module.action_distribution_inputs(embeddings).double().numpy()
                probabilities = torch.from_numpy(logits).softmax(dim=-1).numpy()
                actions = logits[:count].argmax(axis=1)
                values = {
                    "episode": ids,
                    "step": np.full(count, step),
                    "token": tokens,
                    "randomized_token": randomized_tokens,
                    "previous_action": previous,
                    "coarse": coarse[:, 1],
                    "rho": full[:count, 2],
                    "randomized_rho": full[count:, 2],
                    "logit": logits[:count, 0] - logits[:count, 1],
                    "randomized_logit": logits[count:, 0] - logits[count:, 1],
                    "total_variation": 0.5 * np.abs(
                        probabilities[:count] - probabilities[count:]
                    ).sum(axis=1),
                    "greedy_disagreement": actions != logits[count:].argmax(axis=1),
                }
                for name, value in values.items():
                    rows.setdefault(name, []).append(value.copy())
                if step + 1 < horizon:
                    outcomes = [env.step(int(a)) for env, a in zip(envs, actions)]
                    observations = [outcome[0] for outcome in outcomes]
                    infos = [outcome[4] for outcome in outcomes]
                    if any(outcome[2] or outcome[3] for outcome in outcomes):
                        raise RuntimeError("episode ended before the requested horizon")
                    full = np.einsum(
                        "ni,nij->nj", full, transitions[np.tile(actions, 2)]
                    )
                    coarse = np.einsum("ni,nij->nj", coarse, coarse_transitions[actions])
                    previous = actions
        finally:
            for env in envs:
                env.close()
        print(f"split {split}: {start + count}/{episodes} episodes", flush=True)
    return {name: np.concatenate(parts) for name, parts in rows.items()}


def nuisance_features(data: dict[str, np.ndarray]) -> np.ndarray:
    """Cubic coarse-belief curves within token-pair/previous-action cells."""
    cells = (
        (data["token"] * 3 + data["randomized_token"]) * 3
        + data["previous_action"]
    ).astype(int)
    if np.any(data["previous_action"] < 0):
        raise ValueError("nuisance features require post-reset rows")
    indicators = np.eye(27)[cells]
    coarse = data["coarse"] - 0.5
    powers = np.column_stack([coarse ** power for power in range(4)])
    return (indicators[:, :, None] * powers[:, None, :]).reshape(len(cells), -1)


def association(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict:
    """Bootstrap paired episodes with training-fitted nuisance functions fixed."""
    _, codes = np.unique(groups, return_inverse=True)
    stats = np.stack(
        [np.bincount(codes, weights=v) for v in
         (np.ones(len(x)), x, y, x * x, y * y, x * y)],
        axis=1,
    )

    def calculate(s):
        n, sx, sy, sxx, syy, sxy = np.moveaxis(s, -1, 0)
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        cov = sxy - sx * sy / n
        return cov / vx, cov / np.sqrt(vx * vy)

    slope, correlation = calculate(stats.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(stats), size=(resamples, len(stats)))
    slopes, correlations = calculate(stats[draws].sum(axis=1))
    return {
        "slope": float(slope),
        "slope_ci95": np.quantile(slopes, [0.025, 0.975]).tolist(),
        "correlation": float(correlation),
        "correlation_ci95": np.quantile(correlations, [0.025, 0.975]).tolist(),
        "n_episodes": len(stats),
        "n_rows": len(x),
        "bootstrap": "paired episodes; training-fitted nuisance models fixed",
    }


def analyze(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    warmup: int = WARMUP,
    resamples: int = 2000,
    seed: int = EVALUATION_SEED,
) -> dict:
    results = {}
    for subset in ("all", "same_current_token", "current_C"):
        selected = []
        for data in (train, test):
            mask = data["step"] >= warmup
            if subset == "same_current_token":
                mask &= data["token"] == data["randomized_token"]
            elif subset == "current_C":
                mask &= data["token"] == 2
            selected.append({k: v[mask] for k, v in data.items()})
        fit, heldout = selected
        x_fit = fit["rho"] - fit["randomized_rho"]
        y_fit = fit["logit"] - fit["randomized_logit"]
        x_test = heldout["rho"] - heldout["randomized_rho"]
        y_test = heldout["logit"] - heldout["randomized_logit"]
        z_fit, z_test = nuisance_features(fit), nuisance_features(heldout)
        weight, bias, diagnostics = fit_grouped_affine(
            z_fit, np.column_stack((x_fit, y_fit)), fit["episode"],
            seed=seed, rconds=(1e-10,),
        )
        residual_fit = np.column_stack((x_fit, y_fit)) - (z_fit @ weight + bias)
        residual_test = np.column_stack((x_test, y_test)) - (z_test @ weight + bias)
        x0, y0 = residual_fit.T
        beta = float(x0 @ y0 / (x0 @ x0))
        baseline = z_test @ weight[:, 1] + bias[1]
        prediction = baseline + beta * residual_test[:, 0]
        comparison = paired_comparison(
            prediction, baseline, y_test, heldout["episode"],
            seed=seed, n_resamples=resamples,
        )
        results[subset] = {
            "fit_slope": beta,
            "heldout_partial_association": association(
                *residual_test.T, heldout["episode"], seed=seed, resamples=resamples,
            ),
            "heldout_incremental_prediction": comparison,
            "nuisance_fit": diagnostics,
            "rms_evidence_change": float(np.sqrt(np.mean(x_test ** 2))),
            "rms_logit_change": float(np.sqrt(np.mean(y_test ** 2))),
        }
    results["test_rollout"] = {
        "n_decisions_including_warmup": len(test["step"]),
        "mean_total_variation": float(test["total_variation"].mean()),
        "greedy_disagreements": int(test["greedy_disagreement"].sum()),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.raw_output):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    manifest = json.loads(args.run_manifest.read_text())
    if manifest["runtime"]["seed"] != 42 or "variant_2_ctx32_ent" not in manifest["run_id"]:
        parser.error("expected the original Reward-State Quotient seed-42 run")
    sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha256(args.checkpoint / "module_state.pkl") != EXPECTED_MODULE_SHA256:
        parser.error("checkpoint does not match the archived intervention module")
    torch.set_num_threads(2)
    module = load_module_only(args.checkpoint)
    if module.encoder.lookback != WARMUP:
        raise ValueError("expected the original 128-step effective receptive field")
    episodes = 4 if args.smoke else EPISODES_PER_SPLIT
    horizon = 256 if args.smoke else 1024
    data = [
        collect_pairs(
            module, split=split, episodes=episodes, horizon=horizon,
            seed=EVALUATION_SEED,
        )
        for split in (0, 1)
    ]
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.raw_output,
        **{f"{split}_{key}": value for split, rows in zip(("train", "test"), data)
           for key, value in rows.items()},
    )
    results = analyze(*data, resamples=100 if args.smoke else 2000)
    repository = Path(__file__).resolve().parents[2]
    library = Path(inspect.getfile(load_module_only)).resolve().parents[1]
    revisions = {
        name: {
            "commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "dirty": bool(subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True,
            ).strip()),
        }
        for name, root in (("experiment", repository), ("library", library))
    }
    report = {
        "protocol": {
            "condition": "Reward-State Quotient",
            "training_seed": 42,
            "checkpoint": "final checkpoint_000025, 10192528 sampled steps",
            "hypothesis": (
                "original minus randomized Noop-minus-Tilt-X logit increases "
                "with original minus randomized full-filter reward-state evidence"
            ),
            "expected_sign": "positive",
            "evaluation_seed": EVALUATION_SEED,
            "episodes_per_split": episodes,
            "episode_length": horizon,
            "warmup": WARMUP,
            "policy": "greedy baseline controls the sole environment",
            "randomization": "independent fair A/B replacement; C and baseline actions fixed",
            "evidence": (
                "rho minus randomized-history rho; common coarse belief cancels. "
                "Randomized-history rho applies the ORIGINAL emission model to "
                "the altered symbols; it is NOT the posterior for a known randomized channel."
            ),
            "controls": (
                "separate cubic coarse-belief curves in original-token, "
                "randomized-token, previous-action cells; fit on training episodes only"
            ),
            "regression": (
                "residualize evidence and logit differences against the same controls; "
                "fit one signed slope on training episodes and score held-out episodes"
            ),
            "subsets": "all primary; same current token and current C secondary",
            "limitations": (
                "one seed and final checkpoint; altered histories are distribution-shifted; "
                "association does not identify filtering computation or establish causality"
            ),
        },
        "provenance": {
            "run_id": manifest["run_id"],
            "training_revisions": manifest["git"],
            "analysis_revisions": revisions,
            "module_state_sha256": EXPECTED_MODULE_SHA256,
            "run_manifest_sha256": sha256(args.run_manifest),
            "analysis_source_sha256": sha256(__file__),
            "raw_data_sha256": sha256(args.raw_output),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        name: {
            "fit_slope": result["fit_slope"],
            **result["heldout_partial_association"],
            "prediction_improvement": result["heldout_incremental_prediction"]["mse_improvement"],
        }
        for name, result in results.items() if name != "test_rollout"
    }, indent=2))


if __name__ == "__main__":
    main()
