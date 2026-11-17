"""Validate the complete checkpoint grid and compare transformer results to PR #23."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .analyze import HERE, ROOT, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    revision = "origin/devin/1794867412-eps02-decoding-figures"
    def reference(name):
        path = f"experiments/persistent_nuisance_eps02/{name}.json"
        raw = subprocess.check_output(["git", "-C", str(args.paper), "show", f"{revision}:{path}"])
        return json.loads(raw), {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    probes, probe_source = reference("ladder_reprobes")
    comparison, comparison_source = reference("coarse_comparison")
    rows = [read(path) for path in sorted((HERE / "results").glob("variant_*.json"))]
    expected = sorted(set(range(0, 102, 5)) | {1, 2, 3, 4, 101})
    deltas, finals = [], []
    for architecture in ("transformer", "gru128", "gru92", "memoryless"):
        for variant in (1, 2, 3):
            group = sorted([r for r in rows if r["architecture"] == architecture and r["condition"] == f"variant_{variant}"], key=lambda r: r["checkpoint_index"])
            assert [r["checkpoint_index"] for r in group] == expected
            for row in group:
                assert row["seed"] == 42
                assert row["checkpoint_step"] == row["checkpoint_index"] * 19922944
                for score in row["scores"].values():
                    assert np.isfinite(score["error"]) and score["error"] > 0
                if architecture == "transformer":
                    old, = [r for r in probes["conditions"][row["condition"]]["seeds"]["42"] if r["checkpoint_step"] == row["checkpoint_step"]]
                    assert row["checkpoint_sha256"] == old["checkpoint_sha256"]
                    targets = [("b", "belief")]
                    if "coarse_b" in old["scalars"]:
                        targets.append(("c", "coarse_b"))
                    for target, key in targets:
                        a, b = row["scores"][target]["error"], old["scalars"][key]["error"]
                        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)
                        deltas.append(abs(a-b))
            finals.append(group[-1])
    for old in comparison["results"]:
        if old["seed"] != 42:
            continue
        row, = [r for r in finals if r["architecture"] == "transformer" and r["condition"] == old["condition"]]
        for key, target in (("belief", "b"), ("merged_token", "c")):
            np.testing.assert_allclose(row["scores"][target]["error"], old[key]["error"], rtol=1e-8, atol=1e-10)
    for row in finals:
        transformer, = [r for r in finals if r["architecture"] == "transformer" and r["condition"] == row["condition"]]
        row["reward_gap_to_transformer"] = row["greedy_reward_per_step"] - transformer["greedy_reward_per_step"]
        row["reward_below_90pct_transformer"] = row["greedy_reward_per_step"] < .9 * transformer["greedy_reward_per_step"]
    summary = {"protocol": {"seed": 42, "policy": "greedy", "fit_rows": 60000, "test_rows": 80000,
                 "warmup_per_episode": 64, "episode_length": 1024, "ridge": 1e-6,
                 "matmul_precision": "highest", "probe_key": "split(fold_in(PRNGKey(42), 2))",
                 "b": "three-state token/action filter", "c": "merged-A/B token/action filter state-3 probability; exact quotient only for variants 1,2",
                 "representations": {"transformer": "post-final-LayerNorm embedding", "gru128": "top recurrent hidden state (pre-final-LayerNorm)", "gru92": "top recurrent hidden state (pre-final-LayerNorm)", "memoryless": "post-final-LayerNorm 64-wide residual MLP embedding; current token only"},
                 "reward": "held-out 80k greedy probe rows, excludes 64 warmup steps per episode; not complete-episode occupancy",
                 "reward_gate": "descriptive flag below 90% of same-variant transformer reward; not a significance test"},
               "analysis_git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               "analysis_results_merge": "d20d13dc172b4959477c4ea12f919520209144f2 (uncommitted merge)",
               "paper_reference_revision": subprocess.check_output(["git", "-C", str(args.paper), "rev-parse", revision], text=True).strip(),
               "reference_files": [probe_source, comparison_source],
               "transformer_comparisons": len(deltas), "max_abs_transformer_error_difference": max(deltas),
               "archived_reward_comparison": {
                   "max_abs_difference_all_checkpoints": max(abs(r["greedy_reward_per_step"] - r["saved_greedy_reward_per_step"]) for r in rows),
                   "max_abs_difference_finals": max(abs(r["greedy_reward_per_step"] - r["saved_greedy_reward_per_step"]) for r in finals),
                   "note": "CPU highest-precision greedy rollouts need not match archived GPU rollouts. Cause of differences not investigated; both measurements retained."},
               "checkpoint_indices": expected, "result_count": len(rows), "finals": finals,
               "curves": [{k: r[k] for k in ("architecture", "condition", "checkpoint_index", "checkpoint_step", "scores", "greedy_reward_per_step")} for r in rows]}
    write(HERE / "results" / "summary.json", summary)
    print(f"Validated {len(rows)} checkpoint results, 12 arms x {len(expected)} checkpoints; all checkpoint bytes/SHA-256 verified against B2 manifests.")
    print(f"PR23 transformer comparisons: {len(deltas)}; max absolute error difference: {max(deltas):.12g}")
    print("architecture condition b_error c_error greedy_reward gap_to_transformer gate_below_90pct")
    for r in finals:
        print(r["architecture"], r["condition"], f'{r["scores"]["b"]["error"]:.9f}', f'{r["scores"]["c"]["error"]:.9f}', f'{r["greedy_reward_per_step"]:.6f}', f'{r["reward_gap_to_transformer"]:+.6f}', r["reward_below_90pct_transformer"])


if __name__ == "__main__":
    main()
