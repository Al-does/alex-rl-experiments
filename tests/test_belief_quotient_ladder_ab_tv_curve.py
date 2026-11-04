import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_tv_curve import (
    checkpoint_entries,
    evaluation_spec,
    evaluate_checkpoint,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_on_policy_tv import collect
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_pattern_checkpoint import sha256
from experiments.mess3_reward_state_action_symmetry_cycle_7.plot_ab_tv_curve import (
    curve_rows,
)


def inventory_fixture():
    names = ["initial_checkpoint"] + [f"checkpoint_{i:06d}" for i in range(26)]
    manifest = {
        "files": [{
            "relative_path": f"{name}/learner_group/learner/rl_module/default_policy/module_state.pkl",
        } for name in names]
    }
    progress = [{
        "training_iteration": str(i),
        "env_runners/num_env_steps_sampled_lifetime": str(i * 123456),
    } for i in range(1, 27)]
    return manifest, progress


def test_checkpoint_inventory_includes_zero_step_and_uses_actual_progress_steps():
    manifest, progress = inventory_fixture()
    entries = checkpoint_entries(manifest, progress)
    assert len(entries) == 27
    assert entries[0] == {
        "name": "initial_checkpoint", "training_iteration": 0, "agent_steps": 0,
    }
    assert entries[1]["agent_steps"] == 123456
    assert entries[-1]["agent_steps"] == 26 * 123456
    assert entries[-1]["training_iteration"] == 26


def test_inventory_rejects_missing_checkpoints_and_nonincreasing_steps():
    manifest, progress = inventory_fixture()
    missing = copy.deepcopy(manifest)
    missing["files"].pop()
    with pytest.raises(ValueError, match="checkpoints 0-25"):
        checkpoint_entries(missing, progress)
    progress[2]["env_runners/num_env_steps_sampled_lifetime"] = "1"
    with pytest.raises(ValueError, match="strictly increasing"):
        checkpoint_entries(manifest, progress)


def report_fixture():
    manifest, progress = inventory_fixture()
    result = {
        "total_variation": {"mean": 0.1, "ci95": [0.08, 0.12]},
        "rms_logit_preference_change": 0.2,
        "greedy_disagreements": 2, "n_decisions": 100,
        "baseline_entropy_nats": 0.5,
        "baseline_mean_max_action_probability": 0.7,
    }
    return {
        "points": [{
            "checkpoint": entry,
            "results": {"greedy": result, "sampled": result},
        } for entry in checkpoint_entries(manifest, progress)]
    }


def test_plot_rows_preserve_all_checkpoints_both_modes_and_intervals():
    rows = curve_rows(report_fixture())
    assert len(rows) == 54
    assert rows[0]["checkpoint"] == "initial_checkpoint"
    assert rows[0]["agent_steps"] == 0
    assert rows[-1]["mode"] == "sampled"
    assert rows[-1]["ci95_high"] == 0.12
    assert rows[-1]["greedy_disagreement_fraction"] == 0.02


def test_plot_rejects_incomplete_curve_and_invalid_intervals():
    report = report_fixture()
    incomplete = copy.deepcopy(report)
    incomplete["points"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        curve_rows(incomplete)
    report["points"][0]["results"]["greedy"]["total_variation"]["ci95"] = [0.2, 0.3]
    with pytest.raises(ValueError, match="invalid TV"):
        curve_rows(report)


def test_resume_requires_matching_protocol_and_intact_raw_data(tmp_path):
    raw = tmp_path / "raw.npz"
    raw.write_bytes(b"verified fixture")
    entry = {"name": "initial_checkpoint"}
    spec = {"driver_sha256": "original"}
    point = {
        "checkpoint": entry, "spec": spec,
        "raw": {"greedy": {
            "path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        }},
    }
    folder = tmp_path / "points"
    folder.mkdir()
    (folder / "initial_checkpoint.json").write_text(json.dumps(point))
    assert evaluate_checkpoint(entry, str(tmp_path), spec, None) == point
    with pytest.raises(ValueError, match="different protocol"):
        evaluate_checkpoint(entry, str(tmp_path), {"driver_sha256": "changed"}, None)
    raw.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="raw-data hash mismatch"):
        evaluate_checkpoint(entry, str(tmp_path), spec, None)


def test_collector_hash_uses_underlying_source_not_torch_decorator():
    spec = evaluation_spec(False)
    source = Path(inspect.getfile(inspect.unwrap(collect)))
    assert source.name == "ab_on_policy_tv.py"
    assert spec["collector_sha256"] == sha256(source)
