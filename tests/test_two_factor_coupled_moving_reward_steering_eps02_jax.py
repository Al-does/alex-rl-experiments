import numpy as np

from experiments.two_factor_coupled_moving_reward_steering_eps02_jax_2026_11 import steering as s
from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11 import steering as coupled


def test_process_is_v2_dynamics_with_noisy_state_3_emission():
    process, v2 = s.source_process(), coupled.base.source_process("reward_both")
    np.testing.assert_array_equal(np.asarray(process.transitions), np.asarray(v2.transitions))
    np.testing.assert_allclose(np.asarray(process.emission)[8].reshape(3, 3),
                               np.outer([.01, .01, .98], [.01, .01, .98]), atol=1e-7)
    assert np.asarray(v2.emission)[8, 8] == 1
    for kind in coupled.ACTION_MAPS:
        np.testing.assert_array_equal(coupled.q_tables("xor", kind, process=process),
                                      coupled.q_tables("xor", kind))


def test_source_runs_are_the_15_joint_reward_seeds():
    runs = s.source_runs()
    assert [run["seed"] for run in runs] == list(range(42, 57))
    assert all(run["remote_prefix"].startswith(f"experiments/{s.SOURCE_STUDY}/{s.SOURCE_LEAF}/")
               for run in runs)


def test_evaluate_uses_the_supplied_process():
    import jax

    from experiments.belief_quotient_history_jax_2026_11 import models

    spec = models.ModelSpec(kind="transformer", obs_dim=18, n_actions=9, width=8, n_layers=1,
                            context_len=4, chunk_len=4)
    params = models.init_params(spec, jax.random.PRNGKey(0))
    rewards = [coupled.evaluate(spec, params, reward="xor", episodes=4, seed=0, policy="map",
                                process=process)["total_reward"]
               for process in (None, s.source_process())]
    assert rewards[0] != rewards[1]


def test_tuning_curves_average_gain_over_agents():
    from experiments.two_factor_coupled_moving_reward_steering_eps02_jax_2026_11.summarize import tuning_curves

    def report(steered):
        cell = lambda total: {"total_reward": total, "factor_action_change_fraction": [0.0, 0.5]}
        grid = {"discounted_joint": {"0.0": cell(0.5), "1.0": cell(steered)}}
        return {"rewards": {reward: {"strength_tuning": {"grid": grid}} for reward in ("xor", "moving")}}

    row = tuning_curves([report(0.6), report(0.8)])["xor"]["discounted_joint"]["1.0"]
    assert row["total_reward_mean"] == 0.7
    assert row["gain_over_unsteered_mean"] == 0.2
    assert row["gain_over_unsteered_se_across_agents"] == 0.1
    assert row["factor_action_change_fraction_mean"] == [0.0, 0.5]
