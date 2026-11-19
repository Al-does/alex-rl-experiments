import pytest

from experiments.rocksample_jax_2026_10 import belief_steering_continuation_report as report


def comparison(tmp_path, episodes=4):
    mapping = dict(source_run="original", seed=2, continuation_run="continued")
    report.cont.write(tmp_path/"manifest.json", dict(mappings=[mapping], confirmation_episodes=episodes))
    for update in (29, 316):
        protocols = {}
        for protocol, representation in report.PROTOCOLS:
            sites = {}
            for site in range(4):
                conditions = ("intact", "steered", "control") + (("random",) if protocol == "v5" else ())
                sites[str(site)] = [{direction: {c: dict(episodes=episodes, target_probability=.5,
                    mean_return=3., nontarget=dict(good=dict(count=2, picked=1), bad=dict(count=2, picked=0)))
                    for c in conditions} for direction in report.DIRECTIONS} for rock in range(7)]
            if protocol == "v4":
                protocols[protocol] = dict(sites=sites)
            else:
                protocols.setdefault(protocol, dict(representations={}))['representations'][representation] = sites
        state = dict(mapping=mapping, checkpoint=dict(update=update), **protocols)
        report.cont.write(tmp_path/f"original_seed2_update{update}.json", state)


def test_load_and_pooled_comparison(tmp_path):
    comparison(tmp_path)
    _, agents = report.load(tmp_path)
    assert len(agents) == 1
    assert report.pooled(agents[0], 316, "v5", "joint", 2, "Good→Bad") == dict(
        target=.5, mean_return=3., nontarget_good=.5, nontarget_bad=0.)


def test_rejects_empty_comparison(tmp_path):
    report.cont.write(tmp_path/"manifest.json", dict(mappings=[]))
    with pytest.raises(ValueError, match="no agents"):
        report.load(tmp_path)


def test_rejects_incomplete_episode_budget(tmp_path):
    comparison(tmp_path)
    path = tmp_path/"original_seed2_update316.json"
    state = report.cont.read(path)
    state['v4']['sites']['0'][0]['Good→Bad']['steered']['episodes'] = 1
    report.cont.write(path, state)
    with pytest.raises(ValueError, match="episode budget"):
        report.load(tmp_path)
