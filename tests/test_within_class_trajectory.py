import numpy as np
import pytest

from experiments.within_class_belief_probes_2026_11.trajectory import select_steps, reward_interval, evaluate


def test_prespecified_checkpoint_selection_covers_early_midpoint_and_final():
    steps=np.arange(102)*19922944
    selected=select_steps(steps)
    assert len(selected)==12 and selected==sorted(set(selected))
    assert selected[0]==0 and selected[-1]==steps[-1] and steps[50] in selected
    assert steps[1] in selected
    assert select_steps([20,0,10,10],12)==[0,10,20]
    for count in range(3,20):
        chosen=select_steps(steps,count)
        assert len(chosen)==count and chosen[0]==steps[0] and chosen[-1]==steps[-1]
    with pytest.raises(ValueError):select_steps([])
    with pytest.raises(ValueError):select_steps([0,1,2],2)


def test_reward_interval_weights_unequal_episode_lengths():
    reward=np.array([0.,1.,1.,1.,0.]);episodes=np.array([0,0,1,1,2])
    result=reward_interval(reward,episodes,n=100,seed=4)
    weights=np.random.default_rng(4).multinomial(3,np.full(3,1/3),size=100)
    means=(weights@np.array([1.,2.,0.]))/(weights@np.array([2.,2.,1.]))
    np.testing.assert_allclose(result["ci95"],np.quantile(means,[.025,.975]))
    assert result["mean"]==.6 and result["n_episodes"]==3
    assert reward_interval(np.ones(5),episodes,n=10)["ci95"]==[1.,1.]
    with pytest.raises(ValueError):reward_interval([],[])
    with pytest.raises(ValueError):reward_interval(reward,episodes[:-1])
    with pytest.raises(ValueError):reward_interval(reward,episodes,n=0)


def test_trajectory_metrics_refit_and_bootstrap_all_requested_views():
    rng=np.random.default_rng(3)
    h=[rng.normal(size=(n,4)) for n in [40,20]]
    b=[rng.dirichlet(np.ones(3),size=n) for n in [40,20]]
    classes=[np.arange(n)%2 for n in [40,20]]
    w=rng.normal(size=(4,3));bias=np.zeros(3)
    result,pred=evaluate(h,b,classes,w,bias,n=10)
    assert pred.shape==(20,3) and np.isfinite(pred).all()
    assert result["head_null_dimension"]==2 and not result["one_class"]
    assert set(result["metrics"])=={"full_h","head_blind","logits","global_belief"}
    assert set(result["metrics"]["full_h"])=={"belief","delta","rho"}
    for targets in result["metrics"].values():
        for metric in targets.values():
            assert metric["n"]==20 and metric["excluded_rows"]==0
            assert metric["bootstrap"]["n_resamples"]==10
    single=[np.zeros(n,dtype=int) for n in [40,20]]
    one,_=evaluate(h,b,single,w,bias,n=10)
    assert one["one_class"]
    np.testing.assert_allclose(one["metrics"]["full_h"]["belief"]["r2"],one["metrics"]["global_belief"]["belief"]["r2"])
    with pytest.raises(ValueError):evaluate([h[0][:0],h[1]],[b[0][:0],b[1]],[classes[0][:0],classes[1]],w,bias,n=10)
