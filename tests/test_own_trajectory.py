import numpy as np
import pytest

from experiments.within_class_belief_probes_2026_11.own_trajectory import statistics, draws, contrast, point_score


def test_statistics_and_independent_contrast():
    rng=np.random.default_rng(2);y=rng.normal(size=(60,3));p=y+rng.normal(size=y.shape)*.2
    c=np.arange(60)%3;e=np.arange(60)//10
    s=statistics(y,p,c,e)
    value,_=point_score(s)
    sst=sum(((y[c==k]-y[c==k].mean(0))**2).sum() for k in range(3))
    np.testing.assert_allclose(value,1-((y-p)**2).sum()/sst)
    d=contrast(s,s,n=100)
    assert d['difference']==0 and d['left_seed']!=d['right_seed']
    assert d['ci95'][0]<0<d['ci95'][1]
    a,_=draws(s,100,20261120);b,_=draws(s,100,20261121)
    np.testing.assert_allclose(d['ci95'],np.quantile(a-b,[.025,.975]))


def test_class_standardization_removes_count_mix_not_variance():
    y=np.tile(np.array([[-1.],[1.]]),(6,1));c=np.repeat([0,1],[4,8]);p=y.copy();p[c==0]=0
    s=statistics(y,p,c,np.arange(12)//2)
    value,per=point_score(s,np.array([.5,.5,0]))
    np.testing.assert_allclose(value,.5);np.testing.assert_allclose(per[:2],[0,1])
    np.testing.assert_allclose(point_score(s)[0],2/3)
    samples,_=draws(s,100,proportions=np.array([.5,.5,0]))
    assert np.isnan(samples).any()


def test_undefined_and_invalid_inputs():
    with pytest.raises(ValueError):statistics(np.empty((0,3)),np.empty((0,3)),[],[])
    with pytest.raises(ValueError):statistics(np.ones((2,1)),np.ones((3,1)),[0,0],[0,1])
    with pytest.raises(ValueError):statistics(np.ones((2,1)),np.ones((2,1)),[0,4],[0,1])
    s=statistics(np.ones((4,1)),np.zeros((4,1)),np.zeros(4,int),np.arange(4))
    assert np.isnan(point_score(s)[0])
    with pytest.raises(ValueError):draws(s,n=0)
    with pytest.raises(ValueError):draws(s,proportions=[1,1,1])
