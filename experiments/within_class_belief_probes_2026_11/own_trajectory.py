"""Own-policy-only PR219 trajectory and independently resampled contrasts."""

import hashlib
import json
import subprocess

import jax
import numpy as np

from .metrics import refit, score, bootstrap
from .nuisance_v2 import ROOT, RAW, selected_runs
from .temp15 import restore, rows
from .temp15_analyze import perturb, replay
from .trajectory import collect

OUTPUT=ROOT/"results/ladder_nuisance_v2_temp15_own_trajectory.json"


def statistics(y, prediction, classes, episodes):
    y,prediction=np.asarray(y,float),np.asarray(prediction,float)
    classes,episodes=np.asarray(classes),np.asarray(episodes)
    if y.ndim!=2 or y.shape!=prediction.shape or not len(y) or classes.shape!=(len(y),) or episodes.shape!=(len(y),):
        raise ValueError("nonempty aligned targets, predictions, classes and episodes required")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or not np.isin(classes,[0,1,2]).all():
        raise ValueError("finite targets/predictions and three-action labels required")
    _,group=np.unique(episodes,return_inverse=True)
    count=np.zeros((group.max()+1,3));total=np.zeros((*count.shape,y.shape[1]))
    square=np.zeros_like(count);error=np.zeros_like(count)
    for k in range(3):
        mask=classes==k
        np.add.at(count[:,k],group[mask],1)
        np.add.at(total[:,k],group[mask],y[mask])
        np.add.at(square[:,k],group[mask],(y[mask]**2).sum(1))
        np.add.at(error[:,k],group[mask],((y[mask]-prediction[mask])**2).sum(1))
    return count,total,square,error


def draws(stats,n=10000,seed=20261120,proportions=None):
    if n<1:raise ValueError("positive bootstrap count required")
    count,total,square,error=stats
    w=np.random.default_rng(seed).multinomial(len(count),np.full(len(count),1/len(count)),size=n)
    c=w@count;s=np.einsum("rg,gkd->rkd",w,total)
    variance=w@square-(s*s).sum(2)/np.maximum(c,1)
    errors=w@error
    return ratios(c,variance,errors,proportions)


def ratios(count,variance,error,proportions=None):
    with np.errstate(divide="ignore",invalid="ignore"):
        per=1-error/np.where(variance>1e-20,variance,np.nan)
        if proportions is None:
            denom=variance.sum(-1);numer=error.sum(-1)
        else:
            q=np.asarray(proportions,float)
            if q.shape!=(3,) or np.any(q<0) or not np.isclose(q.sum(),1):
                raise ValueError("normalized three-class proportions required")
            denom=np.sum(np.where(q>0,variance/np.maximum(count,1)*q,0),axis=-1)
            numer=np.sum(np.where(q>0,error/np.maximum(count,1)*q,0),axis=-1)
            missing=np.any((q>0)&(count==0),axis=-1)
            denom=np.where(missing,np.nan,denom)
        pooled=1-numer/np.where(denom>1e-20,denom,np.nan)
    return pooled,per


def interval(values):
    values=np.asarray(values);valid=values[np.isfinite(values)]
    return {"ci95":np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
            "valid_resamples":len(valid),"n_resamples":values.size}


def point_score(stats,proportions=None):
    c,s,square,error=[x.sum(0) for x in stats]
    variance=square-(s*s).sum(-1)/np.maximum(c,1)
    pooled,per=ratios(c,variance,error,proportions)
    return float(pooled),[float(v) if np.isfinite(v) else None for v in per]


def contrast(left,right,n=10000,proportions=None):
    a,ac=draws(left,n,20261120,proportions)
    b,bc=draws(right,n,20261121,proportions)
    av,ap=point_score(left,proportions);bv,bp=point_score(right,proportions)
    return {"left_r2":av,"right_r2":bv,"difference":av-bv,**interval(a-b),
            "left_seed":20261120,"right_seed":20261121,
            "per_class":{str(k):{"left_r2":ap[k],"right_r2":bp[k],
                "difference":ap[k]-bp[k],**interval(ac[:,k]-bc[:,k])}
                for k in range(3) if ap[k] is not None and bp[k] is not None}}


def main():
    baseline=json.loads((ROOT/"results/ladder_nuisance_v2_temp15_trajectory.json").read_text())
    paths={(r.parent.parent.name,json.loads((r/"resolved_recipe.json").read_text())["study"]):r for r in selected_runs()}
    arms=[]
    for source in baseline["arms"]:
        run=paths[(source["arm"],source["study"])];raw=RAW/source["study"]
        directory=raw/source["arm"];cache=directory/"own_trajectory";cache.mkdir(exist_ok=True)
        peak=max(source["points"],key=lambda p:p["own_history"]["metrics"]["full_h"]["belief"]["r2"])["step"]
        stages={0,peak,source["final_reference_step"]};points=[];stats={}
        for p in source["points"]:
            step=p["step"];spec,process,params,provenance=restore(run,step,raw_root=raw)
            data=collect(run,step,spec,process,params,raw)
            beliefs=[rows(d,"belief",n) for d,n in zip(data,(60000,80000))]
            classes=[rows(d,"action",n) for d,n in zip(data,(60000,80000))]
            h=[rows(d,"embedding",n) for d,n in zip(data,(60000,80000))]
            targets=[np.concatenate([b,b[:,:1]-b[:,1:2],b[:,2:3]],axis=1).astype(float) for b in beliefs]
            path=cache/f"{step}_prediction.npz"
            archived=directory/"final_predictions.npz"
            if path.exists():
                with np.load(path) as saved:prediction=saved["prediction"]
            elif step==source["final_reference_step"] and archived.exists():
                with np.load(archived) as saved:prediction=saved["full_h"][:,:3]
                np.savez_compressed(path,prediction=prediction)
            else:
                prediction=refit(h[0],targets[0],h[1],*classes)[:,:3]
                np.savez_compressed(path,prediction=prediction)
            valid=np.isfinite(prediction).all(1);episodes=np.arange(80000)//960
            stat=statistics(beliefs[1][valid],prediction[valid],classes[1][valid],episodes[valid]);stats[step]=stat
            point={"step":step,"provenance":provenance,**p["own_history"],"greedy_reward":p["greedy_reward"],"controls":{}}
            np.testing.assert_allclose(score(beliefs[1][valid],prediction[valid],classes[1][valid])["r2"],point["metrics"]["full_h"]["belief"]["r2"],rtol=0,atol=1e-12)
            _,per=draws(stat,2000)
            point["per_class_intervals"]={str(k):interval(per[:,k]) for k in np.unique(classes[1])}
            if step in stages:
                for index,mode in enumerate(["history_shuffle","random_input","label_permutation"]):
                    cp=cache/f"{step}_{mode}.json"
                    if cp.exists():control=json.loads(cp.read_text())
                    else:
                        if step==source["final_reference_step"] and archived.exists():
                            with np.load(archived) as saved:pred=saved[mode][:,:3]
                        elif mode=="label_permutation":
                            pred=refit(h[0],targets[0],h[1],*classes,permute_seed=853)[:,:3]
                        else:
                            changed=[perturb(d["obs"],mode,20261106+10*index+i) for i,d in enumerate(data)]
                            features=[rows({"h":np.asarray(replay(spec,params,obs))},"h",n) for obs,n in zip(changed,(60000,80000))]
                            pred=refit(*[features[0],targets[0],features[1]],*classes)[:,:3]
                        mask=np.isfinite(pred).all(1)
                        control={**score(beliefs[1][mask],pred[mask],classes[1][mask]),"excluded_rows":int((~mask).sum()),
                            "bootstrap":bootstrap(beliefs[1][mask],pred[mask],classes[1][mask],episodes[mask],n=2000)}
                        np.savez_compressed(cache/f"{step}_{mode}_prediction.npz",prediction=pred)
                        cp.write_text(json.dumps(control,indent=2,allow_nan=False)+"\n")
                    point["controls"][mode]=control
            points.append(point);print("OWN",source["study"],source["arm"],step,flush=True)
        final=source["final_reference_step"]
        peak_counts=stats[peak][0].sum(0);final_counts=stats[final][0].sum(0)
        shared=(peak_counts>0)&(final_counts>0);q=np.where(shared,final_counts,0);q=q/q.sum()
        identifiable=bool(shared[final_counts>0].all())
        mixed=contrast(stats[peak],stats[final],proportions=q)
        mixed.update(full_final_class_mix_identifiable=identifiable,
                     full_final_weighted_difference=mixed["difference"] if identifiable else None,
                     missing_peak_classes=np.flatnonzero((final_counts>0)&~shared).tolist())
        arms.append({k:source[k] for k in ["study","arm","training_epsilon","run_id","b2","checkpoint_inventory"]}|{
            "points":points,"peak_step":peak,"peak_checkpoint":f"checkpoints/step_{peak:010d}.npz",
            "peak_minus_final":contrast(stats[peak],stats[final]),"init_minus_peak":contrast(stats[0],stats[peak]),
            "class_standardized_peak_minus_final":{
                "shared_classes":np.flatnonzero(shared).tolist(),"final_proportions_on_shared_support":q.tolist(),
                "final_mass_covered":float(final_counts[shared].sum()/final_counts.sum()),
                "peak_mass_covered":float(peak_counts[shared].sum()/peak_counts.sum()),
                **mixed}})
        jax.clear_caches()
    report={"schema_version":1,"protocol":{
        "primary":"Own-policy histories only; no fixed-history scores used. Same 12 selected checkpoints as original trajectory.",
        "sampling":"60K independent fit / 80K test rows per checkpoint; 64 warmup, full 1024-step episodes, 84 test clusters including truncated last episode; ridge 1e-6; greedy actions; full three-state belief, delta=b1-b2, rho=b3.",
        "bootstrap":"2000 pointwise draws; 10000 unpaired contrast draws with independently resampled episodes, class means recomputed and probes fixed. Underlying checkpoint rollouts reuse PRNG stream keys: independent resampling is the requested convention, not a claim that these common-random-number samples are strictly independent.",
        "class_standardization":"R2=1-sum(q_k*SSE_k/n_k)/sum(q_k*SST_k/n_k). q is final test class proportions, restricted and renormalized to shared peak/final support. Zero-presence bootstrap replicates are invalid, not silently renormalized. This controls class proportions, not within-class visitation or target variance.",
        "selection":"Peak maximizes observed own-history full-h within-class R2 among 12 checkpoints; contrasts condition on selected peak, no selection/multiplicity correction; one training seed only.",
        "controls":"Own-history init/peak/final only. One fixed input-control draw per split; seeds 20261106+10*index+split, labels853. Targets/classes remain original own-policy targets/classes. Input controls are distribution shifts, not required zero nulls.",
        "contrast_signs":"peak_minus_final=peak-final; init_minus_peak=init-peak (negative means rise)."},
        "provenance":baseline["provenance"]|{"own_analysis_parent_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"own_source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"own_trajectory.py",ROOT/"own_trajectory_report.py"]}},"arms":arms}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("Wrote",OUTPUT,flush=True)


if __name__=="__main__":main()
