"""Own-history and fixed-final-history PR219 checkpoint trajectories."""

import argparse
import hashlib
import json
import platform
import subprocess

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout
from .metrics import affine, refit, score, null_basis, bootstrap
from .nuisance_v2 import ROOT, RAW, REF, selected_runs, validate_recipe
from .temp15 import restore, rows
from .temp15_analyze import replay

OUTPUT = ROOT/"results/ladder_nuisance_v2_temp15_trajectory.json"


def select_steps(steps, count=12):
    steps = sorted(set(steps))
    if not steps or count < 3:
        raise ValueError("saved steps and at least three selected checkpoints required")
    if len(steps) <= count:
        return steps
    last = len(steps)-1
    early = np.geomspace(1, max(1, last/4), max(2, count//2-1))
    late = np.linspace(last/4, last, max(2, count-count//2))
    keep = {0, last//2, last, *np.rint(early).astype(int), *np.rint(late).astype(int)}
    protected = {0, last//2, last}
    while len(keep) > count:
        ordered = sorted(keep)
        remove = min((i for i in ordered if i not in protected),
                     key=lambda i: min(i-ordered[ordered.index(i)-1], ordered[ordered.index(i)+1]-i))
        keep.remove(remove)
    while len(keep) < count:
        a,b = max(zip(sorted(keep)[:-1], sorted(keep)[1:]), key=lambda pair: pair[1]-pair[0])
        keep.add((a+b)//2)
    return [steps[i] for i in sorted(keep)]


def reward_interval(reward, episodes, n=2000, seed=20261106):
    reward, episodes = np.asarray(reward), np.asarray(episodes)
    if reward.ndim != 1 or episodes.shape != reward.shape or not len(reward) or n < 1:
        raise ValueError("aligned reward/episode rows and positive resample count required")
    _, group = np.unique(episodes, return_inverse=True)
    counts = np.bincount(group)
    totals = np.bincount(group, weights=reward)
    weights = np.random.default_rng(seed).multinomial(len(counts), np.full(len(counts), 1/len(counts)), size=n)
    means = (weights @ totals)/(weights @ counts)
    return {"mean": float(reward.mean()), "ci95": np.quantile(means, [.025,.975]).tolist(),
            "n_resamples": n, "n_episodes": len(counts), "seed": seed}


def collect(run, step, spec, process, params, raw):
    data=[]
    for split,count,key in zip(("fit","test"),(60000,80000),jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42),2))):
        path=raw/run.parent.parent.name/f"{step}_{split}.npz"
        saved=None
        if path.exists():
            with np.load(path) as loaded:
                saved={k:loaded[k] for k in loaded.files}
        if saved is None or "reward" not in saved:
            records=_probe_rollout(spec,process,params,key,-(-count//960),1024)
            fresh={k:np.asarray(records[k]) for k in ("obs","action","embedding","belief","reward")}
            if saved is not None:
                for k in saved:
                    np.testing.assert_array_equal(fresh[k],saved[k])
            saved=fresh
            np.savez_compressed(path,**saved)
        data.append(saved)
    return data


def evaluate(embeddings, beliefs, classes, weight, bias, n=2000):
    fit,test=[np.asarray(h,dtype=float) for h in embeddings]
    targets=[np.concatenate([b,b[:,:1]-b[:,1:2],b[:,2:3]],axis=1).astype(float) for b in beliefs]
    basis=null_basis(weight.T)
    features={"full_h":(fit,test),"head_blind":(fit@basis,test@basis),
              "logits":(fit@weight+bias,test@weight+bias)}
    predictions={k:refit(a,targets[0],b,*classes) for k,(a,b) in features.items()}
    predictions["global_belief"]=affine(fit,targets[0][:,:3],test)
    metrics={}
    episodes=np.arange(len(test))//960
    specifications=[("full_h","belief",slice(0,3)),("head_blind","belief",slice(0,3)),
                    ("logits","belief",slice(0,3)),("full_h","delta",slice(3,4)),
                    ("full_h","rho",slice(4,5)),("global_belief","belief",slice(0,3))]
    for feature,target,sl in specifications:
        p=predictions[feature][:,sl]
        valid=np.isfinite(p).all(1)
        c=np.zeros(valid.sum(),dtype=int) if feature=="global_belief" else classes[1][valid]
        y=targets[1][valid,sl]
        metrics.setdefault(feature,{})[target]={**score(y,p[valid],c),
            "excluded_rows":int((~valid).sum()),
            "bootstrap":bootstrap(y,p[valid],c,episodes[valid],n=n),
            "per_class":{str(k):score(y[c==k],p[valid][c==k]) for k in np.unique(c)}}
    return {"metrics":metrics,"fit_counts":np.bincount(classes[0],minlength=3).tolist(),
            "test_counts":np.bincount(classes[1],minlength=3).tolist(),
            "one_class":len(np.unique(np.concatenate(classes)))==1,"head_null_dimension":basis.shape[1]},predictions["full_h"][:,:3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points",type=int,default=12)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--peak-bootstrap",type=int,default=10000)
    args=parser.parse_args()
    if args.points < 10 or args.bootstrap < 2 or args.peak_bootstrap < 2:
        parser.error("at least ten points and at least two bootstrap resamples required")
    records=[]
    for run in selected_runs():
        recipe=json.loads((run/"resolved_recipe.json").read_text());validate_recipe(recipe)
        study=recipe["study"];raw=RAW/study;directory=raw/run.parent.parent.name
        summary=json.loads((run/"condition_summary.json").read_text())
        final_step=summary["checkpoint_reports"][-1]["checkpoint_step"]
        spec,process,params,final_provenance=restore(run,final_step,raw_root=raw)
        final=collect(run,final_step,spec,process,params,raw)
        final_b=[rows(d,"belief",n) for d,n in zip(final,(60000,80000))]
        final_c=[rows(d,"action",n) for d,n in zip(final,(60000,80000))]
        manifest=json.loads((directory/"durability_manifest.json").read_text())
        entries=[e for e in manifest["files"] if e["relative_path"].startswith("checkpoints/step_") and e["relative_path"].endswith(".npz")]
        by_step={int(e["relative_path"].split("step_")[1].split(".")[0]):e for e in entries}
        steps=select_steps(by_step,args.points)
        remote=json.loads((run/"run_manifest.json").read_text())["remote_artifacts"]
        result={"study":study,"arm":run.parent.parent.name,"run_dir":str(run),"run_id":final_provenance["run_id"],
                "training_epsilon":recipe["training"]["exploration_epsilon"],"b2":remote,
                "durability_sha256":final_provenance["durability_sha256"],"final_reference_step":final_step,
                "checkpoint_inventory":[{"step":step,"selected":step in steps,**by_step[step]} for step in sorted(by_step)],"points":[]}
        cache=directory/f"trajectory_p{args.points}_b{args.bootstrap}"
        cache.mkdir(exist_ok=True)
        for step in steps:
            path=cache/f"{step}.json"
            if path.exists():
                point=json.loads(path.read_text())
            else:
                spec,process,params,provenance=restore(run,step,raw_root=raw)
                own=collect(run,step,spec,process,params,raw)
                own_b=[rows(d,"belief",n) for d,n in zip(own,(60000,80000))]
                own_c=[rows(d,"action",n) for d,n in zip(own,(60000,80000))]
                embeddings=[rows(d,"embedding",n) for d,n in zip(own,(60000,80000))]
                weight,bias=np.asarray(params["policy"]["w"],dtype=float),np.asarray(params["policy"]["b"],dtype=float)
                own_metrics,_=evaluate(embeddings,own_b,own_c,weight,bias,args.bootstrap)
                if step==final_step:
                    fixed_embeddings=embeddings
                else:
                    fixed_embeddings=[rows({"h":np.asarray(replay(spec,params,d["obs"]))},"h",n) for d,n in zip(final,(60000,80000))]
                fixed_metrics,pred=evaluate(fixed_embeddings,final_b,final_c,weight,bias,args.bootstrap)
                reward=reward_interval(rows(own[1],"reward",80000),np.arange(80000)//960,n=args.bootstrap)
                point={"step":step,"provenance":provenance,"own_history":own_metrics,"fixed_history":fixed_metrics,"greedy_reward":reward}
                np.savez_compressed(cache/f"{step}_fixed_prediction.npz",prediction=pred)
                path.write_text(json.dumps(point,indent=2,allow_nan=False)+"\n")
            result["points"].append(point)
            print("POINT",study,run.parent.parent.name,step,point["own_history"]["metrics"]["full_h"]["belief"]["r2"],point["fixed_history"]["metrics"]["full_h"]["belief"]["r2"],point["greedy_reward"]["mean"],flush=True)
        peak=max(result["points"],key=lambda p:p["fixed_history"]["metrics"]["full_h"]["belief"]["r2"])
        with np.load(cache/f"{peak['step']}_fixed_prediction.npz") as d:peak_pred=d["prediction"]
        with np.load(cache/f"{final_step}_fixed_prediction.npz") as d:final_pred=d["prediction"]
        valid=np.isfinite(peak_pred).all(1)&np.isfinite(final_pred).all(1)
        y=final_b[1][valid];c=final_c[1][valid];episodes=(np.arange(80000)//960)[valid]
        result["fixed_peak_minus_final"]={"peak_step":peak["step"],"final_step":final_step,
            "peak_checkpoint":by_step[peak["step"]]["relative_path"],
            "difference":score(y,peak_pred[valid],c)["r2"]-score(y,final_pred[valid],c)["r2"],
            "excluded_paired_rows":int((~valid).sum()),
            "selection_caveat":"Peak selected by observed full-h fixed-history R2 among sampled checkpoints; paired CI conditions on this chosen peak and fixed probes, not corrected for peak selection or unsampled checkpoints.",
            **bootstrap(y,peak_pred[valid],c,episodes,n=args.peak_bootstrap,comparison=final_pred[valid])}
        records.append(result)
        jax.clear_caches()
    baseline=json.loads((ROOT/"results/ladder_nuisance_v2_temp15_within_class.json").read_text())
    report={"schema_version":1,"protocol":{**baseline["protocol"],
        "classes":"Own-history uses each checkpoint's own greedy classes; fixed-history uses final-policy classes for every encoder.",
        "matched_init":"Fixed-history trajectory generalizes matched initialization to every selected encoder, with exact final-policy streams/targets/classes, empty episode caches, warmup included and probes refit.",
        "controls":"Not recomputed for trajectories; controls are in the separate within-class final/init analysis.",
        "control_seeds":None,
        "bootstrap":f"{args.bootstrap} pointwise episode resamples; {args.peak_bootstrap} paired peak-minus-final resamples, probes fixed; seed 20261106.",
        "intermediate_selection":"Prespecified index-based early geometric / late even grid, including saved midpoint; all selected steps listed in each arm.",
        "selection":"Prespecified index-based hybrid: geometric early and evenly spaced late checkpoints, plus init/midpoint/final; no decoding-dependent selection. All durable checkpoints listed; selected points only analyzed.",
        "selected_point_budget":args.points,"point_bootstrap":args.bootstrap,"peak_bootstrap":args.peak_bootstrap,
        "trajectory_scope":"No additional controls; full h, head-blind, logits, delta, rho and independent global belief probe refit at each point on own and fixed histories. Fixed histories/targets/classes always final-policy streams; own greedy reward reported separately.",
        "uncertainty":"Pointwise whole-test-episode percentile bootstrap; class means recomputed, probes fixed. Peak-minus-final uses paired episodes and fixed probes; selected-peak CI is exploratory, not selection-adjusted. One training seed only.",
        "reward":"Realized pre-transition reward per scored greedy rollout step, not expected reward from belief; 64-step warmup excluded; fixed-history rewards belong to final policy, not replayed encoder."},
        "provenance":{**baseline["provenance"],"source_revision":REF,
            "analysis_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "python":platform.python_version(),"jax":jax.__version__,"numpy":np.__version__,"devices":str(jax.devices()),
            "trajectory_source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"trajectory.py",ROOT/"trajectory_report.py",ROOT/"nuisance_v2.py",ROOT/"metrics.py",ROOT/"temp15.py",ROOT/"temp15_analyze.py"]}},"arms":records}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("Wrote",OUTPUT,flush=True)


if __name__=="__main__":
    main()
