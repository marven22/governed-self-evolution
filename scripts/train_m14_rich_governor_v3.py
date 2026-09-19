"""Fit V3 from replicated, paired capability transitions.

Selection is leave-one-controller-out and treats either a protected-capability
drop or utility below HOLD as harmful.  This script is development-only: its
reported cross-validation scores are not a final held-out claim.
"""
from __future__ import annotations

import argparse, glob, json, pickle
from pathlib import Path
from typing import Any
import numpy as np

from train_m14_grammar_governor import TASKS, TARGETS, fit_kernel_ridge, predict, utility

STATE_KEYS = ("bc_loss_reach", "bc_loss_pick_place", "policy_gradient_norm_reach", "policy_gradient_norm_pick_place", "policy_gradient_cosine")


def cid(row: dict[str, Any]) -> str: return str(row["context"].get("controller_id", f"seed{row['context']['seed']}"))

def read(pattern: str) -> list[dict[str, Any]]:
    rows=[]
    for path in sorted(glob.glob(pattern)): rows.extend(json.loads(Path(path).read_text()))
    if not rows: raise ValueError(f"no transition files match {pattern!r}")
    return rows

def feat(row: dict[str, Any]) -> np.ndarray:
    u=row["update"]; r=u["retention"]; state=row["pre_update_state"]
    return np.array([row["pre_capability"][TASKS[0]],row["pre_capability"][TASKS[1]],u["task_data"]["reach_fraction"],u["optimization"]["gradient_steps"]/1500,np.log10(u["optimization"]["learning_rate"])+4,float(r["freeze_encoder"]),np.log1p(r["prior_policy_l2"])/np.log(11),float(r["protect_policy_during_model_update"]),*[float(u["target"]==t) for t in TARGETS],*[float(state[k]) for k in STATE_KEYS]],dtype=np.float64)

def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average paired replicates, preserving controller identity and action."""
    by_rep: dict[tuple[str,int],dict[str,dict[str,Any]]] = {}
    for row in rows: by_rep.setdefault((cid(row),int(row["context"]["replicate_id"])),{})[row["context"]["candidate_id"]]=row
    groups: dict[tuple[str,str],list[dict[str,Any]]] = {}
    for (controller,rep), actions in by_rep.items():
        holds=[r for r in actions.values() if r["context"]["candidate_label"]=="HOLD"]
        if len(holds)!=1 or len(actions)!=16: raise ValueError(f"incomplete paired replicate {controller}/{rep}")
        hold=holds[0]
        for action,row in actions.items():
            clone=dict(row); clone["hold_capability"]=hold["post_capability"]; clone["delta"]=np.array([row["post_capability"][t]-hold["post_capability"][t] for t in TASKS]); groups.setdefault((controller,action),[]).append(clone)
    output=[]
    for (_, _), values in groups.items():
        base=values[0]
        base["pre_capability"]={t:float(np.mean([v["pre_capability"][t] for v in values])) for t in TASKS}
        base["pre_update_state"]={k:float(np.mean([v["pre_update_state"][k] for v in values])) for k in STATE_KEYS}
        base["hold_capability"]={t:float(np.mean([v["hold_capability"][t] for v in values])) for t in TASKS}
        base["delta"]=np.mean([v["delta"] for v in values],axis=0); base["replicates"]=len(values)
        output.append(base)
    return output

def fit_ensemble(rows: list[dict[str,Any]], gamma: float, ridge: float, size: int, seed: int) -> list[dict[str,Any]]:
    grouped={}
    for row in rows: grouped.setdefault(cid(row),[]).append(row)
    names=sorted(grouped); rng=np.random.default_rng(seed); result=[]
    for _ in range(size):
        sample=rng.choice(names,len(names),replace=True); chosen=[row for name in sample for row in grouped[name]]
        result.append(fit_kernel_ridge(np.stack([feat(r) for r in chosen]),np.stack([r["delta"] for r in chosen]),gamma,ridge))
    return result

def choose(rows: list[dict[str,Any]], ensemble: list[dict[str,Any]] | None, z: float, margin: float, predicted: np.ndarray | None = None) -> dict[str,Any]:
    if predicted is None:
        if ensemble is None: raise ValueError("ensemble or predicted values required")
        x=np.stack([feat(r) for r in rows]); pred=np.stack([predict(m,x) for m in ensemble],axis=1)
    else: pred=predicted
    by={}
    for i,row in enumerate(rows): by.setdefault(cid(row),[]).append(i)
    decisions=[]
    for name,ids in by.items():
        rs=[rows[i] for i in ids]; hi=next(j for j,r in enumerate(rs) if r["context"]["candidate_label"]=="HOLD"); demand=rs[0]["demand"]; d=pred[ids]
        advantage=np.array([[utility(v,demand) for v in member] for member in d]); lcb=advantage.mean(1)-z*advantage.std(1)
        eps=np.array([rs[0]["constraints"].get("max_reach_drop_from_hold",0),rs[0]["constraints"].get("max_pick_place_drop_from_hold",0)])
        safe=np.all(d.mean(1)-z*d.std(1)>=-eps,axis=1); allowed=safe&(lcb>margin); allowed[hi]=True; lcb[hi]=0.; selected=int(np.argmax(np.where(allowed,lcb,-np.inf))); row=rs[selected]
        actual_adv=utility(row["delta"],demand); retention=bool(np.all(row["delta"]>=-eps)); harmful=(selected!=hi) and ((actual_adv<0) or not retention)
        oracle=max(utility(r["delta"],demand) for r in rs if np.all(r["delta"]>=-eps))
        decisions.append({"controller":name,"chosen":row["context"]["candidate_id"],"hold":selected==hi,"advantage":float(actual_adv),"harmful":harmful,"retention":retention,"regret":float(oracle-actual_adv)})
    return {"per_controller":decisions,"harmful_rate":float(np.mean([d["harmful"] for d in decisions])),"mean_advantage":float(np.mean([d["advantage"] for d in decisions])),"mean_regret":float(np.mean([d["regret"] for d in decisions])),"abstentions":int(sum(d["hold"] for d in decisions))}

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data-glob",required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--model",type=Path,required=True); p.add_argument("--ensemble-size",type=int,default=48); a=p.parse_args()
    raw=read(a.data_glob); rows=aggregate(raw); controllers=sorted({cid(r) for r in rows}); trials=[]
    for gamma in (.03,.1,.3):
      for ridge in (.01,.1):
       # Bootstrap prediction depends on the model hyperparameters and held
       # controller, but not on the later z/margin decision rule. Cache it.
       fold_predictions={}
       for held in controllers:
        train=[r for r in rows if cid(r)!=held]; val=[r for r in rows if cid(r)==held]
        ensemble=fit_ensemble(train,gamma,ridge,a.ensemble_size,20260918)
        x=np.stack([feat(r) for r in val]); fold_predictions[held]=(val,np.stack([predict(m,x) for m in ensemble],axis=1))
       for z in (0.,.5,1.,1.64,2.):
        for margin in (0.,.01,.03,.05):
         folds=[]
         for held in controllers:
          val,predicted=fold_predictions[held]
          folds.extend(choose(val,None,z,margin,predicted)["per_controller"])
         trials.append({"gamma":gamma,"ridge":ridge,"lower_z":z,"advantage_margin":margin,"harmful_rate":float(np.mean([x["harmful"] for x in folds])),"mean_advantage":float(np.mean([x["advantage"] for x in folds])),"mean_regret":float(np.mean([x["regret"] for x in folds])),"abstentions":int(sum(x["hold"] for x in folds)),"per_controller":folds})
    safe=[t for t in trials if t["harmful_rate"]<=.10]
    selected=sorted(safe or trials,key=lambda t:(-t["mean_advantage"],t["mean_regret"],t["harmful_rate"]))[0] if safe else sorted(trials,key=lambda t:(t["harmful_rate"],-t["mean_advantage"],t["mean_regret"]))[0]
    ensemble=fit_ensemble(rows,selected["gamma"],selected["ridge"],a.ensemble_size,20260918)
    model={"model_type":"V3 replicated controller-bootstrap relative transition ensemble","tasks":TASKS,"state_keys":STATE_KEYS,"selection":{k:selected[k] for k in ("gamma","ridge","lower_z","advantage_margin")},"epsilon":{"mw-reach":.05,"mw-pick-place":.05},"training_controllers":controllers,"ensemble":ensemble}
    report={"protocol_version":"m14-governor-v3","raw_rows":len(raw),"aggregated_rows":len(rows),"controllers":controllers,"replicates_per_controller_action":2,"harm_target":.10,"selected":selected,"trials":trials}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+"\n");a.model.parent.mkdir(parents=True,exist_ok=True)
    with a.model.open("wb") as f: pickle.dump(model,f)
    print(json.dumps({"selected":model["selection"],"harmful_rate":selected["harmful_rate"],"mean_advantage":selected["mean_advantage"],"abstentions":selected["abstentions"]}))

if __name__=="__main__": main()
