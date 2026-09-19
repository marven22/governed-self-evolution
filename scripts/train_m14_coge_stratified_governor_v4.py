"""Fit COGE with leave-one-parent-out selection on stratified transitions."""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
from train_m14_grammar_governor import fit_kernel_ridge, predict
from train_m14_rich_governor_v3 import TASKS, aggregate, choose, cid, feat, read

def parent(row): return str(row['context'].get('parent_controller_id', cid(row)))

def fit_parent_bootstrap(rows,gamma,ridge,size,seed):
    grouped={}
    for row in rows: grouped.setdefault(parent(row),[]).append(row)
    names=sorted(grouped); rng=np.random.default_rng(seed); ensemble=[]
    for _ in range(size):
        sample=rng.choice(names,len(names),replace=True)
        chosen=[r for name in sample for r in grouped[name]]
        ensemble.append(fit_kernel_ridge(np.stack([feat(r) for r in chosen]),np.stack([r['delta'] for r in chosen]),gamma,ridge))
    return ensemble

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-glob',required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--ensemble-size',type=int,default=64); a=p.parse_args()
    raw=read(a.data_glob); rows=aggregate(raw); parents=sorted({parent(r) for r in rows}); trials=[]
    if len(parents)<3: raise ValueError('need at least three independent parents')
    for gamma in (.01,.03,.1,.3):
      for ridge in (.001,.01,.1):
       cached={}
       for held in parents:
        train=[r for r in rows if parent(r)!=held]; val=[r for r in rows if parent(r)==held]
        ens=fit_parent_bootstrap(train,gamma,ridge,a.ensemble_size,20260919)
        cached[held]=(val,np.stack([predict(m,np.stack([feat(r) for r in val])) for m in ens],axis=1))
       for z in (0.,.5,1.,1.64,2.):
        for margin in (0.,.01,.03,.05):
         folds=[]
         for held in parents:
          val,pred=cached[held]; folds.extend(choose(val,None,z,margin,pred)['per_controller'])
         trials.append({'gamma':gamma,'ridge':ridge,'lower_z':z,'advantage_margin':margin,'harmful_rate':float(np.mean([x['harmful'] for x in folds])),'mean_advantage':float(np.mean([x['advantage'] for x in folds])),'mean_regret':float(np.mean([x['regret'] for x in folds])),'abstentions':int(sum(x['hold'] for x in folds)),'per_controller':folds})
    safe=[t for t in trials if t['harmful_rate']<=.10]
    selected=sorted(safe or trials,key=lambda t:(-t['mean_advantage'],t['mean_regret'],t['harmful_rate']))[0] if safe else sorted(trials,key=lambda t:(t['harmful_rate'],-t['mean_advantage'],t['mean_regret']))[0]
    ensemble=fit_parent_bootstrap(rows,selected['gamma'],selected['ridge'],a.ensemble_size,20260919)
    model={'model_type':'COGE V4 parent-bootstrap relative transition ensemble','tasks':TASKS,'selection':{k:selected[k] for k in ('gamma','ridge','lower_z','advantage_margin')},'epsilon':{'mw-reach':.05,'mw-pick-place':.05},'training_parents':parents,'training_controllers':sorted({cid(r) for r in rows}),'ensemble':ensemble}
    report={'protocol_version':'m14-coge-v4','raw_rows':len(raw),'aggregated_rows':len(rows),'parents':parents,'replicates_per_controller_action':2,'harm_target':.10,'selected':selected,'trials':trials}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2)+'\n'); a.model.parent.mkdir(parents=True,exist_ok=True)
    with a.model.open('wb') as f: pickle.dump(model,f)
    print(json.dumps({'selected':model['selection'],'harmful_rate':selected['harmful_rate'],'mean_advantage':selected['mean_advantage'],'mean_regret':selected['mean_regret'],'abstentions':selected['abstentions']}))
if __name__=='__main__': main()
