#!/usr/bin/env python3
"""Certificate-backed V2-versus-cross-fitted-V3 edit duels."""
from __future__ import annotations
import argparse,json,pickle
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run_compilergym_feasibility import evaluate_fixed_policy
from train_rlcompopt_pairwise_governor_v2 import candidate_features
from evaluate_rlcompopt_shadow_v3_crossfit import reconstruct,EPS

def choose(payload,rows):
 x=np.vstack([candidate_features(r) for r in rows]); pos=payload['positive_model'].predict_proba(x)[:,1]; risk=payload['blame_model'].predict_proba(x)[:,1]; ranks=[]
 for i in range(len(rows)):
  other=[j for j in range(len(rows)) if j!=i]; ranks.append(payload['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in other]))[:,1].mean() if other else .5)
 utility=pos*np.asarray(ranks)-risk; i=int(np.argmax(utility)); return (i if utility[i]>0 and pos[i]>=.5 else None)
def certified(r):
 v=r.get('validation',{}); return bool(r.get('validation_available') and v.get('okay') and v.get('benchmark_semantics_validated'))
def main():
 p=argparse.ArgumentParser()
 for n in ('base_model','development_ledger','selection_ledger','model_db','trajectory_data','vocab_db','report'): p.add_argument('--'+n.replace('_','-'),dest=n,type=Path,required=True)
 p.add_argument('--rollout',type=Path,required=True,action='append'); a=p.parse_args()
 base=pickle.loads(a.base_model.read_bytes()); dev=json.loads(a.development_ledger.read_text()); sel=json.loads(a.selection_ledger.read_text()); base_rows=[r for r in dev['transactions'] if r['outcome']['certified_child']]; by_program=defaultdict(list)
 for source in (dev,sel):
  for r in source['transactions']:
   if r['outcome']['certified_child'] and r['controller'].get('parent_rank',0) in (1,2,3): by_program[r['benchmark']].append(r)
 rollouts=[json.loads(x.read_text()) for x in a.rollout]; programs=sorted({l['benchmark'] for r in rollouts for l in r['lineages']})
 from rlcompopt.model_testing import Environment
 runner=Environment(str(a.model_db),None,0,str(a.vocab_db),max_step=100,benchmarks=[],train_dataset_path=str(a.trajectory_data),sampling=False); coreset=[[int(x) for x in s] for s in runner.actionseqs]; duels=[]
 try:
  for program in programs:
   extra=reconstruct(rollouts,runner,coreset,program); rows=base_rows+extra; x=np.vstack([candidate_features(r) for r in rows]); y=np.asarray([r['outcome']['delta_reward']>EPS for r in rows]); head=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=4000,class_weight='balanced')).fit(x,y); v3=dict(base); v3['positive_model']=head
   contexts=defaultdict(list)
   for r in by_program[program]: contexts[(r['controller'].get('parent_rank',0),r['controller']['parent_coreset_index'])].append(r)
   for key,rows in contexts.items():
    i2,i3=choose(base,rows),choose(v3,rows)
    if i2==i3: continue
    parent=rows[0]['controller']['parent_actions']; parent_eval=evaluate_fixed_policy(program,parent)
    candidates={'v2':rows[i2]['child_actions'] if i2 is not None else parent,'v3':rows[i3]['child_actions'] if i3 is not None else parent}; outcomes={}
    for name,actions in candidates.items():
     result=parent_eval if actions==parent else evaluate_fixed_policy(program,actions); outcomes[name]={'certified':certified(result),'gain':float(result['total_reward']-parent_eval['total_reward'])}
    winner='tie'
    if outcomes['v2']['certified'] and outcomes['v3']['certified']:
     if outcomes['v3']['gain']>outcomes['v2']['gain']+EPS: winner='v3'
     elif outcomes['v2']['gain']>outcomes['v3']['gain']+EPS: winner='v2'
    duels.append({'program':program,'parent_rank':key[0],'v2_action':'hold' if i2 is None else 'edit','v3_action':'hold' if i3 is None else 'edit','v2':outcomes['v2'],'v3':outcomes['v3'],'winner':winner})
 finally: runner.env.close(); runner.model.connection.close()
 diff=np.asarray([d['v3']['gain']-d['v2']['gain'] for d in duels]); rng=np.random.default_rng(20260922); boots=np.asarray([rng.choice(diff,len(diff),replace=True).mean() for _ in range(10000)]) if len(diff) else np.asarray([0.])
 report={'protocol':'rlcompopt-governor-challenger-duels-v1','scope':'engineering counterfactual tournament; V3 not deployed','duels':duels,'summary':{'disagreements':len(duels),'wins':dict(Counter(d['winner'] for d in duels)),'mean_v3_minus_v2_gain':float(diff.mean()) if len(diff) else 0.,'bootstrap_95_lower_bound':float(np.quantile(boots,.025)),'promotion_decision':'promote_v3' if len(diff)>=8 and np.quantile(boots,.025)>0 else 'retain_v2'}}
 a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report['summary'],sort_keys=True))
if __name__=='__main__': main()
