#!/usr/bin/env python3
"""Controlled, certificate-backed self-repair benchmark for frozen V2."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np
from rlcompopt_edit_grammar import apply_edit,generate_candidates
from run_compilergym_feasibility import evaluate_fixed_policy
from train_rlcompopt_pairwise_governor_v2 import candidate_features
EPS=.01
def cert(r):
 v=r.get('validation',{}); return bool(r.get('validation_available') and v.get('okay') and v.get('benchmark_semantics_validated'))
def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); t.replace(p)
def main():
 p=argparse.ArgumentParser();
 for n in ('model_db','trajectory_data','vocab_db','split','model','output'): p.add_argument('--'+n.replace('_','-'),dest=n,type=Path,required=True)
 p.add_argument('--cohort',choices=('selection','development'),default='selection'); p.add_argument('--max-programs',type=int,default=None); p.add_argument('--candidate-budget',type=int,default=24); p.add_argument('--donor-limit',type=int,default=10); a=p.parse_args()
 model=pickle.loads(a.model.read_bytes());
 from rlcompopt.model_testing import Environment
 benchmarks=json.loads(a.split.read_text())[a.cohort]; benchmarks=benchmarks[:a.max_programs] if a.max_programs else benchmarks; runner=Environment(str(a.model_db),None,0,str(a.vocab_db),max_step=100,benchmarks=[],train_dataset_path=str(a.trajectory_data),sampling=False); coreset=[[int(x) for x in s] for s in runner.actionseqs]; contexts=[]; a.output.parent.mkdir(parents=True,exist_ok=True)
 try:
  for b in benchmarks:
   obs=runner.reset(b); ordered=[int(x) for x in runner.get_model_action(obs)]; donors=[coreset[i] for i in ordered[:a.donor_limit]]; original=coreset[ordered[0]]; original_eval=evaluate_fixed_policy(b,original)
   for damage_name,position in (('delete_middle',len(original)//2),('delete_late',len(original)-1)):
    damaged=original[:position]+original[position+1:]; damage_eval=evaluate_fixed_policy(b,damaged); damage_delta=float(damage_eval['total_reward']-original_eval['total_reward']); record={'benchmark':b,'damage':{'name':damage_name,'position':position},'original_reward':original_eval['total_reward'],'damaged_reward':damage_eval['total_reward'],'damage_delta':damage_delta,'eligible':bool(cert(original_eval) and cert(damage_eval) and damage_delta < -EPS)}
    if record['eligible']:
     edits=generate_candidates(benchmark=b,parent=damaged,donors=donors,candidate_budget=a.candidate_budget,max_actions=64,donor_limit=a.donor_limit); rows=[]
     for e in edits:
      child=apply_edit(damaged,donors,e); rows.append({'benchmark':b,'controller':{'parent_coreset_index':ordered[0],'parent_rank':0,'parent_actions':damaged,'ranked_donor_indices':ordered[:a.donor_limit],'autophase':[int(x) for x in obs]},'edit':{k:v for k,v in e.items() if k!='child_actions'},'child_actions':child})
     x=np.vstack([candidate_features(r) for r in rows]); pos=model['positive_model'].predict_proba(x)[:,1]; risk=model['blame_model'].predict_proba(x)[:,1]; rank=[]
     for i in range(len(rows)):
      other=[j for j in range(len(rows)) if j!=i]; rank.append(model['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in other]))[:,1].mean() if other else .5)
     utility=pos*np.asarray(rank)-risk; selected=int(np.argmax(utility)); hold=not(bool(utility[selected]>0 and pos[selected]>=.5)); outcomes=[]
     for row in rows:
      ev=evaluate_fixed_policy(b,row['child_actions']); outcomes.append({'certified':cert(ev),'reward':ev['total_reward'],'repair_gain':float(ev['total_reward']-damage_eval['total_reward'])})
     valid=[o['repair_gain'] for o in outcomes if o['certified']]; chosen_gain=0. if hold else max(0.,outcomes[selected]['repair_gain']); headroom=original_eval['total_reward']-damage_eval['total_reward']; record.update({'candidate_count':len(rows),'v2_action':'hold' if hold else 'edit','v2_repair_gain':chosen_gain,'random_expected_repair_gain':float(np.mean([max(0.,x) for x in valid])),'oracle_repair_gain':max(0.,max(valid)),'recovery_fraction':float(chosen_gain/headroom) if headroom>0 else None,'selected_outcome':None if hold else outcomes[selected],'selected_edit':None if hold else rows[selected]['edit']})
    contexts.append(record); write(a.output,{'protocol':'rlcompopt-controlled-repair-v1','scope':'engineering repair benchmark; frozen V2, no MPC, no model fitting','cohort':a.cohort,'contexts':contexts})
 finally: runner.env.close(); runner.model.connection.close()
 eligible=[c for c in contexts if c['eligible']]; summary={'contexts':len(contexts),'eligible_damages':len(eligible),'mean_v2_repair_gain':float(np.mean([c['v2_repair_gain'] for c in eligible])) if eligible else 0.,'mean_random_expected_repair_gain':float(np.mean([c['random_expected_repair_gain'] for c in eligible])) if eligible else 0.,'mean_oracle_repair_gain':float(np.mean([c['oracle_repair_gain'] for c in eligible])) if eligible else 0.,'output':str(a.output)}; print(json.dumps(summary,sort_keys=True))
if __name__=='__main__': main()
