#!/usr/bin/env python3
"""Horizon-2 receding-horizon MPC over grammar-valid controller edits.

The governor is frozen. MPC only plans in edit space: it scores a bounded beam
of prospective first edits plus their best prospective second edit, executes
only the selected first edit, then certificates and replans from reality.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import Any
import numpy as np
from rlcompopt_edit_grammar import apply_edit, generate_candidates
from run_compilergym_feasibility import evaluate_fixed_policy
from train_rlcompopt_pairwise_governor_v2 import candidate_features
EPSILON=.01
def cert(r):
 v=r.get('validation',{}); return bool(r.get('validation_available') and v.get('okay') and v.get('benchmark_semantics_validated'))
def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); t.replace(p)
def score(payload, benchmark, rank, index, obs, parent, donors, donor_indices, budget):
 edits=generate_candidates(benchmark=benchmark,parent=parent,donors=donors,candidate_budget=budget,max_actions=64,donor_limit=len(donors)); rows=[]
 for e in edits:
  child=apply_edit(parent,donors,e); rows.append({'benchmark':benchmark,'controller':{'parent_coreset_index':index,'parent_rank':rank,'parent_actions':parent,'ranked_donor_indices':donor_indices,'autophase':[int(x) for x in obs]},'edit':{k:v for k,v in e.items() if k!='child_actions'},'child_actions':child})
 x=np.vstack([candidate_features(r) for r in rows]); pos=payload['positive_model'].predict_proba(x)[:,1]; risk=payload['blame_model'].predict_proba(x)[:,1]; ranks=[]
 for i in range(len(rows)):
  other=[j for j in range(len(rows)) if j!=i]; ranks.append(payload['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in other]))[:,1].mean() if other else .5)
 utility=pos*np.asarray(ranks)-risk
 for i,row in enumerate(rows): row['prediction']={'positive_probability':float(pos[i]),'blame_probability':float(risk[i]),'pairwise_rank_score':float(ranks[i]),'utility':float(utility[i]),'eligible':bool(utility[i]>0 and pos[i]>=.5)}
 return rows
def main():
 p=argparse.ArgumentParser();
 for n in ('model_db','trajectory_data','vocab_db','split','model','output'): p.add_argument('--'+n.replace('_','-'),dest=n,type=Path,required=True)
 p.add_argument('--parent-ranks',default='1,2,3'); p.add_argument('--max-steps',type=int,default=3); p.add_argument('--candidate-budget',type=int,default=24); p.add_argument('--beam-width',type=int,default=4); p.add_argument('--gamma',type=float,default=.9); p.add_argument('--donor-limit',type=int,default=10); p.add_argument('--max-programs',type=int,default=None); a=p.parse_args()
 payload=pickle.loads(a.model.read_bytes());
 if payload.get('protocol')!='rlcompopt-pairwise-governor-v2': raise SystemExit('frozen v2 model required')
 from rlcompopt.model_testing import Environment
 benchmarks=json.loads(a.split.read_text())['selection']; benchmarks=benchmarks[:a.max_programs] if a.max_programs else benchmarks; ranks=[int(x) for x in a.parent_ranks.split(',')]
 runner=Environment(str(a.model_db),None,0,str(a.vocab_db),max_step=100,benchmarks=[],train_dataset_path=str(a.trajectory_data),sampling=False); coreset=[[int(x) for x in s] for s in runner.actionseqs]; lineages=[]; a.output.parent.mkdir(parents=True,exist_ok=True)
 try:
  for benchmark in benchmarks:
   obs=runner.reset(benchmark); ordered=[int(x) for x in runner.get_model_action(obs)]; donor_indices=ordered[:a.donor_limit]; donors=[coreset[i] for i in donor_indices]
   for origin_rank in ranks:
    origin_index=ordered[origin_rank]; parent=coreset[origin_index]; lineage={'benchmark':benchmark,'origin_parent_rank':origin_rank,'origin_parent_coreset_index':origin_index,'steps':[],'termination':None}
    for step in range(a.max_steps):
     parent_eval=evaluate_fixed_policy(benchmark,parent)
     if not cert(parent_eval): lineage['termination']='parent_not_certified'; break
     first=score(payload,benchmark,origin_rank,origin_index,obs,parent,donors,donor_indices,a.candidate_budget); eligible=[r for r in first if r['prediction']['eligible']]
     if not eligible: lineage['steps'].append({'step':step,'action':'hold','parent_reward':parent_eval['total_reward']}); lineage['termination']='mpc_hold'; break
     beam=sorted(eligible,key=lambda r:r['prediction']['utility'],reverse=True)[:a.beam_width]
     options=[]
     for candidate in beam:
      second=score(payload,benchmark,origin_rank,origin_index,obs,candidate['child_actions'],donors,donor_indices,a.candidate_budget); second_utility=max([r['prediction']['utility'] for r in second if r['prediction']['eligible']],default=0.0); path=float(candidate['prediction']['utility']+a.gamma*max(0.,second_utility)); options.append((path,candidate,second_utility))
     path,proposal,next_utility=max(options,key=lambda x:x[0]); child_eval=evaluate_fixed_policy(benchmark,proposal['child_actions']); delta=float(child_eval['total_reward']-parent_eval['total_reward']); accept=bool(cert(child_eval) and delta>EPSILON)
     lineage['steps'].append({'step':step,'action':'promote' if accept else 'rollback','parent_reward':parent_eval['total_reward'],'child_reward':child_eval['total_reward'],'delta_reward':delta,'child_certified':cert(child_eval),'edit':proposal['edit'],'immediate_prediction':proposal['prediction'],'best_predicted_second_step_utility':second_utility,'two_step_path_utility':path,'beam_considered':len(beam)})
     if not accept: lineage['termination']='rejected_or_not_improved'; break
     parent=proposal['child_actions']
    else: lineage['termination']='step_budget_reached'
    lineages.append(lineage); write(a.output,{'protocol':'rlcompopt-edit-mpc-pilot-v1','scope':'engineering pilot on diagnostic selection programs; no model fitting','frozen_governor':str(a.model),'planner':{'horizon':2,'beam_width':a.beam_width,'gamma':a.gamma},'lineages':lineages})
 finally: runner.env.close(); runner.model.connection.close()
 promotions=sum(s['action']=='promote' for l in lineages for s in l['steps']); print(json.dumps({'lineages':len(lineages),'promotions':promotions,'terminations':{x:sum(l['termination']==x for l in lineages) for x in sorted({l['termination'] for l in lineages})},'output':str(a.output)},sort_keys=True))
if __name__=='__main__': main()
