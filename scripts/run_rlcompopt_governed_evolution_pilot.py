#!/usr/bin/env python3
"""Execute a certificate-gated, multi-step evolution pilot with frozen governor v2.

This is an engineering rollout, not a new training-set collector. A proposed
child is promoted only after semantic certification and measured improvement;
otherwise its parent is retained and the lineage terminates in HOLD.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import Any
import numpy as np

from rlcompopt_edit_grammar import apply_edit, generate_candidates
from run_compilergym_feasibility import evaluate_fixed_policy
from train_rlcompopt_pairwise_governor_v2 import candidate_features

EPSILON = 0.01

def certified(record: dict[str, Any]) -> bool:
    v=record.get('validation',{}); return bool(record.get('validation_available') and v.get('okay') and v.get('benchmark_semantics_validated'))

def write(path: Path, value: dict[str, Any]) -> None:
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); tmp.replace(path)

def choose(payload: dict[str,Any], benchmark: str, origin_rank: int, origin_index: int, observation, parent, donors, donor_indices, budget: int):
    edits=generate_candidates(benchmark=benchmark,parent=parent,donors=donors,candidate_budget=budget,max_actions=64,donor_limit=len(donors))
    rows=[]
    for edit in edits:
        child=apply_edit(parent,donors,edit)
        rows.append({'benchmark':benchmark,'controller':{'parent_coreset_index':origin_index,'parent_rank':origin_rank,'parent_actions':parent,'ranked_donor_indices':donor_indices,'autophase':[int(x) for x in observation]},'edit':{k:v for k,v in edit.items() if k!='child_actions'},'child_actions':child})
    x=np.vstack([candidate_features(r) for r in rows]); pos=payload['positive_model'].predict_proba(x)[:,1]; risk=payload['blame_model'].predict_proba(x)[:,1]
    rank=[]
    for i in range(len(rows)):
        other=[j for j in range(len(rows)) if j!=i]
        rank.append(payload['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in other]))[:,1].mean() if other else .5)
    utility=pos*np.asarray(rank)-risk; i=int(np.argmax(utility))
    return rows[i], {'positive_probability':float(pos[i]),'blame_probability':float(risk[i]),'pairwise_rank_score':float(rank[i]),'utility':float(utility[i]),'candidate_count':len(rows),'eligible':bool(utility[i]>0 and pos[i]>=.5)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-db',type=Path,required=True); p.add_argument('--trajectory-data',type=Path,required=True); p.add_argument('--vocab-db',type=Path,required=True); p.add_argument('--split',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--parent-ranks',default='1,2,3'); p.add_argument('--max-steps',type=int,default=3); p.add_argument('--candidate-budget',type=int,default=24); p.add_argument('--donor-limit',type=int,default=10); p.add_argument('--max-programs',type=int,default=None); p.add_argument('--cohort',choices=('selection','development'),default='selection'); a=p.parse_args()
    payload=pickle.loads(a.model.read_bytes())
    if payload.get('protocol')!='rlcompopt-pairwise-governor-v2': raise SystemExit('Frozen v2 model required')
    from rlcompopt.model_testing import Environment
    split=json.loads(a.split.read_text()); benchmarks=split[a.cohort];
    if a.max_programs: benchmarks=benchmarks[:a.max_programs]
    ranks=[int(x) for x in a.parent_ranks.split(',')]
    runner=Environment(str(a.model_db),None,0,str(a.vocab_db),max_step=100,benchmarks=[],train_dataset_path=str(a.trajectory_data),sampling=False)
    coreset=[[int(x) for x in seq] for seq in runner.actionseqs]; lineages=[]; a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
      for benchmark in benchmarks:
        observation=runner.reset(benchmark); ranked=[int(x) for x in runner.get_model_action(observation)]; donor_indices=ranked[:a.donor_limit]; donors=[coreset[i] for i in donor_indices]
        for origin_rank in ranks:
          origin_index=ranked[origin_rank]; parent=coreset[origin_index]; lineage={'benchmark':benchmark,'origin_parent_rank':origin_rank,'origin_parent_coreset_index':origin_index,'steps':[],'termination':None}
          for step in range(a.max_steps):
            parent_eval=evaluate_fixed_policy(benchmark,parent)
            if not certified(parent_eval): lineage['termination']='parent_not_certified'; break
            proposal, predicted=choose(payload,benchmark,origin_rank,origin_index,observation,parent,donors,donor_indices,a.candidate_budget)
            if not predicted['eligible']:
              lineage['steps'].append({'step':step,'action':'hold','parent_reward':parent_eval['total_reward'],'prediction':predicted}); lineage['termination']='governor_hold'; break
            child_eval=evaluate_fixed_policy(benchmark,proposal['child_actions']); delta=float(child_eval['total_reward']-parent_eval['total_reward']); accepted=bool(certified(child_eval) and delta>EPSILON)
            lineage['steps'].append({'step':step,'action':'promote' if accepted else 'rollback','parent_reward':parent_eval['total_reward'],'child_reward':child_eval['total_reward'],'delta_reward':delta,'edit':proposal['edit'],'prediction':predicted,'child_certified':certified(child_eval)})
            if not accepted: lineage['termination']='rejected_or_not_improved'; break
            parent=proposal['child_actions']
          else: lineage['termination']='step_budget_reached'
          lineages.append(lineage); write(a.output,{'protocol':'rlcompopt-governed-evolution-pilot-v1','scope':f'engineering pilot on {a.cohort} programs; not final evaluation','cohort':a.cohort,'frozen_model':str(a.model),'parameters':{'parent_ranks':ranks,'max_steps':a.max_steps,'candidate_budget':a.candidate_budget},'lineages':lineages})
    finally:
      runner.env.close(); runner.model.connection.close()
    promoted=sum(s['action']=='promote' for l in lineages for s in l['steps']); print(json.dumps({'lineages':len(lineages),'promotions':promoted,'terminations':{x:sum(l['termination']==x for l in lineages) for x in sorted({l['termination'] for l in lineages})},'output':str(a.output)},sort_keys=True))
if __name__=='__main__': main()
