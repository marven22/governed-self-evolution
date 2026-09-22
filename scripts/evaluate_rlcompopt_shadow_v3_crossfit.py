#!/usr/bin/env python3
"""Cross-fitted test of whether shadow V3 improves over incumbent V2.

For each rollout program, the V3 positive head is rebuilt using live evidence
from *other* programs only. Candidate outcomes from the held-out program are
then revealed solely for measurement. This tests the value of governor updates,
not MPC.
"""
from __future__ import annotations
import argparse,json,pickle
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rlcompopt_edit_grammar import apply_edit
from train_rlcompopt_pairwise_governor_v2 import candidate_features

EPS=.01
def reconstruct(rollouts, runner, coreset, excluded):
 out=[]
 for rollout in rollouts:
  for lineage in rollout['lineages']:
   b=lineage['benchmark']
   if b==excluded: continue
   obs=runner.reset(b); ordered=[int(x) for x in runner.get_model_action(obs)]; donors=[coreset[i] for i in ordered[:10]]; parent=coreset[lineage['origin_parent_coreset_index']]
   for step in lineage['steps']:
    if step['action']=='hold': continue
    child=apply_edit(parent,donors,step['edit']); out.append({'benchmark':b,'controller':{'parent_coreset_index':lineage['origin_parent_coreset_index'],'parent_rank':lineage['origin_parent_rank'],'parent_actions':parent,'ranked_donor_indices':ordered[:10],'autophase':[int(x) for x in obs]},'edit':step['edit'],'child_actions':child,'outcome':{'certified_child':bool(step['child_certified']),'delta_reward':float(step['delta_reward'])}})
    if step['action']=='promote': parent=child
 return out
def decision(payload, rows):
 x=np.vstack([candidate_features(r) for r in rows]); pos=payload['positive_model'].predict_proba(x)[:,1]; risk=payload['blame_model'].predict_proba(x)[:,1]; delta=np.asarray([r['outcome']['delta_reward'] for r in rows]); groups=defaultdict(list)
 for i,r in enumerate(rows): c=r['controller']; groups[(r['benchmark'],c.get('parent_rank',0),c['parent_coreset_index'])].append(i)
 gains=[]; labels=[]
 for idx in groups.values():
  ranks=[]
  for i in idx:
   other=[j for j in idx if j!=i]; ranks.append(payload['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in other]))[:,1].mean() if other else .5)
  util=pos[idx]*np.asarray(ranks)-risk[idx]; j=idx[int(np.argmax(util))]; edit=bool(util.max()>0 and pos[j]>=.5); gains.append(max(0.,float(delta[j])) if edit else 0.); labels.append(rows[j]['outcome']['label'] if edit else 'hold')
 return float(np.mean(gains)),Counter(labels)
def main():
 p=argparse.ArgumentParser();
 for n in ('base_model','development_ledger','selection_ledger','model_db','trajectory_data','vocab_db','report'): p.add_argument('--'+n.replace('_','-'),dest=n,type=Path,required=True)
 p.add_argument('--rollout',type=Path,required=True,action='append'); a=p.parse_args()
 base=pickle.loads(a.base_model.read_bytes()); dev=json.loads(a.development_ledger.read_text()); sel=json.loads(a.selection_ledger.read_text()); base_rows=[r for r in dev['transactions'] if r['outcome']['certified_child']]; candidate_by_program=defaultdict(list)
 for source in (dev,sel):
  for r in source['transactions']:
   if r['outcome']['certified_child']: candidate_by_program[r['benchmark']].append(r)
 rollouts=[json.loads(x.read_text()) for x in a.rollout]; programs=sorted({l['benchmark'] for r in rollouts for l in r['lineages']})
 from rlcompopt.model_testing import Environment
 runner=Environment(str(a.model_db),None,0,str(a.vocab_db),max_step=100,benchmarks=[],train_dataset_path=str(a.trajectory_data),sampling=False); coreset=[[int(x) for x in s] for s in runner.actionseqs]; results=[]
 try:
  for program in programs:
   extra=reconstruct(rollouts,runner,coreset,program); rows=base_rows+extra; x=np.vstack([candidate_features(r) for r in rows]); y=np.asarray([r['outcome']['delta_reward']>EPS for r in rows]); head=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=4000,class_weight='balanced')); head.fit(x,y); challenger=dict(base); challenger['positive_model']=head
   v2_gain,v2_labels=decision(base,candidate_by_program[program]); v3_gain,v3_labels=decision(challenger,candidate_by_program[program]); results.append({'program':program,'incremental_rollout_rows':len(extra),'v2_gain':v2_gain,'v3_gain':v3_gain,'difference':v3_gain-v2_gain,'v2_selected_outcomes':dict(v2_labels),'v3_selected_outcomes':dict(v3_labels)})
 finally: runner.env.close(); runner.model.connection.close()
 diff=np.asarray([r['difference'] for r in results]); rng=np.random.default_rng(20260922); boots=np.asarray([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(10000)])
 report={'protocol':'rlcompopt-shadow-v3-crossfit-v1','purpose':'diagnostic cross-fitted comparison of governor evolution without MPC','programs':len(results),'mean_v2_gain':float(np.mean([r['v2_gain'] for r in results])),'mean_v3_gain':float(np.mean([r['v3_gain'] for r in results])),'mean_difference_v3_minus_v2':float(diff.mean()),'bootstrap_95_lower_bound':float(np.quantile(boots,.025)),'bootstrap_95_upper_bound':float(np.quantile(boots,.975)),'program_results':results,'promotion_decision':'promote_v3' if np.quantile(boots,.025)>0 else 'retain_v2','limitations':['V2 was originally developed using some of these ledgers; this is an engineering diagnostic, not final external evidence.','Only the positive head is adapted because rollout logs lack counterfactual sibling outcomes.']}
 a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:report[k] for k in ('programs','mean_v2_gain','mean_v3_gain','mean_difference_v3_minus_v2','bootstrap_95_lower_bound','bootstrap_95_upper_bound','promotion_decision')},sort_keys=True))
if __name__=='__main__': main()
