#!/usr/bin/env python3
"""Train a non-deployed V3 challenger from certified rollout decisions.

V3 updates only the positive-improvement head. Pairwise ranking and blame are
kept from V2 because rollout logs reveal only V2's selected action, not every
counterfactual candidate. This prevents selected-action feedback from silently
rewriting all of the governor.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rlcompopt_edit_grammar import apply_edit
from train_rlcompopt_pairwise_governor_v2 import candidate_features

EPSILON=.01
def main():
 p=argparse.ArgumentParser(); p.add_argument('--base-model',type=Path,required=True); p.add_argument('--development-ledger',type=Path,required=True); p.add_argument('--rollout',type=Path,required=True,action='append'); p.add_argument('--model-db',type=Path,required=True); p.add_argument('--trajectory-data',type=Path,required=True); p.add_argument('--vocab-db',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
 base=pickle.loads(a.base_model.read_bytes())
 if base.get('protocol')!='rlcompopt-pairwise-governor-v2': raise SystemExit('V2 base model required')
 development=json.loads(a.development_ledger.read_text()); base_rows=[r for r in development['transactions'] if r['outcome']['certified_child']]
 rollouts=[json.loads(path.read_text()) for path in a.rollout]
 from rlcompopt.model_testing import Environment
 runner=Environment(str(a.model_db),None,0,str(a.vocab_db),max_step=100,benchmarks=[],train_dataset_path=str(a.trajectory_data),sampling=False)
 coreset=[[int(x) for x in seq] for seq in runner.actionseqs]; additions=[]
 try:
  for rollout in rollouts:
   for lineage in rollout['lineages']:
    benchmark=lineage['benchmark']; observation=runner.reset(benchmark); ordered=[int(x) for x in runner.get_model_action(observation)]; donors=[coreset[i] for i in ordered[:10]]; parent=coreset[lineage['origin_parent_coreset_index']]
    for step in lineage['steps']:
     if step['action']=='hold': continue
     edit=step['edit']; child=apply_edit(parent,donors,edit); delta=float(step['delta_reward']); certified=bool(step['child_certified'])
     additions.append({'benchmark':benchmark,'controller':{'parent_coreset_index':lineage['origin_parent_coreset_index'],'parent_rank':lineage['origin_parent_rank'],'parent_actions':parent,'ranked_donor_indices':ordered[:10],'autophase':[int(x) for x in observation]},'edit':edit,'child_actions':child,'outcome':{'certified_child':certified,'delta_reward':delta,'blame':False,'label':'certified_improvement' if certified and delta>EPSILON else 'certified_non_improvement'}})
     if step['action']=='promote': parent=child
 finally: runner.env.close(); runner.model.connection.close()
 rows=base_rows+additions; x=np.vstack([candidate_features(r) for r in rows]); y=np.asarray([r['outcome']['delta_reward']>EPSILON for r in rows],dtype=int)
 positive=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=4000,class_weight='balanced')); positive.fit(x,y)
 payload=dict(base); payload.update({'protocol':'rlcompopt-shadow-governor-v3','positive_model':positive,'shadow_only':True,'update_contract':'positive head updated from reconstructed certified selected-action rollout transitions; pairwise and blame heads frozen from V2'})
 a.model.parent.mkdir(parents=True,exist_ok=True); a.report.parent.mkdir(parents=True,exist_ok=True)
 with a.model.open('wb') as f: pickle.dump(payload,f)
 report={'protocol':payload['protocol'],'deployed':False,'base_model':str(a.base_model),'rollout_sources':[str(x) for x in a.rollout],'training':{'development_candidate_rows':len(base_rows),'reconstructed_rollout_rows':len(additions),'total_rows':len(rows),'positive_rows':int(y.sum()),'feature_dim':int(x.shape[1])},'safeguards':['pairwise head frozen: no counterfactual sibling outcomes in rollout data','blame head frozen: parent repeat-stability attribution was not recollected in rollout','V3 cannot replace V2 without blocked challenger evaluation']}
 a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
