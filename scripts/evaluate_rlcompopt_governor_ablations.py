#!/usr/bin/env python3
"""Controlled selection-ledger ablations for the prospective pairwise governor."""
from __future__ import annotations
import argparse, hashlib, json, pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from train_rlcompopt_pairwise_governor_v2 import candidate_features

VARIANTS = {
    "child_recipe_no_pairwise": "positive-probability minus blame risk",
    "pairwise_no_blame": "positive-probability times sibling-rank score",
    "full_v2": "positive-probability times sibling-rank score minus blame risk",
}

def evaluate_variant(name, x, rows, groups, payload):
    pos=payload['positive_model'].predict_proba(x)[:,1]
    risk=payload['blame_model'].predict_proba(x)[:,1]
    delta=np.asarray([r['outcome']['delta_reward'] for r in rows])
    gains=[]; random=[]; oracle=[]; labels=[]; actions=[]
    for indices in groups.values():
        ranks=[]
        for i in indices:
            others=[j for j in indices if j!=i]
            ranks.append(payload['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in others]))[:,1].mean() if others else .5)
        ranks=np.asarray(ranks)
        if name=='child_recipe_no_pairwise': utility=pos[indices]-risk[indices]
        elif name=='pairwise_no_blame': utility=pos[indices]*ranks
        else: utility=pos[indices]*ranks-risk[indices]
        k=int(np.argmax(utility)); selected=indices[k]
        do_edit=utility[k]>0 and pos[selected]>=.5
        gain=max(0.,float(delta[selected])) if do_edit else 0.
        gains.append(gain); random.append(float(np.maximum(0,delta[indices]).mean())); oracle.append(float(max(0,delta[indices].max())))
        actions.append('edit' if do_edit else 'hold'); labels.append(rows[selected]['outcome']['label'] if do_edit else 'hold')
    return {'mean_certified_gain':float(np.mean(gains)),'over_hold':float(np.mean(gains)),'over_random_expected':float(np.mean(gains)-np.mean(random)),'regret_to_oracle':float(np.mean(oracle)-np.mean(gains)),'actions':dict(Counter(actions)),'selected_outcomes':dict(Counter(labels))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--v2-model',type=Path,required=True); p.add_argument('--selection-ledger',type=Path,required=True); p.add_argument('--selection-audit',type=Path,required=True); p.add_argument('--v1-report',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    raw=a.selection_ledger.read_bytes(); audit=json.loads(a.selection_audit.read_text())
    if not audit.get('passed_for_gain_and_blame_governor') or hashlib.sha256(raw).hexdigest()!=audit.get('ledger_sha256'): raise SystemExit('selection audit mismatch')
    ledger=json.loads(raw)
    if ledger.get('cohort')!='selection only': raise SystemExit('selection-only ledger required')
    payload=pickle.loads(a.v2_model.read_bytes()); rows=[r for r in ledger['transactions'] if r['outcome']['certified_child']]; x=np.vstack([candidate_features(r) for r in rows])
    groups=defaultdict(list)
    for i,r in enumerate(rows): c=r['controller']; groups[(r['benchmark'],c.get('parent_rank',0),c['parent_coreset_index'])].append(i)
    results={name:evaluate_variant(name,x,rows,groups,payload) for name in VARIANTS}
    v1=json.loads(a.v1_report.read_text())
    results['v1_coarse']={'mean_certified_gain':v1['mean_certified_gain']['governor'],'over_hold':v1['governor_over_hold'],'over_random_expected':v1['governor_over_random_expected'],'regret_to_oracle':v1['governor_regret_to_oracle'],'actions':{'edit':round(v1['governor_edit_rate']*v1['contexts']),'hold':round((1-v1['governor_edit_rate'])*v1['contexts'])}}
    report={'protocol':'rlcompopt-governor-v2-ablation-v1','selection_ledger_sha256':hashlib.sha256(raw).hexdigest(),'contexts':len(groups),'shared_training':'All learned components were fitted only on the development ledger. This selection ledger was used only for diagnosis.','variants':VARIANTS,'results':results,'limitations':['This is an adaptive development-validation comparison, not final held-out evidence.']}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({'contexts':len(groups),'results':results},sort_keys=True))
if __name__=='__main__': main()
