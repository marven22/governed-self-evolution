#!/usr/bin/env python3
"""One-shot evaluation of frozen pairwise governor v2 on selection evidence."""
from __future__ import annotations
import argparse, hashlib, json, pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from train_rlcompopt_pairwise_governor_v2 import candidate_features

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--selection-ledger',type=Path,required=True); p.add_argument('--selection-audit',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    audit=json.loads(a.selection_audit.read_text()); raw=a.selection_ledger.read_bytes()
    if not audit.get('passed_for_gain_and_blame_governor') or hashlib.sha256(raw).hexdigest()!=audit.get('ledger_sha256'): raise SystemExit('Refusing failed/changed selection ledger')
    ledger=json.loads(raw)
    if ledger.get('cohort')!='selection only': raise SystemExit('Selection-only ledger required')
    payload=pickle.loads(a.model.read_bytes())
    if payload.get('protocol')!='rlcompopt-pairwise-governor-v2': raise SystemExit('Unexpected model')
    rows=[r for r in ledger['transactions'] if r['outcome']['certified_child']]; x=np.vstack([candidate_features(r) for r in rows])
    if x.shape[1]!=payload['feature_dim']: raise SystemExit('Feature mismatch')
    pos=payload['positive_model'].predict_proba(x)[:,1]; risk=payload['blame_model'].predict_proba(x)[:,1]; delta=np.asarray([r['outcome']['delta_reward'] for r in rows])
    groups=defaultdict(list)
    for i,r in enumerate(rows): c=r['controller']; groups[(r['benchmark'],c.get('parent_rank',0),c['parent_coreset_index'])].append(i)
    observed=[]; random=[]; oracle=[]; decisions=[]
    for context,indices in sorted(groups.items()):
        # Candidate's predicted fraction of sibling wins is a pairwise ranking score.
        rank=[]
        for i in indices:
            opponents=[j for j in indices if j!=i]
            if opponents:
                diff=np.vstack([x[i]-x[j] for j in opponents]); wins=payload['pair_model'].predict_proba(diff)[:,1].mean()
            else: wins=0.5
            rank.append(wins)
        utility=pos[indices]*np.asarray(rank)-risk[indices]
        k=int(np.argmax(utility)); chosen=indices[k]
        # The estimated probability of a positive edit is compared to HOLD's known zero gain.
        do_edit=utility[k]>0.0 and pos[chosen]>=0.5
        gain=max(0.0,float(delta[chosen])) if do_edit else 0.0
        observed.append(gain); random.append(float(np.mean(np.maximum(0,delta[indices])))); oracle.append(float(max(0,np.max(delta[indices]))))
        decisions.append({'benchmark':context[0],'parent_rank':context[1],'candidate_count':len(indices),'action':'edit' if do_edit else 'hold','observed_gain':gain,'selected_label':rows[chosen]['outcome']['label'] if do_edit else 'hold','predicted_positive_probability':float(pos[chosen]),'predicted_blame_probability':float(risk[chosen]),'pairwise_rank_score':float(rank[k]),'utility':float(utility[k]),'oracle_gain':oracle[-1]})
    report={'protocol':'rlcompopt-pairwise-governor-v2-selection-evaluation','model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'selection_ledger_sha256':hashlib.sha256(raw).hexdigest(),'contexts':len(groups),'mean_certified_gain':{'governor':float(np.mean(observed)),'hold':0.0,'random_expected':float(np.mean(random)),'oracle':float(np.mean(oracle))},'governor_over_hold':float(np.mean(observed)),'governor_over_random_expected':float(np.mean(observed)-np.mean(random)),'governor_regret_to_oracle':float(np.mean(oracle)-np.mean(observed)),'actions':dict(Counter(d['action'] for d in decisions)),'selection_outcomes':dict(Counter(d['selected_label'] for d in decisions)),'context_decisions':decisions,'limitations':['Selection is diagnostic because v1 results influenced v2 design; do not make a final generalization claim.']}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:report[k] for k in ('contexts','mean_certified_gain','governor_over_hold','governor_over_random_expected','governor_regret_to_oracle','actions','selection_outcomes')},sort_keys=True))
if __name__=='__main__': main()
