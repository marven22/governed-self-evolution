#!/usr/bin/env python3
"""Archive-conditioned V4 ablation, trained only on development evidence."""
from __future__ import annotations
import argparse,json,pickle
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from train_rlcompopt_pairwise_governor_v2 import candidate_features

K=16; ALPHA=.25; EPS=.01
def main():
 p=argparse.ArgumentParser(); p.add_argument('--base-model',type=Path,required=True); p.add_argument('--development-ledger',type=Path,required=True); p.add_argument('--selection-ledger',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
 base=pickle.loads(a.base_model.read_bytes()); dev=json.loads(a.development_ledger.read_text()); sel=json.loads(a.selection_ledger.read_text())
 archive=[r for r in dev['transactions'] if r['outcome']['certified_child']]; query=[r for r in sel['transactions'] if r['outcome']['certified_child']]
 xa=np.vstack([candidate_features(r) for r in archive]); ya=np.asarray([r['outcome']['delta_reward']>EPS for r in archive],float); scaler=StandardScaler().fit(xa); za=scaler.transform(xa); x=scaler.transform(np.vstack([candidate_features(r) for r in query]))
 base_pos=base['positive_model'].predict_proba(np.vstack([candidate_features(r) for r in query]))[:,1]; risk=base['blame_model'].predict_proba(np.vstack([candidate_features(r) for r in query]))[:,1]
 corrected=[]; supports=[]
 for q,p0 in zip(x,base_pos):
  dist=np.sum((za-q)**2,axis=1); idx=np.argpartition(dist,min(K,len(dist))-1)[:K]; w=1/(np.sqrt(dist[idx])+1e-6); local=float(np.dot(w,ya[idx])/w.sum()); corrected.append(float(np.clip(p0+ALPHA*(local-p0),0,1))); supports.append(float(np.mean(np.sqrt(dist[idx]))))
 raw=np.vstack([candidate_features(r) for r in query]); ranks=[]
 for i in range(len(query)):
  # Ranking stays frozen; only the archive-calibrated positive belief changes.
  # The ranking model is evaluated against all candidates, then grouped below.
  ranks.append(0.)
 groups=defaultdict(list)
 for i,r in enumerate(query): c=r['controller']; groups[(r['benchmark'],c.get('parent_rank',0),c['parent_coreset_index'])].append(i)
 delta=np.asarray([r['outcome']['delta_reward'] for r in query]); gains=[]; base_gains=[]; actions=[]; labels=[]
 for idx in groups.values():
  rank=[]
  for i in idx:
   others=[j for j in idx if j!=i]; d=np.vstack([raw[i]-raw[j] for j in others]); rank.append(base['pair_model'].predict_proba(d)[:,1].mean() if others else .5)
  util=np.asarray([corrected[i] for i in idx])*np.asarray(rank)-risk[idx]; j=idx[int(np.argmax(util))]; edit=bool(util.max()>0 and corrected[j]>=.5); gains.append(max(0.,float(delta[j])) if edit else 0.); actions.append('edit' if edit else 'hold'); labels.append(query[j]['outcome']['label'] if edit else 'hold')
  b_util=base_pos[idx]*np.asarray(rank)-risk[idx]; bj=idx[int(np.argmax(b_util))]; be=bool(b_util.max()>0 and base_pos[bj]>=.5); base_gains.append(max(0.,float(delta[bj])) if be else 0.)
 report={'protocol':'rlcompopt-archive-governor-v4-ablation-v1','training':'development ledger only','mechanism':{'neighbors':K,'alpha':ALPHA,'description':'bounded KNN correction of V2 positive-improvement probability; pairwise and blame heads frozen'},'contexts':len(groups),'mean_gain':{'v4':float(np.mean(gains)),'v2':float(np.mean(base_gains))},'v4_minus_v2':float(np.mean(gains)-np.mean(base_gains)),'actions':dict(Counter(actions)),'selected_outcomes':dict(Counter(labels)),'mean_neighbor_distance':float(np.mean(supports)),'limitations':['Selection was previously used diagnostically; this is an ablation, not final evidence.','No V4 deployment or controller rollout is authorized by this result.']}
 a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
