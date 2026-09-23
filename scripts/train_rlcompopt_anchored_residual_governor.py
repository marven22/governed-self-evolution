#!/usr/bin/env python3
"""Train a conservative Csmith override model relative to frozen V2."""
from __future__ import annotations
import argparse,json,pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_rlcompopt_pairwise_governor_v2 import candidate_features

def v2_choice(model,x):
 p=model['positive_model'].predict_proba(x)[:,1]; r=model['blame_model'].predict_proba(x)[:,1]
 q=[]
 for i in range(len(x)):
  others=[j for j in range(len(x)) if j!=i]
  q.append(model['pair_model'].predict_proba(np.vstack([x[i]-x[j] for j in others]))[:,1].mean())
 u=p*np.asarray(q)-r; i=int(np.argmax(u)); return None if not(u[i]>0 and p[i]>=.5) else i
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 rows=json.loads(a.ledger.read_text())['transactions']; base=pickle.loads(a.base.read_bytes()); groups=defaultdict(list)
 for r in rows: groups[(r['benchmark'],r['controller']['parent_rank'],r['controller']['parent_coreset_index'])].append(r)
 xs=[];ys=[]; contexts=[]
 for key,g in groups.items():
  x=np.vstack([candidate_features(r) for r in g]); chosen=v2_choice(base,x); baseline=0. if chosen is None else max(0.,float(g[chosen]['outcome']['delta_reward']))
  xs.extend(x);ys.extend(float(r['outcome']['delta_reward'])-baseline for r in g);contexts.extend([key]*len(g))
 x=np.vstack(xs); y=np.asarray(ys); model=make_pipeline(StandardScaler(),BayesianRidge());model.fit(x,y)
 payload={'protocol':'rlcompopt-anchored-residual-v1','base_model':str(a.base),'advantage_model':model,'margin':.01,'contract':'override V2 only when a conservative residual advantage bound exceeds margin; otherwise retain V2'}
 a.model.parent.mkdir(parents=True,exist_ok=True);a.report.parent.mkdir(parents=True,exist_ok=True)
 pickle.dump(payload,a.model.open('wb')); report={'protocol':payload['protocol'],'contexts':len(groups),'rows':len(y),'positive_override_examples':int((y>.01).sum()),'mean_residual_advantage':float(y.mean()),'margin':.01,'limitation':'Training-only artifact; requires blocked calibration before any promotion.'};a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
