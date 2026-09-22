#!/usr/bin/env python3
"""Discover bounded governor scoring strategies; engineering diagnostic only."""
from __future__ import annotations
import argparse,json,pickle,random
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from train_rlcompopt_pairwise_governor_v2 import candidate_features

K=16; EPS=.01
def evaluate(strategy,base,archive,query,za,scaler):
 raw=np.vstack([candidate_features(r) for r in query]); x=scaler.transform(raw); p=base['positive_model'].predict_proba(raw)[:,1]; risk=base['blame_model'].predict_proba(raw)[:,1]; local=[]
 y=np.asarray([r['outcome']['delta_reward']>EPS for r in archive],float)
 for q in x:
  d=np.sum((za-q)**2,axis=1); ix=np.argpartition(d,K-1)[:K]; w=1/(np.sqrt(d[ix])+1e-6); local.append(float(w@y[ix]/w.sum()))
 groups=defaultdict(list)
 for i,r in enumerate(query): c=r['controller']; groups[(r['benchmark'],c.get('parent_rank',0),c['parent_coreset_index'])].append(i)
 delta=np.asarray([r['outcome']['delta_reward'] for r in query]); gains=[]; edits=0
 for ix in groups.values():
  rank=[]
  for i in ix:
   oth=[j for j in ix if j!=i]; rank.append(base['pair_model'].predict_proba(np.vstack([raw[i]-raw[j] for j in oth]))[:,1].mean() if oth else .5)
  belief=np.clip(p[ix]+strategy['archive_weight']*(np.asarray(local)[ix]-p[ix]),0,1); u=belief*np.asarray(rank)-strategy['risk_weight']*risk[ix]; j=ix[int(np.argmax(u))]; act=u.max()>0 and belief[list(ix).index(j)]>=strategy['threshold']; edits+=int(act); gains.append(max(0.,float(delta[j])) if act else 0.)
 return float(np.mean(gains)),edits
def main():
 p=argparse.ArgumentParser(); p.add_argument('--base-model',type=Path,required=True); p.add_argument('--development-ledger',type=Path,required=True); p.add_argument('--selection-ledger',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--generations',type=int,default=3); p.add_argument('--population',type=int,default=12); a=p.parse_args()
 base=pickle.loads(a.base_model.read_bytes()); dev=json.loads(a.development_ledger.read_text()); sel=json.loads(a.selection_ledger.read_text()); archive=[r for r in dev['transactions'] if r['outcome']['certified_child']]; query=[r for r in sel['transactions'] if r['outcome']['certified_child']]; xa=np.vstack([candidate_features(r) for r in archive]); scaler=StandardScaler().fit(xa); za=scaler.transform(xa)
 rng=random.Random(20260922); population=[{'risk_weight':1.,'archive_weight':0.,'threshold':.5,'origin':'incumbent'}]
 while len(population)<a.population: population.append({'risk_weight':rng.choice([.5,.75,1.,1.25,1.5,2.]),'archive_weight':rng.choice([-.5,-.25,0.,.25,.5]),'threshold':rng.choice([.35,.5,.65]),'origin':'grammar_seed'})
 history=[]
 for generation in range(a.generations):
  scored=[]
  for s in population:
   gain,edits=evaluate(s,base,archive,query,za,scaler); scored.append({**s,'mean_gain':gain,'edits':edits})
  scored.sort(key=lambda s:(s['mean_gain'],-s['risk_weight']),reverse=True); history.append({'generation':generation,'strategies':scored}); elites=scored[:max(2,a.population//3)]; population=[{k:v for k,v in e.items() if k in ('risk_weight','archive_weight','threshold')} for e in elites]
  while len(population)<a.population:
   parent=rng.choice(elites); population.append({'risk_weight':max(.1,min(3.,parent['risk_weight']+rng.choice([-.25,0,.25]))),'archive_weight':max(-1.,min(1.,parent['archive_weight']+rng.choice([-.25,0,.25]))),'threshold':max(.1,min(.9,parent['threshold']+rng.choice([-.15,0,.15]))),'origin':'mutation'})
 champion=max(history[-1]['strategies'],key=lambda s:s['mean_gain']); report={'protocol':'rlcompopt-governor-strategy-grammar-v1','scope':'adaptive engineering discovery on previously used selection evidence; not deployment evidence','grammar':{'score':'clip(p + archive_weight*(local_archive_p-p),0,1)*pairwise_rank - risk_weight*risk','parameters':{'risk_weight':[.1,3.],'archive_weight':[-1.,1.],'threshold':[.1,.9]}},'generations':history,'champion':champion,'incumbent_reference':{'risk_weight':1.,'archive_weight':0.,'threshold':.5}}
 a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({'champion':champion,'incumbent_gain':next(s['mean_gain'] for s in history[0]['strategies'] if s.get('origin')=='incumbent')},sort_keys=True))
if __name__=='__main__': main()
