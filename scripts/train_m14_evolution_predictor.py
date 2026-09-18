"""First uncertainty-aware capability-transition predictor for M14."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pickle

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); rows=json.loads(a.data.read_text()); exact=[r for r in rows if r['pre_capability_is_exact']]
 if len(exact)<16: raise ValueError(f'Need at least 16 exact transition rows; found {len(exact)}')
 families=sorted({r['update']['family'] for r in exact})
 X=[];Y=[]
 for r in exact:
  one_hot=[float(r['update']['family']==family) for family in families]
  X.append([1.,r['pre_capability']['mw-reach'],r['pre_capability']['mw-pick-place'],r['update']['steps']/1500.,*one_hot])
  Y.append([r['post_capability']['mw-reach'],r['post_capability']['mw-pick-place']])
 X,Y=np.asarray(X),np.asarray(Y); ridge=1e-3*np.eye(X.shape[1]); weights=np.linalg.solve(X.T@X+ridge,X.T@Y)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('wb') as f: pickle.dump({'weights':weights,'families':families,'feature_order':['bias','pre_reach','pre_pick','steps_normalized',*families]},f)
 print(json.dumps({'exact_rows':len(exact),'features':len(X[0]),'targets':2}))
if __name__=='__main__':main()
