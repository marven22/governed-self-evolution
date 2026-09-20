"""Small archive-conditioned transition baseline; parent-held-out only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from train_m14_rich_governor_v3 import STATE_KEYS,TASKS
def x(r):
 u=r['update'];return np.array([r['pre_capability'][t] for t in TASKS]+[r['pre_update_state'][k] for k in STATE_KEYS]+[u['task_data']['reach_fraction'],u['optimization']['gradient_steps']/250],float)
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); rows=json.loads(a.archive.read_text()); parents=sorted({r['parent_id'] for r in rows}); report=[]
 for held in parents:
  train=[r for r in rows if r['parent_id']!=held];test=[r for r in rows if r['parent_id']==held]; X=np.stack([x(r) for r in train]);Y=np.array([r['outcome']['utility_delta'] for r in train]); lam=.1; w=np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@Y); pred=np.stack([x(r) for r in test])@w
  report.extend({'parent':held,'controller':r['controller_id'],'label':r['label'],'actual':float(r['outcome']['utility_delta']),'predicted':float(q),'committed':r['outcome']['committed']} for r,q in zip(test,pred))
 payload={'protocol':'m14-transactional-dynamics-v5','records':len(rows),'parents':parents,'leave_one_parent_out':report,'mae':float(np.mean([abs(r['actual']-r['predicted']) for r in report]))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'records':len(rows),'parents':parents,'lopo_mae':payload['mae']}))
if __name__=='__main__':main()
