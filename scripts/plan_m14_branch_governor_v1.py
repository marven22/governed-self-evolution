"""Freeze a V5 governor recommendation before any blind edit outcome is read."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from envs import make_env
from tdmpc2 import TDMPC2
from m14_transactional_v5 import paired_success
from run_m14_mt2_behavior_cloning import TASKS, cfg_for
from run_m14_mt2_grammar_sweep import load_candidates
from train_m14_branch_governor_v1 import features

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--candidates',type=Path,required=True); p.add_argument('--controller-id',required=True); p.add_argument('--controller-seed',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--episodes',type=int,default=50); a=p.parse_args()
 cfg=cfg_for(a.controller_seed); env=make_env(cfg); agent=TDMPC2(cfg); agent.load(a.checkpoint)
 seeds={t:[a.seed+10000*i+j for j in range(a.episodes)] for i,t in enumerate(TASKS)}; before=paired_success(agent,env,seeds); env.close()
 weights=np.asarray(json.loads(a.model.read_text())['weights'],float); choices=[]
 for c in load_candidates(a.candidates):
  if c['label']=='HOLD': continue
  row={'pre_capability':{t:float(before[t].mean()) for t in TASKS},'step':0,'update_spec':{'reach_fraction':c['spec'].reach_fraction,'gradient_steps':c['spec'].gradient_steps}}
  pred=features(row)@weights; safe=bool(np.all(pred[1:]>=-.05)); score=float(pred[0]) if safe and pred[0]>0 else 0.0
  choices.append({'label':c['label'],'predicted_utility':float(pred[0]),'predicted_capability_delta':{t:float(pred[i+1]) for i,t in enumerate(TASKS)},'safe':safe,'score':score})
 best=max(choices,key=lambda x:x['score']); chosen=best['label'] if best['score']>0 else 'HOLD'
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'protocol':'m14-branch-governor-v1-blind-plan','controller_id':a.controller_id,'pre_capability':{t:float(before[t].mean()) for t in TASKS},'chosen':chosen,'choices':choices},indent=2)+'\n'); print(json.dumps({'controller':a.controller_id,'chosen':chosen}))
if __name__=='__main__': main()
