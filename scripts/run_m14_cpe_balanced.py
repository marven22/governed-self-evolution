"""Balanced paired state--edit collection for Certified Pessimistic Evolution.

Every grammar edit is independently branched from the same frozen controller
state.  This deliberately avoids path-dependent supervision.
"""
from __future__ import annotations
import argparse, json, random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from m14_transactional_v5 import certify, make_exact_episode_bank, rollback, snapshot, success_from_exact_episode_bank
from m14_v3_state import plasticity_state
from run_m14_mt2_behavior_cloning import TASKS, cfg_for
from run_m14_mt2_grammar_sweep import apply_update, load_candidates
from run_m14_mt2_warmstart import collect

def record(cert):
 return {'utility_delta':cert.utility_delta,'utility_lcb':cert.utility_lcb,'capability_delta':cert.capability_delta,'capability_lcb':cert.capability_lcb}

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--candidates',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--controller-id',required=True); p.add_argument('--parent-id',required=True); p.add_argument('--controller-seed',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--demo-episodes',type=int,default=50); p.add_argument('--episodes',type=int,default=50); p.add_argument('--replicates',type=int,default=2); p.add_argument('--alpha',type=float,default=.05); p.add_argument('--epsilon',type=float,default=.05); a=p.parse_args()
 specs=load_candidates(a.candidates); cfg=cfg_for(a.controller_seed); cfg.steps,cfg.buffer_size,cfg.batch_size=10000,20000,64; env=make_env(cfg); agent=TDMPC2(cfg); agent.load(a.checkpoint)
 tr,obs,acts,tasks=collect(env,a.demo_episodes); replay=Buffer(cfg)
 for x in tr: replay.add(x)
 state=plasticity_state(agent,obs,acts,tasks); base=snapshot(agent); rows=[]
 try:
  for rep in range(a.replicates):
   for j,candidate in enumerate(specs):
    seed=a.seed+100000*rep+1000*j; seeds={t:[seed+10000*i+k for k in range(a.episodes)] for i,t in enumerate(TASKS)}
    rollback(agent,base); bank=make_exact_episode_bank(agent,env,seeds); before=success_from_exact_episode_bank(agent,env,bank)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); apply_update(agent,obs,acts,tasks,replay,candidate['spec']); after=success_from_exact_episode_bank(agent,env,bank)
    cert=certify(before,after,{TASKS[0]:.5,TASKS[1]:.5},{TASKS[0]:a.epsilon,TASKS[1]:a.epsilon},a.alpha)
    rows.append({'schema':'m14-cpe-balanced-v1','controller_id':a.controller_id,'parent_id':a.parent_id,'replicate':rep,'label':candidate['label'],'update_spec':asdict(candidate['spec']),'pre_capability':{t:float(before[t].mean()) for t in TASKS},'pre_update_state':state,'certificate':record(cert),'paired_episode_count':a.episodes})
    rollback(agent,base); a.run_dir.mkdir(parents=True,exist_ok=True); (a.run_dir/'transitions.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
 finally: env.close()
 a.run_dir.mkdir(parents=True,exist_ok=True); (a.run_dir/'transitions.json').write_text(json.dumps(rows,indent=2)+'\n'); (a.run_dir/'completion.json').write_text(json.dumps({'complete':True,'records':len(rows)},indent=2)+'\n'); print(json.dumps({'records':len(rows),'controller':a.controller_id}))
if __name__=='__main__': main()
