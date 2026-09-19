"""Execute one reversible V5 micro-edit on a development checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from m14_update_grammar import UpdateSpec
from m14_transactional_v5 import paired_success, certify, rollback, snapshot
from run_m14_mt2_behavior_cloning import TASKS, cfg_for
from run_m14_mt2_grammar_sweep import apply_update, load_candidates
from run_m14_mt2_warmstart import collect

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--candidates',type=Path,required=True); p.add_argument('--candidate-label',required=True); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--controller-seed',type=int,required=True); p.add_argument('--execution-seed',type=int,required=True); p.add_argument('--demo-episodes',type=int,default=50); p.add_argument('--verify-episodes',type=int,default=50); p.add_argument('--alpha',type=float,default=.05); p.add_argument('--epsilon',type=float,default=.05); a=p.parse_args()
    records=load_candidates(a.candidates); selected=[r for r in records if r['label']==a.candidate_label]
    if len(selected)!=1 or selected[0]['label']=='HOLD': raise ValueError('select exactly one non-HOLD typed micro-edit')
    spec: UpdateSpec=selected[0]['spec']; cfg=cfg_for(a.controller_seed); cfg.steps,cfg.buffer_size,cfg.batch_size=10000,20000,64
    env=make_env(cfg); agent=TDMPC2(cfg); agent.load(a.checkpoint)
    trajectories,obs,actions,tasks=collect(env,a.demo_episodes); replay=Buffer(cfg)
    for trajectory in trajectories: replay.add(trajectory)
    seeds={task:[a.execution_seed+10000*i+j for j in range(a.verify_episodes)] for i,task in enumerate(TASKS)}
    before=paired_success(agent,env,seeds); state=snapshot(agent); apply_update(agent,obs,actions,tasks,replay,spec); after=paired_success(agent,env,seeds)
    certificate=certify(before,after,{TASKS[0]:.3,TASKS[1]:.7},{TASKS[0]:a.epsilon,TASKS[1]:a.epsilon},a.alpha)
    if not certificate.committed: rollback(agent,state)
    payload={'protocol':'m14-transactional-v5','candidate_id':selected[0]['id'],'label':selected[0]['label'],'controller_seed':a.controller_seed,'execution_seed':a.execution_seed,'verify_episodes':a.verify_episodes,'alpha':a.alpha,'epsilon':a.epsilon,'certificate':{'utility_delta':certificate.utility_delta,'utility_lcb':certificate.utility_lcb,'capability_delta':certificate.capability_delta,'capability_lcb':certificate.capability_lcb,'committed':certificate.committed}}
    a.run_dir.mkdir(parents=True,exist_ok=True); (a.run_dir/'result.json').write_text(json.dumps(payload,indent=2)+'\n')
    if certificate.committed: agent.save(a.run_dir/'committed.pt')
    env.close(); print(json.dumps(payload['certificate']))
if __name__=='__main__': main()
