"""Execute a short sequence of verified micro-edits from one controller state.

The proposal order is an explicit interface for the future MPC planner.  This
runner is agnostic: every proposed edit is still subject to commit/rollback.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from m14_transactional_v5 import paired_success,certify,rollback,snapshot
from run_m14_mt2_behavior_cloning import TASKS,cfg_for
from run_m14_mt2_grammar_sweep import apply_update,load_candidates
from run_m14_mt2_warmstart import collect

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--proposal-order',nargs='+',required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--controller-seed',type=int,required=True);p.add_argument('--execution-seed',type=int,required=True);p.add_argument('--verify-episodes',type=int,default=50);p.add_argument('--demo-episodes',type=int,default=50);p.add_argument('--alpha',type=float,default=.05);p.add_argument('--epsilon',type=float,default=.05);a=p.parse_args()
 records={r['label']:r for r in load_candidates(a.candidates)}
 if any(x not in records or x=='HOLD' for x in a.proposal_order):raise ValueError('proposal order must contain non-HOLD grammar labels')
 cfg=cfg_for(a.controller_seed);cfg.steps,cfg.buffer_size,cfg.batch_size=10000,20000,64;env=make_env(cfg);agent=TDMPC2(cfg);agent.load(a.checkpoint)
 trajectories,obs,actions,tasks=collect(env,a.demo_episodes);replay=Buffer(cfg)
 for tr in trajectories:replay.add(tr)
 history=[]
 for step,label in enumerate(a.proposal_order):
  seeds={task:[a.execution_seed+100000*step+10000*i+j for j in range(a.verify_episodes)] for i,task in enumerate(TASKS)}
  before=paired_success(agent,env,seeds); state=snapshot(agent); apply_update(agent,obs,actions,tasks,replay,records[label]['spec']);after=paired_success(agent,env,seeds)
  cert=certify(before,after,{TASKS[0]:.3,TASKS[1]:.7},{TASKS[0]:a.epsilon,TASKS[1]:a.epsilon},a.alpha)
  if not cert.committed:rollback(agent,state)
  history.append({'step':step,'label':label,'certificate':{'utility_delta':cert.utility_delta,'utility_lcb':cert.utility_lcb,'capability_delta':cert.capability_delta,'capability_lcb':cert.capability_lcb,'committed':cert.committed}})
  if not cert.committed: break
 a.run_dir.mkdir(parents=True,exist_ok=True);(a.run_dir/'trajectory.json').write_text(json.dumps({'protocol':'m14-transactional-sequence-v5','proposal_order':a.proposal_order,'history':history},indent=2)+'\n');
 if history and history[-1]['certificate']['committed']:agent.save(a.run_dir/'final_committed.pt')
 env.close();print(json.dumps({'attempts':len(history),'commits':sum(x['certificate']['committed'] for x in history)}))
if __name__=='__main__':main()
