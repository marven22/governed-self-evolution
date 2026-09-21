"""Horizon-one branch--certify--commit evolution control."""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from m14_transactional_v5 import paired_success,certify,rollback,snapshot
from run_m14_mt2_behavior_cloning import TASKS,cfg_for
from run_m14_mt2_grammar_sweep import apply_update,load_candidates
from run_m14_mt2_warmstart import collect
def ev(agent,env,seed,n):
 return paired_success(agent,env,{t:[seed+10000*i+j for j in range(n)] for i,t in enumerate(TASKS)})
def cert_record(cert):
 return {'utility_delta':cert.utility_delta,'utility_lcb':cert.utility_lcb,'capability_delta':cert.capability_delta,'capability_lcb':cert.capability_lcb,'committed':cert.committed}
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--labels',nargs='+',default=['MICRO_25_75R','MICRO_25_25R','MICRO_25_50R']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--controller-id',default=None);p.add_argument('--controller-seed',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--max-commits',type=int,default=3);p.add_argument('--screen-episodes',type=int,default=50);p.add_argument('--confirm-episodes',type=int,default=200);p.add_argument('--alpha',type=float,default=.05);p.add_argument('--epsilon',type=float,default=.05);a=p.parse_args(); specs={r['label']:r['spec'] for r in load_candidates(a.candidates)}
 cfg=cfg_for(a.controller_seed);cfg.steps,cfg.buffer_size,cfg.batch_size=10000,20000,64;env=make_env(cfg);agent=TDMPC2(cfg);agent.load(a.checkpoint);tr,obs,acts,tasks=collect(env,50);replay=Buffer(cfg)
 for x in tr:replay.add(x)
 history=[]
 for step in range(a.max_commits):
  base=snapshot(agent); candidates=[]
  for j,label in enumerate(a.labels):
   before=ev(agent,env,a.seed+100000*step+1000*j,a.screen_episodes);apply_update(agent,obs,acts,tasks,replay,specs[label]);after=ev(agent,env,a.seed+100000*step+1000*j,a.screen_episodes);screen=certify(before,after,{TASKS[0]:.3,TASKS[1]:.7},{TASKS[0]:a.epsilon,TASKS[1]:a.epsilon},a.alpha);state=snapshot(agent);rollback(agent,base)
   # Only branches with observed nonnegative utility and retention consume the
   # larger confirmation budget; all other branches are rejected cheaply.
   promising=screen.utility_delta>0 and all(v>=-a.epsilon for v in screen.capability_delta.values())
   final=screen
   if promising:
    # Confirmation must compare the original branch point to its edited
    # successor on exactly the same task instances.
    confirm_seed=a.seed+500000+100000*step+1000*j
    rollback(agent,base); before2=ev(agent,env,confirm_seed,a.confirm_episodes)
    rollback(agent,state); after2=ev(agent,env,confirm_seed,a.confirm_episodes)
    final=certify(before2,after2,{TASKS[0]:.3,TASKS[1]:.7},{TASKS[0]:a.epsilon,TASKS[1]:a.epsilon},a.alpha);rollback(agent,base)
   candidates.append((label,state,screen,final,promising,{
    'controller_id':a.controller_id or a.run_dir.name,
    'parent_id':f'p{a.controller_seed}', 'step':step, 'label':label,
    'pre_capability':{t:float(before[t].mean()) for t in TASKS},
    'update_spec':asdict(specs[label]), 'screen_certificate':cert_record(screen),
    'confirmation_certificate':cert_record(final) if promising else None,
   }))
  accepted=[x for x in candidates if x[3].committed]
  if not accepted:
   history.append({'step':step,'committed':False,'branches':[dict(x[5],promising=x[4],confirmed=x[3].committed) for x in candidates]});break
  chosen=max(accepted,key=lambda x:x[3].utility_lcb);rollback(agent,chosen[1]);history.append({'step':step,'committed':True,'chosen':chosen[0],'utility_lcb':chosen[3].utility_lcb,'branches':[{'label':x[0],'screen_delta':x[2].utility_delta,'promising':x[4],'confirmed':x[3].committed} for x in candidates]})
  history[-1]['branches']=[dict(x[5],promising=x[4],confirmed=x[3].committed) for x in candidates]
 a.run_dir.mkdir(parents=True,exist_ok=True);(a.run_dir/'trajectory.json').write_text(json.dumps({'protocol':'m14-branch-certify-v5','history':history},indent=2)+'\n');
 if history and history[-1]['committed']:agent.save(a.run_dir/'final_committed.pt')
 env.close();print(json.dumps({'steps':len(history),'commits':sum(x['committed'] for x in history)}))
if __name__=='__main__':main()
