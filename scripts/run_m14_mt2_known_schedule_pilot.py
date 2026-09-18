from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from run_m14_mt2_behavior_cloning import TASKS,cfg_for,evaluate
from run_m14_mt2_warmstart import collect

DEMAND={"mw-reach":.30,"mw-pick-place":.70}
def bc(agent,obs,actions,tasks,updates,protected):
    params=list(agent.model._pi.parameters())+list(agent.model._task_emb.parameters())
    if not protected: params+=list(agent.model._encoder.parameters())
    opt=torch.optim.Adam(params,lr=3e-4); obs,actions,tasks=obs.cuda(),actions.cuda(),tasks.cuda()
    for _ in range(updates):
        groups=[torch.where(tasks==i)[0] for i in range(2)]
        ids=torch.cat([g[torch.randint(len(g),(128,),device='cuda')] for g in groups if len(g)])
        z=agent.model.encode(obs[ids],tasks[ids]); _,info=agent.model.pi(z,tasks[ids]); loss=torch.nn.functional.mse_loss(info['mean'],actions[ids]); loss.backward(); opt.step();opt.zero_grad(set_to_none=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--eval-episodes',type=int,default=50);p.add_argument('--demo-episodes',type=int,default=100);a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=True)
 cfg=cfg_for(a.seed);cfg.steps=10000;cfg.buffer_size=20000;cfg.batch_size=64;env=make_env(cfg)
 base_agent=TDMPC2(cfg);base_agent.load(a.checkpoint); pre_capability=evaluate(base_agent,env,a.eval_episodes)
 trajectories,obs,actions,tasks=collect(env,a.demo_episodes); results={}
 for name in ('HOLD','POLICY_PICK','POLICY_PROTECTED','MODEL_PROTECTED'):
  agent=TDMPC2(cfg);agent.load(a.checkpoint)
  if name=='POLICY_PICK': bc(agent,obs[tasks==1],actions[tasks==1],torch.zeros_like(tasks[tasks==1])+1,a.updates,False)
  elif name=='POLICY_PROTECTED': bc(agent,obs,actions,tasks,a.updates,True)
  elif name=='MODEL_PROTECTED':
   replay=Buffer(cfg)
   for tr in trajectories[len(trajectories)//2:]: replay.add(tr)
   protected={k:v.detach().clone() for k,v in agent.model.state_dict().items() if isinstance(v,torch.Tensor) and k.startswith(('_encoder','_pi','_task_emb'))}
   for _ in range(a.updates):
    agent.update(replay); state=agent.model.state_dict();state.update(protected);agent.model.load_state_dict(state)
  metrics=evaluate(agent,env,a.eval_episodes); utility=sum(DEMAND[t]*metrics[t]['success'] for t in TASKS); feasible=metrics['mw-reach']['success']>=0.15
  results[name]={'metrics':metrics,'utility':utility,'feasible':feasible};agent.save(a.run_dir/f'{name.lower()}.pt')
 metadata={'seed':a.seed,'update_steps':a.updates,'demo_episodes_per_task':a.demo_episodes,'eval_episodes':a.eval_episodes,'checkpoint':str(a.checkpoint),'pre_capability':pre_capability}
 (a.run_dir/'results.json').write_text(json.dumps({'metadata':metadata,'demand':DEMAND,'results':results},indent=2));(a.run_dir/'completion.json').write_text(json.dumps({'complete':True}));env.close()
if __name__=='__main__':main()
