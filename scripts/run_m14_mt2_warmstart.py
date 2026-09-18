"""Warm-start a shared TD-MPC2 world model without losing the DAgger policy."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
from tensordict import TensorDict

from common.buffer import Buffer
from envs import make_env
from envs.wrappers.timeout import Timeout
from tdmpc2 import TDMPC2
from run_m14_mt2_behavior_cloning import TASKS, POLICIES, cfg_for, evaluate

Timeout.max_episode_steps = property(lambda self: self._max_episode_steps, lambda self, value: setattr(self, "_max_episode_steps", value))

def collect(env, episodes):
    trajectories, flat_obs, flat_actions, flat_tasks = [], [], [], []
    for task_idx, policy_cls in enumerate(POLICIES):
        expert = policy_cls()
        for _ in range(episodes):
            obs, actions, rewards, done = [env.reset(task_idx)], [], [], False
            while not done:
                action = torch.from_numpy(expert.get_action(obs[-1].cpu().numpy()).astype("float32"))
                nxt, reward, done, _ = env.step(action); actions.append(action); rewards.append(reward); obs.append(nxt)
                flat_obs.append(obs[-2]); flat_actions.append(action); flat_tasks.append(task_idx)
            trajectories.append(TensorDict({"obs": torch.stack(obs), "action": torch.stack(actions + [actions[-1]]), "reward": torch.stack(rewards + [torch.tensor(0.)]), "terminated": torch.zeros(len(obs)), "task": torch.full((len(obs),), task_idx, dtype=torch.long)}, batch_size=[len(obs)]))
    return trajectories, torch.stack(flat_obs), torch.stack(flat_actions), torch.tensor(flat_tasks, dtype=torch.long)

def bc_step(agent, obs, actions, tasks, optim):
    groups = [torch.where(tasks == i)[0] for i in range(len(TASKS))]
    ids = torch.cat([g[torch.randint(len(g), (128,), device="cuda")] for g in groups])
    z = agent.model.encode(obs[ids], tasks[ids]); _, info = agent.model.pi(z, tasks[ids]); loss = torch.nn.functional.mse_loss(info["mean"], actions[ids])
    loss.backward(); torch.nn.utils.clip_grad_norm_(list(agent.model._encoder.parameters()) + list(agent.model._pi.parameters()) + list(agent.model._task_emb.parameters()), 20); optim.step(); optim.zero_grad(set_to_none=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--demo-episodes", type=int, default=100); p.add_argument("--updates", type=int, default=5000); p.add_argument("--eval-episodes", type=int, default=50); p.add_argument("--seed", type=int, default=101)
    a=p.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True); cfg=cfg_for(a.seed); cfg.steps=max(a.updates, 10_000); cfg.buffer_size=20_000; cfg.batch_size=64; env=make_env(cfg); agent=TDMPC2(cfg); agent.load(a.checkpoint)
    trajectories, obs, actions, tasks=collect(env, a.demo_episodes); replay=Buffer(cfg)
    for trajectory in trajectories: replay.add(trajectory)
    # The DAgger-validated policy, encoder, and task embedding define the
    # protected capability substrate. Learn predictive/value components only;
    # otherwise TD actor gradients can erase both skills before planning is
    # competent enough to constrain them.
    protected = {k: v.detach().clone() for k, v in agent.model.state_dict().items()
                 if isinstance(v, torch.Tensor) and k.startswith(("_encoder", "_pi", "_task_emb"))}
    for _ in range(a.updates):
        agent.update(replay)
        state = agent.model.state_dict(); state.update(protected); agent.model.load_state_dict(state)
    agent.model.eval(); agent.cfg.mpc=False; direct=evaluate(agent,env,a.eval_episodes); agent.cfg.mpc=True; mpc=evaluate(agent,env,a.eval_episodes)
    result={"direct_policy":direct,"mpc":mpc,"updates":a.updates,"demo_episodes_per_task":a.demo_episodes}; (a.run_dir/"results.json").write_text(json.dumps(result,indent=2)); agent.save(a.run_dir/"warmstart.pt"); (a.run_dir/"completion.json").write_text(json.dumps({"complete":True})); env.close()
if __name__=="__main__": main()
