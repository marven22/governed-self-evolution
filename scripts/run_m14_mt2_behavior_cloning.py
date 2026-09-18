"""Balanced two-task behavior-cloning gate for the M14 shared substrate."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from common import MODEL_SIZE
from envs import make_env
from envs.wrappers.timeout import Timeout
from metaworld.policies import SawyerReachV2Policy, SawyerPickPlaceV2Policy
from tdmpc2 import TDMPC2

Timeout.max_episode_steps = property(lambda self: self._max_episode_steps, lambda self, value: setattr(self, "_max_episode_steps", value))
TASKS, POLICIES = ("mw-reach", "mw-pick-place"), (SawyerReachV2Policy, SawyerPickPlaceV2Policy)

def cfg_for(seed):
    cfg = OmegaConf.load("/home/vmargapu/src/tdmpc2/tdmpc2/config.yaml")
    cfg.task, cfg.tasks, cfg.multitask, cfg.seed = "m14-mt2", list(TASKS), True, seed
    cfg.model_size, cfg.task_dim, cfg.mpc, cfg.compile = 5, 96, False, False
    cfg.enable_wandb, cfg.save_csv, cfg.save_agent, cfg.save_video = False, False, False, False
    cfg.bin_size = (cfg.vmax-cfg.vmin)/(cfg.num_bins-1)
    for key, value in MODEL_SIZE[cfg.model_size].items(): cfg[key] = value
    return cfg

def collect(env, episodes):
    obs, actions, tasks, rows = [], [], [], {}
    for task_idx, (task, policy_cls) in enumerate(zip(TASKS, POLICIES)):
        policy, task_rows = policy_cls(), []
        for ep in range(episodes):
            state, done, ret = env.reset(task_idx), False, 0.
            while not done:
                action = torch.from_numpy(policy.get_action(state.cpu().numpy()).astype(np.float32))
                obs.append(state); actions.append(action); tasks.append(task_idx)
                state, reward, done, info = env.step(action); ret += float(reward)
            task_rows.append({"episode": ep, "return": ret, "success": float(info["success"])})
        rows[task] = task_rows
    return torch.stack(obs), torch.stack(actions), torch.tensor(tasks, dtype=torch.long), rows

@torch.no_grad()
def evaluate(agent, env, episodes):
    result = {}
    agent.cfg.mpc = False
    for task_idx, task in enumerate(TASKS):
        returns, successes = [], []
        for _ in range(episodes):
            obs, done, ret, t = env.reset(task_idx), False, 0., 0
            while not done:
                action = agent.act(obs, t0=t == 0, eval_mode=True, task=task_idx)
                obs, reward, done, info = env.step(action); ret += float(reward); t += 1
            returns.append(ret); successes.append(float(info["success"]))
        result[task] = {"return": float(np.mean(returns)), "success": float(np.mean(successes))}
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--demo-episodes", type=int, default=50); p.add_argument("--updates", type=int, default=5000); p.add_argument("--eval-episodes", type=int, default=20); p.add_argument("--seed", type=int, default=101)
    a = p.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = cfg_for(a.seed); env = make_env(cfg); agent = TDMPC2(cfg)
    obs, actions, tasks, expert_rows = collect(env, a.demo_episodes)
    torch.save({"obs": obs, "action": actions, "task": tasks}, a.run_dir / "expert_mt2.pt")
    expert_summary = {task: {"success": float(np.mean([x["success"] for x in rows])), "return": float(np.mean([x["return"] for x in rows]))} for task, rows in expert_rows.items()}
    optim = torch.optim.Adam(list(agent.model._encoder.parameters()) + list(agent.model._pi.parameters()) + list(agent.model._task_emb.parameters()), lr=3e-4)
    obs, actions, tasks = obs.cuda(), actions.cuda(), tasks.cuda(); agent.model.train()
    losses = []
    for _ in range(a.updates):
        # Exactly balanced task minibatch prevents reach from dominating the actor.
        idx0 = torch.randint(0, int((tasks == 0).sum()), (128,), device="cuda")
        idx1 = torch.randint(0, int((tasks == 1).sum()), (128,), device="cuda")
        ids0, ids1 = torch.where(tasks == 0)[0][idx0], torch.where(tasks == 1)[0][idx1]
        ids = torch.cat((ids0, ids1)); z = agent.model.encode(obs[ids], tasks[ids]); _, info = agent.model.pi(z, tasks[ids])
        loss = torch.nn.functional.mse_loss(info["mean"], actions[ids]); loss.backward(); torch.nn.utils.clip_grad_norm_(optim.param_groups[0]["params"], 20); optim.step(); optim.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
    agent.model.eval(); final = evaluate(agent, env, a.eval_episodes)
    result = {"expert": expert_summary, "bc_loss_final": float(np.mean(losses[-100:])), "policy_eval": final}
    (a.run_dir / "results.json").write_text(json.dumps(result, indent=2)); agent.save(a.run_dir / "policy_bc.pt"); (a.run_dir / "completion.json").write_text(json.dumps({"complete": True})); env.close()

if __name__ == "__main__": main()
