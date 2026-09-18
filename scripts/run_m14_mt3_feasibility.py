"""Online, task-labelled three-task TD-MPC2 substrate feasibility run.

This is intentionally separate from upstream TD-MPC2's offline mt80 trainer.
It establishes only that a shared task-conditioned model can collect, replay,
update, and evaluate the three M14 MetaWorld tasks.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common.buffer import Buffer
from envs import make_env
from envs.wrappers.timeout import Timeout
from tdmpc2 import TDMPC2

Timeout.max_episode_steps = property(lambda self: self._max_episode_steps, lambda self, value: setattr(self, "_max_episode_steps", value))
TASKS = ("mw-reach", "mw-pick-place", "mw-door-open")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cfg_for(seed, steps):
    cfg = OmegaConf.load("/home/vmargapu/src/tdmpc2/tdmpc2/config.yaml")
    cfg.task, cfg.tasks, cfg.multitask = "m14-mt3", list(TASKS), True
    cfg.seed, cfg.steps, cfg.model_size, cfg.task_dim = seed, steps, 5, 96
    cfg.mpc, cfg.compile, cfg.enable_wandb, cfg.save_csv, cfg.save_agent, cfg.save_video = True, False, False, False, False, False
    cfg.batch_size, cfg.buffer_size, cfg.eval_episodes = 64, max(steps, 10_000), 5
    cfg.bin_size = (cfg.vmax-cfg.vmin)/(cfg.num_bins-1)
    for key, value in MODEL_SIZE[cfg.model_size].items(): cfg[key] = value
    return cfg

def td(obs, env, task_idx, action=None, reward=None, terminated=None):
    action = env.rand_act() if action is None else action
    reward = torch.tensor(float("nan")) if reward is None else reward
    terminated = torch.tensor(float("nan")) if terminated is None else terminated
    return TensorDict({"obs": obs.unsqueeze(0).cpu(), "action": action.unsqueeze(0).cpu(),
                       "reward": reward.unsqueeze(0).cpu(), "terminated": terminated.unsqueeze(0).cpu(),
                       # One scalar task label per transition. Buffer.sample()
                       # converts this to one task ID per sampled trajectory.
                       "task": torch.tensor([task_idx], dtype=torch.long)}, batch_size=(1,))

@torch.no_grad()
def evaluate(agent, env, episodes):
    result = {}
    for task_idx, task in enumerate(TASKS):
        rewards, successes = [], []
        for _ in range(episodes):
            obs, done, total, t = env.reset(task_idx), False, 0., 0
            while not done:
                action = agent.act(obs, t0=t == 0, eval_mode=True, task=task_idx)
                obs, reward, done, info = env.step(action); total += float(reward); t += 1
            rewards.append(total); successes.append(float(info["success"]))
        result[task] = {"return": float(np.mean(rewards)), "success": float(np.mean(successes))}
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--eval-every", type=int, default=5_000)
    a = p.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed); cfg = cfg_for(a.seed, a.steps); env = make_env(cfg); agent = TDMPC2(cfg); replay = Buffer(cfg)
    records, global_step, episode, last_eval_step = [], 0, 0, 0
    while global_step < a.steps:
        task_idx = episode % len(TASKS); obs, done, trajectory, t = env.reset(task_idx), False, [], 0
        trajectory.append(td(obs, env, task_idx))
        while not done and global_step < a.steps:
            action = env.rand_act() if global_step < cfg.seed_steps else agent.act(obs, t0=t == 0, task=task_idx)
            obs, reward, done, info = env.step(action); trajectory.append(td(obs, env, task_idx, action, reward, info["terminated"]))
            t += 1; global_step += 1
            if replay.num_eps > 0 and global_step >= cfg.seed_steps: agent.update(replay)
        replay.add(torch.cat(trajectory)); episode += 1
        # Evaluation resets all task environments, so run it only between
        # collection episodes; never invalidate an in-progress trajectory.
        if global_step - last_eval_step >= a.eval_every:
            records.append({"step": global_step, "metrics": evaluate(agent, env, cfg.eval_episodes)})
            (a.run_dir / "progress.json").write_text(json.dumps(records, indent=2))
            last_eval_step = global_step
    final = {"config": {"seed": a.seed, "steps": a.steps, "tasks": TASKS}, "evaluations": records,
             "final": evaluate(agent, env, cfg.eval_episodes)}
    (a.run_dir / "results.json").write_text(json.dumps(final, indent=2)); agent.save(a.run_dir / "final.pt")
    (a.run_dir / "completion.json").write_text(json.dumps({"complete": True})); env.close()

if __name__ == "__main__": main()
