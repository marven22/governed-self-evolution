"""Validate MetaWorld experts through the exact TD-MPC2 wrapper used by M14."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from envs import make_env
from envs.wrappers.timeout import Timeout
from metaworld.policies import SawyerReachV2Policy, SawyerPickPlaceV2Policy, SawyerPushV2Policy

Timeout.max_episode_steps = property(lambda self: self._max_episode_steps, lambda self, value: setattr(self, "_max_episode_steps", value))
POLICIES = {"mw-reach": SawyerReachV2Policy, "mw-pick-place": SawyerPickPlaceV2Policy, "mw-push": SawyerPushV2Policy}

def make(task, seed):
    cfg = OmegaConf.load("/home/vmargapu/src/tdmpc2/tdmpc2/config.yaml")
    cfg.task, cfg.seed, cfg.model_size, cfg.multitask, cfg.task_dim = task, seed, 5, False, 0
    cfg.compile, cfg.enable_wandb, cfg.save_csv, cfg.save_agent, cfg.save_video = False, False, False, False, False
    cfg.tasks, cfg.bin_size = [task], (cfg.vmax-cfg.vmin)/(cfg.num_bins-1)
    return make_env(cfg)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--episodes", type=int, default=20); p.add_argument("--seed", type=int, default=101)
    a = p.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True); result = {}
    for task, policy_cls in POLICIES.items():
        env, policy, rows = make(task, a.seed), policy_cls(), []
        for ep in range(a.episodes):
            obs, done, ret = env.reset(), False, 0.
            while not done:
                action = torch.from_numpy(policy.get_action(obs.cpu().numpy()).astype(np.float32))
                obs, reward, done, info = env.step(action); ret += float(reward)
            rows.append({"episode": ep, "return": ret, "success": float(info["success"])})
        env.close(); result[task] = rows
    summary = {task: {"success": float(np.mean([x["success"] for x in rows])), "return": float(np.mean([x["return"] for x in rows]))} for task, rows in result.items()}
    (a.run_dir / "expert_episodes.json").write_text(json.dumps(result, indent=2)); (a.run_dir / "summary.json").write_text(json.dumps(summary, indent=2)); (a.run_dir / "completion.json").write_text(json.dumps({"complete": True}))

if __name__ == "__main__": main()
