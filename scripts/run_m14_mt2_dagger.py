"""Task-balanced DAgger correction for the two-capability M14 policy gate."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from envs import make_env
from envs.wrappers.timeout import Timeout
from tdmpc2 import TDMPC2
from run_m14_mt2_behavior_cloning import TASKS, POLICIES, cfg_for, evaluate

Timeout.max_episode_steps = property(lambda self: self._max_episode_steps, lambda self, value: setattr(self, "_max_episode_steps", value))

def train(agent, obs, actions, tasks, updates):
    optim = torch.optim.Adam(list(agent.model._encoder.parameters()) + list(agent.model._pi.parameters()) + list(agent.model._task_emb.parameters()), lr=3e-4)
    obs, actions, tasks = obs.cuda(), actions.cuda(), tasks.cuda(); agent.model.train()
    for _ in range(updates):
        groups = [torch.where(tasks == task_idx)[0] for task_idx in range(len(TASKS))]
        ids = torch.cat([group[torch.randint(len(group), (128,), device="cuda")] for group in groups])
        z = agent.model.encode(obs[ids], tasks[ids]); _, info = agent.model.pi(z, tasks[ids])
        loss = torch.nn.functional.mse_loss(info["mean"], actions[ids]); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.model._encoder.parameters()) + list(agent.model._pi.parameters()) + list(agent.model._task_emb.parameters()), 20)
        optim.step(); optim.zero_grad(set_to_none=True)
    agent.model.eval()

@torch.no_grad()
def aggregate_policy_states(agent, env, episodes):
    states, labels, task_ids = [], [], []
    agent.cfg.mpc = False
    for task_idx, policy_cls in enumerate(POLICIES):
        expert = policy_cls()
        for _ in range(episodes):
            obs, done, t = env.reset(task_idx), False, 0
            while not done:
                states.append(obs); labels.append(torch.from_numpy(expert.get_action(obs.cpu().numpy()).astype("float32"))); task_ids.append(task_idx)
                action = agent.act(obs, t0=t == 0, eval_mode=True, task=task_idx)
                obs, _, done, _ = env.step(action); t += 1
    return torch.stack(states), torch.stack(labels), torch.tensor(task_ids, dtype=torch.long)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--expert-data", type=Path, required=True); p.add_argument("--rounds", type=int, default=3); p.add_argument("--rollout-episodes", type=int, default=10); p.add_argument("--updates-per-round", type=int, default=2000); p.add_argument("--eval-episodes", type=int, default=50); p.add_argument("--seed", type=int, default=101)
    a = p.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True); cfg = cfg_for(a.seed); env = make_env(cfg); agent = TDMPC2(cfg); agent.load(a.checkpoint)
    data = torch.load(a.expert_data, weights_only=False); obs, actions, tasks = data["obs"], data["action"], data["task"]
    history = [{"round": 0, "dataset_steps": int(len(obs)), "eval": evaluate(agent, env, a.eval_episodes)}]
    for round_idx in range(1, a.rounds + 1):
        s, a_star, task = aggregate_policy_states(agent, env, a.rollout_episodes); obs, actions, tasks = torch.cat((obs, s)), torch.cat((actions, a_star)), torch.cat((tasks, task))
        train(agent, obs, actions, tasks, a.updates_per_round)
        history.append({"round": round_idx, "dataset_steps": int(len(obs)), "eval": evaluate(agent, env, a.eval_episodes)})
        (a.run_dir / "progress.json").write_text(json.dumps(history, indent=2))
    torch.save({"obs": obs, "action": actions, "task": tasks}, a.run_dir / "dagger_data.pt"); agent.save(a.run_dir / "policy_dagger.pt")
    (a.run_dir / "results.json").write_text(json.dumps({"history": history}, indent=2)); (a.run_dir / "completion.json").write_text(json.dumps({"complete": True})); env.close()

if __name__ == "__main__": main()
