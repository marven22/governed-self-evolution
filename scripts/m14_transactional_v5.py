"""Transactional verification primitives for governed persistent updates.

These functions deliberately separate *proposal* from *commitment*. A planner
may be wrong; only a paired empirical certificate can make an update durable.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from math import log, sqrt
from typing import Mapping

import numpy as np
import torch

from run_m14_mt2_behavior_cloning import TASKS


@dataclass(frozen=True)
class Certificate:
    utility_delta: float
    utility_lcb: float
    capability_delta: dict[str, float]
    capability_lcb: dict[str, float]
    committed: bool


def snapshot(agent) -> dict[str, torch.Tensor]:
    """Exact reversible policy state; no failed edit can persist."""
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in agent.model.state_dict().items()}


def rollback(agent, state: Mapping[str, torch.Tensor]) -> None:
    agent.model.load_state_dict(state)
    agent.model.eval()


def paired_success(agent, env, episode_seeds: Mapping[str, list[int]]) -> dict[str, np.ndarray]:
    """Evaluate on identical task-instance seeds for competing controller states."""
    result: dict[str, np.ndarray] = {}
    agent.cfg.mpc = False
    for task_idx, task in enumerate(TASKS):
        values=[]
        for seed in episode_seeds[task]:
            # MetaWorld's unwrapped environment owns its random-vector RNG.
            raw = env.envs[task_idx].unwrapped
            raw.seed(int(seed))
            obs, done, t = env.reset(task_idx), False, 0
            while not done:
                action = agent.act(obs, t0=t == 0, eval_mode=True, task=task_idx)
                obs, _, done, info = env.step(action); t += 1
            values.append(float(info['success']))
        result[task] = np.asarray(values, dtype=np.float64)
    return result


def make_exact_episode_bank(agent, env, episode_seeds: Mapping[str, list[int]]):
    """Capture post-reset MuJoCo states once for exact paired comparisons."""
    bank = {}
    agent.cfg.mpc = False
    for task_idx, task in enumerate(TASKS):
        entries = []
        raw = env.envs[task_idx].unwrapped
        for seed in episode_seeds[task]:
            raw.seed(int(seed))
            obs = env.reset(task_idx)
            # MetaWorld's physics snapshot omits reward/task variables such as
            # _target_pos. Preserve the reset-time values that affect reward
            # and termination as well as MuJoCo's physical state.
            names = ('_target_pos', '_last_rand_vec', 'obj_init_pos',
                     'obj_init_angle', 'init_tcp', 'init_left_pad',
                     'init_right_pad', 'goal', 'num_resets')
            task_state = {name: deepcopy(getattr(raw, name)) for name in names if hasattr(raw, name)}
            entries.append((raw.get_env_state(), task_state, obs.detach().clone() if isinstance(obs, torch.Tensor) else np.array(obs, copy=True)))
        bank[task] = entries
    return bank


def success_from_exact_episode_bank(agent, env, bank) -> dict[str, np.ndarray]:
    """Roll out an agent from saved simulator starts, never from a new reset."""
    result = {}
    agent.cfg.mpc = False
    for task_idx, task in enumerate(TASKS):
        values = []
        raw = env.envs[task_idx].unwrapped
        for state, task_state, initial_obs in bank[task]:
            # reset only restores wrapper bookkeeping; the saved MuJoCo state is
            # immediately reinstated and the saved observation is authoritative.
            env.reset(task_idx)
            raw.set_env_state(state)
            for name, value in task_state.items():
                setattr(raw, name, deepcopy(value))
            obs, done, t = initial_obs.detach().clone() if isinstance(initial_obs, torch.Tensor) else np.array(initial_obs, copy=True), False, 0
            while not done:
                action = agent.act(obs, t0=t == 0, eval_mode=True, task=task_idx)
                obs, _, done, info = env.step(action); t += 1
            values.append(float(info['success']))
        result[task] = np.asarray(values, dtype=np.float64)
    return result


def certify(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], demand: Mapping[str, float], epsilon: Mapping[str, float], alpha: float) -> Certificate:
    """Paired Hoeffding certificate with an explicit family-wise alpha budget.

    Success differences lie in [-1, 1].  The bound is conservative by design:
    a proposed edit is committed only when every protected capability clears it.
    """
    delta={task: float(np.mean(after[task]-before[task])) for task in TASKS}
    radius={task: sqrt(2.0 * log(2.0 * len(TASKS) / alpha) / len(before[task])) for task in TASKS}
    lcb={task: delta[task]-radius[task] for task in TASKS}
    utility_delta=sum(float(demand[t])*delta[t] for t in TASKS)
    utility_radius=sum(float(demand[t])*radius[t] for t in TASKS)
    utility_lcb=utility_delta-utility_radius
    committed=utility_lcb>0.0 and all(lcb[t]>=-float(epsilon[t]) for t in TASKS)
    return Certificate(utility_delta,utility_lcb,delta,lcb,committed)
