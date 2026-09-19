"""Transactional verification primitives for governed persistent updates.

These functions deliberately separate *proposal* from *commitment*. A planner
may be wrong; only a paired empirical certificate can make an update durable.
"""
from __future__ import annotations

from dataclasses import dataclass
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
