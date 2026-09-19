"""Pre-update plasticity probes for the V3 governed-evolution state."""
from __future__ import annotations

from typing import Any

import torch


def _task_loss_and_grads(agent: Any, obs: torch.Tensor, actions: torch.Tensor, tasks: torch.Tensor, task: int, max_examples: int) -> tuple[float, list[torch.Tensor | None]]:
    ids = torch.where(tasks == task)[0][:max_examples]
    if not len(ids):
        raise ValueError(f"no demonstrations for task {task}")
    parameters = list(agent.model._pi.parameters()) + list(agent.model._task_emb.parameters())
    agent.model.zero_grad(set_to_none=True)
    z = agent.model.encode(obs[ids].cuda(), tasks[ids].cuda())
    _, info = agent.model.pi(z, tasks[ids].cuda())
    loss = torch.nn.functional.mse_loss(info["mean"], actions[ids].cuda())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return float(loss.detach().cpu()), [None if value is None else value.detach() for value in gradients]


def plasticity_state(agent: Any, obs: torch.Tensor, actions: torch.Tensor, tasks: torch.Tensor, *, max_examples: int = 1024) -> dict[str, float]:
    """Return task losses and policy-gradient conflict before self-modification.

    The two capability scores alone are not Markov for an update's consequence.
    These probes expose whether the two task objectives locally agree or fight
    in the current policy, without performing an update.
    """
    was_training = agent.model.training
    agent.model.eval()
    try:
        reach_loss, reach_gradients = _task_loss_and_grads(agent, obs, actions, tasks, 0, max_examples)
        pick_loss, pick_gradients = _task_loss_and_grads(agent, obs, actions, tasks, 1, max_examples)
        dot = norm_reach = norm_pick = 0.0
        for first, second in zip(reach_gradients, pick_gradients):
            if first is None or second is None:
                continue
            dot += float(torch.sum(first * second).cpu())
            norm_reach += float(torch.sum(first * first).cpu())
            norm_pick += float(torch.sum(second * second).cpu())
        denominator = (norm_reach * norm_pick) ** 0.5
        return {
            "schema_version": "m14-plasticity-state-v1",
            "bc_loss_reach": reach_loss,
            "bc_loss_pick_place": pick_loss,
            "policy_gradient_norm_reach": norm_reach ** 0.5,
            "policy_gradient_norm_pick_place": norm_pick ** 0.5,
            "policy_gradient_cosine": dot / denominator if denominator > 1e-12 else 0.0,
        }
    finally:
        agent.model.train(was_training)
