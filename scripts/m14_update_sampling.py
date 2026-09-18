"""Sampling primitives shared by parameterized MetaWorld updates."""
from __future__ import annotations

import torch


def sample_task_balanced_ids(tasks: torch.Tensor, reach_fraction: float, batch_size: int = 128) -> torch.Tensor:
    """Sample an exact two-task replay mixture, including 0% and 100% edges."""
    groups = [torch.where(tasks == task_idx)[0] for task_idx in range(2)]
    reach_count = int(round(batch_size * reach_fraction))
    counts = (reach_count, batch_size - reach_count)
    sampled = []
    for group, count in zip(groups, counts):
        if count:
            if not len(group):
                raise ValueError("requested task has no demonstration transitions")
            sampled.append(group[torch.randint(len(group), (count,), device="cuda")])
    return torch.cat(sampled)
