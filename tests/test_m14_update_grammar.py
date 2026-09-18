from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from m14_update_grammar import UpdateSpec, baseline_specs
from m14_update_sampling import sample_task_balanced_ids


def test_baselines_are_valid_and_unique() -> None:
    specs = baseline_specs()
    assert set(specs) == {"HOLD", "POLICY_PICK", "POLICY_PROTECTED", "MODEL_PROTECTED"}
    assert len({spec.identifier for spec in specs.values()}) == len(specs)
    assert specs["HOLD"].gradient_steps == 0


def test_invalid_hold_and_world_model_penalty_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero gradient_steps"):
        UpdateSpec("hold", 0.5, 10, 3e-4, True, 0.0, True).validate()
    with pytest.raises(ValueError, match="world_model-only"):
        UpdateSpec("world_model", 0.5, 10, 3e-4, True, 0.1, True).validate()


def test_sampling_honors_zero_and_full_replay_boundaries(monkeypatch) -> None:
    # Avoid a CUDA dependency in the pure grammar test.
    original_randint = torch.randint
    monkeypatch.setattr(torch, "randint", lambda high, size, device=None: original_randint(high, size))
    tasks = torch.tensor([0] * 8 + [1] * 8)
    assert torch.all(tasks[sample_task_balanced_ids(tasks, 0.0, batch_size=16)] == 1)
    assert torch.all(tasks[sample_task_balanced_ids(tasks, 1.0, batch_size=16)] == 0)
