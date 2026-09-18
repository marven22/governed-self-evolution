from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from m14_update_grammar import UpdateSpec, baseline_specs


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
