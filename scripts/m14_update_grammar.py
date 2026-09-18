"""Typed, auditable grammar for persistent MT2 controller updates.

The grammar is deliberately an *action language*, not an unrestricted code
generation interface.  Every instance is serializable, validated, and maps to
a known update executor in ``run_m14_mt2_grammar_sweep.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


GRAMMAR_VERSION = "m14-update-grammar-v1"
TARGETS = {"hold", "policy", "world_model", "policy_and_world_model"}


@dataclass(frozen=True)
class UpdateSpec:
    """A valid persistent-update action for the two-task MetaWorld substrate."""

    target: str
    reach_fraction: float
    gradient_steps: int
    learning_rate: float
    freeze_encoder: bool
    prior_policy_l2: float
    protect_policy_during_model_update: bool

    def validate(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}; got {self.target!r}")
        if not 0.0 <= self.reach_fraction <= 1.0:
            raise ValueError("reach_fraction must lie in [0, 1]")
        if self.target == "hold":
            if self.gradient_steps != 0:
                raise ValueError("hold updates must use zero gradient_steps")
            return
        if not 1 <= self.gradient_steps <= 5_000:
            raise ValueError("gradient_steps must lie in [1, 5000]")
        if not 1e-5 <= self.learning_rate <= 1e-2:
            raise ValueError("learning_rate must lie in [1e-5, 1e-2]")
        if not 0.0 <= self.prior_policy_l2 <= 10.0:
            raise ValueError("prior_policy_l2 must lie in [0, 10]")
        if self.target == "world_model" and self.prior_policy_l2 != 0.0:
            raise ValueError("prior_policy_l2 applies to policy updates, not world_model-only updates")

    @property
    def pick_place_fraction(self) -> float:
        return 1.0 - self.reach_fraction

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "grammar_version": GRAMMAR_VERSION,
            "target": self.target,
            "task_data": {
                "reach_fraction": self.reach_fraction,
                "pick_place_fraction": self.pick_place_fraction,
            },
            "optimization": {
                "gradient_steps": self.gradient_steps,
                "learning_rate": self.learning_rate,
            },
            "retention": {
                "freeze_encoder": self.freeze_encoder,
                "prior_policy_l2": self.prior_policy_l2,
                "protect_policy_during_model_update": self.protect_policy_during_model_update,
            },
        }

    @property
    def identifier(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return f"u-{sha256(encoded).hexdigest()[:12]}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "UpdateSpec":
        # Accept the nested on-disk representation, but require every semantic
        # field.  This prevents silent defaults from changing an experiment.
        task_data = raw.get("task_data", raw)
        optimization = raw.get("optimization", raw)
        retention = raw.get("retention", raw)
        spec = cls(
            target=str(raw["target"]),
            reach_fraction=float(task_data["reach_fraction"]),
            gradient_steps=int(optimization["gradient_steps"]),
            learning_rate=float(optimization["learning_rate"]),
            freeze_encoder=bool(retention["freeze_encoder"]),
            prior_policy_l2=float(retention["prior_policy_l2"]),
            protect_policy_during_model_update=bool(retention["protect_policy_during_model_update"]),
        )
        spec.validate()
        return spec


def baseline_specs() -> dict[str, UpdateSpec]:
    """Earlier fixed choices represented exactly as grammar instances."""
    return {
        "HOLD": UpdateSpec("hold", 0.5, 0, 3e-4, True, 0.0, True),
        "POLICY_PICK": UpdateSpec("policy", 0.0, 1_000, 3e-4, False, 0.0, False),
        "POLICY_PROTECTED": UpdateSpec("policy", 0.5, 1_000, 3e-4, True, 0.0, False),
        "MODEL_PROTECTED": UpdateSpec("world_model", 0.5, 1_000, 3e-4, True, 0.0, True),
    }


def canonical_record(spec: UpdateSpec, *, label: str | None = None) -> dict[str, Any]:
    record = {"id": spec.identifier, "update": spec.to_dict()}
    if label is not None:
        record["label"] = label
    return record
