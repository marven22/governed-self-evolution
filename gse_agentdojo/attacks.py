"""Model-agnostic prompt-injection attack.

Upstream ``important_instructions_no_model_name`` resolves the target model's
display name in its constructor and raises for models it does not know, even
though it then discards that name.  This class produces byte-identical
injections without that lookup, so any configured model can be attacked.
"""
from __future__ import annotations

from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME, FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.task_suite.task_suite import TaskSuite

from .common import sha256_text

SUPPORTED_ATTACKS = ("important_instructions_no_model_name",)


class ModelAgnosticImportantInstructions(FixedJailbreakAttack):
    name = "important_instructions_no_model_name"

    def __init__(self, task_suite: TaskSuite) -> None:
        # The target pipeline is only used upstream to resolve the model name.
        super().__init__(ImportantInstructionsAttack._JB_STRING, task_suite, target_pipeline=None)  # type: ignore[arg-type]
        self.user_name = "Emma Johnson"
        self.model_name = DEFAULT_MODEL_NAME


def load_attack(name: str, task_suite: TaskSuite) -> FixedJailbreakAttack:
    if name not in SUPPORTED_ATTACKS:
        raise ValueError(f"attack must be one of {SUPPORTED_ATTACKS}; got {name!r}")
    return ModelAgnosticImportantInstructions(task_suite)


def attack_fingerprint(attack: FixedJailbreakAttack) -> dict[str, str]:
    return {
        "name": attack.name,
        "template_sha256": sha256_text(attack.jailbreak),
        "user_name": attack.user_name,
        "model_name": attack.model_name,
    }
