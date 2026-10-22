#!/usr/bin/env python3
"""Write the versioned, self-describing grammar file the runner loads updates from.

Every named update is stored in canonical form with its content-hash id, beside
the exact system-message text and tool rules each level applies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gse_agentdojo.grammar import (
    DELIMITER_CLOSE,
    DELIMITER_OPEN,
    DESTRUCTIVE_TOOLS,
    GRAMMAR_VERSION,
    ISOLATION_LEVELS,
    ISOLATION_TEXT,
    PERMISSION_LEVELS,
    RECIPIENT_ARGUMENTS,
    canonical_record,
    named_updates,
)


def build_grammar() -> dict:
    return {
        "grammar_version": GRAMMAR_VERSION,
        "families": {
            "instruction_isolation": {
                "levels": {str(level): name for level, name in ISOLATION_LEVELS.items()},
                "system_message_addendum": {str(level): text for level, text in ISOLATION_TEXT.items()},
                "level_2_tool_output_delimiters": [DELIMITER_OPEN, DELIMITER_CLOSE],
            },
            "tool_permission": {
                "levels": {str(level): name for level, name in PERMISSION_LEVELS.items()},
                "level_1_blocked_tools": list(DESTRUCTIVE_TOOLS),
                "level_2_recipient_arguments": {tool: list(args) for tool, args in RECIPIENT_ARGUMENTS.items()},
                "level_2_known_recipients": "user's contact list, the user's own address, addresses in the user's request",
            },
            "memory_retrieval_policy": {
                "status": "reserved",
                "reason": "AgentDojo cases are independent episodes; cross-task memory needs an ordered protocol first.",
            },
        },
        "updates": [canonical_record(name, update) for name, update in named_updates().items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/agentdojo_update_grammar_v1.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_grammar(), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
