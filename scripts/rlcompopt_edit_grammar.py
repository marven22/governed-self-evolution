"""Bounded symbolic edit grammar for RLCompOpt action sequences.

The grammar deliberately exposes *operators* rather than a fixed menu of full
optimization recipes.  Its deterministic sampler is only a foundation-data
collector; a learned governor will later generate/rank the same AST language.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


GRAMMAR_VERSION = "rlcompopt-symbolic-edit-grammar-v1"


def edit_program_id(edit: dict[str, Any]) -> str:
    """Return a reusable symbolic-template identifier, not an instance ID."""
    return ":".join(
        str(edit[key])
        for key in ("op", "position_bucket", "donor_rank", "segment_length")
        if key in edit
    )


def apply_edit(parent: Sequence[int], donors: Sequence[Sequence[int]], edit: dict[str, Any]) -> list[int]:
    """Apply a validated AST. The caller enforces the global sequence budget."""
    actions = list(parent)
    op = edit["op"]
    position = int(edit.get("position", 0))
    if op == "insert":
        return actions[:position] + [int(edit["action"])] + actions[position:]
    if op == "delete":
        return actions[:position] + actions[position + 1 :]
    if op == "replace":
        return actions[:position] + [int(edit["action"])] + actions[position + 1 :]
    if op == "splice":
        donor = list(donors[int(edit["donor_rank"])])
        start = int(edit["donor_start"])
        length = int(edit["segment_length"])
        return actions[:position] + donor[start : start + length] + actions[position:]
    raise ValueError(f"Unknown edit operator: {op}")


def _bucket(position: int, length: int) -> str:
    if length <= 1 or position * 3 < length:
        return "early"
    if position * 3 > length * 2:
        return "late"
    return "middle"


def _stable_offset(benchmark: str, parent: Sequence[int]) -> int:
    message = f"{GRAMMAR_VERSION}|{benchmark}|{','.join(map(str, parent))}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def generate_candidates(
    *,
    benchmark: str,
    parent: Sequence[int],
    donors: Sequence[Sequence[int]],
    candidate_budget: int,
    max_actions: int,
) -> list[dict[str, Any]]:
    """Create a deterministic, stratified pool of novel grammar instances.

    Donors are the frozen controller's ranked coreset sequences.  We draw
    operands from several donor ranks and positions so the data collection is
    broad before a governor has learned a proposal distribution.
    """
    if not parent:
        raise ValueError("Parent action sequence must be non-empty")
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    if not donors:
        raise ValueError("At least one donor sequence is required")

    offset = _stable_offset(benchmark, parent)
    donor_count = min(5, len(donors))
    positions = sorted({0, len(parent) // 2, len(parent)})
    interior_positions = sorted({0, len(parent) // 2, len(parent) - 1})
    raw: list[dict[str, Any]] = []

    # Delete supplies controlled negative/neutral local counterfactuals.
    for position in interior_positions:
        raw.append({
            "op": "delete",
            "position": position,
            "position_bucket": _bucket(position, len(parent)),
        })

    # Insert and replace use operands observed in high-ranked controller
    # sequences, rather than an arbitrary global action menu.
    for donor_rank in range(donor_count):
        donor = list(donors[donor_rank])
        if not donor:
            continue
        source_positions = sorted({0, len(donor) // 2, len(donor) - 1})
        for source_position in source_positions:
            action = int(donor[source_position])
            for position in positions:
                raw.append({
                    "op": "insert",
                    "action": action,
                    "donor_rank": donor_rank,
                    "donor_position": source_position,
                    "position": position,
                    "position_bucket": _bucket(position, len(parent)),
                })
            for position in interior_positions:
                raw.append({
                    "op": "replace",
                    "action": action,
                    "donor_rank": donor_rank,
                    "donor_position": source_position,
                    "position": position,
                    "position_bucket": _bucket(position, len(parent)),
                })

        # Short splices are expressive enough to create novel recipes while
        # remaining locally attributable and computationally bounded.
        for segment_length in (1, 2):
            if len(donor) < segment_length:
                continue
            starts = sorted({0, max(0, len(donor) // 2 - 1), len(donor) - segment_length})
            for start in starts:
                for position in positions:
                    raw.append({
                        "op": "splice",
                        "donor_rank": donor_rank,
                        "donor_start": start,
                        "segment_length": segment_length,
                        "position": position,
                        "position_bucket": _bucket(position, len(parent)),
                    })

    # Round-robin across operator families.  A raw list ordered by construction
    # would otherwise collect mostly splices or mostly local edits, creating a
    # biased foundation ledger before the governor exists.
    families = ("delete", "insert", "replace", "splice")
    by_family = {family: [edit for edit in raw if edit["op"] == family] for family in families}
    for family, edits in by_family.items():
        if edits:
            shift = (offset + len(family)) % len(edits)
            by_family[family] = edits[shift:] + edits[:shift]
    ordered: list[dict[str, Any]] = []
    cursor = 0
    while any(by_family.values()):
        family = families[cursor % len(families)]
        if by_family[family]:
            ordered.append(by_family[family].pop(0))
        cursor += 1
    selected: list[dict[str, Any]] = []
    seen_children: set[tuple[int, ...]] = {tuple(parent)}
    seen_templates: set[str] = set()
    for edit in ordered:
        child = apply_edit(parent, donors, edit)
        if not child or len(child) > max_actions:
            continue
        child_key = tuple(child)
        template = edit_program_id(edit)
        # Prefer coverage of edit-program templates, then permit more than one
        # instance as needed to meet the requested evaluation budget.
        if child_key in seen_children:
            continue
        if template in seen_templates and len(selected) < candidate_budget // 2:
            continue
        instance = dict(edit)
        instance["grammar_version"] = GRAMMAR_VERSION
        instance["edit_program_id"] = template
        instance["child_actions"] = child
        selected.append(instance)
        seen_children.add(child_key)
        seen_templates.add(template)
        if len(selected) >= candidate_budget:
            break
    return selected
