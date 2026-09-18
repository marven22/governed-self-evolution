"""Score a precommitted held-out governor choice after all candidate outcomes exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def result_for(rows: list[dict], *, candidate_id: str | None = None, label: str | None = None) -> dict:
    matches = [row for row in rows if (candidate_id is None or row["context"]["candidate_id"] == candidate_id) and (label is None or row["context"]["candidate_label"] == label)]
    if len(matches) != 1:
        raise ValueError("expected exactly one matching candidate outcome")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, rows = json.loads(args.plan.read_text()), json.loads(args.transitions.read_text())
    chosen = result_for(rows, candidate_id=plan["chosen_candidate_id"])
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise ValueError("held-out candidate set has no feasible actions")
    oracle = max(feasible, key=lambda row: row["utility"])
    baselines = {label: result_for(rows, label=label) for label in ("HOLD", "POLICY_PROTECTED", "MODEL_PROTECTED")}
    payload = {
        "score_version": "m14-governor-heldout-score-v1",
        "planned_candidate_id": plan["chosen_candidate_id"],
        "planned_label": plan["chosen_label"],
        "predicted": {"capability": plan["chosen_predicted_capability"], "utility": plan["chosen_predicted_utility"], "feasible": plan["chosen_predicted_feasible"]},
        "actual": {"capability": chosen["post_capability"], "utility": chosen["utility"], "feasible": chosen["feasible"]},
        "oracle": {"candidate_id": oracle["context"]["candidate_id"], "label": oracle["context"]["candidate_label"], "utility": oracle["utility"], "feasible": oracle["feasible"]},
        "regret": oracle["utility"] - chosen["utility"],
        "baselines": {label: {"utility": row["utility"], "feasible": row["feasible"]} for label, row in baselines.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"actual_utility": chosen["utility"], "actual_feasible": chosen["feasible"], "regret": payload["regret"]}))


if __name__ == "__main__":
    main()
