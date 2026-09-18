"""Precommit a V2 conservative grammar decision from an exact HOLD outcome."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from m14_update_grammar import UpdateSpec
from train_m14_baseline_anchored_governor import TASKS, predict_ensemble
from train_m14_grammar_governor import utility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pre-transition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    pre_rows = json.loads(args.pre_transition.read_text())
    if len(pre_rows) != 1 or pre_rows[0]["context"]["candidate_label"] != "HOLD":
        raise ValueError("pre-transition must contain exactly one HOLD row")
    pre = pre_rows[0]
    records = json.loads(args.candidates.read_text())["candidates"]
    rows, ids, labels = [], [], []
    for record in records:
        spec = UpdateSpec.from_dict(record["update"])
        if spec.identifier != record["id"]:
            raise ValueError("candidate identity mismatch")
        rows.append({"pre_capability": pre["pre_capability"], "update": spec.to_dict()})
        ids.append(spec.identifier)
        labels.append(record.get("label"))
    member_delta = predict_ensemble(model["ensemble"], rows)
    demand = pre["demand"]
    hold_capability = np.array([pre["post_capability"][task] for task in TASKS])
    member_advantage = np.array([[utility(value, demand) for value in member] for member in member_delta])
    mean_advantage, std_advantage = member_advantage.mean(axis=1), member_advantage.std(axis=1)
    lower_z, margin = model["selection"]["lower_z"], model["selection"]["advantage_margin"]
    advantage_lcb = mean_advantage - lower_z * std_advantage
    reach_lcb = hold_capability[0] + member_delta[:, :, 0].mean(axis=1) - lower_z * member_delta[:, :, 0].std(axis=1)
    eligible = (advantage_lcb > margin) & (reach_lcb >= pre["constraints"]["min_reach_retention"])
    hold_index = labels.index("HOLD")
    eligible[hold_index] = True
    mean_advantage[hold_index] = advantage_lcb[hold_index] = 0.0
    reach_lcb[hold_index] = hold_capability[0]
    chosen = int(np.argmax(np.where(eligible, advantage_lcb, -np.inf)))
    predicted_capability = hold_capability + member_delta.mean(axis=1)
    predicted_capability[hold_index] = hold_capability
    payload = {
        "plan_version": "m14-baseline-anchored-governor-plan-v2",
        "model_selection": model["selection"],
        "pre_capability": pre["pre_capability"],
        "observed_hold_capability": pre["post_capability"],
        "demand": demand,
        "chosen_candidate_id": ids[chosen],
        "chosen_label": labels[chosen],
        "chosen_predicted_capability": {task: float(predicted_capability[chosen, index]) for index, task in enumerate(TASKS)},
        "chosen_predicted_utility": float(utility(predicted_capability[chosen], demand)),
        "chosen_predicted_feasible": bool(reach_lcb[chosen] >= pre["constraints"]["min_reach_retention"]),
        "chosen_predicted_advantage_over_hold": float(mean_advantage[chosen]),
        "chosen_advantage_lcb": float(advantage_lcb[chosen]),
        "candidate_predictions": [
            {"id": identifier, "label": label, "advantage_mean": float(mean), "advantage_std": float(std), "advantage_lcb": float(lcb), "reach_lcb": float(reach), "eligible": bool(ok)}
            for identifier, label, mean, std, lcb, reach, ok in zip(ids, labels, mean_advantage, std_advantage, advantage_lcb, reach_lcb, eligible)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"chosen": payload["chosen_label"], "advantage_lcb": payload["chosen_advantage_lcb"]}))


if __name__ == "__main__":
    main()
