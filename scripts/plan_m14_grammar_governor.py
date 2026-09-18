"""Precommit a grammar-governor decision from a held-out pre-state measurement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from m14_update_grammar import UpdateSpec
from train_m14_grammar_governor import feature, predict, utility


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
        raise ValueError("pre-transition must be exactly one HOLD evaluation row")
    pre_row = pre_rows[0]
    candidates = json.loads(args.candidates.read_text())["candidates"]
    pseudo_rows, ids, labels = [], [], []
    for candidate in candidates:
        spec = UpdateSpec.from_dict(candidate["update"])
        if spec.identifier != candidate["id"]:
            raise ValueError("candidate identity mismatch")
        pseudo_rows.append({"pre_capability": pre_row["pre_capability"], "update": spec.to_dict()})
        ids.append(spec.identifier); labels.append(candidate.get("label"))
    predictions = np.clip(predict(model, np.stack([feature(row) for row in pseudo_rows])), 0.0, 1.0)
    demand, minimum = pre_row["demand"], pre_row["constraints"]["min_reach_retention"]
    predicted_utility = np.array([utility(capability, demand) for capability in predictions])
    predicted_safe = predictions[:, 0] >= minimum
    hold_index = labels.index("HOLD")
    chosen_index = int(np.argmax(np.where(predicted_safe, predicted_utility, -np.inf))) if predicted_safe.any() else hold_index
    payload = {
        "plan_version": "m14-governor-heldout-plan-v1",
        "model_selection": model["selection"],
        "pre_capability": pre_row["pre_capability"],
        "demand": demand,
        "min_reach_retention": minimum,
        "chosen_candidate_id": ids[chosen_index],
        "chosen_label": labels[chosen_index],
        "chosen_predicted_capability": {task: float(predictions[chosen_index, index]) for index, task in enumerate(model["tasks"])},
        "chosen_predicted_utility": float(predicted_utility[chosen_index]),
        "chosen_predicted_feasible": bool(predicted_safe[chosen_index]),
        "candidate_predictions": [{"id": identifier, "label": label, "capability": {task: float(prediction[index]) for index, task in enumerate(model["tasks"])}, "utility": float(value), "feasible": bool(safe)} for identifier, label, prediction, value, safe in zip(ids, labels, predictions, predicted_utility, predicted_safe)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"chosen_candidate_id": payload["chosen_candidate_id"], "chosen_label": payload["chosen_label"], "predicted_utility": payload["chosen_predicted_utility"]}))


if __name__ == "__main__":
    main()
