"""Precommit COGE's opportunity-gated micro-update from a HOLD probe.

The learned model predicts capability *change relative to HOLD*.  The planner
first asks whether any non-HOLD action has a conservative positive benefit
while retaining both skills.  Only then does it rank eligible interventions.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from m14_update_grammar import UpdateSpec
from train_m14_grammar_governor import predict, utility
from train_m14_rich_governor_v3 import STATE_KEYS, TASKS, feat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--pre-transition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.model.open("rb") as f:
        model = pickle.load(f)
    pre_rows = json.loads(a.pre_transition.read_text())
    if len(pre_rows) != 1 or pre_rows[0]["context"]["candidate_label"] != "HOLD":
        raise ValueError("pre-transition must contain exactly one HOLD row")
    pre = pre_rows[0]
    if "pre_update_state" not in pre or any(k not in pre["pre_update_state"] for k in STATE_KEYS):
        raise ValueError("COGE requires the complete pre_update_state from the HOLD probe")
    records = json.loads(a.candidates.read_text())["candidates"]
    ids, labels, rows, steps = [], [], [], []
    for record in records:
        spec = UpdateSpec.from_dict(record["update"])
        if record["id"] != spec.identifier:
            raise ValueError("candidate identity mismatch")
        rows.append({"pre_capability": pre["pre_capability"], "pre_update_state": pre["pre_update_state"], "update": spec.to_dict()})
        ids.append(spec.identifier); labels.append(record["label"]); steps.append(spec.gradient_steps)
    if labels.count("HOLD") != 1:
        raise ValueError("candidate grammar must include exactly one HOLD")
    pred = np.stack([predict(member, np.stack([feat(row) for row in rows])) for member in model["ensemble"]], axis=1)
    demand = pre["demand"]
    mean, std = pred.mean(axis=1), pred.std(axis=1)
    advantage = np.array([[utility(value, demand) for value in members] for members in pred])
    z = float(model["selection"]["lower_z"]); margin = float(model["selection"]["advantage_margin"])
    adv_mean, adv_std = advantage.mean(axis=1), advantage.std(axis=1)
    adv_lcb = adv_mean - z * adv_std
    eps = np.array([pre["constraints"].get("max_reach_drop_from_hold", model["epsilon"][TASKS[0]]), pre["constraints"].get("max_pick_place_drop_from_hold", model["epsilon"][TASKS[1]])])
    retention_lcb = mean - z * std
    safe = np.all(retention_lcb >= -eps, axis=1)
    hold = labels.index("HOLD")
    eligible_nonhold = safe & (adv_lcb > margin)
    eligible_nonhold[hold] = False
    opportunity = bool(np.any(eligible_nonhold))
    if opportunity:
        # Prefer the smallest intervention when conservative benefits tie.
        order = sorted(range(len(ids)), key=lambda i: (-adv_lcb[i], steps[i], ids[i]))
        chosen = next(i for i in order if eligible_nonhold[i])
    else:
        chosen = hold
    hold_cap = np.array([pre["post_capability"][task] for task in TASKS])
    predicted_cap = hold_cap + mean[chosen]
    payload = {
        "plan_version": "m14-coge-v3.1", "model_selection": model["selection"],
        "pre_capability": pre["pre_capability"], "observed_hold_capability": pre["post_capability"], "demand": demand,
        "opportunity_detected": opportunity, "opportunity_rule": "non-HOLD requires retention LCB >= -epsilon and utility-advantage LCB > margin",
        "chosen_candidate_id": ids[chosen], "chosen_label": labels[chosen],
        "chosen_predicted_capability": {task: float(predicted_cap[i]) for i, task in enumerate(TASKS)},
        "chosen_predicted_advantage_over_hold": float(adv_mean[chosen]), "chosen_advantage_lcb": float(adv_lcb[chosen]),
        "candidate_predictions": [
            {"id": ids[i], "label": labels[i], "gradient_steps": steps[i], "advantage_mean": float(adv_mean[i]), "advantage_std": float(adv_std[i]), "advantage_lcb": float(adv_lcb[i]), "retention_lcb": {task: float(retention_lcb[i, j]) for j, task in enumerate(TASKS)}, "safe": bool(safe[i]), "eligible_nonhold": bool(eligible_nonhold[i])}
            for i in range(len(ids))],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"chosen": payload["chosen_label"], "opportunity_detected": opportunity, "advantage_lcb": payload["chosen_advantage_lcb"]}))


if __name__ == "__main__":
    main()
