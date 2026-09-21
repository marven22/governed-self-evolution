"""Train a deliberately small, parent-held-out branch-governor baseline.

This is a development diagnostic, not a final controller: records that did not
receive costly confirmation retain their screened outcome and are therefore
lower-fidelity supervision.  The paired certificate still gates every commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ("mw-reach", "mw-pick-place")


def features(row: dict) -> np.ndarray:
    update = row["update_spec"]
    return np.array([
        1.0,
        row["pre_capability"][TASKS[0]], row["pre_capability"][TASKS[1]],
        row["step"] / 3.0, update["reach_fraction"],
        update["gradient_steps"] / 250.0,
    ], dtype=float)


def target(row: dict) -> np.ndarray:
    outcome = row["outcome"]
    return np.array([outcome["utility_delta"], *[outcome["capability_delta"][t] for t in TASKS]], dtype=float)


def fit(rows: list[dict], ridge: float) -> np.ndarray:
    x = np.stack([features(r) for r in rows])
    y = np.stack([target(r) for r in rows])
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)


def evaluate(rows: list[dict], weights: np.ndarray) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["controller_id"], row["step"]), []).append(row)
    predictions, decisions = [], []
    for key, choices in grouped.items():
        scores = []
        for row in choices:
            predicted = features(row) @ weights
            actual = target(row)
            predictions.append({"parent": row["parent_id"], "label": row["label"], "actual": actual.tolist(), "predicted": predicted.tolist()})
            safe = bool(np.all(predicted[1:] >= -0.05))
            scores.append(float(predicted[0]) if safe and predicted[0] > 0 else 0.0)
        selected = int(np.argmax(scores))
        actual = target(choices[selected]) if scores[selected] > 0 else np.zeros(3)
        decisions.append({"controller_id": key[0], "step": key[1], "chosen": choices[selected]["label"] if scores[selected] > 0 else "HOLD", "screen_utility": float(actual[0]), "screen_retention": bool(np.all(actual[1:] >= -0.05))})
    return predictions, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()
    rows = json.loads(args.archive.read_text())
    parents = sorted({r["parent_id"] for r in rows})
    folds, all_predictions, all_decisions = [], [], []
    for held in parents:
        train = [r for r in rows if r["parent_id"] != held]
        test = [r for r in rows if r["parent_id"] == held]
        predictions, decisions = evaluate(test, fit(train, args.ridge))
        mae = float(np.mean([np.abs(np.asarray(p["actual"]) - np.asarray(p["predicted"])) for p in predictions]))
        folds.append({"held_parent": held, "records": len(test), "mae": mae, "decisions": decisions})
        all_predictions += predictions; all_decisions += decisions
    model = fit(rows, args.ridge)
    report = {"protocol": "m14-branch-governor-v1-development", "records": len(rows), "parents": parents, "features": ["bias", "reach", "pick_place", "step", "reach_fraction", "gradient_steps"], "targets": ["screen_or_confirmation_utility", "reach_delta", "pick_place_delta"], "leave_one_parent_out": folds, "lopo_mae": float(np.mean([x["mae"] for x in folds])), "development_decisions": all_decisions, "warning": "Unconfirmed branches have screen-only targets; do not make blind-test claims from this report."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps({"protocol": report["protocol"], "weights": model.tolist(), "ridge": args.ridge, "features": report["features"], "development_parents": parents}, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "parents": parents, "lopo_mae": report["lopo_mae"], "proposed_non_hold": sum(d["chosen"] != "HOLD" for d in all_decisions)}))


if __name__ == "__main__":
    main()
