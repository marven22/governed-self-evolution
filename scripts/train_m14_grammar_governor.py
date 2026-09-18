"""Fit and internally validate a controller-held-out grammar transition governor.

The model is intentionally small: RBF kernel ridge regression predicts the
two post-update capabilities from an exact pre-state and an update grammar
action. Hyperparameters are selected only on held-out *development*
controllers, then the selected model is refit on all development controllers.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


TASKS = ("mw-reach", "mw-pick-place")
TARGETS = ("hold", "policy", "world_model", "policy_and_world_model")


def feature(row: dict[str, Any]) -> np.ndarray:
    update = row["update"]
    retention = update["retention"]
    target = update["target"]
    return np.array([
        row["pre_capability"][TASKS[0]],
        row["pre_capability"][TASKS[1]],
        update["task_data"]["reach_fraction"],
        update["optimization"]["gradient_steps"] / 1_500.0,
        np.log10(update["optimization"]["learning_rate"]) + 4.0,
        float(retention["freeze_encoder"]),
        np.log1p(retention["prior_policy_l2"]) / np.log(11.0),
        float(retention["protect_policy_during_model_update"]),
        *[float(target == item) for item in TARGETS],
    ], dtype=np.float64)


def target(row: dict[str, Any]) -> np.ndarray:
    return np.array([row["post_capability"][task] for task in TASKS], dtype=np.float64)


def fit_kernel_ridge(x: np.ndarray, y: np.ndarray, gamma: float, ridge: float) -> dict[str, Any]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    distances = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-gamma * distances)
    alpha = np.linalg.solve(kernel + ridge * np.eye(len(z)), y)
    return {"mean": mean, "scale": scale, "train_z": z, "alpha": alpha, "gamma": gamma, "ridge": ridge}


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    distances = np.sum((z[:, None, :] - model["train_z"][None, :, :]) ** 2, axis=2)
    kernel = np.exp(-model["gamma"] * distances)
    return kernel @ model["alpha"]


def utility(capability: np.ndarray, demand: dict[str, float]) -> float:
    return float(sum(demand[task] * capability[index] for index, task in enumerate(TASKS)))


def evaluate_decisions(rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    per_controller: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {}
    for row, prediction in zip(rows, predictions):
        per_controller.setdefault(f"seed{row['context']['seed']}", []).append((row, prediction))
    outcomes = []
    for controller, items in sorted(per_controller.items()):
        demand = items[0][0]["demand"]
        predicted_utility = np.array([utility(prediction, demand) for _, prediction in items])
        predicted_safe = np.array([prediction[0] >= row["constraints"]["min_reach_retention"] for row, prediction in items])
        hold_index = next(index for index, (row, _) in enumerate(items) if row["context"]["candidate_label"] == "HOLD")
        chosen_index = int(np.argmax(np.where(predicted_safe, predicted_utility, -np.inf))) if predicted_safe.any() else hold_index
        actual_utilities = np.array([row["utility"] for row, _ in items])
        actual_feasible = np.array([row["feasible"] for row, _ in items])
        oracle_utility = float(actual_utilities[actual_feasible].max())
        chosen_row = items[chosen_index][0]
        outcomes.append({
            "controller": controller,
            "chosen_candidate_id": chosen_row["context"]["candidate_id"],
            "chosen_label": chosen_row["context"]["candidate_label"],
            "chosen_actual_utility": float(chosen_row["utility"]),
            "chosen_actual_feasible": bool(chosen_row["feasible"]),
            "oracle_utility": oracle_utility,
            "regret": oracle_utility - float(chosen_row["utility"]),
            "hold_utility": float(items[hold_index][0]["utility"]),
        })
    return {
        "per_controller": outcomes,
        "mean_regret": float(np.mean([outcome["regret"] for outcome in outcomes])),
        "safety_violations": int(sum(not outcome["chosen_actual_feasible"] for outcome in outcomes)),
        "mean_chosen_utility": float(np.mean([outcome["chosen_actual_utility"] for outcome in outcomes])),
        "mean_hold_utility": float(np.mean([outcome["hold_utility"] for outcome in outcomes])),
    }


def load_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(Path(path).read_text()))
    if not rows:
        raise ValueError(f"no transition files matched {pattern!r}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    rows = load_rows(args.data_glob)
    by_controller = {f"seed{row['context']['seed']}" for row in rows}
    expected = set(split["final_fit_controllers"])
    if by_controller != expected:
        raise ValueError(f"data controllers {sorted(by_controller)} do not match frozen development split {sorted(expected)}")
    train_rows = [row for row in rows if f"seed{row['context']['seed']}" in split["train_controllers"]]
    validation_rows = [row for row in rows if f"seed{row['context']['seed']}" in split["validation_controllers"]]
    x_train, y_train = np.stack([feature(row) for row in train_rows]), np.stack([target(row) for row in train_rows])
    x_validation, y_validation = np.stack([feature(row) for row in validation_rows]), np.stack([target(row) for row in validation_rows])
    trials = []
    for gamma in (0.03, 0.1, 0.3, 1.0, 3.0):
        for ridge in (1e-3, 1e-2, 1e-1):
            fitted = fit_kernel_ridge(x_train, y_train, gamma, ridge)
            predictions = np.clip(predict(fitted, x_validation), 0.0, 1.0)
            decision = evaluate_decisions(validation_rows, predictions)
            trials.append({"gamma": gamma, "ridge": ridge, "mse": float(np.mean((predictions - y_validation) ** 2)), **decision})
    trials.sort(key=lambda trial: (trial["safety_violations"], trial["mean_regret"], trial["mse"]))
    selected = trials[0]
    x_all, y_all = np.stack([feature(row) for row in rows]), np.stack([target(row) for row in rows])
    final = fit_kernel_ridge(x_all, y_all, selected["gamma"], selected["ridge"])
    final.update({"feature_names": ["pre_reach", "pre_pick", "reach_fraction", "steps_normalized", "log_learning_rate", "freeze_encoder", "log_prior_policy_l2", "protect_policy_during_model_update", *[f"target_{item}" for item in TARGETS]], "tasks": TASKS, "selection": selected, "training_controllers": split["final_fit_controllers"], "held_out_controllers": split["held_out_controllers"]})
    report = {"model_type": "RBF kernel ridge transition predictor", "training_rows_for_selection": len(train_rows), "validation_rows": len(validation_rows), "selected": selected, "trials": trials, "note": "The final serialized model refits the selected hyperparameters on all development controllers. Held-out controllers were not accessed."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with args.model.open("wb") as handle:
        pickle.dump(final, handle)
    print(json.dumps({"selected_gamma": selected["gamma"], "selected_ridge": selected["ridge"], "validation_regret": selected["mean_regret"], "validation_violations": selected["safety_violations"]}))


if __name__ == "__main__":
    main()
