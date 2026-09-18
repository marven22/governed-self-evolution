"""Train V2: an uncertainty-aware, baseline-relative evolution governor."""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from train_m14_grammar_governor import TASKS, feature, fit_kernel_ridge, predict, utility


def controller(row: dict[str, Any]) -> str:
    return f"seed{row['context']['seed']}"


def load_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(Path(path).read_text()))
    if not rows:
        raise ValueError(f"no rows match {pattern!r}")
    return rows


def holds(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    answer: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["context"]["candidate_label"] != "HOLD":
            continue
        name = controller(row)
        if name in answer:
            raise ValueError(f"{name} has multiple HOLD rows")
        answer[name] = row
    expected = {controller(row) for row in rows}
    if set(answer) != expected:
        raise ValueError("every controller requires exactly one HOLD row")
    return answer


def delta(row: dict[str, Any], hold_rows: dict[str, dict[str, Any]]) -> np.ndarray:
    hold = hold_rows[controller(row)]
    return np.array([row["post_capability"][task] - hold["post_capability"][task] for task in TASKS])


def fit_ensemble(rows: list[dict[str, Any]], hold_rows: dict[str, dict[str, Any]], gamma: float, ridge: float, size: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(controller(row), []).append(row)
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    ensemble = []
    for _ in range(size):
        sampled = rng.choice(names, len(names), replace=True)
        chosen = [row for name in sampled for row in grouped[name]]
        x = np.stack([feature(row) for row in chosen])
        y = np.stack([delta(row, hold_rows) for row in chosen])
        ensemble.append(fit_kernel_ridge(x, y, gamma, ridge))
    return ensemble


def predict_ensemble(ensemble: list[dict[str, Any]], rows: list[dict[str, Any]]) -> np.ndarray:
    x = np.stack([feature(row) for row in rows])
    return np.stack([predict(model, x) for model in ensemble], axis=1)


def evaluate(rows: list[dict[str, Any]], predicted_delta: np.ndarray, lower_z: float, margin: float) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(controller(row), []).append(index)
    hold_rows = holds(rows)
    outcomes = []
    for name, indices in sorted(grouped.items()):
        candidates = [rows[index] for index in indices]
        hold_local = next(i for i, row in enumerate(candidates) if row["context"]["candidate_label"] == "HOLD")
        hold = hold_rows[name]
        base = np.array([hold["post_capability"][task] for task in TASKS])
        member_delta = predicted_delta[indices]
        member_advantage = np.array([[utility(value, hold["demand"]) for value in member] for member in member_delta])
        mean_advantage = member_advantage.mean(axis=1)
        lcb_advantage = mean_advantage - lower_z * member_advantage.std(axis=1)
        reach_lcb = base[0] + member_delta[:, :, 0].mean(axis=1) - lower_z * member_delta[:, :, 0].std(axis=1)
        allowed = (lcb_advantage > margin) & (reach_lcb >= hold["constraints"]["min_reach_retention"])
        # HOLD is an observed, exact no-change baseline, not an uncertain prediction.
        allowed[hold_local] = True
        mean_advantage[hold_local] = lcb_advantage[hold_local] = 0.0
        chosen_local = int(np.argmax(np.where(allowed, lcb_advantage, -np.inf)))
        chosen = candidates[chosen_local]
        oracle = max(row["utility"] for row in candidates if row["feasible"])
        outcomes.append({
            "controller": name,
            "chosen_candidate_id": chosen["context"]["candidate_id"],
            "chosen_label": chosen["context"]["candidate_label"],
            "chosen_actual_utility": float(chosen["utility"]),
            "chosen_actual_feasible": bool(chosen["feasible"]),
            "hold_utility": float(hold["utility"]),
            "oracle_utility": float(oracle),
            "regret": float(oracle - chosen["utility"]),
            "actual_advantage_over_hold": float(chosen["utility"] - hold["utility"]),
            "chosen_advantage_lcb": float(lcb_advantage[chosen_local]),
        })
    advantages = [item["actual_advantage_over_hold"] for item in outcomes]
    return {
        "per_controller": outcomes,
        "mean_regret": float(np.mean([item["regret"] for item in outcomes])),
        "mean_chosen_utility": float(np.mean([item["chosen_actual_utility"] for item in outcomes])),
        "mean_hold_utility": float(np.mean([item["hold_utility"] for item in outcomes])),
        "mean_advantage_over_hold": float(np.mean(advantages)),
        "mean_harm_vs_hold": float(np.mean([max(0.0, -value) for value in advantages])),
        "safety_violations": int(sum(not item["chosen_actual_feasible"] for item in outcomes)),
        "abstentions": int(sum(item["chosen_label"] == "HOLD" for item in outcomes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=64)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.ensemble_size < 2:
        raise ValueError("ensemble-size must be at least two")
    split = json.loads(args.split.read_text())
    rows = load_rows(args.data_glob)
    if {controller(row) for row in rows} != set(split["final_fit_controllers"]):
        raise ValueError("data controllers do not match frozen development split")
    train = [row for row in rows if controller(row) in split["train_controllers"]]
    validation = [row for row in rows if controller(row) in split["validation_controllers"]]
    train_holds, validation_holds = holds(train), holds(validation)
    validation_targets = np.stack([delta(row, validation_holds) for row in validation])
    trials = []
    for gamma in (0.03, 0.1, 0.3, 1.0, 3.0):
        for ridge in (1e-3, 1e-2, 1e-1):
            ensemble = fit_ensemble(train, train_holds, gamma, ridge, args.ensemble_size, args.bootstrap_seed)
            predictions = predict_ensemble(ensemble, validation)
            mse = float(np.mean((predictions.mean(axis=1) - validation_targets) ** 2))
            for lower_z in (0.0, 0.5, 1.0):
                for margin in (0.0, 0.02, 0.05):
                    trial = evaluate(validation, predictions, lower_z, margin)
                    trials.append({"gamma": gamma, "ridge": ridge, "lower_z": lower_z, "advantage_margin": margin, "delta_mse": mse, **trial})
    trials.sort(key=lambda item: (item["safety_violations"], item["mean_harm_vs_hold"], item["mean_regret"], item["delta_mse"]))
    selected = trials[0]
    final_ensemble = fit_ensemble(rows, holds(rows), selected["gamma"], selected["ridge"], args.ensemble_size, args.bootstrap_seed)
    model = {
        "model_type": "controller-bootstrap RBF relative transition ensemble",
        "tasks": TASKS,
        "selection": {key: selected[key] for key in ("gamma", "ridge", "lower_z", "advantage_margin")},
        "training_controllers": split["final_fit_controllers"],
        "held_out_controllers": split["held_out_controllers"],
        "ensemble_size": args.ensemble_size,
        "bootstrap_seed": args.bootstrap_seed,
        "ensemble": final_ensemble,
        "decision_rule": "choose a non-HOLD action only if its advantage LCB clears margin and reach LCB passes retention; otherwise HOLD",
    }
    report = {"protocol_version": "m14-baseline-anchored-governor-v2", "training_rows_for_selection": len(train), "validation_rows": len(validation), "selected": selected, "trials": trials, "note": "Controller-held-out development selection only; final fit did not access held-out controllers."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with args.model.open("wb") as handle:
        pickle.dump(model, handle)
    print(json.dumps({"selected": model["selection"], "validation_regret": selected["mean_regret"], "validation_harm": selected["mean_harm_vs_hold"], "validation_abstentions": selected["abstentions"]}))


if __name__ == "__main__":
    main()
