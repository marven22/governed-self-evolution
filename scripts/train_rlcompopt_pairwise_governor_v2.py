#!/usr/bin/env python3
"""Train a prospective-recipe, pairwise governor from development evidence only.

Unlike v1, v2 receives the fully materialized *candidate* recipe. That recipe
is deterministically available after an edit AST is instantiated and before it
is executed; it is not a post-edit outcome feature. The model learns (a) which
candidate beats a sibling and (b) whether an edit beats HOLD.
"""
from __future__ import annotations

import argparse, hashlib, json, pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_rlcompopt_blame_gated_governor import features as v1_features

POSITIONS = 4


def candidate_features(row: dict[str, Any]) -> np.ndarray:
    """Pre-execution feature vector: controller, edit AST, and candidate recipe."""
    parent = np.asarray(row["controller"]["parent_actions"], dtype=int)
    child = np.asarray(row["child_actions"], dtype=int)
    parent_counts = np.bincount(parent, minlength=124)[:124]
    child_counts = np.bincount(child, minlength=124)[:124]
    positional = np.zeros((POSITIONS, 124), dtype=float)
    for position, action in enumerate(child):
        positional[min(POSITIONS - 1, POSITIONS * position // max(1, len(child))), action] += 1.0
    length = np.asarray([len(child), len(child) - len(parent)], dtype=float)
    return np.r_[v1_features(row), child_counts, child_counts - parent_counts, positional.ravel(), length]


def fit(x: np.ndarray, positive: np.ndarray, pair_x: np.ndarray, pair_y: np.ndarray, blame: np.ndarray):
    # Strong regularization is intentional: 8 programs is a small independent sample.
    kwargs = {"C": 0.03, "max_iter": 4000, "class_weight": "balanced"}
    positive_model = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))
    pair_model = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))
    blame_model = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))
    positive_model.fit(x, positive); pair_model.fit(pair_x, pair_y); blame_model.fit(x, blame)
    return positive_model, pair_model, blame_model


def pair_rows(x: np.ndarray, delta: np.ndarray, groups: list[tuple[str, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for i, group in enumerate(groups): grouped[group].append(i)
    differences, labels = [], []
    for indices in grouped.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1:]:
                if abs(delta[left] - delta[right]) <= 1e-12: continue
                differences.append(x[left] - x[right])
                labels.append(int(delta[left] > delta[right]))
                differences.append(x[right] - x[left])
                labels.append(int(delta[right] > delta[left]))
    return np.vstack(differences), np.asarray(labels, dtype=int)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True); p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    raw = a.ledger.read_bytes()
    if not audit.get("passed_for_gain_and_blame_governor") or hashlib.sha256(raw).hexdigest() != audit.get("ledger_sha256"):
        raise SystemExit("Refusing unaudited or changed development ledger")
    ledger = json.loads(raw)
    if ledger.get("cohort") != "development only" or ledger.get("selection_touched"):
        raise SystemExit("v2 may train only on the development cohort")
    rows = [r for r in ledger["transactions"] if r["outcome"]["certified_child"]]
    x = np.vstack([candidate_features(r) for r in rows])
    delta = np.asarray([r["outcome"]["delta_reward"] for r in rows])
    positive = (delta > 0.01).astype(int)
    blame = np.asarray([r["outcome"]["blame"] for r in rows], dtype=int)
    # The original rank-0 seed ledger predates an explicit parent_rank field.
    # Its protocol fixes that field to zero, just as v1 feature extraction did.
    groups = [(r["benchmark"], r["controller"].get("parent_rank", 0), r["controller"]["parent_coreset_index"]) for r in rows]
    px, py = pair_rows(x, delta, groups)
    if min(positive.sum(), blame.sum(), py.sum(), len(py) - py.sum()) == 0: raise SystemExit("Insufficient outcome variation")
    models = fit(x, positive, px, py, blame)
    payload = {"protocol":"rlcompopt-pairwise-governor-v2", "feature_dim":int(x.shape[1]), "positive_model":models[0], "pair_model":models[1], "blame_model":models[2], "contract":"pre-edit controller/edit context plus deterministically materialized candidate recipe", "safety":"hard certificate gate"}
    a.model.parent.mkdir(parents=True, exist_ok=True); a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.model.open("wb") as f: pickle.dump(payload, f)
    report = {"protocol":payload["protocol"], "training":{"certified_rows":len(rows), "contexts":len(set(groups)), "positive_edits":int(positive.sum()), "blamed_edits":int(blame.sum()), "pairwise_examples":len(py), "feature_dim":int(x.shape[1])}, "prospective_feature_justification":"candidate recipe is deterministically known from parent plus edit AST before child execution; outcome/certificate fields are excluded", "limitations":["Development data spans only eight programs.", "Selection ledger is not used for fitting."]}
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__": main()
