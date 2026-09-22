#!/usr/bin/env python3
"""Train an uncertainty-aware, one-step gain-and-blame governor.

The model deliberately excludes child outcomes and post-edit certificates from
its inputs.  Certification remains an execution-time hard gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OPS = ("delete", "insert", "replace", "splice")
BUCKETS = ("early", "middle", "late")


def features(row: dict[str, Any]) -> np.ndarray:
    c, e = row["controller"], row["edit"]
    action_counts = np.bincount(np.asarray(c["parent_actions"], dtype=int), minlength=124)[:124]
    autophase = np.log1p(np.asarray(c["autophase"], dtype=float))
    op = np.asarray([float(e["op"] == x) for x in OPS])
    bucket = np.asarray([float(e.get("position_bucket") == x) for x in BUCKETS])
    rank = np.asarray([float(c.get("parent_rank", 0) == x) for x in range(4)])
    edit_action = np.zeros(124)
    if "action" in e:
        edit_action[int(e["action"])] = 1.0
    numeric = np.asarray([
        len(c["parent_actions"]),
        float(e.get("position", 0)) / max(1, len(c["parent_actions"])),
        float(e.get("donor_rank", -1)),
        float(e.get("donor_position", -1)),
        float(e.get("donor_start", -1)),
        float(e.get("segment_length", 0)),
    ])
    # Low-dimensional interactions let a linear posterior express that the
    # same operator behaves differently at different parent ranks/locations.
    interactions = np.outer(op, np.r_[bucket, rank]).ravel()
    return np.r_[autophase, action_counts, op, bucket, rank, edit_action, numeric, interactions]


def fit_models(x: np.ndarray, delta: np.ndarray, blame: np.ndarray, seed: int, n_bootstrap: int):
    gain = make_pipeline(StandardScaler(), BayesianRidge())
    blame_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=3000, random_state=seed))
    gain.fit(x, delta)
    blame_model.fit(x, blame)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x), size=len(x))
        # A resample containing one class cannot fit logistic regression.
        if len(np.unique(blame[indices])) < 2:
            continue
        g = make_pipeline(StandardScaler(), BayesianRidge())
        b = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=3000, random_state=int(rng.integers(1 << 30))))
        g.fit(x[indices], delta[indices]); b.fit(x[indices], blame[indices])
        bootstrap.append((g, b))
    return gain, blame_model, bootstrap


def score(gain, blame_model, bootstrap, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = gain.predict(x)
    risk = blame_model.predict_proba(x)[:, 1]
    if not bootstrap:
        return mean, risk, np.zeros(len(x))
    predictions = np.vstack([g.predict(x) for g, _ in bootstrap])
    return mean, risk, predictions.std(axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260922)
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    if not audit.get("passed_for_gain_and_blame_governor"):
        raise SystemExit("Refusing to train from a failed ledger audit")
    raw = a.ledger.read_bytes()
    if hashlib.sha256(raw).hexdigest() != audit["ledger_sha256"]:
        raise SystemExit("Ledger changed after audit; re-run audit before training")
    rows = [r for r in json.loads(raw)["transactions"] if r["outcome"]["certified_child"]]
    x = np.vstack([features(r) for r in rows])
    delta = np.asarray([r["outcome"]["delta_reward"] for r in rows], dtype=float)
    blame = np.asarray([r["outcome"]["blame"] for r in rows], dtype=int)
    groups = np.asarray([r["benchmark"] for r in rows])
    if len(np.unique(blame)) != 2:
        raise SystemExit("Need both blamed and non-blamed certified rows")

    # Leave-one-program-out predictions are the only reported internal metric.
    cv_gain, cv_risk, cv_uncertainty = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
    for benchmark in sorted(set(groups)):
        test = groups == benchmark
        train = ~test
        g, b, boot = fit_models(x[train], delta[train], blame[train], a.seed, a.bootstrap)
        cv_gain[test], cv_risk[test], cv_uncertainty[test] = score(g, b, boot, x[test])
    mae = float(mean_absolute_error(delta, cv_gain))
    auc = float(roc_auc_score(blame, cv_risk))

    # Regret is evaluated per parent context. The execution certificate remains
    # a hard gate; all rows here are certified by construction.
    contexts: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        c = r["controller"]
        contexts[(r["benchmark"], c.get("parent_rank", 0), c["parent_coreset_index"])].append(i)
    regret, held_contexts, selected_wins, selected_nonpositive = [], 0, 0, 0
    for indices in contexts.values():
        utility = cv_gain[indices] - cv_uncertainty[indices] - cv_risk[indices]
        best_position = int(np.argmax(utility))
        chosen = indices[best_position]
        best = max(0.0, float(np.max(delta[indices])))
        if utility[best_position] <= 0.0:
            observed = 0.0
            held_contexts += 1
        else:
            observed = max(0.0, float(delta[chosen]))
            selected_wins += int(observed > 0.0)
            selected_nonpositive += int(observed <= 0.0)
        regret.append(best - observed)

    gain, blame_model, bootstrap = fit_models(x, delta, blame, a.seed, a.bootstrap)
    payload = {"protocol": "rlcompopt-blame-gated-governor-v1", "feature_dim": int(x.shape[1]), "gain_model": gain, "blame_model": blame_model, "bootstrap": bootstrap, "feature_contract": "pre-edit controller context and edit AST only", "certificate_policy": "hard execution-time gate; no learned semantic safety head"}
    a.model.parent.mkdir(parents=True, exist_ok=True); a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.model.open("wb") as f: pickle.dump(payload, f)
    report = {"protocol": payload["protocol"], "training": {"certified_rows": len(rows), "blame_rows": int(blame.sum()), "feature_dim": int(x.shape[1]), "bootstrap_models": len(bootstrap)}, "leave_one_program_out": {"gain_mae": mae, "blame_auc": auc, "parent_contexts": len(contexts), "mean_decision_regret": float(np.mean(regret)), "held_contexts": held_contexts, "selected_positive_children": selected_wins, "selected_nonpositive_children": selected_nonpositive}, "limitations": ["Development-only internal validation; do not claim generalization.", "Semantic safety remains certificate-gated because only two unsafe cases exist.", "Archive utility is intentionally excluded to prevent same-ledger outcome leakage."]}
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__": main()
