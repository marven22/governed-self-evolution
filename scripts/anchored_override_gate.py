"""Lower-confidence override rule for a frozen V2 governor."""
from __future__ import annotations
import numpy as np
from train_rlcompopt_anchored_residual_governor import v2_choice
from train_rlcompopt_pairwise_governor_v2 import candidate_features

def decide(v2_model, residual_model, rows, beta: float, margin: float):
    """Return V2 unless a candidate has a conservative residual advantage."""
    x=np.vstack([candidate_features(row) for row in rows])
    incumbent=v2_choice(v2_model,x)
    predictions=np.vstack([m.predict(x) for m in residual_model['bootstrap_ensemble']])
    mean=predictions.mean(0); std=predictions.std(0); lcb=mean-beta*std
    candidate=int(np.argmax(lcb)); override=bool(lcb[candidate]>margin and candidate!=incumbent)
    return {'v2_index':incumbent,'selected_index':candidate if override else incumbent,'override':override,'residual_mean':float(mean[candidate]),'residual_std':float(std[candidate]),'lower_confidence_advantage':float(lcb[candidate]),'beta':beta,'margin':margin}
