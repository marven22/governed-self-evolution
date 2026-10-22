"""Small, auditable building blocks for governed repair experiments.

These functions deliberately separate a *certificate* (what a proposed edit
must preserve) from a *routing decision* (when it is safe to use a cheaper
direct proposal rather than a more expensive fallback/search procedure).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Iterable, Mapping, Sequence

from scipy.stats import beta


@dataclass(frozen=True)
class Quality:
    """Measured quality of an edit on target and protected panels.

    Scores are normalized to ``[0, 1]``.  ``edit_scope`` is a non-negative
    proxy for how much of the persistent policy/configuration was changed.
    """

    target_security: float
    target_utility: float
    protected_security: float
    protected_utility: float
    edit_scope: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name != "edit_scope" and value > 1:
                raise ValueError(f"{name} must be at most one")


def certificate(quality: Quality, *, target_min: float = 1.0, protected_min: float = 1.0) -> bool:
    """Return whether an edit clears explicit target and preservation floors."""

    return (
        quality.target_security >= target_min
        and quality.target_utility >= target_min
        and quality.protected_security >= protected_min
        and quality.protected_utility >= protected_min
    )


def pareto_dominates(left: Quality, right: Quality) -> bool:
    """Whether ``left`` is no worse on all quality coordinates and better on one."""

    left_scores = (
        left.target_security,
        left.target_utility,
        left.protected_security,
        left.protected_utility,
        -left.edit_scope,
    )
    right_scores = (
        right.target_security,
        right.target_utility,
        right.protected_security,
        right.protected_utility,
        -right.edit_scope,
    )
    return all(a >= b for a, b in zip(left_scores, right_scores)) and any(
        a > b for a, b in zip(left_scores, right_scores)
    )


def clopper_pearson_upper(failures: int, trials: int, *, confidence: float) -> float:
    """Exact one-sided binomial upper confidence bound on a failure rate."""

    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("require 0 <= failures <= trials and trials > 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if failures == trials:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, trials - failures))


def authorize_direct_route(
    calibration: Iterable[tuple[float, bool]],
    thresholds: Sequence[float],
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
) -> Mapping[str, object]:
    """Choose a predeclared score threshold under simultaneous risk control.

    Each calibration tuple is ``(direct_route_score, protected_failure)``.
    Candidate thresholds are fixed *before* inspecting the final cohort.
    Among thresholds whose simultaneous Clopper--Pearson upper bound is at
    most ``alpha``, choose the one covering the most calibration contexts.
    """

    rows = list(calibration)
    unique_thresholds = sorted(set(float(value) for value in thresholds))
    if not rows or not unique_thresholds:
        raise ValueError("calibration rows and at least one threshold are required")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must lie strictly between zero and one")
    if any(not isfinite(score) for score, _ in rows):
        raise ValueError("scores must be finite")

    simultaneous_confidence = 1.0 - delta / len(unique_thresholds)
    audits: list[dict[str, object]] = []
    for threshold in unique_thresholds:
        selected = [(score, failed) for score, failed in rows if score >= threshold]
        trials = len(selected)
        failures = sum(failed for _, failed in selected)
        ucb = (
            clopper_pearson_upper(failures, trials, confidence=simultaneous_confidence)
            if trials
            else 1.0
        )
        audits.append(
            {
                "threshold": threshold,
                "selected": trials,
                "failures": failures,
                "protected_failure_ucb": ucb,
                "authorized": bool(trials and ucb <= alpha),
            }
        )

    authorized = [row for row in audits if row["authorized"]]
    winner = max(authorized, key=lambda row: (int(row["selected"]), -float(row["threshold"])), default=None)
    return {
        "alpha": alpha,
        "delta": delta,
        "simultaneous_confidence": simultaneous_confidence,
        "threshold": None if winner is None else winner["threshold"],
        "authorized": winner is not None,
        "audit": audits,
    }


def governor_promotion(
    incumbent: Quality,
    challenger: Quality,
    *,
    incumbent_repairs: int,
    challenger_repairs: int,
) -> bool:
    """Promotion rule: non-regressive quality plus stronger repair yield.

    A challenger may not buy more repairs by degrading either target or
    protected quality.  When repair counts tie, it must use less edit scope.
    """

    if min(incumbent_repairs, challenger_repairs) < 0:
        raise ValueError("repair counts must be non-negative")
    non_regressive = all(
        new >= old
        for new, old in zip(
            (
                challenger.target_security,
                challenger.target_utility,
                challenger.protected_security,
                challenger.protected_utility,
            ),
            (
                incumbent.target_security,
                incumbent.target_utility,
                incumbent.protected_security,
                incumbent.protected_utility,
            ),
        )
    )
    stronger = challenger_repairs > incumbent_repairs or (
        challenger_repairs == incumbent_repairs and challenger.edit_scope < incumbent.edit_scope
    )
    return non_regressive and stronger
