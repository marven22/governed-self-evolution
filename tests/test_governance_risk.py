from gse_governance.risk import (
    Quality,
    authorize_direct_route,
    certificate,
    governor_promotion,
    pareto_dominates,
)


def test_certificate_and_pareto_ordering() -> None:
    clean = Quality(1, 1, 1, 1, 0.1)
    weak = Quality(1, 0.9, 1, 1, 0.2)
    assert certificate(clean)
    assert not certificate(weak)
    assert pareto_dominates(clean, weak)


def test_gate_refuses_underpowered_evidence() -> None:
    result = authorize_direct_route([(0.9, False)] * 2, [0.5, 0.8], alpha=0.05, delta=0.05)
    assert not result["authorized"]
    assert result["threshold"] is None


def test_promotion_is_non_regressive() -> None:
    incumbent = Quality(1, 0.9, 1, 1, 0.2)
    safer_better = Quality(1, 1, 1, 1, 0.2)
    regressive = Quality(0.9, 1, 1, 1, 0.1)
    assert governor_promotion(incumbent, safer_better, incumbent_repairs=4, challenger_repairs=5)
    assert not governor_promotion(incumbent, regressive, incumbent_repairs=4, challenger_repairs=6)
