# Baseline-Anchored Evolution Governor Protocol V2

V1 predicted absolute post-update capability. Its blind result showed that this
can select an unnecessary update for an already strong controller. V2 instead
predicts the paired difference from the same controller's exact observed
`HOLD` transition:

\[
\Delta c(u) = c^+(u) - c^+(\mathrm{HOLD}).
\]

For future-demand vector \(d\), a non-HOLD update is eligible only when

\[
\operatorname{LCB}_z[d^\top\Delta c(u)] > m
\]

and its conservative reach estimate satisfies the retention constraint.
Otherwise the governor selects `HOLD`.

The transition predictor is a controller-bootstrap RBF ensemble. Each member
resamples whole controllers, never correlated individual candidate rows. The
margin, confidence multiplier, and RBF parameters are selected on
controller-held-out development agents only.

For a new controller: run only `HOLD`, precommit a plan, then execute the full
candidate sweep and score advantage versus `HOLD`, regret to the feasible
oracle, harmful-update rate, retention violations, and abstention rate.
Controllers 111 and 114 were used by V1 and are burned; they cannot confirm V2.
