# M14 Blind Held-Out Governor Results (v1)

## Protocol integrity

Controllers 111 and 114 were absent from development fitting and hyperparameter
selection. For each controller, a `HOLD`-only pre-measurement was made first;
the frozen governor wrote `plan.json`; only afterward were all 16 candidate
updates executed. Both plans selected the same protected joint update,
`u-396aad2568d0`. The raw plans, pre-measurements, full outcomes, and scores
are versioned under `data/m14_heldout_*` and `reports/m14_heldout_*`.

## Results

| Held-out controller | Governor utility | Feasible | Oracle utility | Regret | HOLD | Protected policy |
|---|---:|---:|---:|---:|---:|---:|
| 111 | 0.880 | yes | 0.965 | 0.085 | 0.675 | 0.810 |
| 114 | 0.780 | yes | 1.000 | 0.220 | 1.000 | 0.845 |

The governor made no safety violation. It improved materially over HOLD and
the named protected baselines on controller 111. On controller 114, however,
HOLD was the oracle; the governor made an unnecessary persistent update and
lost 0.220 utility.

Mean governor utility is 0.830, mean HOLD utility is 0.838, and mean regret is
0.153. Thus the predeclared blind test does **not** support a claim that this
first governor reliably beats HOLD on new controllers.

## Interpretation

This is a productive negative result, not a collapse of the problem framing:

1. The learned transition model selected a safe update on both new agents.
2. It did not represent the value of *doing nothing* accurately enough when a
   new controller was already near its capability ceiling.
3. Selecting the identical update for both agents is evidence that this small
model is overly population-average-driven and insufficiently conditional on
the starting state.

## Required repair before a stronger claim

The next governor should be uncertainty-aware and compare every update against
HOLD with an explicit advantage margin. It should update only when the lower
confidence bound of predicted improvement over HOLD is positive; otherwise it
must abstain. We also need more controller identities and replicated outcomes
per controller-action pair to estimate uncertainty rather than infer it from a
single observation.
