# M14 Grammar Governor: Internal Controller-Held-Out Validation (v1)

## Frozen selection protocol

The transition governor uses RBF kernel ridge regression to predict post-update
reach and pick-place success from an exact starting capability state and a
canonical grammar action. Hyperparameters were selected on controller identity,
not random rows:

- Fit-selection controllers: 101, 102, 103, 105 (64 rows).
- Internal validation controllers: 106, 112 (32 rows).
- Final fit after selection: all six development controllers.
- Blind final controllers: 111, 114; not accessed during this procedure.

The selected configuration is RBF gamma 0.03 and ridge 0.1. Selection order
was: fewest observed decision safety violations, then least candidate-set
regret, then least capability MSE.

## Internal result

On the two controller-held-out validation agents, the governor chose feasible
updates in both cases:

| Controller | Chosen actual utility | Oracle utility | Regret | HOLD utility |
|---|---:|---:|---:|---:|
| 106 | 0.985 | 1.000 | 0.015 | 0.845 |
| 112 | 1.000 | 1.000 | 0.000 | 0.985 |

Mean regret is 0.0075; observed safety violations are 0; capability MSE is
0.0340. This is sufficient to run the predeclared blind test, but not to make
a strong generalization claim: validation contains only two controllers and
one demand vector.

The machine-readable selection report is
`reports/m14_grammar_governor_internal_v1.json`; the frozen refit model is
`models/m14_grammar_governor_v1.pkl`.
