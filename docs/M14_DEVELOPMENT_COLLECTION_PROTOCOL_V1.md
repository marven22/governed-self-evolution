# M14 Development Collection Protocol (v1)

The controller split is frozen before the following update experiments:

- Development: 101, 102, 103, 105, 106, 112.
- Held out: 111, 114.

The held-out controllers must not receive any grammar update until the blind
decision evaluation. The fixed development design has 16 candidates: four
historical baselines and 12 stratified grammar actions spanning policy,
world-model, and joint update targets; 25%, 50%, and 75% retained-reach data
mixtures; multiple budgets; and retention mechanisms.

Each development controller runs all 16 candidates from its fresh base
checkpoint using 50 demonstrations per task and 20 evaluation episodes per
task. This produces 96 exact transitions. The output directories are
`/home/vmargapu/experiments/m14_grammar_development_v1/seed*/`.

The immediate next operation after collection is a transition-data audit—not
model fitting. We must check completion, schema conformance, capability
variation, and distribution coverage before defining the predictor split.
