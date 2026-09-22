# Certified Pessimistic Evolution: Balanced Collection

Development parents 143--150 provide three correlated stages (BC, R1, R3).
Each stage is a frozen branch point. Every grammar edit, including HOLD, is
applied to a temporary copy from that exact point and evaluated on two
independent matched seed sets. The permanent controller is never changed.

The resulting rows are state--edit--outcome observations, not an evolution
trajectory. Parent-disjoint splits are mandatory for model selection.

`m14_cpe_balanced_development_v1` is preserved for reset debugging only. It
used seed-based resets that failed the HOLD-versus-HOLD invariant. Valid data
collection begins at `m14_cpe_balanced_development_v2_exact`, whose evaluator
restores both exact MuJoCo and task-level reset state for every paired rollout.
