# Transactional Evolution MPC (V5)

V4 is frozen as a negative one-shot prediction result. V5 changes the
commitment semantics, not merely its regressor.

At evolution time the planner proposes a typed 25-step edit. The candidate is
applied to a reversible checkpoint and tested on paired task-instance seeds.
It becomes persistent only if a predeclared paired lower-confidence certificate
shows positive demand-weighted gain and no protected capability drop beyond
its retention budget. Otherwise the exact checkpoint is restored.

The future MPC state is capability, plasticity, demand, posterior dynamics,
and remaining family-wise risk budget. It plans a short edit sequence but
executes only the first transaction; the certified result updates its belief
and it replans. This is the critical distinction between prediction-only V4
and governed self-evolution.

The V5 initial kernel uses a conservative paired Hoeffding certificate. Before
any empirical claim, we will calibrate its episode budget and risk allocation
on a new development bank, then reserve a fresh parent bank for a blind test.
