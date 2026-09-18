# M14 known-schedule self-evolution case (v1)

## Question

Can a persistent-update chooser select an internal change that maximizes
expected future capability when the next workload mix is supplied before the
choice?

## Shared substrate gate

The agent is one task-conditioned TD-MPC2 controller with a shared encoder,
world model, policy, and value heads. Its initial capability state is measured
over the three MetaWorld tasks:

`[mw-reach, mw-pick-place, mw-door-open]`.

No self-update experiment may begin until a held-out evaluation of 20 episodes
per task shows at least 0.40 success on every task. This threshold is a gate,
not a result claim. A controller that is competent at only one task is rejected
because changes to it cannot test governed preservation of several capabilities.

## Runtime design

The stock TD-MPC2 multi-task trainer is deliberately not used: it is an
offline-only `mt30`/`mt80` implementation and expects its published large
pretraining dataset. The M14 substrate instead uses online round-robin data
collection from these three environments. Every stored transition carries a
task ID; the shared TD-MPC2 task embedding receives that ID during both update
and evaluation. Tasks share the same state/action interface, so no padding
confound is introduced.

The feasibility run is 30,000 environment steps (10,000 scheduled per task),
with per-task 5-episode evaluation every 5,000 global steps. It establishes
only that collection, task conditioning, replay, and per-task evaluation work.
It must not be interpreted as a competence result.

## First known schedule

After the substrate gate passes, freeze a copy of the controller at capability
state `c_t`. Reveal the future demand vector before the update:

`d = [0.20 reach, 0.70 pick-place, 0.10 door-open]`.

The agent selects exactly one persistent update from a bounded update grammar,
initially: hold; policy update on fresh pick-place data; dynamics/value update
on the same data; or policy update with reach/door replay. All candidates use
the same data count and gradient-update budget. The selected state becomes
`c_{t+1}` and is evaluated on all three tasks.

Primary utility is `d · c_{t+1}`. Guardrail: no protected task may lose more
than 0.15 absolute success without an explicit override. The future vector is
given in this first case; later cases replace it with a forecast and then a
robust objective under uncertainty.

## Required comparisons

Report hold, random eligible update, always-update-pick-place, and the
demand-aware chooser. Evaluate over at least five independently trained
substrate seeds once the feasibility gate is met. The main outcome is choice
regret against the best feasible update under the supplied demand vector, not
raw parameter-prediction error.
