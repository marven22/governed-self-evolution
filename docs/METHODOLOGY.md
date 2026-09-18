# Methodology: Decision-Aware Governed Self-Evolution

## Question

An agent has useful capabilities already. It may change itself through a
persistent update: training on new evidence, changing its memory/retrieval
rules, or tightening how it calls tools. Which update should it choose?

The important point is that an update is an **action with lasting
consequences**. Optimizing only the newly emphasized task can erase a capability
that matters tomorrow. We want an agent to predict those consequences and make
an update decision that is useful under plausible future demand while obeying
retention and safety constraints.

## Two time scales

There are two separate control problems.

1. **Task control:** the fixed agent acts in its current environment. In
   MetaWorld, this means robot actions; in AgentDojo, tool calls and responses.
2. **Evolution control:** between episodes or task blocks, the system chooses a
   persistent update to its own policy, model, memory, or guardrails.

Our contribution concerns the second layer. A strong task controller such as
TD-MPC2 is a substrate, not the claimed solution.

## Formalization

Let the measured capability state be

\[
c_t = [s_1, \ldots, s_K],
\]

where each component is an independently measured success, utility, or safety
metric. An evolution action \(u_t\), taken in update context \(z_t\), produces

\[
c_{t+1} = f_\theta(c_t, u_t, z_t) + \epsilon.
\]

Future demand is represented by a weight vector \(d\). Given a learned
transition model, the governor selects

\[
u^* = \arg\max_{u \in \mathcal U} d^\top \hat c_{t+1}(c_t,u,z_t)
\]

subject to constraints such as retention \(\hat c_{t+1,j} \ge r_j\), safety,
or resource cost. With uncertain demand, the objective becomes expected or
robust value across a demand distribution.

This makes the correct evaluation target decision quality: constraint
violations, utility regret against the best feasible measured update, and
calibration of the predicted tradeoffs—not only mean-squared error of a
capability predictor.

## MetaWorld first case

The initial MT2 state is

\[
c = [s_{reach}, s_{pick\text{-}place}].
\]

We begin from a shared TD-MPC2 checkpoint and measure its two task successes.
We then apply one of four persistent updates:

- `HOLD`: make no update.
- `POLICY_PICK`: specialize policy learning on pick-place demonstrations.
- `POLICY_PROTECTED`: learn from balanced demonstrations while protecting
  representation/task components.
- `MODEL_PROTECTED`: update world-model components while restoring protected
  policy/representation components.

For each candidate, we measure post-update performance on both tasks. The
pilot shows that `POLICY_PICK` can improve its local training target while
destroying retained reach performance; protected updates can avoid that failure
in the observed cases. This validates the need for governed update selection,
but is not a generality claim yet.

## Transition-data schema

Each row must contain:

```json
{
  "pre_capability": {"capability_name": 0.0},
  "pre_capability_is_exact": true,
  "update": {"family": "...", "steps": 0},
  "context": {"seed": 0, "checkpoint_id": "..."},
  "demand": {"capability_name": 0.0},
  "post_capability": {"capability_name": 0.0},
  "utility": 0.0,
  "feasible": true
}
```

`pre_capability_is_exact` is mandatory. A post-update `HOLD` result is not a
substitute for measuring the state before the update; it may be a useful proxy
for debugging, but it must not silently become ground truth.

## Planned evidence ladder

1. **Interference:** demonstrate multi-capability tradeoffs under persistent
   updates. Completed as a small MetaWorld pilot.
2. **Transition learning:** collect sufficiently varied, exact before/after
   rows; evaluate held-out transition prediction by starting state and context.
3. **Decision learning:** compare the decision-aware governor with HOLD,
   specialized updating, protected heuristic updates, and an oracle over the
   candidate set. Report utility, feasibility, regret, and uncertainty.
4. **Generalization of update language:** replace the four hand-designed
   update families with a parameterized update grammar and learn/rank updates
   within it. The grammar is necessary before claiming strategy discovery.
5. **External validity:** reproduce the same governed-update question in a
   practical tool-using agent domain (AgentDojo), then consider additional
   embodied or operational domains.

## What we should not claim yet

- We have not shown open-ended self-improvement or autonomous invention of
  update strategies.
- The tiny predictor is a data-to-model plumbing check, not a validated
  forecaster.
- MetaWorld alone cannot establish broad practical utility.
- TD-MPC2 is not the proposed general governing mechanism; it is one task
  controller used to create and evaluate transitions.
