# M14 MT3 substrate repair (v1)

## Diagnosis

The 30,000-step round-robin run established that shared task conditioning,
collection, replay, and evaluation work. It did not establish a usable
three-capability substrate: reach reached 0.8 success, while pick-place and
door-open remained at 0.0.

Increasing the identical random-exploration budget is not the repair. Both
failed tasks require ordered, multi-stage contact behavior; before a success,
the TD-MPC2 critic has no reliable trajectory-level evidence about the actions
that yield completion. More of the same data is therefore a weak and
unfalsifiable intervention.

## Causal repair

Use successful, fixed MetaWorld expert demonstrations to solve exploration
without removing the persistent-update question later.

1. **Demonstration validity gate.** Evaluate each benchmark expert for 20
   episodes through the exact wrapped environment. Require >=0.90 success.
   Store 100 complete trajectories/task with task IDs and success metadata.
2. **Imitation gate.** Train only the task-conditioned policy prior on the
   fixed demonstrations with balanced task minibatches. Evaluate the policy
   directly (MPC disabled). Require >=0.50 success on every task. This proves
   that the shared parameterization can represent all three skills before
   value/planning confounds are introduced.
3. **World-model and critic warm start.** Train TD-MPC2's encoder, dynamics,
   reward, and Q functions on the same task-balanced demonstrations while
   retaining the behavior-cloning policy loss. Do not use an unconstrained
   actor-Q update as the sole actor signal.
4. **Online consolidation.** Mix fixed expert trajectories with fresh
   round-robin online trajectories, stratified equally by task. Use 50% expert
   sampling initially and anneal only after each task clears 0.40 online
   success. Evaluate every 10,000 online interactions.

The base-controller gate for self-evolution remains >=0.40 held-out success
on every task (20 episodes/task). No persistent update experiment runs before
this gate passes.

## Necessary controls

- expert policy itself, to validate environment/policy wiring;
- behavior cloning only, to test representational adequacy;
- TD-MPC2 update only on the exact demonstrations, to quantify the benefit of
  the imitation auxiliary;
- full warm start plus online consolidation.

This sequence identifies *which mechanism* fixes the bottleneck. It is not a
single large training run whose outcome cannot be interpreted.
