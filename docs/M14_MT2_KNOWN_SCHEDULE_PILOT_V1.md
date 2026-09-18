# M14 MT2 known-schedule persistent-update pilot (v1)

## Starting substrates

Use the two independently replicated protected warm-start checkpoints:

- seed 101: `m14_mt2_warmstart_v2/warmstart.pt`
- seed 102: `m14_mt2_warmstart_v3_seed102/warmstart.pt`

Each begins with held-out MPC success above 0.5 for both reach and pick-place.

## Revealed future demand

Before the persistent update choice, reveal `d = [0.30 reach, 0.70 pick-place]`.
The primary utility after the chosen update is
`U = 0.30 * success(reach) + 0.70 * success(pick-place)`.
Reach is protected: a candidate with an absolute reach-success loss greater
than 0.15 is infeasible unless explicitly treated as an unconstrained baseline.

## Matched update context

Every non-hold candidate receives the same fixed successful pick-place expert
trajectories, same number of gradient steps, optimizer, and evaluation seeds.
The replay-protected candidate additionally receives matched reach expert
trajectories, balanced in every minibatch. The extra reach data is part of the
intervention definition, not an accidental training-budget advantage.

## Candidates

1. `HOLD`: no persistent change.
2. `POLICY-PICK`: behavior-clone policy on pick-place data only.
3. `POLICY-PROTECTED`: behavior-clone policy on balanced pick-place and reach
   data.
4. `MODEL-PROTECTED`: update dynamics/reward/Q components on pick-place
   trajectories while restoring policy, encoder, and task embedding after each
   gradient step.

Evaluate each resulting persistent controller with MPC on 50 episodes per
task. Report capability vector, utility, feasibility, and regret to the best
feasible candidate. A future decision-aware evolution model is trained only
after several such transitions exist; this pilot supplies the transition data
and tests whether candidate consequences are heterogeneous.
