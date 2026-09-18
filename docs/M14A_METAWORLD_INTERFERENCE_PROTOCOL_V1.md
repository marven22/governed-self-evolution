# M14-A: MetaWorld interference characterization, v1

## Decision

This is a scientific gate, not an efficacy experiment. Its sole question is
whether persistent, heterogeneous world-model updates yield different
cross-task capability profiles. We do not train an Evolution Dynamics Model,
forecast task demand, or claim a lifetime-planning advantage in this phase.

## Frozen design

The configuration is [configs/m14a_metaworld_interference_v1.json](../configs/m14a_metaworld_interference_v1.json).
It uses five predeclared MT10-family tasks: reach, push, pick-place, door-open,
and button-press-topdown. They were selected by interaction primitive before
any adaptation result is observed: free-space motion, object translation,
grasp/lift, articulated-object motion, and constrained contact respectively.

For every base seed, adaptation task, and candidate (`HOLD`, `LOCAL`, `ENC`,
`DYN`), clone the same base world model and apply the frozen adaptation budget.
Evaluate every resulting clone on each of the five tasks using identical
frozen transition sets and matched control evaluation seeds. The raw object of
analysis is therefore a 5 x 4 x 5 matrix, separately for prediction and
control outcomes.

`HOLD` performs no optimisation. `LOCAL` changes only the zero-initialized
rank-8 latent residual adapter. `ENC` changes the encoder only. `DYN` changes
the residual latent dynamics and its reward head only. The behavior-cloned
proposal actor remains frozen for every candidate; model-based control uses it
only to keep MPC action candidates inside the fixed data distribution. Candidate
update mechanisms must have equal
update counts and batch size; the fixed learning rate is not tuned against a
future-demand or planner metric.

## Infrastructure gate

The existing repository's MetaWorld artifacts are cached policy-outcome
studies; they are not a suitable substitute for MuJoCo rollouts or neural
world-model adaptation. The previously used local runtime is preserved at
`/home/vmargapu/metaworld-v2/bin/python` under WSL. The M14-A preflight passed
there on 2026-09-17 for all five frozen tasks (MetaWorld 3.0.0, MuJoCo 3.11.0,
Python 3.11.15). Use that runtime rather than creating a new environment.

From Windows, run:

```bash
wsl -d Ubuntu -- env MUJOCO_GL=egl /home/vmargapu/metaworld-v2/bin/python \
  /mnt/c/Users/vmargapu/OneDrive\ -\ Villanova\ University/Documents/Research/memory-corruption-experiment-harness/scripts/m14a_preflight.py \
  --output /mnt/c/Users/vmargapu/OneDrive\ -\ Villanova\ University/Documents/Research/memory-corruption-experiment-harness/experiments/m14a/preflight.json
```

The preflight must create, reset, and step all five environments and report
the expected observation/action shapes. It records package versions, platform,
and commit hash but writes no scientific results. PyTorch is not currently
installed in that preserved environment, so add a CUDA-capable PyTorch package
there before executing neural world-model training.

## Analysis and go/no-go rule

For each metric, compute paired candidate-minus-HOLD deltas at a fixed base
seed, adaptation task, evaluated task, evaluation seed, and starting state.
Report seed-stratified means and paired bootstrap intervals, all 25 cells, and
the full raw episode ledger. Do not pool task cells as independent samples.

Proceed to M14-B only if all three preregistered conditions hold:

1. At least two updates have distinguishable cross-task capability-delta profiles.
2. The update that maximizes the current-task metric is not always the one with least collateral loss.
3. At least one explicit future-demand weighting reverses the current-task-greedy choice when sandbox outcomes are used.

If any condition fails, publish the matrix and diagnose the failed premise.
No EDM is trained as a rescue mechanism.
