# COGE Stratified Development Bank Protocol (v1)

## Motivation

Seed 124 showed that the original V3 development data did not cover weak but
highly repairable controllers. This protocol supplies controlled evidence for
the opportunity decision without using seed 124 for fitting.

## Parent-run construction

For each fresh parent seed, construct one independent MT2 training run using
the fixed BC-plus-three-round-DAgger procedure. Preserve these three persistent
checkpoints:

| Stage | Checkpoint | Intended role |
|---|---|---|
| weak | `bc/policy_bc.pt` | detect recoverable under-capability |
| intermediate | `dagger/policy_dagger_round1.pt` | decide whether a bounded repair remains useful |
| late | `dagger/policy_dagger_round3.pt` | provide a later-training capability state |

The three stages are correlated views of one parent run. They may enrich
transition learning, but all stages from a parent must remain in the same
development or held-out partition. No leave-one-stage-out estimate is valid.
They are not assumed to be monotonic in competence: strata are assigned from
the measured two-skill capability state, not the stage label.

## Frozen first bank

Build parent seeds 125--130. Before collecting transitions, allocate parents
125--128 to development and 129--130 to the final untouched test partition.
Seed 124 remains an analysis-only counterexample and is excluded from fitting,
selection, and final evaluation.

For every development checkpoint, collect two replicated executions of the
COGE micro grammar with 50 evaluation episodes and paired 0.05 retention
constraints. The revised opportunity model must be selected by leave-one-parent-out validation.
