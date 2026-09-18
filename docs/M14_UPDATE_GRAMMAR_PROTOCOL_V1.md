# M14 Parameterized Update Grammar Protocol (v1)

## Purpose

This protocol replaces a closed list of four hand-written update choices with
a constrained, compositional language for persistent updates to the MT2
controller. The language supports strategy discovery while preserving causal
attribution and reproducibility.

## Action language

Each action has these fields:

| Field | Values | Meaning |
|---|---|---|
| `target` | `hold`, `policy`, `world_model`, `policy_and_world_model` | Which persistent components may learn. |
| `reach_fraction` | 0 to 1 | Fraction of each policy-update batch drawn from retained reach data. |
| `gradient_steps` | 1 to 5,000, or 0 only for HOLD | Update budget. |
| `learning_rate` | 1e-5 to 1e-2 | Learning strength. |
| `freeze_encoder` | Boolean | Retain the shared representation during the update. |
| `prior_policy_l2` | 0 to 10 | Penalty for policy drift from the pre-update controller. |
| `protect_policy_during_model_update` | Boolean | Restore policy/task modules after each world-model update. |

The implementation is `scripts/m14_update_grammar.py`. It serializes each
action canonically and derives an immutable short identifier from its contents.
Invalid combinations are rejected before a run starts.

## Relation to the pilot baselines

The original `HOLD`, `POLICY_PICK`, `POLICY_PROTECTED`, and `MODEL_PROTECTED`
choices are retained as named grammar instances. The new experiment is thus
comparable to the pilot while adding parameterized compositions.

## Candidate generation

Use the seeded coverage sampler first:

```bash
python scripts/generate_m14_update_candidates.py \
  --output configs/m14_candidates_v1.json --num-random 24 --seed 201
```

The output contains all four baselines plus 24 unique samples. It is a
transparent initial sampler, not yet a learned update proposer. Generate
candidate files with multiple seeds, then reserve complete
checkpoint/candidate-generator seeds for held-out decision evaluation.

## Executing a sweep

From the compatible TD-MPC2/MetaWorld environment, with this repository's
`scripts/` directory on `PYTHONPATH`:

```bash
python scripts/run_m14_mt2_grammar_sweep.py \
  --checkpoint /path/to/base.pt \
  --candidates configs/m14_candidates_v1.json \
  --run-dir runs/m14_grammar_seed101 \
  --seed 101
```

The runner measures the base controller before any candidate runs. Every
candidate starts from the same checkpoint and yields a row containing the exact
pre-state, full update specification, execution context, demand, post-state,
utility, and feasibility. It writes these rows to `transitions.json`.

## Experimental design

The first sweep should vary all of the following:

1. At least 8 independently obtained competent base checkpoints.
2. At least 3 candidate-generation seeds per checkpoint.
3. At least 3 execution/evaluation seeds per checkpoint-candidate condition.
4. A frozen evaluation suite and retention threshold chosen before analyzing
   decision results.

The immediate data target is coverage, not a leaderboard. We need enough
diverse exact transitions to estimate both beneficial and harmful regions of
the update space.

## Safety and evidence rules

- Candidate updates are configuration actions, never arbitrary generated code.
- A failed retention constraint is retained in the dataset; it is not deleted.
- The governor may rank only candidates that pass schema validation.
- Development and held-out checkpoints must not overlap.
- Initial evaluation must compare learned selection against HOLD, the four
  named baselines, a random feasible candidate, and a measured candidate-set
  oracle.
