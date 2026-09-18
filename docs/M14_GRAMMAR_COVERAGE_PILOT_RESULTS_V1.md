# M14 Grammar Coverage Pilot Results (v1)

## Purpose and scope

This is the first exact-transition collection using the parameterized update
grammar. It is a small coverage pilot, not an evaluation of a learned
decision-aware governor.

The pilot used two independently trained competent DAgger checkpoints, one
fixed candidate-generator seed, 50 expert demonstration episodes per task, and
20 evaluation episodes per task. Each of the eight candidates began from a
fresh copy of the relevant base checkpoint. The retention constraint was reach
success at least 0.15. Demand was 0.30 reach and 0.70 pick-place.

Raw exact transition rows:

- `data/m14_grammar_coverage_seed101_v2.json`
- `data/m14_grammar_coverage_seed102_v1.json`

An earlier seed-101 implementation run is excluded because its zero-reach
mixture boundary was executed incorrectly (one retained-reach sample per
batch). The executor was corrected, tested, and the seed-101 condition was
rerun as `v2` before producing these results.

## Aggregate results across two checkpoints

| Candidate | Mean reach | Mean pick-place | Mean demand utility | Feasible runs |
|---|---:|---:|---:|---:|
| Policy, 50/50 data, frozen encoder (named `POLICY_PROTECTED`) | 0.900 | 0.975 | 0.952 | 2/2 |
| Policy, 75% reach, frozen encoder, 500 steps, 1e-4 | 1.000 | 0.875 | 0.912 | 2/2 |
| HOLD | 0.775 | 0.875 | 0.845 | 2/2 |
| Policy, 75% reach, trainable encoder, L2 retention 0.1 | 0.900 | 0.825 | 0.847 | 2/2 |
| Protected world-model update | 0.900 | 0.800 | 0.830 | 2/2 |
| Policy + world model, 75% reach, frozen encoder, L2 1.0 | 0.850 | 0.775 | 0.797 | 2/2 |
| Pick-place-only policy specialization (named `POLICY_PICK`) | 0.000 | 0.950 | 0.665 | 0/2 |
| Policy + world model, 60% reach, frozen encoder, 1,500 steps | 0.000 | 0.000 | 0.000 | 0/2 |

## Interpretation

Three conclusions are justified by this pilot:

1. The grammar produces both useful, feasible updates and catastrophic
   updates. It is therefore a meaningful action space for governed selection.
2. The original interference phenomenon survives the grammar refactor:
   pick-place-only specialization is not feasible because it erases reach.
3. The best observed candidate depends on the allowed update construction;
   a low-strength, reach-heavy protected policy update is competitive with the
   named protected baseline.

The results do **not** establish that a transition predictor can extrapolate,
that the apparent ranking will hold across checkpoints, or that an optimizer
has discovered a generally useful self-update strategy. Sixteen rows from two
checkpoints are insufficient for those claims.

## Immediate next data collection

1. Add at least six more independently trained competent base checkpoints.
2. Use additional candidate-generator seeds with deliberate coverage of
   pick-place-heavy, balanced, and reach-heavy mixtures.
3. Replicate candidate execution with additional evaluation seeds.
4. Split entire checkpoints and candidate-generation seeds into development
   and held-out partitions before fitting or evaluating the governor.
