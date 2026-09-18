# M14 Development Transition Data Audit (v1)

## Decision

**Pass for development-only transition-model work.** The collection is
structurally trustworthy enough to proceed to a held-out-by-controller model
split. It is not yet sufficient to support a broad generalization claim.

## Dataset and grain

The audited dataset contains 96 rows: six development controllers times 16
canonical grammar actions. One row is one exact measured pre-capability state,
one base checkpoint, one persistent update action, one execution seed, one
fixed demand vector, and one post-update capability measurement.

Raw files are versioned as `data/m14_development_seed*_v1.json`. The audit
receipt is `reports/m14_development_audit_v1.json`.

## Checks performed and result

| Check | Result |
|---|---|
| Expected controller coverage | 6/6 present and complete |
| Expected action coverage | 16/16 actions for every controller |
| Exact pre-state provenance | 96/96 rows labeled exact |
| Canonical action identity | all row IDs match validated grammar specifications |
| Duplicate controller-action rows | none |
| Score bounds | all pre/post successes in [0, 1] |
| Demand and utility arithmetic | all rows consistent |
| Feasibility labels | all rows agree with stated reach constraint |
| Held-out isolation | no development artifacts for controllers 111 or 114 |

The outcome distribution is useful for modeling rather than degenerate: 77
rows are feasible and 19 are infeasible; utility ranges from 0 to 1; the data
contains 57 distinct post-update capability pairs. Observed reach and
pick-place changes each range from -1.0 to +0.35.

## Material limitations

1. **One execution realization per controller-action pair.** Outcome noise is
   not separately estimated. This is a medium risk for precise ranking among
   close candidates.
2. **Six development controllers.** The data can support a preliminary
   controller-held-out test but has limited statistical power.
3. **One demand vector.** It teaches capability consequences, but not yet
   decision robustness to changing future demand.
4. **Fixed finite grammar.** This validates governed selection within the
   initial update language, not open-ended update invention.

## Required safeguards for the next phase

- Split development controllers by identity for predictor selection; never
  random-split rows from the same controller across train and validation.
- Keep controllers 111 and 114 untouched until the final decision evaluation.
- Report uncertainty and candidate-set regret, not only predictive MSE.
- Add execution-seed replication and demand variation after the first
  controller-held-out result establishes a viable signal.
