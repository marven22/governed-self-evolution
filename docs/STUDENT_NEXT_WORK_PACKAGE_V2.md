# Student work package: build the AgentDojo repair-opportunity ledger

Start from the repository default branch after pulling it. Do **not** rewrite
the frozen baseline matrix or claim success from its static configurations.
Your prior adapter is now the substrate for the next experiment.

## Why the current approach is not enough

The existing work establishes a good, auditable security baseline, but it is
not yet governed continual repair:

- it tests a small fixed menu of persistent configurations;
- the measured static configurations have no target-utility improvement and
  near-zero attack success, hence little or no repair headroom;
- it has no contextual, outcome-conditioned repair generator or governor;
- it has no target/protected capability panels, certified child evaluation,
  calibrated abstention rule, or held-out promotion test; and
- it does not yet test whether later governor versions make better repairs
  without sacrificing existing capabilities.

This is not a criticism of the adapter. The adapter is the part we needed
first. The missing work is to create the *right evidence* for a governor.

## Your task

Build a **repair-opportunity audit and ledger**, not a broad configuration
sweep. For each context, define:

`(base agent, attack family, target task family, protected task family, fault,
legal candidate edits)`.

A context enters the ledger only when all of the following are observed with
the real runner:

1. The faulty/HOLD agent has a measurable deficit on the target panel.
2. At least one grammar-valid candidate improves that target outcome.
3. At least one candidate creates a plausible protected-capability tradeoff
   (security, benign utility, or tool-policy compliance).
4. The evidence includes `attacker_contacted`, task completion, policy
   compliance, normalized cost, full candidate parameters, trace hash, and
   random seed.

Use temporary child configurations for candidate testing. A failed child is
discarded, but its outcome remains valuable negative evidence in the ledger.

## Experimental discipline

- Split **by attack/injection family**, before fitting any selector or
  generator: train, calibration, final. Do not let examples from a family
  leak between splits.
- Predeclare the candidate grammar and trial budget. Do not tune those on the
  final cohort.
- Repeat stochastic contexts at least three times and retain all repetitions.
- Keep a compact `manifest.json` with exact suite version, model name,
  temperature, prompt/config hash, seeds, and trace hashes. Do not commit API
  keys or raw sensitive traces.

## Deliverables

Open a PR with:

1. a documented fault model and grammar extension, if needed;
2. an audit script that produces a reproducible qualified-context report;
3. a JSONL transition ledger with parent/child outcomes and the four-part
   quality vector; and
4. a short GO/NO-GO report: qualified-context count by family, fraction with
   a repair, fraction with a preservation tradeoff, and a clear conclusion
   about whether there is sufficient headroom for training a governor.

Do not train a sophisticated governor yet. A small scoring baseline is fine
only to validate the schema. The scientific objective of this task is to show
that the substrate contains authentic, family-held-out repair decisions.

## How this connects to the shared baseline

Once your ledger passes its audit, the shared `gse_governance` primitives will
be used to certify edits, calibrate whether direct repair is safe to use, and
evaluate promotion of a new governor version without silent capability loss.
