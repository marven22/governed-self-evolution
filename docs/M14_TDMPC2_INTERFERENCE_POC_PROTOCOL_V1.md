# M14 TD-MPC2 Interference POC Protocol v1

## Objective

Establish whether persistent parameter updates to a competent TD-MPC2 agent
produce distinct adaptation--retention tradeoffs. This is a controlled
phenomenon check, not a learned-governor experiment.

## Frozen starting point

- Base controller: completed `mw-reach`, seed 101, 50,000 steps.
- Checkpoint: TD-MPC2 `final.pt` from `m14_tdmpc2_controller_poc_v1`.
- Adaptation task: `mw-pick-place`.
- Retention task: `mw-reach`.
- Held-out diagnostic task: `mw-door-open`.

## Candidate persistent interventions

All candidates begin from byte-identical copies of the base checkpoint and
receive the same adaptation interactions, reset seeds, and update budget.

| Label | Persistent parameters eligible to change |
| --- | --- |
| HOLD | none |
| POLICY | policy prior and Q/value heads |
| DYNAMICS | latent dynamics, reward model, and Q/value heads |
| REPRESENTATION | encoder, dynamics, reward model, Q/value heads, and policy prior |

The policy and value heads are explicitly separated in the implementation even
though TD-MPC2 represents value through Q-functions. These labels are a
controlled diagnostic vocabulary, not the eventual learned update language.

## Budget and measurement

- One seed (`101`) for this POC.
- 5,000 adaptation environment steps on `mw-pick-place` per intervention.
- Exactly 10 deterministic evaluation episodes after the update on each of
  `mw-reach`, `mw-pick-place`, and `mw-door-open`.
- Record success and return. Preserve the copied checkpoints, adaptation
  trajectory seeds, resolved config, and raw per-episode outcomes.

## POC decision rule

Proceed to a three-task matrix only if at least two interventions differ by at
least 0.20 success on either retention or adaptation while remaining within the
same fixed data/update budget. Otherwise, report a null/interchangeable-update
result and redesign the update grammar; do not train an EDM on arbitrary labels.
