# CompilerGym governed-evolution protocol, v1

## Purpose

This domain is the controlled proof setting for governed self-evolution. A
compiler policy selects LLVM optimization passes for a program. The outer
system changes that *persistent policy* only through a branch, paired
evaluation, certificate, and commit/rollback transaction.

The initial objective is LLVM instruction-count reduction, not wall-clock
runtime. Runtime is a later secondary outcome because it is hardware-noisy.

## Feasibility gate

Before controller training or transition collection, run:

```bash
bash scripts/run_compilergym_feasibility_wsl.sh
```

The run is valid only when the report says `passed: true`. This means two
identical compiler-policy rollouts from the same benchmark have byte-identical
records and each transformed program passes CompilerGym reference-output
validation. A failure blocks all evolution-data collection.

The archived CompilerGym LLVM-10 sanitizer binaries are not portable to the
current WSL runtime. Feasibility therefore uses its deterministic
compile-and-reference-output validator and records that profile explicitly.
Sanitizer validation is a separate portability/safety gate; we must not claim
sanitizer coverage from this first environment.

## Three non-overlapping program splits

Programs are assigned once, before any learning:

- **development:** self-generated transition collection and mechanism design;
- **selection:** governor calibration and threshold selection only;
- **held out:** untouched controller-and-program evaluation only.

No held-out program may occur in a controller update, archive row, challenger
query, or governor feature fitting step.

## Evolution transaction

For a parent compiler policy \(C_t\), a state \(s_t\), and an update \(e_t\):

`branch C_t -> apply e_t -> paired evaluation -> correctness/retention certificate -> commit or rollback`

An archive row records the parent identity, program cohort, state diagnostics,
fully specified update, paired scores, correctness result, uncertainty,
certificate result, and descendant identity if committed. Rejected descendants
remain recorded; they are evidence, not discarded failures.

## Initial safe update interface

The first interface is intentionally bounded: replay allocation, failure-focus
weight, gradient-step budget, learning rate/trust-region strength, exploration
weight, and a small adapter update. It is a measurement scaffold, not the
claimed endpoint of self-evolution.

## Certification

Every candidate must satisfy all of the following:

1. CompilerGym validation succeeds for every evaluated program.
2. Mean normalized instruction-count improvement clears a prespecified lower
   confidence bound relative to HOLD.
3. A retention cohort does not exceed a prespecified regression budget.
4. The identical parent/candidate comparison is reproducible from fixed program
   identities and fixed policy-update randomness.

The certificate operates independently of the governor. A high predicted value
never overrides a failed certificate.

## Evaluation sequence

1. Establish deterministic feasibility and validation.
2. Construct a development controller bank with deliberate headroom.
3. Collect balanced accepted and rejected one-step transactions.
4. Train a conservative, uncertainty-aware one-step governor.
5. Test it on held-out controllers and programs against HOLD, random-safe edit,
   fixed heuristic, and a post-hoc oracle.
6. Only after one-step transfer is established, add a capability-boundary
   program challenger and horizon-two receding-horizon evolution MPC.

## Claim boundary

The initial claim is governed selection and certification of bounded persistent
policy improvements. Open-ended update-program generation is a later stage,
enabled by—not assumed by—the certified archive.
