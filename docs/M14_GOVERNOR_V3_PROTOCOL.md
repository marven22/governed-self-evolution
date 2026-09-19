# M14 Governor V3: Calibrated Multi-Capability Retention

V2's five-controller blind result was mixed: it improved mean utility by only
0.004 over `HOLD` and selected one harmful update. V3 is a new protocol, not a
post-hoc relabeling of V2.

## Repairs

1. **Relative retention.** An update must preserve each protected skill within
   a predeclared tolerance of the exact paired `HOLD` capability. An absolute
   reach floor alone is insufficient.
2. **Plasticity state.** Before candidate selection, record per-task behavior
   cloning loss, policy-gradient norms, and reach/pick gradient cosine. These
   are read-only probes of update sensitivity.
3. **Replicated transitions.** Run each controller--candidate pair in multiple
   independently seeded executions and evaluate with more episodes. Aggregate
   replicate outcomes before fitting or calibrating the governor.
4. **Calibration target.** Select abstention parameters by nested
   controller-level validation subject to a prespecified harmful-update target;
   do not select a zero confidence penalty merely because a tiny validation set
   contains no harmful update.

## V3 safety rule

For every protected capability \(j\), a non-HOLD update is eligible only if

\[
\Pr[c_j^+(u) \ge c_j^+(\mathrm{HOLD}) - \epsilon_j] \ge 1-\alpha
\]

and its lower confidence bound on demand-weighted advantage clears the
predeclared margin. Otherwise select `HOLD`.

## Separation

All controllers 117--123 are burned after V2. They may become V3 development
evidence, but V3 confirmation must use newly trained controller identities.

## First development-enrichment run

The first V3 enrichment run uses controllers 117, 120, 121, 122, and 123,
which were already exposed by V2. It executes each of the frozen 16 grammar
actions twice from the original checkpoint, using distinct execution seeds,
50 evaluation episodes per task, and tolerances \(\epsilon_{reach}=
\epsilon_{pick}=0.05\). The controller identity and execution replicate are
recorded separately. These data are development-only and cannot be reused as a
V3 confirmation test.
