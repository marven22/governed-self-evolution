# Blame-Gated Certified Evolution Protocol

## Scope

This protocol collects development-only evidence for a later governor. It does
not make a final generalization claim, tune on the selection cohort, or reuse
the existing held-out cohort, which has already been used for frozen-controller
evaluation.

## Objects

- **Parent**: frozen RLCompOpt controller plus its top-ranked action sequence
  for one program.
- **Symbolic edit program**: an AST from the bounded `insert`, `delete`,
  `replace`, or `splice` grammar.
- **Child**: a parent policy configuration with exactly one AST applied.
- **Certificate**: cBench reference-output-equivalence validation. A failed
  certificate is a hard rejection, never a promotion candidate.
- **Transaction ledger**: immutable parent/edit/child/outcome record.

## Matched evaluation

For each development program, the parent is evaluated twice. It is trusted for
blame only when both executions pass the semantic certificate, agree in reward
within the configured tolerance, and its reward is non-negative. Every child
is then evaluated on the same program against that exact parent.

Let `delta` be the certified child reward minus the first parent reward.

```
unsafe child       := certificate failed
improvement        := certificate passed and delta > epsilon
blame              := trusted parent and certified child and delta < -margin
neutral            := certified child otherwise
```

Thus a departure from a weak or unstable parent is exploration, not
automatically counterproductive evidence. This is the PRAXIS-style
learner-reliability gate adapted to matched compiler transitions.

## Dual archives

The collector emits initial evidence for an **edit-program archive**, grouped
by reusable symbolic template and scored by certified gain minus blame. A later
promotion stage will maintain a separate **descendant archive** of certified
children organized by capability descriptors. The two archives must not be
conflated.

## Evidence boundary

The grammar sampler is deterministic and stratified over operators, locations,
donor ranks, and short donor segments. It is a data-collection policy, not the
governor. A later learned governor will predict gain, safety, blame, and
uncertainty over this exact AST language.
