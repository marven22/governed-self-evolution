# Governed repair baseline (v1)

This branch now combines the real AgentDojo adapter with a small, portable
governed-repair layer in `gse_governance/`. It is a baseline for a research
program, not a claim that the full self-evolving system has been demonstrated
with real LLM rollouts.

## What is already reliable

The merged AgentDojo work gives us an unusually useful experimental substrate:

- a pinned AgentDojo version and a real Workspace LLM runner;
- replayable per-case traces, retries, resumption, cost reports, and tests;
- family-blocked development, selection, and held-out splits;
- a persistent configuration-update grammar; and
- explicit `attacker_contacted` logging. This matters because a task can look
  defended after it has already leaked data.

Its first real-LLM development matrix is a valuable negative result: all
tested static configurations resisted the sampled attacks, none improved task
utility, and the policies were effectively inert in the measured traces. A
governor cannot demonstrate repair where this matrix gives it no observable
repair opportunity.

## What this baseline adds

`gse_governance.risk` contains domain-independent pieces we will reuse in
AgentDojo and other substrates:

1. **Quality vector.** An edit is judged by target security and utility plus
   protected security and utility, not a single reward.
2. **Certificate.** An edit must clear declared target and protection floors
   before it can persist.
3. **Risk-controlled routing.** A direct proposal is used only when a
   predeclared threshold clears a simultaneous exact binomial upper bound on
   protected failures; otherwise the system can fall back to a more careful
   route or abstain.
4. **Promotion rule.** A new governor is promoted only if it is non-regressive
   on all four quality coordinates and produces more repairs (or equal repairs
   with a smaller edit).

These are deliberately small and testable. They do **not** turn a finite
calibration set into a universal safety guarantee. They make the finite-sample
claim explicit and auditable.

## Research boundary

The next milestone is an evidence ledger containing real repair opportunities:
contexts where HOLD has a measurable target deficit, at least one legal edit
can help the target, and at least one plausible edit reveals a preservation
tradeoff. Only after that ledger exists should we train a contextual governor,
calibrate a gate, and evaluate frozen versions on an untouched family-held-out
cohort.
