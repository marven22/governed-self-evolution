# Student Work Package: AgentDojo Extension

## Goal

Build a practical second experimental substrate for governed self-evolution.
AgentDojo evaluates tool-using LLM agents in stateful environments and includes
both ordinary tasks and prompt-injection/security evaluation. This lets us ask:

> Which persistent agent-configuration update improves future tool-use utility
> without making the agent less safe or less policy-compliant?

The student owns the benchmark adapter and transition-data infrastructure. The
core decision-aware evolution model remains shared research infrastructure.

## Why this matters

MetaWorld measures a robot's retained physical skills. AgentDojo tests the
same governing idea in a closer-to-deployment setting: an assistant may alter
its enduring memory rules, tool permissions, or instruction-handling policy.
An update can make benign tasks easier but increase prompt-injection success.
That is a nontrivial utility–security tradeoff, rather than a toy score.

## Milestone 1 — reproducible adapter

Implement a small, pinned AgentDojo experiment runner for one initial suite
(recommended: Workspace). The runner must:

- expose model/provider settings through configuration, not source edits;
- run a fixed documented subset of benign tasks and security cases;
- record package versions, model identifier, prompts/configuration hashes,
  random seeds, elapsed time, token/call cost if available, and raw evaluator
  outputs;
- produce a compact machine-readable result file and a human-readable summary;
- have a smoke test that does not require a paid model call where practical.

Do not expand to all suites or tune models until the small runner is auditable.

## Milestone 2 — persistent update grammar

Represent a controlled, reversible set of persistent configuration updates.
The initial grammar should include only documented and inspectable choices:

1. `HOLD` — unchanged configuration.
2. `INSTRUCTION_ISOLATION(level)` — explicit rules for treating tool content as
   untrusted data.
3. `TOOL_PERMISSION_POLICY(level)` — allow-list or confirmation policy for
   sensitive tool calls.
4. `MEMORY_RETRIEVAL_POLICY(level)` — what may be retained and reused across
   tasks, with isolation constraints.

Every update must be serialized, versioned, and applied before both benign and
security evaluation. Avoid online weight training or unconstrained prompt
search in this first version: they make causality and reproducibility obscure.

## Milestone 3 — transition logger and baseline matrix

For every configuration update, produce a transition row compatible with the
shared schema:

```json
{
  "domain": "agentdojo",
  "pre_capability": {
    "benign_task_success": 0.0,
    "prompt_injection_resistance": 0.0,
    "tool_policy_compliance": 0.0,
    "normalized_cost": 0.0
  },
  "update": {"family": "...", "parameters": {}},
  "context": {"suite": "workspace", "model": "...", "seed": 0},
  "post_capability": {},
  "demand": {},
  "raw_evidence_ref": "..."
}
```

Run a baseline matrix over `HOLD` and the controlled update grammar. The
report must separate development configurations from held-out configurations;
do not choose the final policy on the same attack cases used for evaluation.

## Deliverables

1. `agentdojo/` package or equivalent runner, configuration files, and tests.
2. A schema document and validator for transition rows.
3. A small versioned development dataset, with raw evidence references.
4. `docs/AGENTDOJO_ADAPTER_PROTOCOL_V1.md` and
   `docs/AGENTDOJO_BASELINE_RESULTS_V1.md`.
5. A short presentation explaining one observed utility–security tradeoff and
   the limitations of the evidence.

## Definition of done

The work is complete when another researcher can reproduce one fixed run,
inspect exactly which persistent configuration changed, regenerate the
transition rows, and verify that reported utility and security metrics use
disjoint development and held-out evaluation cases.

## Boundaries

- Do not claim a defense is generally secure from a small benchmark result.
- Do not replace formal security evaluation with an LLM self-judgment.
- Do not make hidden updates to system prompts, tools, or retrieval state.
- Do not treat a failed benign task as equivalent to a successful security
  defense; report both dimensions explicitly.
