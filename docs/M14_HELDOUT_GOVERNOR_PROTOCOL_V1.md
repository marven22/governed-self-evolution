# M14 Blind Held-Out Governor Protocol (v1)

Controllers 111 and 114 are first evaluated with `HOLD` only. The frozen
governor reads that exact pre-capability measurement and writes a `plan.json`
selecting one action from the fixed 16-candidate grammar. Only after that
immutable plan exists does the evaluator run every candidate to reveal actual
outcomes and score the precommitted choice.

The final score reports actual utility and feasibility, candidate-set oracle
utility and regret, and the `HOLD`, `POLICY_PROTECTED`, and
`MODEL_PROTECTED` baselines. No held-out transition outcome is available to the
planner.
