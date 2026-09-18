# Governed Self-Evolution

This repository studies how an already-capable agent should decide whether to
make a *persistent* internal update. The central concern is not merely whether
an update improves the immediately trained skill, but whether it produces a
better and safe capability profile for the tasks the agent will face later.

The first substrate is MetaWorld MT2 (reach and pick-place). It establishes a
controlled example of the core failure mode: an update that improves pick-place
can substantially damage reach. The next substrate is AgentDojo, where an
update can improve ordinary tool-use utility while weakening resistance to
prompt injection or violating tool-use policy.

## Repository layout

- `scripts/` — MetaWorld pilot, transition-building, and predictor-training code.
- `data/` — small, versioned transition datasets; `*_exact_*` records true
  pre-update capability measurements.
- `models/` — the initial small ridge predictor built from the exact dataset.
- `docs/METHODOLOGY.md` — research question, formalization, evidence, and plan.
- `docs/STUDENT_AGENTDOJO_WORK_PACKAGE.md` — parallel work package for the
  AgentDojo extension.
- `docs/M14_*.md` — original MetaWorld protocols and pilot records.

## Current status and evidence boundary

The MetaWorld pilot confirms the *phenomenon*: unprotected specialization can
produce catastrophic retention failure, while protected update rules can often
retain the old skill and improve the target skill. We have also built an
auditable transition dataset and trained a minimal action-conditioned ridge
predictor as a pipeline check.

This is not yet evidence that the learned predictor generalizes or selects the
best update reliably. The exact dataset currently has 16 transition rows from
two seeds and two update budgets. The next MetaWorld stage must diversify
starting checkpoints, update parameters, demands, and evaluation seeds before
we evaluate transition prediction and decision regret on held-out transitions.

## Reproducing the current data pipeline

The scripts run against the official TD-MPC2 source tree plus a compatible
MetaWorld/MuJoCo runtime. They deliberately do **not** vendor TD-MPC2, MuJoCo,
or trained controller checkpoints. Configure those external dependencies first,
then run the experiment scripts from the TD-MPC2 source environment.

To rebuild the committed exact transition table from completed run directories:

```bash
python scripts/build_m14_evolution_transitions.py --runs <run-dir> ... \
  --output data/m14_evolution_transitions_exact_v1.json
```

To train the current initial predictor:

```bash
python scripts/train_m14_evolution_predictor.py \
  --data data/m14_evolution_transitions_exact_v1.json \
  --output models/m14_evolution_predictor_v1.pkl
```

The shell launchers under `scripts/*_wsl.sh` document the prior runtime
assumptions. They are historical launch helpers, not a promise of a portable
one-command installation.

## Reproducibility principles

1. Every update transition records a measured pre-state, update description,
   context, post-state, demand, and evaluation seed.
2. A row with a proxy pre-state is labeled as such and is not used as a primary
   learning target.
3. Decision evaluation is held out by starting state/update context, not just
   by a random row split.
4. Checkpoints and large outputs remain outside Git; publish a manifest with
   immutable paths or hashes when they are needed to reproduce a result.
