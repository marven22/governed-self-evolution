# M14 TD-MPC2 Baseline Qualification Protocol v1

## Purpose

Establish a strong, reproducible controller substrate before testing persistent
self-modification. This protocol qualifies the unmodified official TD-MPC2
implementation; it does not execute any update action, adaptation, EDM, or
governor experiment.

## Immutable upstream sources

- TD-MPC2: `https://github.com/nicklashansen/tdmpc2.git`
  - revision: `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`
  - checkout: `/home/vmargapu/src/tdmpc2`
- Legacy MetaWorld required by TD-MPC2's official adapter:
  `https://github.com/Farama-Foundation/Metaworld.git`
  - revision: `04be337a12305e393c0caf0cbf5ec7755c7c8feb`
  - checkout: `/home/vmargapu/src/tdmpc2-metaworld-legacy`

The official TD-MPC2 MetaWorld adapter uses `gym`, `mujoco-py`, the v2
goal-observable API, and 100-step episodes. It is therefore deliberately
separate from the MetaWorld 3 / MuJoCo 3 M14-A runtime. Neither upstream
checkout is to be edited for baseline qualification.

The maintained upstream checkout pairs its legacy MetaWorld adapter with a
Gymnasium `Timeout` wrapper whose `max_episode_steps` property is read-only.
`scripts/run_tdmpc2_official_wsl.sh` installs a runtime-only writable
descriptor before executing `train.py`. This does not edit the upstream source
or alter TD-MPC2 learning, planning, or network code; it is a documented API
compatibility shim that should be removed if upstream unifies the wrapper APIs.

## Pinned reproduction runtime

- Python virtual environment: `/home/vmargapu/tdmpc2-metaworld-official`
- Python: 3.11.15
- PyTorch: 2.7.1+cu118
- Gym: 0.21.0
- MuJoCo: 2.1.0 with the license supplied by the TD-MPC2 project
- `mujoco-py`: 2.1.2.14
- Cython: 0.29.36 (Cython 3 is incompatible with this `mujoco-py` release)

The initial task mapping is frozen before results:

| M14-A task | Official TD-MPC2 task |
| --- | --- |
| `reach-v3` | `mw-reach` |
| `push-v3` | `mw-push` |
| `pick-place-v3` | `mw-pick-place` |
| `door-open-v3` | `mw-door-open` |
| `button-press-topdown-v3` | `mw-button-press-topdown` |

## Password-free compatibility dependencies

The historical `mujoco-py` CPU/headless builder requires OSMesa headers and
libraries. They are unpacked, without administrator access or global package
installation, under `/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs`. The
launcher exposes them only through `C_INCLUDE_PATH`, `LDFLAGS`, and
`LD_LIBRARY_PATH` for TD-MPC2 processes. MuJoCo 2.1.0 and the license supplied
by the TD-MPC2 project are likewise isolated under `/home/vmargapu/.mujoco`.

Do not share an administrator password. No password is required for this
reproduction path.

## Runtime smoke result

On 2026-09-17, the maintained official TD-MPC2 source completed the following
execution-only smoke run through `scripts/run_tdmpc2_official_wsl.sh`:

```text
task=mw-reach model_size=5 steps=1001 batch_size=256 eval_episodes=1
eval_freq=1000000 save_video=false save_agent=false enable_wandb=false
exp_name=m14_tdmpc2_smoke_seed101 seed=101
```

The run created a 4,955,238-parameter model, completed a 100-step initial
evaluation and ten 100-step seed-data episodes, allocated replay storage on
CUDA, performed the built-in seed-data pretraining update, and exited with
`Training completed successfully`. This establishes runtime compatibility only;
the observed success was 0.0 and is not a controller-performance result.

## Controller learning POC

Before the full qualification sequence, run one fixed feasibility probe:

```text
task=mw-reach model_size=5 seed=101 steps=50000 batch_size=256
eval_episodes=10 eval_freq=10000 save_video=false save_agent=true
enable_wandb=false save_csv=true exp_name=m14_tdmpc2_controller_poc_v1
```

The POC tests whether the official controller shows a learning trajectory in
the isolated runtime and measures actual wall-clock throughput. It is not a
task-suite claim. Its pre-registered evidence criterion is an improvement in
mean deterministic evaluation success over the untrained step-0 value by the
50,000-step endpoint. A flat or degrading curve stops the project from
escalating to the full 15-run qualification grid.

## Qualification sequence

1. Import `gym`, legacy `metaworld`, and `mujoco_py`; construct and reset
   `mw-reach` using the unmodified official adapter.
2. Run a short unmodified TD-MPC2 training smoke test on `mw-reach` only. Its
   purpose is execution verification, not performance measurement.
3. For each of the five frozen tasks, train/evaluate unmodified single-task
   TD-MPC2 using the official defaults except for recorded task, seed, and
   output location.
4. The pre-registered qualification budget is 1,000,000 environment steps per
   task and seed. The fixed seeds are `101`, `202`, and `303`; the model size
   is the official single-task `5` (about 5M parameters). Evaluation uses the
   official 100-step horizon, 20 deterministic episodes, and the same frozen
   task/reset protocol for every run.
5. Declare the controller qualified only if every task has final mean success
   at least `0.80` over the three seeds and no individual seed below `0.60`.
   Report per-seed success, mean, and standard deviation. No task-specific
   architecture or hyperparameter changes are permitted. These thresholds and
   the budget are frozen before inspecting final results.

Only after this gate can the project introduce fixed persistent-update probes,
then a parameterized update grammar. Results from a failed or unqualified
controller are calibration artifacts, not evidence for governed self-evolution.
