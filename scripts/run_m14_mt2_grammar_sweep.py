"""Execute exact, auditable capability transitions from the M14 update grammar.

Run this from a TD-MPC2/MetaWorld environment with this repository's scripts
directory on ``PYTHONPATH``. It evaluates the base checkpoint before applying
any candidate; a HOLD result is never used as a pre-state proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from m14_update_grammar import UpdateSpec
from m14_update_sampling import sample_task_balanced_ids
from run_m14_mt2_behavior_cloning import TASKS, cfg_for, evaluate
from run_m14_mt2_warmstart import collect
from m14_v3_state import plasticity_state


def load_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    result = []
    for candidate in payload["candidates"]:
        spec = UpdateSpec.from_dict(candidate["update"])
        if candidate["id"] != spec.identifier:
            raise ValueError(f"candidate id does not match canonical update: {candidate['id']}")
        result.append({"id": spec.identifier, "label": candidate.get("label"), "spec": spec})
    if len({candidate["id"] for candidate in result}) != len(result):
        raise ValueError("candidate file contains duplicate updates")
    return result


def policy_update(agent: TDMPC2, obs: torch.Tensor, actions: torch.Tensor, tasks: torch.Tensor, spec: UpdateSpec) -> None:
    """Behavioral update with explicit replay mixture and parameter-drift retention."""
    encoder_params = list(agent.model._encoder.parameters())
    policy_params = list(agent.model._pi.parameters()) + list(agent.model._task_emb.parameters())
    trainable = policy_params + ([] if spec.freeze_encoder else encoder_params)
    original_requires_grad = [parameter.requires_grad for parameter in encoder_params]
    for parameter in encoder_params:
        parameter.requires_grad_(not spec.freeze_encoder)
    optimizer = torch.optim.Adam(trainable, lr=spec.learning_rate)
    prior = [parameter.detach().clone() for parameter in policy_params]
    obs, actions, tasks = obs.cuda(), actions.cuda(), tasks.cuda()
    agent.model.train()
    try:
        for _ in range(spec.gradient_steps):
            ids = sample_task_balanced_ids(tasks, spec.reach_fraction)
            z = agent.model.encode(obs[ids], tasks[ids])
            _, info = agent.model.pi(z, tasks[ids])
            loss = torch.nn.functional.mse_loss(info["mean"], actions[ids])
            if spec.prior_policy_l2:
                drift = sum(torch.mean((parameter - initial) ** 2) for parameter, initial in zip(policy_params, prior))
                loss = loss + spec.prior_policy_l2 * drift
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 20.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    finally:
        for parameter, requires_grad in zip(encoder_params, original_requires_grad):
            parameter.requires_grad_(requires_grad)
        agent.model.eval()


def world_model_update(agent: TDMPC2, replay: Buffer, spec: UpdateSpec) -> None:
    """TD-MPC2 update with optional restoration of protected capability modules."""
    prefixes: tuple[str, ...] = ()
    if spec.protect_policy_during_model_update:
        prefixes += ("_pi", "_task_emb")
    if spec.freeze_encoder:
        prefixes += ("_encoder",)
    protected = {
        key: value.detach().clone()
        for key, value in agent.model.state_dict().items()
        if isinstance(value, torch.Tensor) and key.startswith(prefixes)
    }
    for _ in range(spec.gradient_steps):
        agent.update(replay)
        if protected:
            state = agent.model.state_dict()
            state.update(protected)
            agent.model.load_state_dict(state)
    agent.model.eval()


def apply_update(agent: TDMPC2, obs: torch.Tensor, actions: torch.Tensor, tasks: torch.Tensor, replay: Buffer, spec: UpdateSpec) -> None:
    if spec.target == "hold":
        return
    if spec.target in {"policy", "policy_and_world_model"}:
        policy_update(agent, obs, actions, tasks, spec)
    if spec.target in {"world_model", "policy_and_world_model"}:
        world_model_update(agent, replay, spec)


def compact(metrics: dict) -> dict[str, float]:
    return {task: float(metrics[task]["success"]) for task in TASKS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controller-seed", type=int, help="Identity of the trained controller; defaults to --seed.")
    parser.add_argument("--replicate-id", type=int, default=0)
    parser.add_argument("--demo-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--reach-demand", type=float, default=0.30)
    parser.add_argument("--min-reach-retention", type=float, default=0.15)
    parser.add_argument("--max-reach-drop-from-hold", type=float)
    parser.add_argument("--max-pick-place-drop-from-hold", type=float)
    parser.add_argument("--record-plasticity-state", action="store_true")
    parser.add_argument("--candidate-label", help="Run one named baseline/candidate label only (smoke testing).")
    parser.add_argument("--save-agents", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.reach_demand <= 1.0:
        raise ValueError("--reach-demand must lie in [0, 1]")

    controller_seed = args.seed if args.controller_seed is None else args.controller_seed
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    candidates = load_candidates(args.candidates)
    if args.candidate_label:
        candidates = [candidate for candidate in candidates if candidate["label"] == args.candidate_label]
        if len(candidates) != 1:
            raise ValueError("--candidate-label must select exactly one named candidate")
    if (args.max_reach_drop_from_hold is not None or args.max_pick_place_drop_from_hold is not None) and candidates[0]["label"] != "HOLD":
        raise ValueError("relative-retention runs require HOLD as the first candidate")
    demand = {TASKS[0]: args.reach_demand, TASKS[1]: 1.0 - args.reach_demand}
    cfg = cfg_for(controller_seed)
    cfg.steps, cfg.buffer_size, cfg.batch_size = 10_000, 20_000, 64
    env = make_env(cfg)
    base_agent = TDMPC2(cfg); base_agent.load(args.checkpoint)
    pre_metrics = evaluate(base_agent, env, args.eval_episodes)
    trajectories, obs, actions, tasks = collect(env, args.demo_episodes)
    pre_update_state = plasticity_state(base_agent, obs, actions, tasks) if args.record_plasticity_state else None
    replay = Buffer(cfg)
    for trajectory in trajectories:
        replay.add(trajectory)

    rows, results, hold_capability = [], {}, None
    metadata = {"grammar_version": "m14-update-grammar-v1", "checkpoint": str(args.checkpoint), "seed": controller_seed, "execution_seed": args.seed, "replicate_id": args.replicate_id, "demo_episodes_per_task": args.demo_episodes, "eval_episodes": args.eval_episodes, "pre_capability": pre_metrics}
    try:
        for candidate in candidates:
            spec = candidate["spec"]
            agent = TDMPC2(cfg); agent.load(args.checkpoint)
            apply_update(agent, obs, actions, tasks, replay, spec)
            post_metrics = evaluate(agent, env, args.eval_episodes)
            post_capability = compact(post_metrics)
            utility = sum(demand[task] * post_capability[task] for task in TASKS)
            feasible = post_capability[TASKS[0]] >= args.min_reach_retention
            if candidate["label"] == "HOLD":
                hold_capability = post_capability
            if hold_capability is not None:
                if args.max_reach_drop_from_hold is not None:
                    feasible = feasible and post_capability[TASKS[0]] >= hold_capability[TASKS[0]] - args.max_reach_drop_from_hold
                if args.max_pick_place_drop_from_hold is not None:
                    feasible = feasible and post_capability[TASKS[1]] >= hold_capability[TASKS[1]] - args.max_pick_place_drop_from_hold
            context = {"seed": controller_seed, "execution_seed": args.seed, "replicate_id": args.replicate_id, "checkpoint": str(args.checkpoint), "demo_episodes_per_task": args.demo_episodes, "eval_episodes": args.eval_episodes, "candidate_id": candidate["id"], "candidate_label": candidate["label"]}
            constraints = {"min_reach_retention": args.min_reach_retention, "max_reach_drop_from_hold": args.max_reach_drop_from_hold, "max_pick_place_drop_from_hold": args.max_pick_place_drop_from_hold}
            row = {"domain": "metaworld-mt2", "pre_capability": compact(pre_metrics), "pre_capability_is_exact": True, "update": spec.to_dict(), "context": context, "demand": demand, "post_capability": post_capability, "utility": utility, "feasible": feasible, "constraints": constraints}
            if pre_update_state is not None:
                row["pre_update_state"] = pre_update_state
            rows.append(row)
            results[candidate["id"]] = {"label": candidate["label"], "metrics": post_metrics, "utility": utility, "feasible": feasible}
            if args.save_agents:
                agent.save(args.run_dir / f"{candidate['id']}.pt")
            # Persist incrementally: an interrupted multi-hour sweep remains
            # an auditable partial dataset rather than a lost experiment.
            (args.run_dir / "transitions.partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            (args.run_dir / "progress.json").write_text(json.dumps({"completed": len(rows), "total": len(candidates), "completed_ids": [row["context"]["candidate_id"] for row in rows]}, indent=2) + "\n")
    finally:
        env.close()

    (args.run_dir / "transitions.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.run_dir / "results.json").write_text(json.dumps({"metadata": metadata, "demand": demand, "results": results}, indent=2) + "\n")
    (args.run_dir / "completion.json").write_text(json.dumps({"complete": True, "transitions": len(rows)}, indent=2) + "\n")
    print(json.dumps({"transitions": len(rows), "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    main()
