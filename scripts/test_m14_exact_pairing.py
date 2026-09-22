"""Fail fast unless unchanged-versus-unchanged exact pairing is identical."""
from __future__ import annotations
import argparse
from pathlib import Path
from envs import make_env
from tdmpc2 import TDMPC2
from m14_transactional_v5 import make_exact_episode_bank, success_from_exact_episode_bank
from run_m14_mt2_behavior_cloning import TASKS, cfg_for

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--controller-seed',type=int,required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--episodes',type=int,default=20); a=p.parse_args()
 cfg=cfg_for(a.controller_seed); env=make_env(cfg); agent=TDMPC2(cfg); agent.load(a.checkpoint)
 seeds={t:[a.seed+10000*i+j for j in range(a.episodes)] for i,t in enumerate(TASKS)}
 bank=make_exact_episode_bank(agent,env,seeds); first=success_from_exact_episode_bank(agent,env,bank); second=success_from_exact_episode_bank(agent,env,bank); env.close()
 mismatches={t:int((first[t]!=second[t]).sum()) for t in TASKS}
 if any(mismatches.values()): raise RuntimeError(f'exact pairing invariant failed: {mismatches}')
 print({'exact_pairing':'passed','episodes':a.episodes,'mismatches':mismatches})
if __name__=='__main__': main()
