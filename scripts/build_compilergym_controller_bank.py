"""Create a reproducible, diverse bank of bootstrap LLVM pass policies."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    import compiler_gym
    env=compiler_gym.make('llvm-v0',observation_space='Autophase',reward_space='IrInstructionCountOz')
    try: action_count=int(env.action_space.n)
    finally: env.close()
    policies=[{'controller_id':'cg-hold','kind':'hold','seed':None,'actions':[]}]
    for i,horizon in enumerate((2,4,6,8,10,12,16),start=1):
        seed=41000+i
        rng=random.Random(seed)
        policies.append({'controller_id':f'cg-bootstrap-{i:02d}','kind':'fixed-pass-sequence','seed':seed,'actions':[rng.randrange(action_count) for _ in range(horizon)]})
    bank={'protocol':'compilergym-bootstrap-controller-bank-v1','action_space_size':action_count,'policies':policies,'note':'Bootstrap policies establish certified headroom; learned feature-conditioned policies follow after transactional feasibility.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(bank,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'controllers':len(policies),'action_space_size':action_count,'output':str(a.output)}))
if __name__=='__main__': main()
