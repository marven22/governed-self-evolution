"""Build auditable self-evolution transition rows from completed pilot results."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); rows=[]
    for run in a.runs:
        payload=json.loads((run/'results.json').read_text()); demand=payload['demand']; metadata=payload.get('metadata',{})
        hold=payload['results']['HOLD']['metrics']
        pre=metadata.get('pre_capability',hold)
        exact_pre='pre_capability' in metadata
        update_steps=metadata.get('update_steps',500 if 'u500' in run.name else 2000)
        for update,outcome in payload['results'].items():
            m=outcome['metrics']; rows.append({'run':run.name,'seed':metadata.get('seed'),'pre_capability':{task:pre[task]['success'] for task in demand},'pre_capability_is_exact':exact_pre,'update':{'family':update,'steps':update_steps},'demand':demand,'post_capability':{task:m[task]['success'] for task in demand},'utility':outcome['utility'],'feasible':outcome['feasible']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
