"""Verify complete paired COGE development transitions before fitting."""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--data-glob',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    files=sorted(glob.glob(a.data_glob)); errors=[]; by_controller=defaultdict(list); rows=0
    for name in files:
        data=json.loads(Path(name).read_text()); rows+=len(data)
        if not data: errors.append(f'{name}: empty'); continue
        ctx=data[0].get('context',{}); cid=ctx.get('controller_id'); parent=ctx.get('parent_controller_id')
        if len(data)!=13: errors.append(f'{cid}: expected 13 rows, found {len(data)}')
        if not cid or not parent: errors.append(f'{name}: missing controller or parent identity')
        ids=[r.get('context',{}).get('candidate_id') for r in data]
        if len(ids)!=len(set(ids)) or sum(r.get('context',{}).get('candidate_label')=='HOLD' for r in data)!=1: errors.append(f'{cid}: invalid candidate grammar')
        if any('pre_update_state' not in r for r in data): errors.append(f'{cid}: missing plasticity state')
        by_controller[cid].append((parent,set(ids)))
    for cid,reps in by_controller.items():
        if len(reps)!=2: errors.append(f'{cid}: expected two replications, found {len(reps)}')
        elif reps[0][0]!=reps[1][0] or reps[0][1]!=reps[1][1]: errors.append(f'{cid}: replication identity or candidate mismatch')
    parents=sorted({reps[0][0] for reps in by_controller.values() if reps})
    result={'audit_version':'m14-coge-stratified-v1','files':len(files),'rows':rows,'controllers':sorted(by_controller),'parents':parents,'errors':errors,'passed':not errors}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'passed':not errors,'files':len(files),'rows':rows,'parents':parents,'errors':len(errors)}))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
