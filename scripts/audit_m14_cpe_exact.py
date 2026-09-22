"""Fail-closed quality audit for exact paired CPE transition files."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

REQUIRED_STATE = {'bc_loss_reach','bc_loss_pick_place','policy_gradient_norm_reach','policy_gradient_norm_pick_place','policy_gradient_cosine'}

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--transitions',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--expected-labels',type=int,default=13); p.add_argument('--expected-replicates',type=int,default=2); a=p.parse_args()
 rows=json.loads(a.transitions.read_text()); errors=[]
 if len(rows)!=a.expected_labels*a.expected_replicates: errors.append(f'expected {a.expected_labels*a.expected_replicates} rows, found {len(rows)}')
 cells=set(); labels=set(); hold=[]
 for i,row in enumerate(rows):
  missing={'controller_id','parent_id','replicate','label','update_spec','pre_capability','pre_update_state','certificate','paired_episode_count'}-set(row)
  if missing: errors.append(f'row {i}: missing {sorted(missing)}'); continue
  cell=(row['replicate'],row['label']); labels.add(row['label'])
  if cell in cells: errors.append(f'row {i}: duplicate replicate/label {cell}')
  cells.add(cell)
  if set(row['pre_update_state']) < REQUIRED_STATE: errors.append(f'row {i}: incomplete plasticity state')
  values=list(row['pre_capability'].values())+list(row['certificate']['capability_delta'].values())+[row['certificate']['utility_delta']]
  if not all(math.isfinite(float(x)) for x in values): errors.append(f'row {i}: non-finite outcome')
  if row['paired_episode_count'] <= 0: errors.append(f'row {i}: invalid episode count')
  if row['label']=='HOLD': hold.append(row)
 if len(labels)!=a.expected_labels: errors.append(f'expected {a.expected_labels} labels, found {len(labels)}')
 if len(hold)!=a.expected_replicates: errors.append(f'expected {a.expected_replicates} HOLD rows, found {len(hold)}')
 for row in hold:
  c=row['certificate']; exact=[c['utility_delta'],*c['capability_delta'].values()]
  if any(abs(float(x))>1e-12 for x in exact): errors.append(f'HOLD invariant failed at replicate {row["replicate"]}: {exact}')
 report={'dataset':str(a.transitions),'rows':len(rows),'labels':sorted(labels),'hold_rows':len(hold),'valid':not errors,'errors':errors}
 a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
 if errors: raise SystemExit(1)
if __name__=='__main__': main()
