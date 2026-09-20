"""Materialize a PRAXIS-style archive from transactional V5 outcomes."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--results-glob',required=True);p.add_argument('--state-glob',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 states={}
 for f in glob.glob(a.state_glob):
  for r in json.loads(Path(f).read_text()): states.setdefault((r['context'].get('controller_id'),r['context']['candidate_label']),r)
 archive=[]
 for f in glob.glob(a.results_glob):
  r=json.loads(Path(f).read_text()); parts=Path(f).parts; controller=parts[-3]; label=r['label']; s=states.get((controller,label))
  if s is None: raise ValueError(f'missing state transition for {controller}/{label}')
  c=r['certificate']; archive.append({'archive_version':'m14-transactional-v5','controller_id':controller,'parent_id':f"p{r['controller_seed']}",'update_id':r['candidate_id'],'label':label,'pre_capability':s['pre_capability'],'pre_update_state':s['pre_update_state'],'update':s['update'],'certificate':c,'outcome':{'utility_delta':c['utility_delta'],'capability_delta':c['capability_delta'],'committed':c['committed']},'provenance':{'transaction_result':f,'state_transition':next(x for x in glob.glob(a.state_glob) if controller in x)}})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(archive,indent=2)+'\n');print(json.dumps({'records':len(archive),'committed':sum(x['outcome']['committed'] for x in archive),'parents':sorted(set(x['parent_id'] for x in archive))}))
if __name__=='__main__':main()
