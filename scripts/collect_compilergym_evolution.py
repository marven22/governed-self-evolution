"""Collect exact paired compiler-policy evolution transactions."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
from run_compilergym_feasibility import evaluate_fixed_policy

def ok(r):
 v=r['validation']; return bool(r['validation_available'] and v['okay'] and v['benchmark_semantics_validated'] and not v['benchmark_semantics_validation_failed'])
def edit(actions,e):
 a=list(actions); op=e['op']
 if op=='hold': return a
 if op=='truncate': return a[:-e['count']] if len(a)>e['count'] else []
 if op=='append': return a+[e['action']]
 if op=='replace_last': return a[:-1]+[e['action']] if a else [e['action']]
 raise ValueError(op)
def evaluate(actions,programs):
 rows=[]
 for b in programs:
  r=evaluate_fixed_policy(b,actions); rows.append({'benchmark':b,'reward':r['total_reward'],'certified':ok(r),'validation':r['validation']})
 return rows
def main():
 p=argparse.ArgumentParser(); p.add_argument('--bank',type=Path,required=True);p.add_argument('--grammar',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epsilon',type=float,default=.01);p.add_argument('--retention-floor',type=float,default=-.05);a=p.parse_args()
 bank=json.loads(a.bank.read_text()); grammar=json.loads(a.grammar.read_text()); programs=json.loads(a.split.read_text())['development']; rows=[]
 for parent in bank['policies']:
  parent_eval=evaluate(parent['actions'],programs)
  if not all(x['certified'] for x in parent_eval): raise RuntimeError(f"uncertified parent {parent['controller_id']}")
  pre=[x['reward'] for x in parent_eval]
  for spec in grammar['edits']:
   child_actions=edit(parent['actions'],spec); child_eval=evaluate(child_actions,programs)
   post=[x['reward'] for x in child_eval]; delta=[q-p for p,q in zip(pre,post)]; mean=statistics.mean(delta); se=statistics.stdev(delta)/math.sqrt(len(delta)) if len(delta)>1 else 0.; lcb=mean-1.96*se
   valid=all(x['certified'] for x in child_eval); retained=min(delta)>=a.retention_floor
   accepted=bool(valid and lcb>a.epsilon and retained and spec['op']!='hold')
   rows.append({'protocol':'compilergym-evolution-transition-v1','parent_id':parent['controller_id'],'label':spec['label'],'update_spec':spec,'pre_state':{'mean_reward':statistics.mean(pre),'min_reward':min(pre),'action_length':len(parent['actions']),'program_rewards':dict(zip(programs,pre))},'child_actions':child_actions,'paired_programs':programs,'certificate':{'valid_evaluation':valid,'utility_delta':mean,'utility_lcb_95':lcb,'per_program_delta':dict(zip(programs,delta)),'retention_floor':a.retention_floor,'retention_passed':retained,'epsilon':a.epsilon,'accepted':accepted},'parent_evaluation':parent_eval,'child_evaluation':child_eval})
   a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'rows':len(rows),'accepted':sum(r['certificate']['accepted'] for r in rows),'output':str(a.output)}))
if __name__=='__main__': main()
