"""Certify initial capability and headroom for bootstrap compiler controllers."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from run_compilergym_feasibility import evaluate_fixed_policy


def certified(record: dict) -> bool:
    v=record['validation']
    return bool(record['validation_available'] and v['okay'] and v['benchmark_semantics_validated'] and not v['benchmark_semantics_validation_failed'])


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--bank',type=Path,required=True); p.add_argument('--split',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    bank=json.loads(a.bank.read_text()); split=json.loads(a.split.read_text()); programs=split['development']
    rows=[]
    for policy in bank['policies']:
        outcomes=[]
        for benchmark in programs:
            result=evaluate_fixed_policy(benchmark,policy['actions'])
            outcomes.append({'benchmark':benchmark,'reward':result['total_reward'],'certified':certified(result),'validation':result['validation']})
        rows.append({'controller_id':policy['controller_id'],'kind':policy['kind'],'actions':policy['actions'],'mean_reward':statistics.mean(x['reward'] for x in outcomes),'min_reward':min(x['reward'] for x in outcomes),'all_certified':all(x['certified'] for x in outcomes),'outcomes':outcomes})
    hold=next(row for row in rows if row['controller_id']=='cg-hold')['mean_reward']
    for row in rows: row['headroom_vs_hold']=row['mean_reward']-hold
    failures=[row['controller_id'] for row in rows if not row['all_certified']]
    report={'protocol':'compilergym-bootstrap-bank-evaluation-v1','development_programs':programs,'held_out_touched':False,'rows':rows,'passed':not failures,'certificate_failures':failures}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'controllers':len(rows),'held_out_touched':False,'passed':report['passed'],'mean_rewards':{r['controller_id']:r['mean_reward'] for r in rows}}))
    if failures: raise SystemExit(f'uncertified bootstrap controllers: {failures}')
if __name__=='__main__': main()
