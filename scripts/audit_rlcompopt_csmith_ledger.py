#!/usr/bin/env python3
"""Fail-closed audit for the fixed Csmith development ledger."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 raw=a.ledger.read_bytes(); d=json.loads(raw); rows=d['transactions']; labels=Counter(r['outcome']['label'] for r in rows); ops=Counter(r['edit']['op'] for r in rows)
 keys=[(r['benchmark'],r['controller'].get('parent_rank',0),tuple(r['child_actions'])) for r in rows]
 cert_bad=[r for r in rows if not (r['outcome']['certified_child'] and r['child_certificate']['validation']['okay'] and r['child_certificate']['validation']['benchmark_semantics_validated'])]
 failures=[]
 if d.get('cohort')!='development only' or d.get('selection_touched'): failures.append('ledger is not development-only')
 if len(rows)!=420: failures.append(f'expected 420 rows, found {len(rows)}')
 if len({r['benchmark'] for r in rows})!=18: failures.append('expected 18 distinct development seeds')
 if len(keys)!=len(set(keys)): failures.append('duplicate transition')
 if cert_bad: failures.append(f'{len(cert_bad)} uncertified children')
 if labels['certified_improvement']<20: failures.append('fewer than 20 improvements')
 if labels['certified_regression_blamed']<20: failures.append('fewer than 20 blamed regressions')
 if set(ops)!={'delete','insert','replace','splice'}: failures.append('incomplete grammar coverage')
 report={'protocol':'rlcompopt-csmith-ledger-audit-v1','ledger_sha256':hashlib.sha256(raw).hexdigest(),'passed_for_gain_and_blame_governor':not failures,'failures':failures,'metrics':{'transactions':len(rows),'programs':len({r['benchmark'] for r in rows}),'labels':dict(labels),'operators':dict(ops),'certificate_failures':len(cert_bad),'duplicates':len(keys)-len(set(keys))}}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True))
 if failures: raise SystemExit('Csmith ledger audit failed')
if __name__=='__main__':main()
