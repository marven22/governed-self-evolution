#!/usr/bin/env python3
"""Build a post-deployment archive summary and fail closed on governor updates.

Rollout records are evidence for a future shadow challenger, not permission to
silently update the deployed governor.  Independence is counted by program,
not by raw edit count.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rollout',type=Path,required=True); p.add_argument('--gate',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    rollout=json.loads(a.rollout.read_text()); gate=json.loads(a.gate.read_text())
    lineages=rollout.get('lineages',[]); decisions=[(lineage,step) for lineage in lineages for step in lineage.get('steps',[])]
    actions=Counter(step.get('action') for _,step in decisions); programs=sorted({lineage['benchmark'] for lineage,_ in decisions})
    certificates=[step.get('child_certified') for _,step in decisions if step.get('action') in ('promote','rollback')]
    requirements={
      'new_decision_contexts':{'observed':len(decisions),'required':gate['minimum_new_decision_contexts'],'passes':len(decisions)>=gate['minimum_new_decision_contexts']},
      'distinct_programs':{'observed':len(programs),'required':gate['minimum_distinct_programs'],'passes':len(programs)>=gate['minimum_distinct_programs']},
      'certified_promotions':{'observed':actions['promote'],'required':gate['minimum_certified_promotions'],'passes':actions['promote']>=gate['minimum_certified_promotions']},
      'rejections_or_rollbacks':{'observed':actions['rollback'],'required':gate['minimum_rejections_or_rollbacks'],'passes':actions['rollback']>=gate['minimum_rejections_or_rollbacks']},
      'semantic_certificate_integrity':{'observed':sum(x is False for x in certificates),'required':0,'passes':not any(x is False for x in certificates)},
    }
    ready=all(item['passes'] for item in requirements.values())
    report={'protocol':'rlcompopt-governor-update-readiness-v1','scope':'post-deployment rollout archive; no governor fitting or promotion performed','source_rollout':str(a.rollout),'gate':gate,'archive_summary':{'lineages':len(lineages),'decision_contexts':len(decisions),'distinct_programs':programs,'action_counts':dict(sorted(actions.items()))},'requirements':requirements,'shadow_challenger_training_ready':ready,'incumbent_action':'retain_v2','reason':'All readiness gates passed; shadow training may begin.' if ready else 'Insufficient diverse post-deployment evidence. Retain V2 and continue archiving.'}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({'ready':ready,'archive_summary':report['archive_summary'],'unmet':[k for k,v in requirements.items() if not v['passes']]},sort_keys=True))
if __name__=='__main__': main()
