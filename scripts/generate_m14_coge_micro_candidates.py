"""Generate a small, auditable V3.1 micro-update grammar design."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from m14_update_grammar import UpdateSpec, canonical_record, baseline_specs

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    items=[canonical_record(baseline_specs()["HOLD"],label="HOLD")]
    # Bounded, policy-only updates isolate gradual plasticity from the more
    # destructive joint/world-model changes observed in V2/V3 development.
    for steps in (25,50,100,250):
        for reach_fraction in (.25,.5,.75):
            spec=UpdateSpec("policy",reach_fraction,steps,1e-4,True,1.0,True)
            items.append(canonical_record(spec,label=f"MICRO_{steps}_{int(reach_fraction*100)}R"))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({"grammar_version":"m14-update-grammar-v1","protocol":"m14-coge-v3.1","candidates":items},indent=2)+"\n")
    print(json.dumps({"candidates":len(items)}))
if __name__=="__main__": main()
