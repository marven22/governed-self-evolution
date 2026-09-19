"""Score one precommitted COGE choice against HOLD and the feasible oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def only(rows: list[dict], **match: str) -> dict:
    found = [r for r in rows if all(r["context"].get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise ValueError(f"expected one matching row for {match}, found {len(found)}")
    return found[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--transitions", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); plan = json.loads(a.plan.read_text()); rows = json.loads(a.transitions.read_text())
    chosen = only(rows, candidate_id=plan["chosen_candidate_id"]); hold = only(rows, candidate_label="HOLD")
    feasible = [r for r in rows if r["feasible"]]
    if not feasible: raise ValueError("no feasible candidates")
    oracle = max(feasible, key=lambda r: r["utility"])
    payload = {"score_version": "m14-coge-heldout-v3.1", "chosen": {"candidate_id": plan["chosen_candidate_id"], "label": plan["chosen_label"], "opportunity_detected": plan["opportunity_detected"], "utility": chosen["utility"], "capability": chosen["post_capability"], "feasible": chosen["feasible"]}, "hold": {"utility": hold["utility"], "capability": hold["post_capability"], "feasible": hold["feasible"]}, "oracle": {"candidate_id": oracle["context"]["candidate_id"], "label": oracle["context"]["candidate_label"], "utility": oracle["utility"], "feasible": oracle["feasible"]}, "advantage_over_hold": chosen["utility"] - hold["utility"], "regret": oracle["utility"] - chosen["utility"]}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"choice": plan["chosen_label"], "advantage_over_hold": payload["advantage_over_hold"], "regret": payload["regret"], "feasible": chosen["feasible"]}))


if __name__ == "__main__": main()
