"""Flatten corrected branch--certify histories into learning transitions.

Each row describes one candidate edit from a measured controller state.  The
confirmation result, rather than the inexpensive screen, is the learning
target whenever a candidate reached confirmation.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in sorted(glob.glob(args.trajectory_glob)):
        payload = json.loads(Path(name).read_text())
        for event in payload["history"]:
            for branch in event["branches"]:
                if "pre_capability" not in branch:
                    raise ValueError(
                        f"{name} lacks learning context; use only replay logs "
                        "written by the enriched branch-certify runner."
                    )
                target = branch["confirmation_certificate"] or branch["screen_certificate"]
                rows.append({
                    "archive_version": "m14-branch-archive-v1",
                    "source_trajectory": name,
                    "controller_id": branch["controller_id"],
                    "parent_id": branch["parent_id"],
                    "step": branch["step"],
                    "label": branch["label"],
                    "pre_capability": branch["pre_capability"],
                    "update_spec": branch["update_spec"],
                    "screen_certificate": branch["screen_certificate"],
                    "confirmation_certificate": branch["confirmation_certificate"],
                    "outcome": target,
                    "was_confirmed": branch["confirmed"],
                    "was_committed": bool(event["committed"] and event.get("chosen") == branch["label"]),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({
        "records": len(rows),
        "parents": sorted({r["parent_id"] for r in rows}),
        "confirmed": sum(r["was_confirmed"] for r in rows),
        "committed": sum(r["was_committed"] for r in rows),
    }))


if __name__ == "__main__":
    main()
