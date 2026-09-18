"""Write an auditable capability/qualification manifest for a controller bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("mw-reach", "mw-pick-place")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-success", type=float, default=0.60)
    args = parser.parse_args()
    controllers = []
    for seed_dir in sorted(args.root.glob("seed*")):
        result_path = seed_dir / "dagger" / "results.json"
        completion_path = seed_dir / "dagger" / "completion.json"
        record = {"id": seed_dir.name, "seed": int(seed_dir.name.removeprefix("seed")), "path": str(seed_dir / "dagger" / "policy_dagger.pt"), "complete": completion_path.exists()}
        if result_path.exists():
            history = json.loads(result_path.read_text())["history"]
            final = history[-1]["eval"]
            capabilities = {task: float(final[task]["success"]) for task in TASKS}
            record["capabilities"] = capabilities
            record["qualified"] = record["complete"] and all(value >= args.min_success for value in capabilities.values())
        else:
            record["qualified"] = False
        controllers.append(record)
    payload = {"schema_version": "m14-controller-bank-v1", "qualification": {"min_success_each_task": args.min_success}, "controllers": controllers}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "controller_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"controllers": len(controllers), "qualified": sum(record["qualified"] for record in controllers)}))


if __name__ == "__main__":
    main()
