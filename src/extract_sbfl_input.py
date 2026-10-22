
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "Data" / "Pi0" / "DH_200scenes" / "run_1"
OUTPUT = ROOT / "input" / "Pi0" / "DH_200scenes" / "run_1"

OUTPUT.mkdir(parents=True, exist_ok=True)

evaluations = list(csv.DictReader(
    (SOURCE / "dh_evaluations.csv").open(
        "r", encoding="utf-8-sig", newline=""
    )
))

scene_dirs = list((SOURCE / "results").iterdir())

episodes = []
trajectory_rows = []
errors = []
seen_ids = set()

for evaluation in evaluations:
    eval_id = evaluation["eval_id"].strip()
    failure = evaluation["failure"].strip()

    if eval_id in seen_ids:
        errors.append(f"Duplicate eval_id: {eval_id}")
        continue
    seen_ids.add(eval_id)

    if failure not in ("0", "1"):
        errors.append(f"Invalid outcome for eval_id={eval_id}")
        continue

    matches = [
        d for d in scene_dirs
        if d.is_dir() and d.name.endswith(f"Scenes_{eval_id}")
    ]

    if len(matches) != 1:
        errors.append(
            f"eval_id={eval_id}: {len(matches)} matching folders"
        )
        continue

    path = (
        matches[0] / "pi0" / "allMetrics" / "0" / "tcp_poses.json"
    )

    try:
        with path.open("r", encoding="utf-8") as f:
            poses = json.load(f)

        if not isinstance(poses, list) or not poses:
            raise ValueError("Empty or invalid trajectory")

        current_rows = []

        for step, pose in enumerate(poses):
            if not isinstance(pose, list) or len(pose) != 7:
                raise ValueError(f"Invalid pose at step {step}")

            if not all(
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and math.isfinite(x)
                for x in pose
            ):
                raise ValueError(f"Non-finite pose at step {step}")

            current_rows.append({
                "eval_id": eval_id,
                "step": step,
                "x": pose[0],
                "y": pose[1],
                "z": pose[2],
            })

        episodes.append({
            "model": "Pi0",
            "algorithm": "DH_200scenes",
            "run_id": "run_1",
            "eval_id": eval_id,
            "failure": int(failure),
            "object_name": evaluation.get("object_name", ""),
            "num_steps": len(current_rows),
        })

        trajectory_rows.extend(current_rows)

    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"eval_id={eval_id}: {exc}")

# Fail safely: do not create incomplete input CSV files.
if errors:
    report = OUTPUT / "extraction_errors.txt"
    report.write_text("\n".join(errors), encoding="utf-8")
    raise RuntimeError(
        f"Extraction failed: {len(errors)} errors. See {report}"
    )

def save_csv(filename, rows, fields):
    with (OUTPUT / filename).open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

save_csv(
    "episodes.csv",
    episodes,
    [
        "model", "algorithm", "run_id", "eval_id",
        "failure", "object_name", "num_steps"
    ],
)

save_csv(
    "trajectories.csv",
    trajectory_rows,
    ["eval_id", "step", "x", "y", "z"],
)

summary = {
    "source": str(SOURCE),
    "total_evaluations": len(evaluations),
    "extracted_episodes": len(episodes),
    "successful_episodes": sum(e["failure"] == 0 for e in episodes),
    "failed_episodes": sum(e["failure"] == 1 for e in episodes),
    "total_trajectory_positions": len(trajectory_rows),
    "errors": len(errors),
}

(OUTPUT / "validation_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nSBFL INPUT EXTRACTION COMPLETED")
print("=" * 45)

for key, value in summary.items():
    print(f"{key}: {value}")

print(f"\nFiles saved to: {OUTPUT}")
