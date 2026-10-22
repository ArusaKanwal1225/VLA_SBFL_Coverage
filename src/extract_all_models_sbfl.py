
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "Pi0": "pi0",
    "EO1": "eo1",
    "Groot": "gr00t",
    "SpatialVLA": "spatialvla",
}

ALGORITHMS = [
    "DH",
    "DH_HybridDT",
    "DH_HybridRF",
    "DH_WeightedRF",
]

BUDGETS = [200, 400]

SOURCE_ROOT = ROOT / "Data"
INPUT_ROOT = ROOT / "input"
REPORT_ROOT = ROOT / "output" / "input_validation"

REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_run(model, model_folder, algorithm, budget, run_number):
    algorithm_dir = f"{algorithm}_{budget}scenes"
    run_id = f"run_{run_number}"

    source = SOURCE_ROOT / model / algorithm_dir / run_id
    destination = INPUT_ROOT / model / algorithm_dir / run_id

    csv_path = source / "dh_evaluations.csv"
    results_dir = source / "results"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing evaluations: {csv_path}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results: {results_dir}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        evaluations = list(csv.DictReader(f))

    required = {"eval_id", "failure"}
    if evaluations and not required.issubset(evaluations[0]):
        raise ValueError("Missing eval_id or failure column")

    scene_map = {}
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        match = re.search(r"Scenes_(\d+)$", folder.name)
        if match:
            scene_map.setdefault(match.group(1), []).append(folder)

    episodes = []
    trajectories = []
    errors = []
    seen = set()

    for row in evaluations:
        eval_id = row["eval_id"].strip()
        failure = row["failure"].strip()

        if eval_id in seen:
            errors.append(f"Duplicate eval_id: {eval_id}")
            continue
        seen.add(eval_id)

        if failure not in ("0", "1"):
            errors.append(f"Invalid failure: eval_id={eval_id}")
            continue

        matches = scene_map.get(eval_id, [])
        if len(matches) != 1:
            errors.append(
                f"eval_id={eval_id}: {len(matches)} matching scene folders"
            )
            continue

        pose_file = (
            matches[0] / model_folder
            / "allMetrics" / "0" / "tcp_poses.json"
        )

        try:
            with pose_file.open("r", encoding="utf-8") as f:
                poses = json.load(f)

            if not isinstance(poses, list) or not poses:
                raise ValueError("Empty trajectory")

            episode_id = (
                f"{model}_{algorithm_dir}_{run_id}_{eval_id}"
            )

            current_rows = []
            for step, pose in enumerate(poses):
                if not isinstance(pose, list) or len(pose) != 7:
                    raise ValueError(f"Invalid pose at step {step}")

                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in pose
                ):
                    raise ValueError(f"Non-finite pose at step {step}")

                current_rows.append({
                    "episode_id": episode_id,
                    "step": step,
                    "x": pose[0],
                    "y": pose[1],
                    "z": pose[2],
                })

            episodes.append({
                "model": model,
                "algorithm": algorithm_dir,
                "budget": budget,
                "run_id": run_id,
                "eval_id": eval_id,
                "episode_id": episode_id,
                "failure": int(failure),
                "object_name": row.get("object_name", ""),
                "num_steps": len(current_rows),
            })
            trajectories.extend(current_rows)

        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"eval_id={eval_id}: {exc}")

    if errors:
        error_file = (
            REPORT_ROOT /
            f"{model}_{algorithm_dir}_{run_id}_errors.txt"
        )
        error_file.write_text("\n".join(errors), encoding="utf-8")
        raise RuntimeError(
            f"{len(errors)} extraction errors; see {error_file}"
        )

    save_csv(
        destination / "episodes.csv",
        [
            "model", "algorithm", "budget", "run_id",
            "eval_id", "episode_id", "failure",
            "object_name", "num_steps"
        ],
        episodes,
    )

    save_csv(
        destination / "trajectories.csv",
        ["episode_id", "step", "x", "y", "z"],
        trajectories,
    )

    summary = {
        "model": model,
        "algorithm": algorithm_dir,
        "run_id": run_id,
        "budget": budget,
        "episodes": len(episodes),
        "successes": sum(e["failure"] == 0 for e in episodes),
        "failures": sum(e["failure"] == 1 for e in episodes),
        "trajectory_positions": len(trajectories),
        "errors": 0,
    }

    (destination / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


def main():
    summaries = []
    failures = []

    for model, model_folder in MODELS.items():
        for budget in BUDGETS:
            for algorithm in ALGORITHMS:
                for run_number in range(1, 11):
                    run_name = f"{algorithm}_{budget}scenes/run_{run_number}"

                    try:
                        summary = extract_run(
                            model, model_folder,
                            algorithm, budget, run_number
                        )
                        summaries.append(summary)
                        print(
                            f"[OK] {model}/{run_name}: "
                            f"{summary['episodes']} episodes"
                        )
                    except Exception as exc:
                        failures.append((model, run_name, str(exc)))
                        print(f"[ERROR] {model}/{run_name}: {exc}")

    fields = [
        "model", "algorithm", "run_id", "budget",
        "episodes", "successes", "failures",
        "trajectory_positions", "errors"
    ]

    save_csv(
        REPORT_ROOT / "all_models_summary.csv",
        fields,
        summaries,
    )

    print("\n" + "=" * 55)
    print("ALL-MODEL SBFL INPUT EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"Successful runs: {len(summaries)} / 320")
    print(f"Failed runs: {len(failures)}")
    print(f"Total episodes: {sum(s['episodes'] for s in summaries)}")
    print(
        "Total trajectory positions: "
        f"{sum(s['trajectory_positions'] for s in summaries)}"
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
