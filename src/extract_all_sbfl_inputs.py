
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ALGORITHMS = [
    "DH_400scenes",
    "DH_HybridDT_400scenes",
    "DH_HybridRF_400scenes",
    "DH_WeightedRF_400scenes",
     "DH_200scenes",
    "DH_HybridDT_200scenes",
    "DH_HybridRF_200scenes",
    "DH_WeightedRF_200scenes",
]

SOURCE_ROOT = ROOT / "Data" / "EO1"
OUTPUT_ROOT = ROOT / "input" / "EO1"
REPORT_DIR = ROOT / "output" / "input_validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def process_run(algorithm, run_number):
    run_id = f"run_{run_number}"
    source = SOURCE_ROOT / algorithm / run_id
    destination = OUTPUT_ROOT / algorithm / run_id

    evaluation_file = source / "dh_evaluations.csv"
    results_folder = source / "results"

    if not evaluation_file.is_file():
        raise FileNotFoundError(evaluation_file)

    if not results_folder.is_dir():
        raise FileNotFoundError(results_folder)

    with evaluation_file.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        evaluations = list(csv.DictReader(f))

    scene_lookup = {}

    for folder in results_folder.iterdir():
        if not folder.is_dir():
            continue

        # Extract the numeric Scenes_N suffix.
        suffix = folder.name.rsplit("Scenes_", 1)

        if len(suffix) != 2 or not suffix[1].isdigit():
            continue

        scene_id = suffix[1]
        scene_lookup.setdefault(scene_id, []).append(folder)

    episodes = []
    trajectories = []
    problems = []
    seen_ids = set()

    for evaluation in evaluations:
        eval_id = evaluation["eval_id"].strip()
        failure = evaluation["failure"].strip()

        if eval_id in seen_ids:
            problems.append(f"Duplicate eval_id={eval_id}")
            continue

        seen_ids.add(eval_id)

        if failure not in ("0", "1"):
            problems.append(f"Invalid failure label: {eval_id}")
            continue

        matches = scene_lookup.get(eval_id, [])

        if len(matches) != 1:
            problems.append(
                f"eval_id={eval_id}: {len(matches)} scene matches"
            )
            continue

        pose_file = (
            matches[0]
            / "pi0"
            / "allMetrics"
            / "0"
            / "tcp_poses.json"
        )

        try:
            with pose_file.open("r", encoding="utf-8") as f:
                poses = json.load(f)

            if not isinstance(poses, list) or not poses:
                raise ValueError("Empty trajectory")

            current = []

            for step, pose in enumerate(poses):
                if not isinstance(pose, list) or len(pose) != 7:
                    raise ValueError(f"Invalid pose at step {step}")

                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in pose
                ):
                    raise ValueError(
                        f"Invalid numeric value at step {step}"
                    )

                current.append({
                    "eval_id": eval_id,
                    "step": step,
                    "x": pose[0],
                    "y": pose[1],
                    "z": pose[2],
                })

            episodes.append({
                "model": "Pi0",
                "algorithm": algorithm,
                "run_id": run_id,
                "eval_id": eval_id,
                "episode_id": f"Pi0_{algorithm}_{run_id}_{eval_id}",
                "failure": int(failure),
                "object_name": evaluation.get("object_name", ""),
                "num_steps": len(current),
            })

            trajectories.extend(current)

        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"eval_id={eval_id}: {exc}")

    if problems:
        error_path = REPORT_DIR / f"{algorithm}_{run_id}_errors.txt"
        error_path.write_text(
            "\n".join(problems), encoding="utf-8"
        )
        raise RuntimeError(
            f"{algorithm}/{run_id}: {len(problems)} errors. "
            f"See {error_path}"
        )

    write_csv(
        destination / "episodes.csv",
        [
            "model", "algorithm", "run_id", "eval_id",
            "episode_id", "failure", "object_name", "num_steps"
        ],
        episodes,
    )

    write_csv(
        destination / "trajectories.csv",
        ["eval_id", "step", "x", "y", "z"],
        trajectories,
    )

    summary = {
        "algorithm": algorithm,
        "run_id": run_id,
        "evaluations": len(evaluations),
        "episodes": len(episodes),
        "successes": sum(e["failure"] == 0 for e in episodes),
        "failures": sum(e["failure"] == 1 for e in episodes),
        "trajectory_positions": len(trajectories),
        "errors": len(problems),
    }

    (destination / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def main():
    summaries = []
    failed_runs = []

    for algorithm in ALGORITHMS:
        for run_number in range(1, 11):
            try:
                result = process_run(algorithm, run_number)
                summaries.append(result)

                print(
                    f"[OK] {algorithm}/run_{run_number}: "
                    f"{result['episodes']} episodes, "
                    f"{result['failures']} failures"
                )

            except Exception as exc:
                failed_runs.append(
                    (algorithm, run_number, str(exc))
                )
                print(
                    f"[ERROR] {algorithm}/run_{run_number}: {exc}"
                )

    write_csv(
        REPORT_DIR / "all_runs_summary.csv",
        [
            "algorithm", "run_id", "evaluations", "episodes",
            "successes", "failures", "trajectory_positions", "errors"
        ],
        summaries,
    )

    print("\n" + "=" * 55)
    print("SBFL DATASET EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"Successful runs: {len(summaries)} / 40")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Total episodes: {sum(s['episodes'] for s in summaries)}")
    print(
        "Total trajectory positions: "
        f"{sum(s['trajectory_positions'] for s in summaries)}"
    )

    if failed_runs:
        print("\nRuns needing investigation:")
        for algorithm, run_number, error in failed_runs:
            print(f"  {algorithm}/run_{run_number}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
