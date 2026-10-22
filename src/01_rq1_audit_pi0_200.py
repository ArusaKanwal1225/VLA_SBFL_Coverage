"""RQ1 / Step 01: Audit Pi0 200-scene SBFL input datasets.

Run from the new VLA_SBFL_Coverage repository:
    python src/01_rq1_audit_pi0_200.py

Requirements: pandas, numpy, matplotlib.

Inputs: input/Pi0/<algorithm>/run_N/episodes.csv and trajectories.csv
Outputs: output/RQ1/pi0_200/step01_data_audit/ (CSVs, figures, JSON)

This script does NOT split the data or calculate suspiciousness. It checks
input integrity and reports whether a verified scene-group identifier exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGORITHMS = [
    "DH_200scenes",
    "DH_HybridDT_200scenes",
    "DH_HybridRF_200scenes",
    "DH_WeightedRF_200scenes",
]
RUNS_PER_ALGORITHM = 10
EPISODES_PER_RUN = 200
REQUIRED_EPISODE_COLS = {"episode_id", "eval_id", "failure", "num_steps"}
REQUIRED_TRAJECTORY_COLS = {"episode_id", "step", "x", "y", "z"}


def save_figure(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root of SBFL-only repository (defaults to parent of src/)",
    )
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    input_root = root / "input" / "Pi0"
    out = root / "output" / "RQ1" / "pi0_200" / "step01_data_audit"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    episode_tables = []
    run_records = []
    issues = []
    all_ids = set()
    known_scene_groups = True

    def report(algorithm, run_id, code, detail):
        issues.append({"algorithm": algorithm, "run_id": run_id,
                       "issue_code": code, "detail": str(detail)})

    for algorithm in ALGORITHMS:
        for run_num in range(1, RUNS_PER_ALGORITHM + 1):
            run_id = f"run_{run_num}"
            run_dir = input_root / algorithm / run_id
            ep_file = run_dir / "episodes.csv"
            tr_file = run_dir / "trajectories.csv"
            if not ep_file.is_file() or not tr_file.is_file():
                report(algorithm, run_id, "MISSING_INPUT_FILE", run_dir)
                continue

            try:
                ep = pd.read_csv(ep_file, dtype={"episode_id": str, "eval_id": str})
                tr = pd.read_csv(tr_file, dtype={"episode_id": str})
            except Exception as exc:
                report(algorithm, run_id, "CSV_READ_ERROR", exc)
                continue

            ep_missing = REQUIRED_EPISODE_COLS - set(ep.columns)
            tr_missing = REQUIRED_TRAJECTORY_COLS - set(tr.columns)
            if ep_missing or tr_missing:
                report(algorithm, run_id, "MISSING_COLUMNS",
                       f"episodes={sorted(ep_missing)}, trajectories={sorted(tr_missing)}")
                continue

            if len(ep) != EPISODES_PER_RUN:
                report(algorithm, run_id, "UNEXPECTED_EPISODE_COUNT",
                       f"actual={len(ep)}, expected={EPISODES_PER_RUN}")
            if ep["episode_id"].isna().any() or ep["episode_id"].duplicated().any():
                report(algorithm, run_id, "INVALID_EPISODE_IDS", "null or duplicate episode_id")

            duplicates_across_runs = set(ep["episode_id"].dropna()) & all_ids
            if duplicates_across_runs:
                report(algorithm, run_id, "DUPLICATE_GLOBAL_EPISODE_ID",
                       f"{len(duplicates_across_runs)} IDs repeated across runs")
            all_ids.update(ep["episode_id"].dropna())

            labels = pd.to_numeric(ep["failure"], errors="coerce")
            steps_recorded = pd.to_numeric(ep["num_steps"], errors="coerce")
            if not labels.isin([0, 1]).all():
                report(algorithm, run_id, "INVALID_FAILURE_LABEL", "failure must be 0 or 1")
            if not (steps_recorded.notna() & (steps_recorded > 0) & (steps_recorded % 1 == 0)).all():
                report(algorithm, run_id, "INVALID_NUM_STEPS", "positive integer expected")
            ep["failure"] = labels
            ep["num_steps"] = steps_recorded
            ep["algorithm"] = algorithm
            ep["run_id"] = run_id
            ep["model"] = "Pi0"
            ep["scene_group_id"] = ""
            if "scene_group_id" in pd.read_csv(ep_file, nrows=0).columns:
                raw_scene = pd.read_csv(ep_file, usecols=["scene_group_id"], dtype=str)
                if raw_scene["scene_group_id"].notna().all() and raw_scene["scene_group_id"].str.strip().ne("").all():
                    ep["scene_group_id"] = raw_scene["scene_group_id"].values
                else:
                    known_scene_groups = False
            else:
                known_scene_groups = False

            # Coordinate and step integrity checks
            for col in ["step", "x", "y", "z"]:
                tr[col] = pd.to_numeric(tr[col], errors="coerce")
            if not np.isfinite(tr[["step", "x", "y", "z"]].to_numpy(dtype=float)).all():
                report(algorithm, run_id, "INVALID_TRAJECTORY_NUMBERS", "NaN or infinite position/step")
            if not (tr["step"].dropna() % 1 == 0).all():
                report(algorithm, run_id, "NONINTEGER_STEP", "step indexes must be integers")

            ep_ids = set(ep["episode_id"].dropna())
            tr_ids = set(tr["episode_id"].dropna())
            if ep_ids != tr_ids:
                report(algorithm, run_id, "EPISODE_TRAJECTORY_ID_MISMATCH",
                       f"missing trajectories={len(ep_ids - tr_ids)}; orphan trajectories={len(tr_ids - ep_ids)}")

            stats = tr.groupby("episode_id", dropna=False)["step"].agg(
                observed_steps="size", min_step="min", max_step="max", unique_steps="nunique"
            )
            joined = ep.set_index("episode_id")[["num_steps"]].join(stats, how="left")
            if (joined["observed_steps"].ne(joined["num_steps"]) |
                joined["min_step"].ne(0) |
                joined["max_step"].ne(joined["num_steps"] - 1) |
                joined["unique_steps"].ne(joined["num_steps"])).any():
                report(algorithm, run_id, "STEP_SEQUENCE_MISMATCH",
                       "trajectory rows must exactly cover 0..num_steps-1")

            positives = int((labels == 1).sum())
            negatives = int((labels == 0).sum())
            run_records.append({
                "algorithm": algorithm, "run_id": run_id, "episodes": len(ep),
                "successes": negatives, "failures": positives,
                "failure_rate": positives / len(ep) if len(ep) else float("nan"),
                "trajectory_positions": len(tr),
                "mean_steps": float(steps_recorded.mean()),
                "min_steps": float(steps_recorded.min()),
                "max_steps": float(steps_recorded.max()),
                "verified_scene_groups": bool(ep["scene_group_id"].astype(str).str.strip().ne("").all()),
            })
            episode_tables.append(ep[["model", "algorithm", "run_id", "eval_id", "episode_id",
                                      "scene_group_id", "object_name", "failure", "num_steps"]].copy())
            print(f"[CHECKED] {algorithm}/{run_id}: {len(ep)} episodes, {positives} failures")

    runs = pd.DataFrame(run_records)
    episodes = pd.concat(episode_tables, ignore_index=True) if episode_tables else pd.DataFrame()
    issues_df = pd.DataFrame(issues, columns=["algorithm", "run_id", "issue_code", "detail"])
    runs.to_csv(out / "run_summary.csv", index=False)
    episodes.to_csv(out / "episode_manifest.csv", index=False)
    issues_df.to_csv(out / "validation_issues.csv", index=False)

    if not runs.empty:
        alg = runs.groupby("algorithm", as_index=False).agg(
            runs=("run_id", "size"), episodes=("episodes", "sum"),
            successes=("successes", "sum"), failures=("failures", "sum"),
            trajectory_positions=("trajectory_positions", "sum"),
            mean_steps=("mean_steps", "mean"),
        )
        alg["failure_rate"] = alg["failures"] / alg["episodes"]
        alg.to_csv(out / "algorithm_summary.csv", index=False)
        # Figure 1 and its source data: per-algorithm failure rates.
        fig, ax = plt.subplots(figsize=(10, 5))
        short_names = [x.replace("_200scenes", "") for x in alg["algorithm"]]
        bars = ax.bar(short_names, alg["failure_rate"] * 100)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(100, (alg["failure_rate"].max() * 100) + 10))
        ax.set_ylabel("Failed episodes (%)")
        ax.set_title("Pi0 · 200 scenes · failure prevalence by algorithm")
        ax.tick_params(axis="x", labelrotation=15)
        save_figure(fig, figs, "01_failure_rates_by_algorithm")

        # Figure 2 and its source data: rate by run × algorithm.
        pivot = runs.pivot(index="algorithm", columns="run_id", values="failure_rate")
        pivot = pivot.reindex(index=ALGORITHMS,
                              columns=[f"run_{i}" for i in range(1, RUNS_PER_ALGORITHM+1)])
        pivot.to_csv(out / "failure_rate_matrix.csv")
        fig, ax = plt.subplots(figsize=(11, 4))
        heat = ax.imshow(pivot.to_numpy(dtype=float)*100, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(RUNS_PER_ALGORITHM), [f"{i}" for i in range(1, RUNS_PER_ALGORITHM+1)])
        ax.set_yticks(range(len(ALGORITHMS)), [a.replace("_200scenes", "") for a in ALGORITHMS])
        ax.set_xlabel("Run number")
        ax.set_title("Pi0 · 200 scenes · failure rate (%) per run")
        fig.colorbar(heat, ax=ax, label="Failed episodes (%)")
        save_figure(fig, figs, "02_failure_rates_by_run")

        # Figure 3: episode lengths by algorithm. Data in episode_manifest.csv.
        fig, ax = plt.subplots(figsize=(10, 4.5))
        data = [episodes.loc[episodes["algorithm"].eq(a), "num_steps"].dropna().to_numpy()
                for a in ALGORITHMS]
        ax.boxplot(data, labels=[x.replace("_200scenes", "") for x in ALGORITHMS])
        ax.set_ylabel("TCP positions per episode")
        ax.set_title("Pi0 · 200 scenes · trajectory length distribution")
        ax.tick_params(axis="x", labelrotation=15)
        save_figure(fig, figs, "03_trajectory_lengths_by_algorithm")
    else:
        pd.DataFrame(columns=["algorithm", "runs", "episodes", "successes", "failures",
                              "trajectory_positions", "mean_steps", "failure_rate"]).to_csv(
                                  out / "algorithm_summary.csv", index=False)

    critical = len(issues_df) > 0 or len(runs) != 40 or len(episodes) != 8000
    config = {
        "research_question": "RQ1", "step": "01_data_audit",
        "model": "Pi0", "budget": 200, "algorithms": ALGORITHMS,
        "runs_per_algorithm": RUNS_PER_ALGORITHM,
        "expected_total_episodes": 8000,
        "observed_runs": len(runs), "observed_episodes": len(episodes),
        "critical_issues": len(issues_df),
        "scene_group_identifier_available": bool(known_scene_groups and not episodes.empty),
        "train_test_split_created": False,
        "status": "FAIL" if critical else "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "audit_report.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nRQ1 STEP 01 — DATA AUDIT")
    print(f"Runs: {len(runs)}/40 | Episodes: {len(episodes)}/8000 | Critical issues: {len(issues_df)}")
    print(f"Scene group ID available: {config['scene_group_identifier_available']}")
    if not config["scene_group_identifier_available"]:
        print("NOTE: Current input lacks verified scene-group IDs. Do NOT split episodes yet.")
        print("Next: recover scene metadata / define grouping key before train/test splitting.")
    print(f"Status: {config['status']} | Outputs: {out}")
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
