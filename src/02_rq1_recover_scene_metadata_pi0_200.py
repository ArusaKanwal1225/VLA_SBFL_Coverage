"""RQ1 Step 02: Recover candidate scene groups for Pi0 200-scene datasets.

Why: The extracted SBFL input contains episode IDs and labels, but it has no
verified scene-configuration identifier. Do not split until the grouping rule
has been reviewed.

Run (PowerShell from the SBFL-only repository root):
  python src/02_rq1_recover_scene_metadata_pi0_200.py --raw-root "C:\\Users\\akanwal\\source\\repos\\VLA_SBFL_1\\Data"

Reads (no changes to originals):
  input/Pi0/<algorithm>/run_N/episodes.csv
  <raw-root>/Pi0/<algorithm>/run_N/dh_evaluations.csv

Writes:
  input/Pi0/scene_metadata_candidates_200.csv
  output/RQ1/pi0_200/step02_scene_metadata/{...csv,...json,figures/*.png,pdf}

Candidate exact scene identity is object_name + orientation_id + x + y.
These 4 fields are visible in the original DeepHyperion evaluation logs, but
are NOT by themselves proof that every simulator scene aspect is identical.
The 4-decimal-position candidate is ONLY a diagnostic for near duplicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
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
REQUIRED_RAW = {"eval_id", "failure", "x", "y", "orientation_id", "object_name"}
REQUIRED_INPUT = {"eval_id", "episode_id", "failure", "algorithm", "run_id", "model"}


def csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV lacks header: " + str(path))
        return set(reader.fieldnames), list(reader)


def normalize_decimal(value):
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid coordinate {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"non-finite coordinate {value!r}")
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def rounded_decimal(value, digits):
    number = Decimal(str(value).strip())
    unit = Decimal("1").scaleb(-digits)
    return format(number.quantize(unit), "f")


def fingerprint(obj, orientation, x, y, rounded_digits=None):
    if rounded_digits is not None:
        x = rounded_decimal(x, rounded_digits)
        y = rounded_decimal(y, rounded_digits)
    else:
        x = normalize_decimal(x)
        y = normalize_decimal(y)
    # Orientation identifier and object name are categorical.
    parts = [obj.strip(), orientation.strip(), x, y]
    canonical = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return "cfg_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_table(df, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Path to original DeepHyperion Data folder (containing Pi0/)")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    raw = args.raw_root.expanduser().resolve()
    out = root / "output" / "RQ1" / "pi0_200" / "step02_scene_metadata"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    inp = root / "input" / "Pi0"
    issues = []
    records = []
    columns = ["algorithm", "run_id", "eval_id", "episode_id", "failure",
               "object_name", "orientation_id", "x", "y",
               "scene_group_exact_candidate", "scene_group_4dp_diagnostic"]

    for algorithm in ALGORITHMS:
        for run_i in range(1, 11):
            run_id = f"run_{run_i}"
            episode_path = inp / algorithm / run_id / "episodes.csv"
            raw_path = raw / "Pi0" / algorithm / run_id / "dh_evaluations.csv"
            try:
                episode_cols, episode_rows = csv_rows(episode_path)
                raw_cols, raw_rows = csv_rows(raw_path)
                if not REQUIRED_INPUT.issubset(episode_cols):
                    raise ValueError(f"missing input fields: {sorted(REQUIRED_INPUT - episode_cols)}")
                if not REQUIRED_RAW.issubset(raw_cols):
                    raise ValueError(f"missing raw fields: {sorted(REQUIRED_RAW - raw_cols)}")
                if len(episode_rows) != 200 or len(raw_rows) != 200:
                    raise ValueError(f"unexpected records: input={len(episode_rows)} raw={len(raw_rows)}")

                eps = {}
                for e in episode_rows:
                    eid = str(e["eval_id"]).strip()
                    if eid in eps:
                        raise ValueError(f"duplicate input eval_id: {eid}")
                    eps[eid] = e
                seen_raw = set()
                current = []
                for r in raw_rows:
                    eid = str(r["eval_id"]).strip()
                    if eid in seen_raw:
                        raise ValueError(f"duplicate raw eval_id: {eid}")
                    seen_raw.add(eid)
                    if eid not in eps:
                        raise ValueError(f"raw eval_id missing from input: {eid}")
                    e = eps[eid]
                    if str(r["failure"]).strip() != str(e["failure"]).strip():
                        raise ValueError(f"failure label mismatch: eval_id={eid}")
                    if e["algorithm"].strip() != algorithm or e["run_id"].strip() != run_id:
                        raise ValueError(f"input algorithm/run mismatch: eval_id={eid}")
                    if e["model"].strip().lower() != "pi0":
                        raise ValueError(f"input model mismatch: eval_id={eid}")
                    obj = r["object_name"].strip()
                    orientation = r["orientation_id"].strip()
                    if not obj or not orientation:
                        raise ValueError(f"blank object/orientation: eval_id={eid}")
                    x = normalize_decimal(r["x"])
                    y = normalize_decimal(r["y"])
                    current.append({
                        "algorithm": algorithm, "run_id": run_id, "eval_id": eid,
                        "episode_id": e["episode_id"].strip(), "failure": int(r["failure"]),
                        "object_name": obj, "orientation_id": orientation,
                        "x": x, "y": y,
                        "scene_group_exact_candidate": fingerprint(obj, orientation, x, y),
                        "scene_group_4dp_diagnostic": fingerprint(obj, orientation, x, y, 4),
                    })
                if set(eps) != seen_raw:
                    raise ValueError(f"input IDs missing from raw: {len(set(eps)-seen_raw)}")
                records.extend(current)
                print(f"[CHECKED] {algorithm}/{run_id}: {len(current)} matched scene records")
            except (OSError, ValueError, KeyError, csv.Error) as exc:
                issues.append({"algorithm": algorithm, "run_id": run_id, "error": str(exc)})
                print(f"[ERROR] {algorithm}/{run_id}: {exc}")

    issues_df = pd.DataFrame(issues, columns=["algorithm", "run_id", "error"])
    write_table(issues_df, out / "validation_issues.csv")
    if issues or len(records) != 8000:
        print(f"\nBLOCKED: matched {len(records)}/8000 episodes, problems={len(issues)}")
        print("No candidate scene manifest was written. Resolve source-data inconsistencies first.")
        return 1

    all_eps = pd.DataFrame.from_records(records, columns=columns)
    if all_eps["episode_id"].duplicated().any():
        print("BLOCKED: duplicate global episode IDs; no scene manifest written")
        return 1
    # Save independently useful, simulator-free grouping metadata in input/.
    write_table(all_eps, inp / "scene_metadata_candidates_200.csv")
    write_table(all_eps, out / "scene_metadata_candidates_200.csv")

    rows = []
    grouping = ["algorithm", "scene_group_exact_candidate"]
    grouped = all_eps.groupby(grouping, sort=False, dropna=False)
    for (algorithm, group_id), group in grouped:
        rows.append({"algorithm": algorithm, "scene_group_exact_candidate": group_id,
                     "episodes_in_group": len(group),
                     "distinct_runs": group["run_id"].nunique(),
                     "failures": int(group["failure"].sum()),
                     "successes": int((group["failure"] == 0).sum()),
                     "object_name": group["object_name"].iloc[0],
                     "orientation_id": group["orientation_id"].iloc[0],
                     "x": group["x"].iloc[0], "y": group["y"].iloc[0]})
    group_table = pd.DataFrame(rows)
    write_table(group_table, out / "exact_candidate_group_details.csv")

    summary = []
    for alg in ALGORITHMS:
        d = all_eps.loc[all_eps["algorithm"] == alg]
        g = group_table.loc[group_table["algorithm"] == alg]
        duplicate_group = g.loc[g["episodes_in_group"] > 1]
        summary.append({
            "algorithm": alg, "episodes": len(d),
            "exact_candidate_unique_groups": d["scene_group_exact_candidate"].nunique(),
            "rounded4_diagnostic_unique_groups": d["scene_group_4dp_diagnostic"].nunique(),
            "exact_candidate_groups_repeated": len(duplicate_group),
            "episodes_in_repeated_exact_groups": int(duplicate_group["episodes_in_group"].sum()),
            "exact_candidate_groups_across_runs": int((g["distinct_runs"] > 1).sum()),
            "max_episodes_in_exact_group": int(g["episodes_in_group"].max()),
        })
    stats = pd.DataFrame(summary)
    write_table(stats, out / "algorithm_scene_group_summary.csv")

    # Figure 1: repeated scene-configuration candidates (source CSV: algorithm_scene_group_summary.csv).
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [s.replace("_200scenes", "") for s in stats["algorithm"]]
    bars = ax.bar(labels, stats["episodes_in_repeated_exact_groups"])
    ax.set_ylabel("Episodes belonging to repeated candidate groups")
    ax.set_title("Pi0, 200 scenes - repeated exact scene configurations")
    ax.tick_params(axis="x", labelrotation=15)
    for b in bars:
        ax.annotate(str(int(b.get_height())), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(10, float(stats["episodes_in_repeated_exact_groups"].max()) * 1.15))
    save_figure(fig, figs, "01_repeated_scene_candidates")

    # Figure 2: exact vs rounded diagnostic identifiers; values from summary CSV.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pos = np.arange(len(stats))
    ax.bar(pos - 0.18, stats["exact_candidate_unique_groups"], width=0.36,
           label="Exact config")
    ax.bar(pos + 0.18, stats["rounded4_diagnostic_unique_groups"], width=0.36,
           label="Rounded x,y (4 decimal, diagnostic)")
    ax.set_xticks(pos, labels, rotation=15)
    ax.set_ylabel("Distinct candidate scene configurations")
    ax.set_title("Pi0, 200 scenes - sensitivity of grouping to coordinate rounding")
    ax.legend()
    save_figure(fig, figs, "02_exact_vs_rounded_candidate_groups")

    # Figure 3: scene configuration coordinates, NOT robot TCP coverage.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, alg in zip(axes.flat, ALGORITHMS):
        d = all_eps[all_eps["algorithm"] == alg]
        ax.scatter(pd.to_numeric(d["x"]), pd.to_numeric(d["y"]), s=7, alpha=0.35)
        ax.set_title(alg.replace("_200scenes", ""))
        ax.set_xlabel("Scene x parameter")
        ax.set_ylabel("Scene y parameter")
    fig.suptitle("Pi0 scene-generation parameters (not TCP trajectory coverage)")
    save_figure(fig, figs, "03_scene_configuration_xy")

    report = {
        "step": "02_scene_metadata_recovery",
        "model": "Pi0", "budget": 200,
        "runs_checked": 40, "episodes_matched": int(len(all_eps)),
        "critical_errors": len(issues),
        "identity_fields": ["object_name", "orientation_id", "x", "y"],
        "candidate_group_definition": "exact string-normalized numeric scene parameters",
        "rounded4_group": "diagnostic only, NOT approved for splitting",
        "grouping_rule_verified_as_full_simulator_scene_identity": False,
        "train_test_split_created": False,
        "status": "CANDIDATES_READY_FOR_REVIEW",
        "raw_data_root": str(raw),
    }
    (out / "scene_metadata_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nRQ1 STEP 02 - SCENE METADATA RECOVERY")
    print(f"Runs matched: 40/40 | Episodes matched: {len(all_eps)}/8000 | Critical errors: {len(issues)}")
    for s in summary:
        print(f"  {s['algorithm']}: exact_groups={s['exact_candidate_unique_groups']}, "
              f"repeated_episodes={s['episodes_in_repeated_exact_groups']}, "
              f"cross_run_groups={s['exact_candidate_groups_across_runs']}")
    print("Status: CANDIDATES_READY_FOR_REVIEW; do not split until grouping rule is checked")
    print("Output:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
