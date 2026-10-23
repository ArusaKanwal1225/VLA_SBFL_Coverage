#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""RQ1 Pi0/400 Step 03: review candidate scene groups and build grouped 70/30 splits.

Run from VLA_SBFL_Coverage root (PowerShell):
    python .\src\RQ1_Pi0_400\03_rq1_review_and_split_pi0_400.py --repo-root .

After verifying the scene key and reviewing duplicate/near-duplicate reports:
    python .\src\RQ1_Pi0_400\03_rq1_review_and_split_pi0_400.py --repo-root . --confirm-scene-key

If near-duplicate candidates are reported, review the detailed CSV first.
If you agree to CONSERVATIVELY KEEP the near-duplicate pairs in the SAME
train/test partition, explicitly add:
    --accept-reviewed-near-duplicates
This flag MERGES each flagged near-duplicate group for splitting (it never ignores it).

Input:
    input/Pi0/scene_metadata_candidates_400.csv
    input/Pi0/<algorithm>_400scenes/run_1..10/episodes.csv
Output:
    output/RQ1/pi0_400/step03_grouped_splits/
        scene_key_review_summary.csv, repeated_scene_episodes.csv, etc.
        split_manifest.csv, split_summary.csv, per_run_split_summary.csv,
        group_leakage_checks.csv, step03_report.json, figures/*.png, *.pdf

Candidate key = object_name + orientation_id + x + y; this is NOT proof of
physical scene identity. No train/test split is made without explicit approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL = "Pi0"
BUDGET = 400
ALGORITHMS = (
    "DH_400scenes", "DH_HybridDT_400scenes",
    "DH_HybridRF_400scenes", "DH_WeightedRF_400scenes",
)
SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
KEY = "scene_group_exact_candidate"
NEAR_KEY = "scene_group_4dp_diagnostic"
SPLIT_KEY = "scene_group_split"
NECESSARY_COLUMNS = (
    "algorithm", "run_id", "eval_id", "episode_id", "failure",
    "object_name", "orientation_id", "x", "y", KEY,
)


def csv_out(data, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(target, index=False, encoding="utf-8")


def figure_out(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rounded4(value):
    try:
        val = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric scene coordinate: {value!r}") from exc
    if not val.is_finite():
        raise ValueError(f"Non-finite scene coordinate: {value!r}")
    return format(val.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN), ".4f")


def make_near_key(row):
    parts = [
        str(row["object_name"]).strip(), str(row["orientation_id"]).strip(),
        rounded4(row["x"]), rounded4(row["y"]),
    ]
    raw = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return "near4_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_and_validate(root):
    metadata = root / "input" / MODEL / "scene_metadata_candidates_400.csv"
    if not metadata.is_file():
        raise ValueError(f"Missing metadata file: {metadata}")
    df = pd.read_csv(metadata, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = set(NECESSARY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError("Missing metadata columns: " + ", ".join(sorted(missing)))
    if len(df) != 16000 or df.episode_id.duplicated().any():
        raise ValueError("Expected exactly 16,000 unique episode IDs")
    if df[list(NECESSARY_COLUMNS)].eq("").any().any():
        raise ValueError("One or more required metadata values are blank")
    if not df.failure.isin(("0", "1")).all():
        raise ValueError("Failure labels must be 0 or 1")
    if set(df.algorithm) != set(ALGORITHMS):
        raise ValueError(f"Unexpected algorithms: {sorted(set(df.algorithm))}")

    # This field was not in the user's 400-scene metadata screenshot.
    # Reconstruct it from known scene variables for diagnostics only.
    df[NEAR_KEY] = df.apply(make_near_key, axis=1)

    for algorithm in ALGORITHMS:
        section = df[df.algorithm == algorithm]
        if len(section) != 4000:
            raise ValueError(f"{algorithm}: expected 4,000 metadata episodes, got {len(section)}")
        for n in range(1, 11):
            run_id = f"run_{n}"
            sub = section[section.run_id == run_id]
            if len(sub) != 400 or sub.eval_id.nunique() != 400:
                raise ValueError(f"{algorithm}/{run_id}: expected 400 unique evaluations")
            ep_path = root / "input" / MODEL / algorithm / run_id / "episodes.csv"
            if not ep_path.is_file():
                raise ValueError("Missing input episodes file: " + str(ep_path))
            ep = pd.read_csv(ep_path, dtype=str, keep_default_na=False,
                             usecols=["episode_id", "failure"])
            joined = sub[["episode_id", "failure"]].merge(
                ep, on="episode_id", how="outer", suffixes=("_metadata", "_input"),
                indicator=True, validate="one_to_one",
            )
            if (len(joined) != 400 or (joined["_merge"] != "both").any()
                    or (joined.failure_metadata != joined.failure_input).any()):
                raise ValueError(f"Episode IDs / labels mismatch: {algorithm}/{run_id}")

    # A candidate hash must consistently refer to its input fields.
    counts = (df.groupby(["algorithm", KEY], sort=False)
              [["object_name", "orientation_id", "x", "y"]].nunique(dropna=False))
    if (counts > 1).any().any():
        raise ValueError("Exact candidate key maps to inconsistent scene fields")
    return df


def review(df, output, figs):
    sizes = (df.groupby(["algorithm", KEY], sort=False)
             .agg(group_size=("episode_id", "size"),
                  distinct_runs=("run_id", "nunique"),
                  failures=("failure", lambda s: int((s == "1").sum())),
                  successes=("failure", lambda s: int((s == "0").sum())))
             .reset_index())
    repeated = sizes[sizes.group_size > 1].copy()
    repeated_ep = df.merge(repeated[["algorithm", KEY]], on=["algorithm", KEY], how="inner")
    near = (df.groupby(["algorithm", NEAR_KEY])
            .agg(exact_groups=(KEY, "nunique"), episodes=("episode_id", "size"))
            .reset_index())
    near = near[near.exact_groups > 1].copy()
    near_ep = df.merge(near[["algorithm", NEAR_KEY]], on=["algorithm", NEAR_KEY], how="inner")

    rows = []
    for alg in ALGORITHMS:
        a = sizes[sizes.algorithm == alg]
        rows.append({
            "algorithm": alg, "episodes": 4000,
            "exact_groups": len(a),
            "repeated_exact_groups": int((a.group_size > 1).sum()),
            "episodes_in_repeated_groups": int(a.loc[a.group_size > 1, "group_size"].sum()),
            "cross_run_exact_groups": int((a.distinct_runs > 1).sum()),
            "rounded4_groups_containing_multiple_exact_groups": int((near.algorithm == alg).sum()),
        })
    summary = pd.DataFrame(rows)
    csv_out(sizes, output / "all_exact_group_sizes.csv")
    csv_out(repeated, output / "repeated_exact_groups.csv")
    csv_out(repeated_ep.sort_values(["algorithm", KEY, "run_id"]),
            output / "repeated_scene_episodes.csv")
    csv_out(near, output / "near_duplicate_group_summary.csv")
    csv_out(near_ep.sort_values(["algorithm", NEAR_KEY, "run_id"]),
            output / "near_duplicate_episodes.csv")
    csv_out(summary, output / "scene_key_review_summary.csv")

    labels = [a.replace("_400scenes", "") for a in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, summary.episodes_in_repeated_groups)
    for bar in bars:
        ax.annotate(str(int(bar.get_height())),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    ax.set_ylabel("Episodes belonging to repeated exact scene groups")
    ax.set_ylim(0, max(3, float(summary.episodes_in_repeated_groups.max()) * 1.25))
    ax.tick_params(axis="x", labelrotation=15)
    ax.set_title("Pi0 400: repeated candidate scene configurations")
    figure_out(fig, figs, "01_repeated_exact_groups")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xx = np.arange(4)
    ax.bar(xx - 0.18, summary.repeated_exact_groups, width=.36, label="Repeated exact groups")
    ax.bar(xx + 0.18, summary.rounded4_groups_containing_multiple_exact_groups,
           width=.36, label="Near-duplicate groups (diagnostic)")
    ax.set_xticks(xx, labels, rotation=15)
    ax.set_ylabel("Number of groups")
    ax.set_title("Pi0 400: exact and near-duplicate scene candidates")
    ax.legend()
    figure_out(fig, figs, "02_exact_and_near_duplicate_groups")

    print("\nRQ1 Pi0 400 - SCENE GROUP REVIEW")
    print(summary.to_string(index=False))
    if not repeated_ep.empty:
        print("\nREPEATED EXACT EPISODES (sample, full CSV saved):")
        cols = ["algorithm", "run_id", "eval_id", "failure", "object_name", "orientation_id", "x", "y"]
        print(repeated_ep[cols].head(20).to_string(index=False))
    if not near.empty:
        print(f"\nREVIEW NEEDED: {len(near)} near-duplicate candidate groups. ")
        print("Inspect near_duplicate_episodes.csv before approving the grouping rule.")
    return summary, near


def assign_groups(subset, seed):
    """Approximately stratified by run and outcome; identical keys remain intact."""
    rng = np.random.default_rng(seed)
    groups = []
    for key, group in subset.groupby(SPLIT_KEY, sort=True):
        failed = int((group.failure == "1").sum())
        passed = int((group.failure == "0").sum())
        category = "mixed" if failed and passed else ("fail" if failed else "pass")
        primary_run = min(group.run_id, key=lambda x: int(x.rsplit("_", 1)[1]))
        groups.append((key, primary_run, category))
    strata = {}
    for key, run, category in groups:
        strata.setdefault((run, category), []).append(key)
    test_keys = set()
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        choose = int(round(0.30 * len(keys)))
        for j in rng.choice(len(keys), size=choose, replace=False):
            test_keys.add(keys[int(j)])
    return subset[SPLIT_KEY].map(lambda value: "test" if value in test_keys else "train")


def build_splits(df, output, figs, accept_near):
    # Conservative grouping: if multiple exact scene keys share the same rounded
    # XY+object+orientation key, keep every episode in that near-duplicate cluster
    # in one train/test partition. Do not merely acknowledge the cluster.
    df = df.copy()
    df[SPLIT_KEY] = df[KEY]
    near_exact_counts = df.groupby(["algorithm", NEAR_KEY])[KEY].transform("nunique")
    near_mask = near_exact_counts > 1
    if near_mask.any() and not accept_near:
        raise ValueError("Near-duplicate scenes must be reviewed and explicitly merged")
    df.loc[near_mask, SPLIT_KEY] = "merged_" + df.loc[near_mask, NEAR_KEY]
    # Check grouping integrity before generating any partitions.
    if (df.groupby(["algorithm", KEY])[SPLIT_KEY].nunique() > 1).any():
        raise ValueError("One exact scene candidate maps to multiple split groups")
    if (df.loc[near_mask].groupby(["algorithm", NEAR_KEY])[SPLIT_KEY].nunique() > 1).any():
        raise ValueError("A flagged near-duplicate group was not fully merged")
    csv_out(df.loc[near_mask, ["algorithm", "run_id", "eval_id", "episode_id",
                                "object_name", "orientation_id", "x", "y",
                                KEY, NEAR_KEY, SPLIT_KEY]],
            output / "reviewed_near_duplicate_merge.csv")
    csv_out(df[["algorithm", "episode_id", KEY, NEAR_KEY, SPLIT_KEY]],
            output / "split_group_mapping.csv")
    print(f"[GROUPING] Merged near-duplicate episodes: {int(near_mask.sum())} "
          f"across {df.loc[near_mask, ['algorithm', NEAR_KEY]].drop_duplicates().shape[0]} clusters")
    manifests, totals, per_run, leakage = [], [], [], []
    for alg in ALGORITHMS:
        section = df[df.algorithm == alg].copy()
        for seed in SEEDS:
            assignments = assign_groups(section, seed)
            result = section[["algorithm", "run_id", "eval_id", "episode_id",
                              "failure", "object_name", KEY, NEAR_KEY, SPLIT_KEY]].copy()
            result.insert(0, "seed", seed)
            result["split"] = list(assignments)
            train = result[result.split == "train"]
            test = result[result.split == "test"]
            both = set(train[SPLIT_KEY]).intersection(test[SPLIT_KEY])
            exact_both = set(train[KEY]).intersection(test[KEY])
            near_both = set(train[NEAR_KEY]).intersection(test[NEAR_KEY])
            shared_eps = set(train.episode_id).intersection(test.episode_id)
            if both or exact_both or near_both or shared_eps or not (1150 <= len(test) <= 1250):
                raise ValueError(f"{alg}, seed={seed}: train/test leakage or size imbalance "
                                 f"(test={len(test)}, shared_groups={len(both)})")
            if train.failure.nunique() != 2 or test.failure.nunique() != 2:
                raise ValueError(f"{alg}, seed={seed}: a split contains only one outcome class")
            totals.append({
                "algorithm": alg, "seed": seed,
                "train_episodes": len(train), "test_episodes": len(test),
                "train_failures": int((train.failure == "1").sum()),
                "test_failures": int((test.failure == "1").sum()),
                "train_failure_rate": float((train.failure == "1").mean()),
                "test_failure_rate": float((test.failure == "1").mean()),
                "train_scene_groups": int(train[SPLIT_KEY].nunique()),
                "test_scene_groups": int(test[SPLIT_KEY].nunique()),
                "shared_scene_groups": len(both),
                "shared_exact_candidate_groups": len(exact_both),
                "shared_rounded4_candidate_groups": len(near_both),
            })
            leakage.append({
                "algorithm": alg, "seed": seed,
                "shared_scene_groups": len(both),
                "shared_exact_candidate_groups": len(exact_both),
                "shared_rounded4_candidate_groups": len(near_both),
                "shared_episode_ids": len(shared_eps),
            })
            for rid, rsub in result.groupby("run_id"):
                test_rows = rsub[rsub.split == "test"]
                per_run.append({
                    "algorithm": alg, "seed": seed, "run_id": rid,
                    "train_episodes": int((rsub.split == "train").sum()),
                    "test_episodes": len(test_rows),
                    "test_failure_rate": (float((test_rows.failure == "1").mean())
                                          if len(test_rows) else float("nan")),
                })
            manifests.append(result)
            print(f"[SPLIT] {alg} seed={seed}: train={len(train)} "
                  f"test={len(test)} shared_scene_groups={len(both)}")

    joined = pd.concat(manifests, ignore_index=True)
    counts = pd.DataFrame(totals)
    run_counts = pd.DataFrame(per_run)
    leak = pd.DataFrame(leakage)
    if len(joined) != 160000 or len(counts) != 40:
        raise ValueError("Split manifest must contain 160,000 rows across 40 splits")
    csv_out(joined, output / "split_manifest.csv")
    csv_out(counts, output / "split_summary.csv")
    csv_out(run_counts, output / "per_run_split_summary.csv")
    csv_out(leak, output / "group_leakage_checks.csv")

    labels = [a.replace("_400scenes", "") for a in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    a = counts.groupby("algorithm")[["train_episodes", "test_episodes"]].mean().reindex(ALGORITHMS)
    xx = np.arange(4)
    ax.bar(xx - 0.2, a.train_episodes, width=0.4, label="Training")
    ax.bar(xx + 0.2, a.test_episodes, width=0.4, label="Testing")
    ax.set_xticks(xx, labels, rotation=15)
    ax.set_ylabel("Mean number of episodes across seeds")
    ax.set_title("Pi0 400: grouped 70/30 split sizes")
    ax.legend()
    figure_out(fig, figs, "03_train_test_sizes")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for alg in ALGORITHMS:
        sub = counts[counts.algorithm == alg]
        ax.plot(sub.seed, sub.test_failure_rate, marker="o", label=alg.replace("_400scenes", ""))
    ax.set_xlabel("Split seed")
    ax.set_ylabel("Held-out failure prevalence")
    ax.set_ylim(0, 1)
    ax.set_title("Pi0 400: test failure prevalence by split")
    ax.legend()
    figure_out(fig, figs, "04_test_failure_fraction_by_seed")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = (run_counts.groupby(["algorithm", "run_id"]).test_episodes.mean()
            .unstack("run_id").reindex(index=ALGORITHMS,
                                       columns=[f"run_{i}" for i in range(1, 11)]))
    im = ax.imshow(vals.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(4), labels)
    ax.set_xticks(range(10), [f"run_{i}" for i in range(1, 11)], rotation=45)
    ax.set_title("Pi0 400: average held-out episodes per run")
    fig.colorbar(im, ax=ax, label="Test episodes")
    figure_out(fig, figs, "05_test_episodes_by_run")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="SBFL project root with input/, src/, output/")
    parser.add_argument("--confirm-scene-key", action="store_true",
                        help="Confirm after reviewing candidate scene fields and duplicates")
    parser.add_argument("--accept-reviewed-near-duplicates", action="store_true",
                        help="After reviewing near_duplicate_episodes.csv, merge near-identical scenes into one split group")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = root / "output" / "RQ1" / "pi0_400" / "step03_grouped_splits"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = {
        "step": "03_scene_review_grouped_split", "model": MODEL, "budget": BUDGET,
        "scene_group_fields": ["object_name", "orientation_id", "x", "y"],
        "identity_caveat": "Candidate configuration key, not independently proven simulator identity",
        "near_duplicate_handling": "Accepted near-duplicate candidates are merged into the same train/test grouping key",
        "seeds": list(SEEDS), "confirmed": bool(args.confirm_scene_key),
        "status": "NOT_STARTED",
    }
    try:
        df = load_and_validate(root)
        summary, near = review(df, output, figures)
        report.update({"episodes_reviewed": len(df),
                       "near_duplicate_groups": len(near),
                       "summary_by_algorithm": summary.to_dict(orient="records")})
        if not args.confirm_scene_key:
            report["status"] = "PENDING_SCENE_KEY_REVIEW"
            print("\nREVIEW COMPLETE: no train/test splits created.")
            print("Review repeated_scene_episodes.csv and near_duplicate_episodes.csv.")
            print("Then run with --confirm-scene-key once scene fields are approved.")
        elif len(near) and not args.accept_reviewed_near_duplicates:
            report["status"] = "BLOCKED_NEAR_DUPLICATES"
            print("\nBLOCKED: near-duplicate candidates need manual review before splitting.")
            print("After review, add --accept-reviewed-near-duplicates to KEEP the near-duplicate cluster TOGETHER.")
            rc = 2
        else:
            counts = build_splits(df, output, figures,
                                  accept_near=args.accept_reviewed_near_duplicates)
            report["status"] = "PASS_GROUPED_SPLITS_CREATED"
            report["split_combinations"] = len(counts)
            print(f"\nSUCCESS: {len(counts)}/40 grouped train/test splits; all overlap checks passed.")
            rc = 0
        if not args.confirm_scene_key:
            rc = 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["status"] = "FAILED"
        report["error"] = str(exc)
        print("BLOCKED:", exc, file=sys.stderr)
        rc = 1
    (output / "step03_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Outputs:", output)
    return rc


if __name__ == "__main__":
    sys.exit(main())
