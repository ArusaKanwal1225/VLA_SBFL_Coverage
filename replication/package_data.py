#!/usr/bin/env python3
"""Create the video-free Zenodo archive from an unmodified research checkout.

Maintainer tool. --output - writes ZIP64 bytes to stdout (progress goes to stderr).
Files retain their original bytes. No source files are removed by this command.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {
    "rq1": "experiments/RQ1/results/phase_a_primary",
    "rq2": "experiments/RQ2_baselines/results/all_models",
    "rq3": "experiments/RQ2/results/phase_a_granularity",
    "rq4": "experiments/RQ3_v2/results/learning_curves",
}
# Same filenames as the frozen baseline catalogue, plus its outcome evidence.
METRICS = {
    "Token_based_token_prob.json", "Token_based_pcs.json",
    "Token_based_deepgini.json", "Token_based_entropy.json",
    "position_instability.json", "velocity_instability.json",
    "acceleration_instability.json", "Execution_Variability.json",
    "TCP_position_instability.json", "TCP_velocity_instability.json",
    "TCP_acceleration_instability.json", "Trajectory_Instability_Gradients.json",
    "Optimal_Trajectory.json", "log.json",
}
COMPACT = {"episodes.csv", "trajectories.csv", "validation_summary.json",
           "dh_evaluations.csv", "dh_summary.json"}
MODELS = ("EO1", "Groot", "Pi0", "SpatialVLA")

def walk(folder):
    for parent, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in {"__pycache__", ".git"})
        for name in sorted(names):
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError("Unexpected symbolic link: " + str(path))
            yield path

def entries(root):
    for model in MODELS:
        base = root / "input" / model
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in walk(base):
            rel = path.relative_to(root / "input")
            # Compact files at the run root; raw baseline files only under results.
            if ((len(rel.parts) == 4 and path.name in COMPACT)
                    or ("results" in rel.parts and path.name in METRICS
                        and path.parent.name == "0" and path.parent.parent.name == "allMetrics")):
                yield path, "data/pickup/" + rel.as_posix()
    base = root / "experiments/RQ5/data"
    for path in walk(base):
        yield path, "data/rq5/" + path.relative_to(base).as_posix()
    for name, location in REFERENCE.items():
        base = root / location
        if not (base / "run.json").is_file():
            raise FileNotFoundError(base / "run.json")
        for path in walk(base):
            rel = path.relative_to(base)
            if rel.parts[0] in {"aggregate", "audit", "tables", "figures"} or str(rel) == "run.json":
                yield path, "data/reference/" + name + "/" + rel.as_posix()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data", type=Path, help="Repackage an installed/updated analysis dataset")
    parser.add_argument("--source-commit", help="Scientific source commit for the data release")
    parser.add_argument("--output", required=True, help="New ZIP path, or - for stdout")
    args = parser.parse_args()
    root = args.root.resolve()
    target = sys.stdout.buffer if args.output == "-" else open(args.output, "xb")
    started = time.monotonic()
    counts = {}
    total = 0
    seen = set()
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="") as manifest:
            writer = csv.writer(manifest, lineterminator="\n")
            writer.writerow(["path", "bytes", "sha256"])
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                if args.data:
                    installed = args.data.resolve()
                    def installed_entries():
                        for path in walk(installed):
                            rel = path.relative_to(installed)
                            if rel.as_posix() not in {"files_manifest.csv", "DATASET.json", "README.txt"}:
                                yield path, "data/" + rel.as_posix()
                    payload_entries = installed_entries()
                else:
                    payload_entries = entries(root)
                for source, name in payload_entries:
                    if name in seen:
                        raise ValueError("Duplicate archive path: " + name)
                    seen.add(name)
                    info = source.stat()
                    digest = hashlib.sha256()
                    size = 0
                    entry = zipfile.ZipInfo(name, (2026, 11, 16, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry._compresslevel = 6
                    entry.external_attr = 0o100644 << 16
                    with source.open("rb") as stream, archive.open(entry, "w", force_zip64=True) as out:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                            size += len(block)
                            out.write(block)
                    after = source.stat()
                    if size != info.st_size or (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError("Source changed during packaging: " + str(source))
                    writer.writerow([name.removeprefix("data/"), size, digest.hexdigest()])
                    total += size
                    group = name.split("/")[1]
                    counts[group] = counts.get(group, 0) + 1
                    if len(seen) % 20000 == 0:
                        print(f"Packed {len(seen):,} files; {total / 1024**3:.2f} GiB raw; "
                              f"{(time.monotonic()-started)/60:.1f} min", file=sys.stderr, flush=True)
                commit = args.source_commit or subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                metadata = {
                    "schema_version": 2, "source_commit": commit,
                    "paper": "VoxelSBFL: Voxel-Level Spectrum-Based Failure Prediction for Vision-Language-Action Models",
                    "rq_mapping": {"rq1": "RQ1", "rq2": "RQ2_baselines", "rq3": "RQ2", "rq4": "RQ3_v2", "rq5": "RQ5"},
                    "exact_statistical_review": True,
                    "packaging_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "files": len(seen), "bytes_uncompressed": total, "files_by_component": counts,
                    "pickup_expected_episodes": 96000, "pickup_expected_runs": 320,
                    "move_near_episodes_per_model": 504,
                    "rq5_selected_N": {"eo1": 223, "gr00t": 246, "pi0": 298, "spatialvla-4b": 342},
                    "rq5_maps": "frozen_reconstructed_pickup_maps_not_historical_originals",
                    "references_are_historical": True,
                    "excluded": ["videos", "images outside reference figures", "checkpoints",
                                 "duplicate raw TCP/action streams", "timing-only logs",
                                 "exploratory outputs", "optional fitted RQ1-RQ3 models"],
                }
                payload = (json.dumps(metadata, indent=2) + "\n").encode()
                archive.writestr("data/DATASET.json", payload)
                if args.data:
                    (args.data/"DATASET.json").write_bytes(payload)
                writer.writerow(["DATASET.json", len(payload), hashlib.sha256(payload).hexdigest()])
                readme = (
                    "VLA SBFL analysis data\n"
                    "Extract with replication/data.py from the matching code release.\n"
                    "pickup/: compact TCP/outcome CSVs plus original baseline metric JSONs.\n"
                    "rq5/: exact frozen Move Near bundle, reconstructed maps, scenes and references.\n"
                    "reference/: historical aggregates, audits, tables and figures.\n"
                    "files_manifest.csv: SHA-256 and byte size of every other payload file.\n"
                    "No videos, simulator dependencies or model checkpoints are needed.\n"
                ).encode()
                archive.writestr("data/README.txt", readme)
                if args.data:
                    (args.data/"README.txt").write_bytes(readme)
                writer.writerow(["README.txt", len(readme), hashlib.sha256(readme).hexdigest()])
                manifest.seek(0)
                with archive.open("data/files_manifest.csv", "w", force_zip64=True) as out:
                    for text in iter(lambda: manifest.read(1024 * 1024), ""):
                        out.write(text.encode())
            if args.data:
                manifest.seek(0)
                temporary_manifest = args.data/"files_manifest.csv.tmp"
                with temporary_manifest.open("w", encoding="utf-8", newline="") as out:
                    shutil.copyfileobj(manifest, out)
                temporary_manifest.replace(args.data/"files_manifest.csv")
        print(json.dumps({"status": "packed", **metadata,
                          "elapsed_seconds": time.monotonic()-started}), file=sys.stderr, flush=True)
    finally:
        if args.output != "-":
            target.close()

if __name__ == "__main__":
    main()
