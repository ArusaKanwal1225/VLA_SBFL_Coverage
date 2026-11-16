#!/usr/bin/env python3
"""Download, safely unpack, or verify the Zenodo analysis data (standard library)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "replication/archive.json"

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def safe_relative(text):
    path = PurePosixPath(text)
    if (not text or "\\" in text or "\0" in text or path.is_absolute()
            or any(x in {"", ".", ".."} or ":" in x for x in text.split("/"))):
        raise ValueError("Unsafe archive/manifest path: " + repr(text))
    return path

def records(stream):
    result = {}
    for row in csv.DictReader(stream):
        relative = safe_relative(row["path"]).as_posix()
        if relative in result or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Duplicate path or invalid digest in manifest: " + relative)
        size = int(row["bytes"])
        if size < 0:
            raise ValueError("Negative file size")
        result[relative] = (size, row["sha256"])
    if not result:
        raise ValueError("Empty dataset manifest")
    return result

def verify(data, prefixes=()):
    data = Path(data).resolve()
    with (data / "files_manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = records(stream)
    selected = {name: pair for name, pair in manifest.items()
                if not prefixes or any(name.startswith(prefix) for prefix in prefixes)}
    if not selected:
        raise ValueError("No matching verification entries")
    for i, (name, (size, digest)) in enumerate(selected.items(), 1):
        path = data / name
        if path.is_symlink() or not path.resolve().is_relative_to(data):
            raise ValueError("Symlink in dataset: " + name)
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            raise ValueError("Data integrity mismatch: " + name)
        if i % 20000 == 0:
            print(f"Verified {i:,}/{len(selected):,} files", flush=True)
    if not prefixes:
        actual = set()
        for parent, dirs, files in os.walk(data, followlinks=False):
            for d in dirs:
                if (Path(parent)/d).is_symlink():
                    raise ValueError("Unexpected directory symlink")
            actual.update((Path(parent)/f).relative_to(data).as_posix()
                          for f in files if Path(parent)/f != data/"files_manifest.csv")
        if actual != manifest.keys():
            raise ValueError("Missing or unlisted dataset files: " + str(sorted(actual ^ manifest.keys())[:5]))
    report = {"status": "verified", "files": len(selected),
              "bytes": sum(v[0] for v in selected.values()), "prefixes": list(prefixes)}
    print(json.dumps(report, indent=2), flush=True)
    return report

def unpack(archive, destination, expected):
    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    final = destination / "data"
    if not expected or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("A trusted SHA-256 is required (--sha256 or replication/archive.json)")
    if sha(archive) != expected.lower():
        raise ValueError("Archive SHA-256 mismatch; no files extracted")
    if final.is_symlink() or (final.exists() and (not final.is_dir() or any(final.iterdir()))):
        raise FileExistsError("Choose a destination without an existing data/ dataset: " + str(final))
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = set()
        files = []
        for info in z.infolist():
            p = safe_relative(info.filename)
            if p.parts[0] != "data" or len(p.parts) < 2 or info.filename in names:
                raise ValueError("Invalid or duplicate ZIP member: " + info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unexpected directory/symlink ZIP member")
            names.add(info.filename)
            files.append(info)
        manifest_bytes = z.read("data/files_manifest.csv")
        manifest = records(io.StringIO(manifest_bytes.decode("utf-8")))
        if names != {"data/" + name for name in manifest} | {"data/files_manifest.csv"}:
            raise ValueError("ZIP payload and manifest differ")
        required = sum(info.file_size for info in files)
        if shutil.disk_usage(destination).free < required + 100*1024**2:
            raise OSError(f"Insufficient disk space; need at least {required/1024**3:.2f} GiB plus margin")
        with tempfile.TemporaryDirectory(prefix=".vla-extract-", dir=destination) as temporary:
            staging = Path(temporary) / "data"
            staging.mkdir()
            for i, info in enumerate(files, 1):
                rel = PurePosixPath(info.filename).relative_to("data")
                path = staging.joinpath(*rel.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with z.open(info) as source, path.open("xb") as target:
                    for block in iter(lambda: source.read(1024*1024), b""):
                        target.write(block)
                        size += len(block)
                        digest.update(block)
                if str(rel) != "files_manifest.csv" and (size, digest.hexdigest()) != manifest[str(rel)]:
                    raise ValueError("Payload SHA-256 mismatch: " + str(rel))
                if i % 20000 == 0:
                    print(f"Extracted and verified {i:,}/{len(files):,} files", flush=True)
            if final.exists():
                final.rmdir()  # Only the previously checked empty directory.
            staging.replace(final)
    print(f"Extracted and verified {len(manifest):,} payload files in {final}")
    return final

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--data", type=Path, default=ROOT/"data")
    p = sub.add_parser("unpack")
    p.add_argument("archive", type=Path)
    p.add_argument("--destination", type=Path, default=ROOT)
    p.add_argument("--sha256")
    p = sub.add_parser("download")
    p.add_argument("--url", help="Zenodo HTTPS download URL; otherwise release metadata")
    p.add_argument("--output", type=Path)
    p.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.data)
        return
    release = json.loads(RELEASE.read_text()) if RELEASE.is_file() else {}
    expected = args.sha256 or release.get("sha256")
    if args.command == "unpack":
        unpack(args.archive, args.destination, expected)
        return
    url = args.url or release.get("url")
    if not url or urllib.parse.urlparse(url).scheme != "https" or not expected:
        parser.error("Set the Zenodo HTTPS URL and SHA-256; the upload has not been published yet")
    target = (args.output or ROOT/release.get("filename", "vla-sbfl-data.zip")).resolve()
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_name(target.name+".partial")
    with urllib.request.urlopen(url, timeout=60) as source, partial.open("xb") as out:
        shutil.copyfileobj(source, out, length=1024*1024)
    if sha(partial) != expected.lower():
        raise ValueError("Download SHA-256 mismatch; retained .partial for inspection")
    partial.replace(target)
    print(target)

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        print("ERROR: "+str(error), file=sys.stderr)
        sys.exit(1)
