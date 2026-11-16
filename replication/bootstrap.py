#!/usr/bin/env python3
"""Recompute bootstrap sensitivity on output copies, preserving frozen references."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src_new")]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--output",type=Path,default=ROOT/"reproduced/bootstrap")
    args=p.parse_args()
    data=args.data.expanduser().resolve();output=args.output.expanduser().resolve()
    if output==data or output.is_relative_to(data):raise ValueError("Output must be outside immutable data")
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    for rq in ["rq1","rq2","rq3"]:
        shutil.copytree(data/"reference"/rq/"aggregate",output/"data/reference"/rq/"aggregate")
    from experiments import bootstrap_review
    bootstrap_review.ROOT=output
    checks={}
    for name in ["RQ1","RQ2","RQ3"]:
        checks[name]=bootstrap_review.audit(name)
        if checks[name]["maximum_reproduction_error"]>1e-12:
            raise ValueError("Historical interval reproduction differs: "+name)
    (output/"verification.json").write_text(json.dumps({"status":"matched","checks":checks,
        "frozen_inputs_modified":False},indent=2)+"\n")
    print(output/"verification.json")
if __name__=="__main__":main()
