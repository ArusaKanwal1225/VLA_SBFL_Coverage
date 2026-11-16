"""Recover per-fold voxel counts from released trajectories without refitting SBFL."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src_new")]
from experiments.RQ1.run_experiment import discover_configurations
from vla_sbfl.dataset import load_trajectory_dataset
from vla_sbfl.voxelizer import VoxelGrid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    output=a.output or a.data/"reference/rq3/training_voxels.csv"
    rows=[]
    configurations=discover_configurations(a.data/"pickup")
    for size in [.01,.02,.05]:
        for i,c in enumerate(configurations,1):
            ds=load_trajectory_dataset([c.directory],grid=VoxelGrid(size,(-1.,-1.,0.),"robot_base"),require_labels=True)
            runs={}
            for run,group in ds.episodes.groupby("run_id",sort=True):
                covered=set()
                for eid in group.episode_id.astype(str):covered.update(ds.visits[eid])
                runs[str(run)]=covered
            support=Counter(voxel for voxels in runs.values() for voxel in voxels)
            for run,covered in runs.items():
                count=len(support)-sum(support[v]==1 for v in covered)
                rows.append({"voxel_size":size,"configuration_id":c.configuration_id,"test_run":run,
                             "vla":c.vla,"training_voxels":count})
            print(f"voxel-size {size}: {i}/{len(configurations)} configurations",flush=True)
    frame=pd.DataFrame(rows).sort_values(["voxel_size","configuration_id","test_run"])
    output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False)
    expected=pd.read_csv(a.data/"reference/rq3/tables/rq2_diagnostics_by_vla.csv")
    actual=frame.groupby(["vla","voxel_size"]).training_voxels.agg(["mean","median","min","max"]).reset_index()
    checks=[]
    for row in expected.to_dict("records"):
        current=actual[(actual.vla==row["vla"]) & np.isclose(actual.voxel_size,row["voxel_size"])].iloc[0]
        for ours,theirs in [("mean","mean_training_voxels"),("median","median_training_voxels"),
                           ("min","minimum_training_voxels"),("max","maximum_training_voxels")]:
            if not np.isclose(current[ours],row[theirs],rtol=0,atol=1e-8):
                checks.append({"vla":row["vla"],"size":row["voxel_size"],"field":ours,
                               "historical":row[theirs],"reconstructed":float(current[ours])})
    report={"source":"current_released_pickup_trajectory_binary_coverage",
            "rows":len(frame),"matched_historical_diagnostics":not checks,"differences":checks,
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "method":"union of nine run-level visited-voxel sets; no outcome-based fitting"}
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    if checks:raise ValueError("Recovered voxel counts differ from historical diagnostic table")
    print("All 960 counts reconcile to the historical voxel-size diagnostic table",flush=True)

if __name__=="__main__":main()
