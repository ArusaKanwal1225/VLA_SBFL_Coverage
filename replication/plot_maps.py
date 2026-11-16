#!/usr/bin/env python3
"""Reproduce Figure 5 from the frozen reconstructed Pick-Up maps, without refitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src_new")]
from vla_sbfl.scoring import empirical_percentiles
from experiments.RQ5.provenance import verify_compatibility,verify_frozen_maps

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps",type=Path,default=ROOT/"data/rq5/reconstructed_pickup_maps")
    p.add_argument("--output",type=Path,default=ROOT/"reproduced/maps/figure_05")
    a=p.parse_args()
    verify_compatibility()
    verify_frozen_maps(a.maps)
    frames=[];summaries={};fingerprints={}
    expected={"EO1":(6291,292),"Groot":(8061,406),"Pi0":(8443,392),"SpatialVLA":(7763,413)}
    keys=["voxel_x","voxel_y","voxel_z"]
    for vla in ["EO1","Groot","Pi0","SpatialVLA"]:
        paths=sorted((a.maps/"folds").glob(vla+"__*/*/model/voxel_scores.csv.gz"))
        if len(paths)!=80:raise ValueError(f"Expected 80 maps for {vla}, found {len(paths)}")
        parts=[]
        for path in paths:
            meta=path.with_name("metadata.json")
            metadata=json.loads(meta.read_text())
            grid=metadata["grid"]
            if grid["voxel_size"]!=.02 or grid["origin"]!=[-1.,-1.,0.]:
                raise ValueError("Unexpected scoring grid")
            frame=pd.read_csv(path,usecols=[*keys,"ef","ep","harmonic_mean"])
            if frame.duplicated(keys).any():raise ValueError("Repeated voxel in a map")
            frame["percentile"]=empirical_percentiles(frame.harmonic_mean.to_numpy())
            frame["support"]=frame.ef+frame.ep
            parts.append(frame[keys+["percentile","support"]])
            fingerprints[meta.relative_to(a.maps).as_posix()]=hashlib.sha256(meta.read_bytes()).hexdigest()
        combined=pd.concat(parts,ignore_index=True)
        stable=combined.groupby(keys).agg(median_percentile=("percentile","median"),
            maps_present=("percentile","size"),median_episode_support=("support","median")).reset_index()
        stable=stable[stable.maps_present>=20].copy()
        stable["vla"]=vla
        for axis,col in zip("xyz",keys):stable[axis]=-1+(.02*(stable[col]+.5)) if axis!="z" else .02*(stable[col]+.5)
        top=int((stable.median_percentile>=.9).sum())
        summaries[vla]={"maps":80,"stable_voxels":len(stable),"top_decile":top,
                        "minimum_maps_present":20,"matches_paper_counts":(len(stable),top)==expected[vla]}
        frames.append(stable)
        print(vla,summaries[vla],flush=True)
    points=pd.concat(frames,ignore_index=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    points.to_csv(a.output.with_suffix(".csv.gz"),index=False,compression="gzip")
    fig=plt.figure(figsize=(11,8.5))
    cmap=plt.get_cmap("magma")
    for i,(vla,title) in enumerate([("EO1","EO1"),("Groot","GR00T"),("Pi0","Pi0"),("SpatialVLA","SpatialVLA")],1):
        ax=fig.add_subplot(2,2,i,projection="3d")
        group=points[points.vla==vla]
        strength=np.sqrt(np.clip(group.median_episode_support/max(float(group.median_episode_support.quantile(.95)),1),0,1))
        colors=cmap(group.median_percentile.to_numpy())
        colors[:,3]=.12+.68*strength
        ax.scatter(group.x,group.y,group.z,c=colors,s=2+10*strength,linewidths=0,depthshade=False,rasterized=True)
        for axis in "xyz":
            low=math.floor(float(points[axis].min())*10)/10
            high=math.ceil(float(points[axis].max())*10)/10
            getattr(ax,"set_"+axis+"lim")(low,high)
            getattr(ax,"set_"+axis+"label")(axis,labelpad=1)
        ax.view_init(elev=22,azim=-60)
        ax.set_box_aspect((1,1,.65))
        ax.tick_params(labelsize=7,pad=0)
        ax.set_title(title+f"\n{len(group):,} stable voxels; {summaries[vla]['top_decile']:,} in top decile",fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,bottom=.20,top=.87,wspace=.01,hspace=.16)
    fig.suptitle("Failure-associated voxel suspiciousness by VLA",fontsize=14,fontweight="bold",y=.97)
    fig.text(.5,.94,"Harmonic Mean | 0.02 m voxels | median within-map percentile across 80 maps per VLA",
             ha="center",fontsize=9)
    cax=fig.add_axes([.23,.105,.54,.018])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax,orientation="horizontal")
    bar.set_label("Median suspiciousness percentile",fontsize=9,labelpad=2)
    fig.text(.5,.015,"Only voxels present in at least 25% of maps; absent voxels are not assigned zero.\n"
             "Marker size and opacity increase with median episode support. Frozen reconstructed Pick-Up maps.",
             ha="center",fontsize=8)
    for suffix in [".png",".pdf"]:fig.savefig(a.output.with_suffix(suffix),dpi=220)
    plt.close(fig)
    report={"models":summaries,"map_origin":"frozen_reconstructed_pickup_maps",
            "normalization":"training-voxel empirical percentile independently per map",
            "aggregation":"median conditional on voxel presence; equal weight across 80 maps",
            "map_metadata_sha256":fingerprints}
    a.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
