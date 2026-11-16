#!/usr/bin/env python3
"""CPU replication of the five paper RQs: verify, tables, stats, smoke, run."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
RQS=("rq1","rq2","rq3","rq4","rq5")
SCRIPTS={"rq1":"RQ1","rq2":"RQ2_baselines","rq3":"RQ2","rq4":"RQ3_v2"}
COMPARISONS={
 "rq1":("configuration_metrics.csv",["configuration_id","formula_id"],["roc_auc","average_precision","mean_known_fraction"]),
 "rq2":("configuration_metrics.csv",["configuration_id","method_id"],["roc_auc","average_precision"]),
 "rq3":("fold_metrics_by_size.csv.gz",["voxel_size","configuration_id","test_run","formula_id"],["roc_auc","average_precision","mean_known_fraction"]),
 "rq4":("configuration_learning_curves.csv",["voxel_size","configuration_id","training_size","formula_id"],["roc_auc","average_precision","calibration_unique_coverage","test_unique_coverage"]),
}

def command(args,script,*tokens):
    argv=[sys.executable,str(ROOT/script),*map(str,tokens)]
    print("\n$ "+" ".join(argv),flush=True)
    if args.dry_run:return
    env=dict(os.environ)
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"src_new"),env.get("PYTHONPATH","")])
    env["VLA_SBFL_RQ1_REFERENCE"]=str(args.data/"reference/rq1")
    env["MPLBACKEND"]="Agg"
    for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:env[name]="1"
    subprocess.run(argv,cwd=ROOT,env=env,check=True)

def table_commands(args,name,source,target,rq1):
    if name=="rq5":
        command(args,"experiments/RQ5/rq5_reproduce.py","tables","--data",args.data/"rq5","--output",target)
        return
    aggregate=target/"paper_aggregate"
    if not args.dry_run:
        from replication.paper import paper_aggregate
        paper_aggregate(name,source/"aggregate",aggregate)
    extra=[]
    if name=="rq2":extra=["--rq1-aggregate-dir",rq1/"aggregate"]
    if name=="rq3":
        extra=["--fine-results",source/"cells/size_0p01","--coarse-results",source/"cells/size_0p05",
               "--primary-results",rq1]
        if args.command in {"tables","stats"} or not (source/"cells/size_0p01/folds").is_dir():
            extra+=["--training-voxels-csv",args.data/"reference/rq3/training_voxels.csv"]
    command(args,f"experiments/{SCRIPTS[name]}/generate_tables.py",
            "--aggregate-dir",aggregate,"--output-dir",target/"tables",*extra)
    command(args,f"experiments/{SCRIPTS[name]}/generate_tables.py",
            "--aggregate-dir",aggregate,"--output-dir",target/"statistical_review",
            "--statistical-review-method","exact")
    if name=="rq4" and not args.no_figures:
        command(args,"experiments/RQ3_v2/generate_plots.py","--result-root",source,
                "--output-directory",target/"figures")

def compare(reference,actual,keys,columns):
    def indexed(path):
        opener=gzip.open if path.suffix==".gz" else open
        with opener(path,"rt",encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
        index={tuple(row[k] for k in keys):row for row in rows}
        if len(index)!=len(rows):raise ValueError("Duplicate metric keys: "+str(path))
        return index
    expected,observed=indexed(reference),indexed(actual)
    if expected.keys()!=observed.keys():return {"status":"different_keys","expected":len(expected),"observed":len(observed)}
    differences=[]
    for key,row in expected.items():
        for col in columns:
            left,right=float(row[col]),float(observed[key][col])
            if not math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-10):
                differences.append({"key":key,"column":col,"reference":left,"actual":right})
    return {"status":"matched" if not differences else "differences","rows":len(expected),
            "columns":columns,"different_values":len(differences),"examples":differences[:10]}

def exact_review(args,name,source):
    flag="--analysis-only" if name=="rq1" else "--statistical-review-only"
    command(args,f"experiments/{SCRIPTS[name]}/run_experiment.py",
            "--output-root",source,flag,"--wilcoxon-method","exact")

def run(args):
    if not args.dry_run and sys.version_info[:2]!=(3,11):
        raise RuntimeError("Use Python 3.11 with requirements.txt")
    if not args.dry_run and not (args.data/"DATASET.json").is_file():
        raise FileNotFoundError("Install the data with: python replication/data.py unpack ARCHIVE.zip")
    if args.command=="verify":
        if args.dry_run:print("Would verify hashes and RQ5 outcomes/maps:",args.data);return
        from replication.data import verify
        verify(args.data)
        command(args,"experiments/RQ5/rq5_reproduce.py","verify","--data",args.data/"rq5")
        return
    report={"status":"running","configuration":{"command":args.command,"rq":args.rq,
        "data":str(args.data),"output":str(args.output),"no_figures":args.no_figures},
        "python":sys.version,"started_at":datetime.now(timezone.utc).isoformat(),"checks":{}}
    receipt=args.output/"replication_run.json"
    if not args.dry_run:
        args.output.mkdir(parents=True,exist_ok=True)
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if args.command!="run" or not args.resume or old["configuration"]!=report["configuration"]:
                raise FileExistsError("Choose a new output directory, or --resume with the same run settings")
        elif any(args.output.iterdir()):raise FileExistsError("Choose an empty output directory")
        receipt.write_text(json.dumps(report,indent=2)+"\n")
        if not args.no_verify:
            from replication.data import verify
            prefixes=("reference/","rq5/reference/") if args.command in {"tables","stats"} else ()
            verify(args.data,prefixes=prefixes)
    reference=args.data/"reference/rq1"
    fitted_rq1=args.output/"rq1"
    for name in RQS:
        if name not in args.rq:continue
        destination=args.output/name
        if args.command in {"tables","stats"}:
            source=args.data/"reference"/name
            if args.command=="stats" and name!="rq5":
                if not args.dry_run:shutil.copytree(source/"aggregate",destination/"aggregate",dirs_exist_ok=True)
                exact_review(args,name,destination)
                if not args.dry_run:
                    checks=[]
                    for actual in sorted((destination/"aggregate").glob("*_exact.csv")):
                        expected=source/"aggregate"/actual.name
                        if not expected.is_file():continue
                        with actual.open(newline="") as stream:fields=next(csv.reader(stream))
                        keys=[field for field in ["formula_id","candidate_formula_id","first_formula_id",
                              "second_formula_id","method_id","voxel_size",
                              "lower_training_size","upper_training_size"] if field in fields]
                        numeric=[field for field in fields if "p_value" in field or "rank_biserial" in field
                                 or field in {"raw_p","holm_p"}]
                        if keys and numeric:
                            checks.append({"file":actual.name,**compare(expected,actual,keys,numeric)})
                    report["checks"][name]={"status":"matched" if all(c["status"]=="matched" for c in checks)
                                           and checks else "differences","exact_reviews":checks}
                source=destination
            table_commands(args,name,source,destination,reference)
            continue
        if name=="rq1":
            command(args,"experiments/RQ1/run_experiment.py","--input-root",args.data/"pickup",
                "--output-root",destination,
                *(["--configuration","EO1__DH__200","--max-folds",1,"--bootstrap-iterations",100]
                  if args.command=="smoke" else []))
        elif name=="rq2":
            rq1=fitted_rq1 if "rq1" in args.rq else reference
            command(args,"experiments/RQ2_baselines/run_experiment.py",
                "--input-root",args.data/"pickup","--raw-root",args.data/"pickup",
                "--rq1-fold-metrics",rq1/"aggregate/fold_metrics.csv.gz","--output-root",destination,
                *(["--configuration","EO1__DH__200","--max-runs",1,"--extract-only"]
                  if args.command=="smoke" else []))
        elif name=="rq3":
            command(args,"experiments/RQ2/run_experiment.py","--input-root",args.data/"pickup",
                "--primary-results",fitted_rq1,"--output-root",
                destination/"configured" if args.command=="smoke" else destination,
                *(["--smoke","--bootstrap-iterations",100] if args.command=="smoke" else []))
        elif name=="rq4":
            command(args,"experiments/RQ3_v2/run_experiment.py","--input-root",args.data/"pickup",
                "--output-root",destination,*(["--smoke"] if args.command=="smoke" else []))
        elif args.command=="smoke":
            command(args,"experiments/RQ5/rq5_reproduce.py","tables","--data",args.data/"rq5","--output",destination)
        else:
            complete=destination/"verification_report.json"
            if args.resume and complete.is_file():
                if json.loads(complete.read_text()).get("status")!="matched_reference":raise ValueError("Invalid RQ5 completion marker")
                print("Skipping completed RQ5 reproduction",flush=True)
            else:
                command(args,"experiments/RQ5/rq5_reproduce.py","run","--data",args.data/"rq5","--output",destination)
        if args.command=="run" and name!="rq5":
            exact_review(args,name,destination)
            rq1=fitted_rq1 if "rq1" in args.rq else reference
            table_commands(args,name,destination,destination,rq1)
            if not args.dry_run:
                filename,keys,columns=COMPARISONS[name]
                report["checks"][name]=compare(args.data/"reference"/name/"aggregate"/filename,
                                               destination/"aggregate"/filename,keys,columns)
                receipt.write_text(json.dumps(report,indent=2)+"\n")
    if not args.dry_run:
        if args.command!="smoke":
            from replication.paper import export_tables
            export_tables(args.output,args.data,args.rq)
        if "rq1" in args.rq and not args.no_figures and args.command!="smoke":
            command(args,"replication/plot_maps.py","--maps",args.data/"rq5/reconstructed_pickup_maps",
                    "--output",args.output/"paper/figure_05")
        mismatch=any(row["status"]!="matched" for row in report["checks"].values())
        report.update(status="reference_differences" if mismatch else "complete",
                      completed_at=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(report,indent=2)+"\n")
        print("\nReplication report:",receipt)
        if mismatch:raise RuntimeError("Recomputed metrics differ from references; inspect replication_run.json")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["verify","tables","stats","smoke","run"])
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--output",type=Path)
    p.add_argument("--rq",nargs="+",choices=RQS,default=list(RQS))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--no-verify",action="store_true",help="Skip rehashing data already verified on extraction")
    p.add_argument("--no-figures",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if len(args.rq)!=len(set(args.rq)):p.error("Repeated RQ names")
    if args.command=="run" and "rq3" in args.rq and "rq1" not in args.rq:
        args.rq.insert(0,"rq1")
        print("Adding RQ1: the voxel-size run reuses its fitted 0.02 m folds.",flush=True)
    args.data=args.data.expanduser().resolve()
    args.output=(args.output or ROOT/"reproduced"/args.command).expanduser().resolve()
    if args.output==args.data or args.output.is_relative_to(args.data):p.error("Output must be outside immutable data")
    run(args)

if __name__=="__main__":
    try:main()
    except (ValueError,OSError,RuntimeError,subprocess.CalledProcessError) as error:
        print("ERROR: "+str(error),file=sys.stderr);sys.exit(1)
