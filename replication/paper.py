"""Prepare paper-format table inputs from the separately labelled exact reviews."""
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil

def read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        return list(reader.fieldnames),list(reader)

def write(path,fields,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        out=csv.DictWriter(stream,fields,lineterminator="\n")
        out.writeheader();out.writerows(rows)

def paper_aggregate(name,source,destination):
    """Work on a copy; frozen unsuffixed historical inputs are never overwritten."""
    source=Path(source);destination=Path(destination)
    shutil.copytree(source,destination,dirs_exist_ok=True)
    promoted=[]
    if name=="rq1":
        fields,rows=read(source/"statistical_comparison.csv")
        _,exact=read(source/"reference_comparisons_exact.csv")
        by_id={row["candidate_formula_id"]:row for row in exact}
        for row in rows:
            if row["formula_id"] not in by_id:continue
            current=by_id[row["formula_id"]]
            for target,key in [
                ("wilcoxon_p_value","wilcoxon_p_value"),
                ("holm_adjusted_p_value","holm_adjusted_p_value"),
                ("rank_biserial_leader_minus_candidate","rank_biserial_reference_minus_candidate"),
                ("nonzero_primary_block_pairs","nonzero_primary_block_pairs"),
                ("practically_equivalent","practically_equivalent")]:
                row[target]=current[key]
        write(destination/"statistical_comparison.csv",fields,rows)
        promoted=["reference_comparisons_exact.csv"]
    else:
        stems={
            "rq2":["statistical_comparison"],
            "rq3":["technique_size_effects","technique_size_interactions"],
            "rq4":["adjacent_size_tests","sample_efficiency_tests","conditional_coverage_tests"],
        }[name]
        for stem in stems:
            shutil.copy2(source/(stem+"_exact.csv"),destination/(stem+".csv"))
            promoted.append(stem+"_exact.csv")
    metadata={"method":"exact","purpose":"Use approved exact p-values in the original publication formatter",
              "input_files":{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in promoted},
              "confidence_intervals":"original pointwise hierarchical intervals retained",
              "frozen_source_modified":False}
    (destination/"paper_statistical_sources.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return destination

ALIASES={
 "rq1":[(3,"rq1_cross_vla_roc_auc.tex"),(4,"rq1_competitive_statistics.tex")],
 "rq2":[(5,"rq2_baseline_performance_top5_rq1.tex")],
 "rq3":[(6,"rq2_reference_by_vla.tex"),(7,"rq2_techniques_by_vla.tex"),
        (8,"rq2_diagnostics_by_vla.tex"),(9,"rq2_global_statistics.tex")],
 "rq4":[(10,"rq3v2_main_coverage_performance_by_vla.tex"),(11,"rq3v2_main_sample_efficiency_by_vla.tex"),
        (12,"rq3v2_main_adjacent_size_effects.tex"),(13,"rq3v2_supplementary_conditional_coverage.tex")],
 "rq5":[(14,"sbfl/roc_auc_table.tex"),(15,"baselines/baseline_top5_sbfl.tex")],
}

def export_tables(output,data,selected):
    output=Path(output);data=Path(data)
    folder=output/"paper";folder.mkdir(parents=True,exist_ok=True)
    aliases=[]
    for rq in selected:
        for number,name in ALIASES[rq]:
            source=output/rq/(name if rq=="rq5" else "tables/"+name)
            if not source.is_file():raise FileNotFoundError(source)
            target=folder/f"table_{number:02d}.tex"
            text=source.read_text()
            if rq=="rq3":text=text.replace("RQ2","RQ3")
            elif rq=="rq4":text=text.replace("RQ3 v2","RQ4").replace("RQ3","RQ4")
            if number==4:
                text=text.replace("Confidence intervals are 95\\%; $p$-values use Holm correction",
                    "Confidence intervals are pointwise 95\\%; $p$-values use exact Wilcoxon tests "
                    "with Holm correction over all 561 unordered technique pairs")
            if number==12:
                _,adjacent=read(output/"rq4/paper_aggregate/adjacent_size_tests.csv")
                by_transition={str(row["lower_training_size"])+"--"+str(row["upper_training_size"]):row
                               for row in adjacent}
                lines=[]
                for line in text.splitlines():
                    transition=line.split(" & ",1)[0]
                    if transition in by_transition:
                        cells=line.split(" & ")
                        value=float(by_transition[transition]["holm_adjusted_p_value"])
                        cells[-2]=f"{value:.5f}" if value>=1e-5 else f"{value:.3g}"
                        line=" & ".join(cells)
                    lines.append(line)
                text="\n".join(lines)+"\n"
            target.write_text(text)
            aliases.append({"table":number,"source":str(source.relative_to(output)),"output":target.name})
    if "rq5" in selected:
        import gzip
        with gzip.open(data/"reference/rq1/audit/episode_manifest.csv.gz","rt",encoding="utf-8-sig",newline="") as stream:
            outcomes={}
            for row in csv.DictReader(stream):
                label=int(row["failure"])
                group=outcomes.setdefault(row["vla"],[0,0]);group[label]+=1
        _,selected_rows=read(data/"rq5/reference/selection/sample_manifest.csv")
        rows=[];tex=[]
        for model,vla,label in [("pi0","Pi0",r"\(\pi_0\)"),("gr00t","Groot","GR00T"),
                                ("eo1","EO1","EO1"),("spatialvla-4b","SpatialVLA","SpatialVLA")]:
            count=outcomes[vla];total=sum(count)
            sample=[row for row in selected_rows if row["model"]==model]
            passed=sum(row["failure"]=="0" for row in sample);failed=len(sample)-passed
            minimum=math.ceil(Fraction(100*total,min(count)))
            record={"model":model,"rq1_pass_percent":100*count[0]/total,"minimum_N":minimum,
                    "selected_pass":passed,"selected_fail":failed,"selected_N":len(sample)}
            rows.append(record)
            tex.append(f"{label} & {record['rq1_pass_percent']:.3f} & {minimum} & {passed} & {failed} & {len(sample)} "+r"\\")
        write(folder/"table_02.csv",list(rows[0]),rows)
        (folder/"table_02.tex").write_text("\n".join([
            r"\begin{table}[ht]",r"\centering",
            r"\caption{Pooled RQ1 success proportions and selected Move-Near evaluation samples.}",
            r"\label{tab:rq5-sample-size}",r"\begin{tabular}{lrrrrr}",r"\toprule",
            r"VLA & RQ1 pass (\%) & Minimum \(N\) & Pass & Fail & Selected \(N\) \\",
            r"\midrule",*tex,r"\bottomrule",r"\end{tabular}",r"\end{table}",""]))
        aliases.append({"table":2,"source":"frozen RQ1 audit and RQ5 final selection","output":"table_02.tex"})
    (folder/"table_manifest.json").write_text(json.dumps({"paper":"TOSEM_Arusa (5).pdf",
        "tables":sorted(aliases,key=lambda row:row["table"]),"statistical_method":"exact",
        "table_1":"conceptual worked example, not an empirical result",
        "figures_1_to_4":"illustrations, architecture and simulator examples; not evaluation outputs"},indent=2)+"\n")
