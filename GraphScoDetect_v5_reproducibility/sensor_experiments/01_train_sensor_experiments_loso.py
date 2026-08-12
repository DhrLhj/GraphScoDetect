#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOSO sensor experiments that reuse the EXACT v5 baseline training implementation.

Key design:
- Dynamically imports the user's current v5 `03_train_all_protocols.py`.
- Calls its load_task/load_folds/train_one_fold/sample_meta/compute_metrics/aggregate_subjects.
- Reads epochs/batch_size/lr/weight_decay/dropout/balanced_loss from completed baseline LOSO run_config.
- Uses the exact baseline LOSO split.
- Full/all-channel configuration is NOT retrained. It is reused from BASE_OUT_ROOT/results,
  so the Full row must exactly equal the completed v5 LOSO baseline after identical summarization.
- Only non-full sensor subsets are newly trained.
"""
from __future__ import annotations
import argparse, importlib.util, json, hashlib, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

SENSOR_CHANNELS={"S1":[0],"S2":[1],"S3":[2,3],"S4":[4,5]}

def sensors(*xs):
    z=[]
    for x in xs:z+=SENSOR_CHANNELS[x]
    return sorted(z)

def registry(task:int, uniaxial_definition:str):
    r={}
    def add(k,ch,s,desc):
        r[k]={"channels":sorted(set(ch)),"sensors":s,"description":desc}
    add("all",[0,1,2,3,4,5],"S1+S2+S3+S4","All channels / baseline reuse")
    add("remove_s1",sensors("S2","S3","S4"),"S2+S3+S4","Remove S1")
    add("remove_s2",sensors("S1","S3","S4"),"S1+S3+S4","Remove S2")
    add("remove_s3",sensors("S1","S2","S4"),"S1+S2+S4","Remove S3")
    add("remove_s4",sensors("S1","S2","S3"),"S1+S2+S3","Remove S4")
    add("vector_only",sensors("S3","S4"),"S3+S4","Vector-only")
    if task==2:
        add("num1",sensors("S4"),"S4","1 sensor")
    elif task==4:
        add("num2",sensors("S1","S3"),"S1+S3","2 sensors")
        add("num1",sensors("S3"),"S3","1 sensor")
    if uniaxial_definition=="image":
        add("uniaxial_only",[0,1,2,4],"S1+S2+S3(ch1)+S4(ch1)","Uniaxial-only")
    else:
        add("uniaxial_only",[0,1],"S1+S2","Uniaxial-only legacy")
    return r

def table_rows(task:int):
    ch=[
        ("All channels","all","S1+S2+S3+S4"),
        ("Remove S1","remove_s1","S2+S3+S4"),
        ("Remove S2","remove_s2","S1+S3+S4"),
        ("Remove S3","remove_s3","S1+S2+S4"),
        ("Remove S4","remove_s4","S1+S2+S3"),
    ]
    if task==2:
        num=[("4 sensors","all","S1+S2+S3+S4"),("3 sensors","remove_s2","S1+S3+S4"),
             ("2 sensors","vector_only","S3+S4"),("1 sensor","num1","S4")]
    else:
        num=[("4 sensors","all","S1+S2+S3+S4"),("3 sensors","remove_s2","S1+S3+S4"),
             ("2 sensors","num2","S1+S3"),("1 sensor","num1","S3")]
    mod=[("Uniaxial-only","uniaxial_only","CURRENT_UNIAXIAL"),
         ("Vector-only","vector_only","S3+S4"),("Combined","all","S1+S2+S3+S4")]
    return {"channel_ablation":ch,"sensor_number":num,"modality":mod}

def load_json(p):
    with Path(p).open("r",encoding="utf-8") as f:return json.load(f)
def save_json(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def canon(ch):return "ch"+"-".join(map(str,ch))

def import_base(script:Path,project_root:Path):
    sys.path.insert(0,str(project_root))
    spec=importlib.util.spec_from_file_location("v5_base_train",str(script))
    if spec is None or spec.loader is None:raise RuntimeError(f"Cannot import {script}")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    required=["load_task","load_folds","normalize_fold","train_one_fold","sample_meta",
              "compute_metrics","aggregate_subjects","config_hash"]
    miss=[x for x in required if not hasattr(m,x)]
    if miss:raise RuntimeError(f"Base train script misses functions: {miss}")
    return m

def exact_baseline_config(base_out:Path,task:int,seeds:List[int]):
    rows=[]
    for seed in seeds:
        for p in sorted((base_out/"results"/f"task{task}"/"loso"/f"seed_{seed}").glob("fold_*/run_config.json")):
            rows.append(load_json(p))
    if not rows:raise FileNotFoundError(f"No baseline LOSO results for task {task}")
    fields=["epochs","batch_size","lr","weight_decay","dropout","balanced_loss","model","n_classes",
            "pretrain_epochs","joint_epochs","resample_len","segment_len","hidden_dim","lstm_hidden",
            "lambda_inter","gamma_class","intra_margin","temperature",
            "dataset_fingerprint","split_fingerprint"]
    for k in fields:
        vals={json.dumps(x.get(k),sort_keys=True,ensure_ascii=False) for x in rows}
        if len(vals)!=1:raise RuntimeError(f"Task {task}: baseline field {k} is not uniform: {vals}")
    x=rows[0]
    return {k:x.get(k) for k in fields}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_out_root",required=True,help="Completed new_label_v5_selected_protocols root")
    ap.add_argument("--base_train_script",required=True,help="Exact v5 03_train_all_protocols.py used by baseline")
    ap.add_argument("--project_root",required=True)
    ap.add_argument("--out_root",required=True)
    ap.add_argument("--tasks",nargs="+",type=int,default=[2,4])
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    ap.add_argument("--num_workers",type=int,default=0,
                    help="Current v5 runner default is 0. Set to the value used for the baseline if you overrode it.")
    ap.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True,
                    help="Current v5 runner default is AMP on. Match the baseline run.")
    ap.add_argument("--uniaxial_definition",choices=["image","legacy"],default="image")
    ap.add_argument("--overwrite",action="store_true")
    ap.add_argument("--limit_folds",type=int,default=0)
    a=ap.parse_args()

    base_out=Path(a.base_out_root).resolve()
    data_root=base_out/"data";split_root=base_out/"splits"
    script=Path(a.base_train_script).resolve();project=Path(a.project_root).resolve();out=Path(a.out_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    B=import_base(script,project)

    run_manifest={
        "base_out_root":str(base_out),"base_train_script":str(script),"project_root":str(project),
        "protocol":"loso","tasks":a.tasks,"seeds":a.seeds,"num_workers":a.num_workers,"amp":a.amp,
        "uniaxial_definition":a.uniaxial_definition,
        "important":"All-channel predictions are reused from baseline; non-full configs call baseline train_one_fold()."
    }
    save_json(run_manifest,out/"run_manifest.json")

    for task in a.tasks:
        Xfull,y,names,samples,cfg,dataset_fp=B.load_task(data_root,task)
        folds,split_path,split_fp=B.load_folds(split_root,task,"loso")
        if a.limit_folds>0:folds=folds[:a.limit_folds]
        bc=exact_baseline_config(base_out,task,a.seeds)
        if dataset_fp!=bc["dataset_fingerprint"]:
            raise RuntimeError(f"Task {task}: current data fingerprint != completed baseline fingerprint")
        if split_fp!=bc["split_fingerprint"]:
            raise RuntimeError(f"Task {task}: current LOSO split fingerprint != baseline fingerprint")
        print(f"\n===== TASK {task} EXACT BASELINE CONFIG =====")
        for k in ["epochs","pretrain_epochs","joint_epochs","batch_size","lr","weight_decay",
                  "dropout","balanced_loss","model","n_classes","resample_len","segment_len",
                  "hidden_dim","lstm_hidden","lambda_inter","gamma_class","intra_margin","temperature"]:
            print(f"{k}: {bc[k]}")
        print(f"num_workers (runner-level, manually matched): {a.num_workers}")
        print(f"amp (runner-level, manually matched): {a.amp}")
        print(f"LOSO folds: {len(folds)}")

        reg=registry(task,a.uniaxial_definition)
        aliases=defaultdict(list)
        for k,v in reg.items():aliases[tuple(v["channels"])].append(k)
        # Full [0..5] is baseline reuse, never train it here.
        subsets=[(tuple(ch),als) for ch,als in aliases.items() if list(ch)!=[0,1,2,3,4,5]]
        print(f"Logical configs={len(reg)}, newly trained unique subsets={len(subsets)}, Full=reused")

        mapping={}
        for k,v in reg.items():
            mapping[k]={**v,"canonical_config":canon(v["channels"]),
                        "source":"baseline" if v["channels"]==[0,1,2,3,4,5] else "sensor_run"}
        save_json({"task":task,"protocol":"loso","split_file":str(split_path),
                   "baseline_exact_config":bc,"config_mapping":mapping,"table_rows":table_rows(task)},
                  out/"config_maps"/f"task{task}.json")

        for cht,als in sorted(subsets):
            ch=list(cht);cc=canon(ch);X=Xfull[...,ch].copy()
            print(f"\n[CONFIG] task={task} {cc} channels={ch} aliases={als}")
            for seed in a.seeds:
                for pos,fold in enumerate(folds):
                    fid,tr,te=B.normalize_fold(fold,pos)
                    rd=out/"results"/f"task{task}"/"loso"/cc/f"seed_{seed}"/f"fold_{fid}"
                    rd.mkdir(parents=True,exist_ok=True)
                    rc={
                        "task":task,"protocol":"loso","seed":seed,"fold":fid,"channels":ch,"aliases":sorted(als),
                        "epochs":bc["epochs"],"batch_size":bc["batch_size"],"lr":bc["lr"],
                        "weight_decay":bc["weight_decay"],"dropout":bc["dropout"],
                        "balanced_loss":bool(bc["balanced_loss"]),"model":bc["model"],"n_classes":bc["n_classes"],
                        "pretrain_epochs":bc.get("pretrain_epochs",20),"joint_epochs":bc.get("joint_epochs",80),
                        "resample_len":bc.get("resample_len",500),"segment_len":bc.get("segment_len",25),
                        "hidden_dim":bc.get("hidden_dim",64),"lstm_hidden":bc.get("lstm_hidden",128),
                        "lambda_inter":bc.get("lambda_inter",0.5),"gamma_class":bc.get("gamma_class",1.0),
                        "intra_margin":bc.get("intra_margin",1.0),"temperature":bc.get("temperature",0.1),
                        "dataset_fingerprint":dataset_fp,"split_fingerprint":split_fp,"split_file":str(split_path),
                        "base_train_script":str(script),"num_workers":a.num_workers,"amp":a.amp,
                    }
                    rc["config_hash"]=B.config_hash(rc)
                    cp=rd/"run_config.json";mp=rd/"metrics.json";pp=rd/"predictions.json"
                    if cp.exists() and mp.exists() and pp.exists() and not a.overwrite:
                        old=load_json(cp)
                        if old.get("config_hash")==rc["config_hash"]:
                            print(f"[SKIP] task={task} seed={seed} fold={fid} {cc}");continue
                        raise RuntimeError(f"Existing result has different config: {rd}")
                    save_json(rc,cp)
                    np.save(rd/"train.npy",tr);np.save(rd/"test.npy",te)
                    print(f"[RUN] task={task} seed={seed} fold={fid} {cc} train={len(tr)} test={len(te)}")
                    t0=time.time()
                    all_subject_ids=[
                        B.sample_meta(samples[i] or {},names[i],i)["subject_key"]
                        for i in range(len(X))
                    ]
                    model,pred,prob,history=B.train_one_fold(
                        X,y,tr,te,int(bc["n_classes"]),
                        "cuda" if __import__("torch").cuda.is_available() else "cpu",
                        int(bc["epochs"]),int(bc["batch_size"]),float(bc["lr"]),float(bc["weight_decay"]),
                        float(bc["dropout"]),int(seed),bool(bc["balanced_loss"]),int(a.num_workers),bool(a.amp),
                        subject_ids=all_subject_ids,
                        pretrain_epochs=int(bc.get("pretrain_epochs",20)),
                        joint_epochs=int(bc.get("joint_epochs",80)),
                        resample_len=int(bc.get("resample_len",500)),
                        segment_len=int(bc.get("segment_len",25)),
                        hidden_dim=int(bc.get("hidden_dim",64)),
                        lstm_hidden=int(bc.get("lstm_hidden",128)),
                        lambda_inter=float(bc.get("lambda_inter",0.5)),
                        gamma_class=float(bc.get("gamma_class",1.0)),
                        intra_margin=float(bc.get("intra_margin",1.0)),
                        temperature=float(bc.get("temperature",0.1)),
                    )
                    yt=y[te];rows=[]
                    for j,src in enumerate(te):
                        m=B.sample_meta(samples[int(src)] or {},names[int(src)],int(src))
                        rows.append({**m,"task":task,"protocol":"loso","seed":seed,"fold":fid,
                                     "y_true":int(yt[j]),"y_pred":int(pred[j]),"correct":int(yt[j]==pred[j]),
                                     "prob":prob[j].tolist()})
                    # Keep the baseline metric implementation.  If the current v5
                    # trainer supports probability-aware AUROC, pass probabilities;
                    # otherwise fall back to the original signature.  The final
                    # sensor summary independently computes AUROC from saved
                    # probabilities, so older baseline code remains compatible.
                    try:
                        sm=B.compute_metrics(
                            yt, pred, int(bc["n_classes"]), cfg["class_names"], prob
                        )
                    except TypeError:
                        sm=B.compute_metrics(
                            yt, pred, int(bc["n_classes"]), cfg["class_names"]
                        )
                    subm,subrows=B.aggregate_subjects(
                        rows, int(bc["n_classes"]), cfg["class_names"]
                    )
                    metrics={**rc,"elapsed_seconds":float(time.time()-t0),"n_train_samples":int(len(tr)),
                             "n_test_samples":int(len(te)),"sample_metrics":sm,"subject_metrics":subm}
                    save_json(metrics,mp)
                    save_json({"task":task,"protocol":"loso","seed":seed,"fold":fid,
                               "predictions":rows,"subject_predictions":subrows},pp)
                    with (rd/"training_history.csv").open("w",encoding="utf-8") as f:
                        f.write("epoch,stage,train_loss,intra_loss,inter_loss,class_loss\n")
                        for h in history:
                            f.write(
                                f"{h['epoch']},{h.get('stage','')},{h['train_loss']:.10f},"
                                f"{h.get('intra_loss',float('nan')):.10f},"
                                f"{h.get('inter_loss',float('nan')):.10f},"
                                f"{h.get('class_loss',float('nan')):.10f}\n"
                            )
                    auc_txt = subm.get("auroc")
                    auc_txt = "NA" if auc_txt is None else f"{float(auc_txt):.4f}"
                    print(
                        f"[DONE] subject acc={subm['accuracy']:.4f} "
                        f"bacc={subm['balanced_accuracy']:.4f} auroc={auc_txt}"
                    )
    print("\n[DONE] Exact-v5 LOSO sensor training finished.")

if __name__=="__main__":
    main()
