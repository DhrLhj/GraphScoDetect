#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect the completed v5 LOSO baseline and extract its exact training configuration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

FIELDS = ["epochs","batch_size","lr","weight_decay","dropout","balanced_loss","model","n_classes",
          "dataset_fingerprint","split_fingerprint","split_file"]

def load(p):
    with Path(p).open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_out_root",required=True)
    ap.add_argument("--tasks",nargs="+",type=int,default=[2,4])
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    ap.add_argument("--save_json",default="")
    a=ap.parse_args()
    root=Path(a.base_out_root).resolve()
    report={}
    for task in a.tasks:
        rows=[]
        for seed in a.seeds:
            for p in sorted((root/"results"/f"task{task}"/"loso"/f"seed_{seed}").glob("fold_*/run_config.json")):
                x=load(p); rows.append((seed,p,x))
        if not rows:
            raise FileNotFoundError(f"No completed baseline LOSO run_config found for task {task}")
        unique={}
        for field in FIELDS:
            vals=[]
            for _,_,x in rows:
                v=x.get(field)
                if field=="split_file":
                    # split_file path itself may be stable; keep it for diagnostics.
                    pass
                vals.append(json.dumps(v,sort_keys=True,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v))
            unique[field]=sorted(set(vals))
        critical=["epochs","batch_size","lr","weight_decay","dropout","balanced_loss","model","n_classes",
                  "dataset_fingerprint","split_fingerprint"]
        bad={k:v for k,v in unique.items() if k in critical and len(v)!=1}
        if bad:
            raise RuntimeError(f"Task {task}: baseline LOSO configurations are not uniform: {bad}")
        n_by_seed=Counter(seed for seed,_,_ in rows)
        first=rows[0][2]

        # Verify that the baseline LOSO is truly complete before a dependent
        # sensor experiment is allowed to start.  This prevents the automatic
        # wait/launcher from proceeding after a crashed or partial baseline run.
        split_file = Path(first["split_file"])
        if not split_file.exists():
            raise FileNotFoundError(
                f"Task {task}: baseline split file recorded in run_config does not exist: {split_file}"
            )
        split_obj = load(split_file)
        if isinstance(split_obj, list):
            expected_folds = len(split_obj)
        elif isinstance(split_obj, dict):
            fold_list = None
            for key in ("folds", "splits", "data"):
                if isinstance(split_obj.get(key), list):
                    fold_list = split_obj[key]
                    break
            if fold_list is None:
                raise RuntimeError(
                    f"Task {task}: cannot determine fold count from split file {split_file}"
                )
            expected_folds = len(fold_list)
        else:
            raise RuntimeError(
                f"Task {task}: unsupported split structure in {split_file}"
            )

        incomplete = {
            int(seed): int(n_by_seed.get(seed, 0))
            for seed in a.seeds
            if int(n_by_seed.get(seed, 0)) != int(expected_folds)
        }
        if incomplete:
            raise RuntimeError(
                f"Task {task}: baseline LOSO is incomplete. "
                f"Expected {expected_folds} folds per seed, got {incomplete}. "
                "Do not start sensor ablation until the baseline run finishes successfully."
            )

        task_report={
            "n_run_configs":len(rows),
            "expected_folds_per_seed":int(expected_folds),
            "n_folds_by_seed":{str(k):int(v) for k,v in sorted(n_by_seed.items())},
            "epochs":first["epochs"],
            "batch_size":first["batch_size"],
            "lr":first["lr"],
            "weight_decay":first["weight_decay"],
            "dropout":first["dropout"],
            "balanced_loss":first["balanced_loss"],
            "model":first["model"],
            "n_classes":first["n_classes"],
            "dataset_fingerprint":first["dataset_fingerprint"],
            "split_fingerprint":first["split_fingerprint"],
            "split_file":first["split_file"],
        }
        report[str(task)]=task_report
        print(f"\n===== BASELINE TASK {task} LOSO =====")
        for k,v in task_report.items():
            print(f"{k}: {v}")
    if a.save_json:
        p=Path(a.save_json);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"\n[SAVED] {p}")

if __name__=="__main__":
    main()
