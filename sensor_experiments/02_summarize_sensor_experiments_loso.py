#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize exact-v5 LOSO sensor experiments.

All pre-existing metrics still come from the user's canonical v5
04_summarize_full_protocol_results.py::metric_bundle().
This script only adds probability-based AUROC on top, so the definitions of
ACC/Sensitivity/Specificity/BAcc/F1 are unchanged.

AUROC:
- Task 2: class-1 (scoliosis) ROC-AUC.
- Task 4: macro one-vs-rest AUROC across valid classes.
- Per-class OVR AUROC is also saved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLASS_NAMES = {
    2: ["normal", "scoliosis"],
    4: ["normal", "mild", "moderate", "severe"],
}


def load_json(p):
    with Path(p).open("r", encoding="utf-8") as f:
        return json.load(f)


def import_summary(script: Path):
    spec = importlib.util.spec_from_file_location("v5_base_summary", str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m, "metric_bundle"):
        raise RuntimeError("Base summary script has no metric_bundle()")
    return m


def read_subject_rows_from_fold_files(paths):
    rows = []
    for p in paths:
        obj = load_json(p)
        seed = int(obj.get("seed", Path(p).parts[-3].replace("seed_", "")))
        fold = str(obj.get("fold", Path(p).parent.name.replace("fold_", "")))
        for r in obj.get("subject_predictions", []):
            if "prob" not in r:
                raise RuntimeError(
                    f"subject_predictions in {p} has no 'prob'; AUROC requires probabilities."
                )
            rows.append({**r, "seed": seed, "fold": fold})
    return rows


def baseline_subject_rows(base_out: Path, task: int, seeds: List[int]):
    rows = []
    for seed in seeds:
        ps = sorted(
            (base_out / "results" / f"task{task}" / "loso" / f"seed_{seed}")
            .glob("fold_*/predictions.json")
        )
        if not ps:
            raise FileNotFoundError(
                f"No baseline LOSO predictions task={task} seed={seed}"
            )
        rows += read_subject_rows_from_fold_files(ps)
    return rows


def sensor_subject_rows(sensor_out: Path, task: int, cc: str, seeds: List[int]):
    rows = []
    for seed in seeds:
        ps = sorted(
            (sensor_out / "results" / f"task{task}" / "loso" / cc / f"seed_{seed}")
            .glob("fold_*/predictions.json")
        )
        if not ps:
            raise FileNotFoundError(
                f"No sensor LOSO predictions task={task} config={cc} seed={seed}"
            )
        rows += read_subject_rows_from_fold_files(ps)
    return rows


def prob(r):
    x = r["prob"]
    if isinstance(x, str):
        x = json.loads(x)
    return np.asarray(x, dtype=float)


def auroc_bundle(y_true: np.ndarray, y_prob: np.ndarray, task: int):
    """Probability-based AUROC plus per-class OVR AUROC."""
    names = CLASS_NAMES[task]
    n_classes = len(names)
    if y_prob.shape != (len(y_true), n_classes):
        raise ValueError(
            f"Task {task}: expected probability shape {(len(y_true), n_classes)}, "
            f"got {y_prob.shape}"
        )

    per_class = []
    aucs = []
    for c, name in enumerate(names):
        binary_true = (y_true == c).astype(int)
        if len(np.unique(binary_true)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(binary_true, y_prob[:, c]))
        aucs.append(auc)
        per_class.append(
            {
                "class_id": c,
                "class_name": name,
                "auroc_ovr": float(auc) if np.isfinite(auc) else np.nan,
                "auroc_defined": bool(np.isfinite(auc)),
            }
        )

    if task == 2:
        # Scoliosis/class 1 is positive.
        auc = aucs[1]
    else:
        valid = [x for x in aucs if np.isfinite(x)]
        auc = float(np.mean(valid)) if valid else np.nan

    return (
        float(auc) if np.isfinite(auc) else np.nan,
        int(sum(np.isfinite(x) for x in aucs)),
        per_class,
    )


def canonical_metrics(Summary, y_true, y_pred, task):
    """
    Preserve all old metric definitions.  We intentionally call the canonical
    metric_bundle with the old 3-argument form first.  If the user's updated
    summary requires a y_prob argument, None is accepted by the patched v5
    implementation.
    """
    try:
        return Summary.metric_bundle(y_true, y_pred, task)
    except TypeError:
        return Summary.metric_bundle(y_true, y_pred, task, None)


def per_seed_and_ensemble(rows, task, Summary):
    seed_metric = []
    seed_auroc_pc = []

    df = pd.DataFrame(rows)
    for seed, g in df.groupby("seed", sort=True):
        yt = g["y_true"].astype(int).to_numpy()
        yp = g["y_pred"].astype(int).to_numpy()
        yprob = np.stack([prob(r) for r in g.to_dict("records")])

        m, pc, cm, cmn = canonical_metrics(Summary, yt, yp, task)
        auc, valid_count, auc_pc = auroc_bundle(yt, yprob, task)
        m = dict(m)
        m["auroc"] = auc
        m["auroc_valid_class_count"] = valid_count

        seed_metric.append({"seed": int(seed), "n_subjects": len(g), **m})
        for x in auc_pc:
            seed_auroc_pc.append(
                {"seed": int(seed), "level": "seed", **x}
            )

    # Exact same seed-ensemble concept as baseline:
    # average subject probabilities across seeds, then argmax.
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["subject_key"])].append(r)

    ens = []
    for key, rs in groups.items():
        truth = {int(r["y_true"]) for r in rs}
        if len(truth) != 1:
            raise RuntimeError(f"Inconsistent labels for {key}: {truth}")
        p = np.stack([prob(r) for r in rs]).mean(axis=0)
        first = rs[0]
        y = next(iter(truth))
        ens.append(
            {
                "subject_key": key,
                "name": first.get("name", ""),
                "center": first.get("center", ""),
                "y_true": y,
                "y_pred": int(p.argmax()),
                "correct": int(int(p.argmax()) == y),
                "n_seeds": len(rs),
                "ensemble_prob": p.tolist(),
            }
        )

    eg = pd.DataFrame(ens)
    yt = eg["y_true"].astype(int).to_numpy()
    yp = eg["y_pred"].astype(int).to_numpy()
    yprob = np.stack([np.asarray(x, dtype=float) for x in eg["ensemble_prob"]])

    em, pc, cm, cmn = canonical_metrics(Summary, yt, yp, task)
    em = dict(em)
    auc, valid_count, auc_pc = auroc_bundle(yt, yprob, task)
    em["auroc"] = auc
    em["auroc_valid_class_count"] = valid_count

    # Merge AUROC into canonical per-class rows without changing other fields.
    pc_by_id = {int(x["class_id"]): dict(x) for x in pc}
    for x in auc_pc:
        cid = int(x["class_id"])
        if cid not in pc_by_id:
            pc_by_id[cid] = {
                "class_id": cid,
                "class_name": x["class_name"],
            }
        pc_by_id[cid]["auroc_ovr"] = x["auroc_ovr"]
        pc_by_id[cid]["auroc_defined"] = x["auroc_defined"]

    return (
        pd.DataFrame(seed_metric),
        eg,
        em,
        list(pc_by_id.values()),
        cm,
        cmn,
        pd.DataFrame(seed_auroc_pc),
    )


def fm(x):
    a = np.asarray(x, dtype=float)
    valid = a[np.isfinite(a)]
    if len(valid) == 0:
        return "NA"
    sd = valid.std(ddof=1) if len(valid) > 1 else 0.0
    return f"{valid.mean():.4f} ± {sd:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out_root", required=True)
    ap.add_argument("--sensor_out_root", required=True)
    ap.add_argument(
        "--base_summary_script",
        required=True,
        help="Exact v5 04_summarize_full_protocol_results.py",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    a = ap.parse_args()

    base = Path(a.base_out_root).resolve()
    out = Path(a.sensor_out_root).resolve()
    Summary = import_summary(Path(a.base_summary_script).resolve())
    sd = out / "summary"
    sd.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "auroc",
    ]

    all_seed = []
    all_ens = []
    all_pc = []
    all_cm = []
    all_subject = []
    all_seed_auc_pc = []
    tables = {}

    names = {
        (2, "channel_ablation"): "Table_S7_binary_channel_ablation_LOSO",
        (4, "channel_ablation"): "Table_S8_fourclass_channel_ablation_LOSO",
        (2, "sensor_number"): "Table_S9_binary_sensor_number_LOSO",
        (4, "sensor_number"): "Table_S10_fourclass_sensor_number_LOSO",
        (2, "modality"): "Table_S11_binary_modality_LOSO",
        (4, "modality"): "Table_S12_fourclass_modality_LOSO",
    }

    cache = {}
    for task in (2, 4):
        obj = load_json(out / "config_maps" / f"task{task}.json")
        cmap = obj["config_mapping"]

        for key, ci in cmap.items():
            cc = ci["canonical_config"]
            if (task, cc) in cache:
                continue

            rows = (
                baseline_subject_rows(base, task, a.seeds)
                if ci["source"] == "baseline"
                else sensor_subject_rows(out, task, cc, a.seeds)
            )

            sm, ens, em, pc, cm, cmn, seed_auc_pc = per_seed_and_ensemble(
                rows, task, Summary
            )
            cache[(task, cc)] = (
                sm, ens, em, pc, cm, cmn, ci["source"]
            )

            for _, r in sm.iterrows():
                all_seed.append(
                    {
                        "task": task,
                        "canonical_config": cc,
                        "source": ci["source"],
                        **r.to_dict(),
                    }
                )

            all_ens.append(
                {
                    "task": task,
                    "canonical_config": cc,
                    "source": ci["source"],
                    "n_subjects": len(ens),
                    **em,
                }
            )

            for r in pc:
                all_pc.append(
                    {
                        "task": task,
                        "canonical_config": cc,
                        "source": ci["source"],
                        **r,
                    }
                )

            if not seed_auc_pc.empty:
                for _, r in seed_auc_pc.iterrows():
                    all_seed_auc_pc.append(
                        {
                            "task": task,
                            "canonical_config": cc,
                            "source": ci["source"],
                            **r.to_dict(),
                        }
                    )

            for i, tname in enumerate(CLASS_NAMES[task]):
                for j, pname in enumerate(CLASS_NAMES[task]):
                    all_cm.append(
                        {
                            "task": task,
                            "canonical_config": cc,
                            "source": ci["source"],
                            "true_class": tname,
                            "pred_class": pname,
                            "count": int(cm[i, j]),
                            "row_normalized": float(cmn[i, j]),
                        }
                    )

            e2 = ens.copy()
            e2["task"] = task
            e2["canonical_config"] = cc
            e2["source"] = ci["source"]
            all_subject.extend(e2.to_dict("records"))

        for fam, defs in obj["table_rows"].items():
            tab = []
            for setting, key, slabel in defs:
                ci = cmap[key]
                cc = ci["canonical_config"]
                sm, ens, em, pc, cm, cmn, source = cache[(task, cc)]

                row = {
                    "setting": setting,
                    "sensors": (
                        ci["sensors"]
                        if slabel == "CURRENT_UNIAXIAL"
                        else slabel
                    ),
                    "channels": ",".join(map(str, ci["channels"])),
                    "source": source,
                    "n_subjects": len(ens),
                }

                # Primary columns = seed probability ensemble, identical to
                # base Main_Subject_Ensemble convention.
                for col in metric_cols:
                    row[col] = em.get(col, np.nan)
                    row[col + "_seed_mean_std"] = (
                        fm(sm[col].values) if col in sm.columns else "NA"
                    )
                tab.append(row)

            tables[(task, fam)] = pd.DataFrame(tab)

    pd.DataFrame(all_seed).to_csv(
        sd / "01_metrics_by_seed.csv", index=False
    )
    pd.DataFrame(all_ens).to_csv(
        sd / "02_ensemble_metrics_primary.csv", index=False
    )
    pd.DataFrame(all_pc).to_csv(
        sd / "03_per_class_ensemble.csv", index=False
    )
    pd.DataFrame(all_cm).to_csv(
        sd / "04_confusion_matrix_ensemble.csv", index=False
    )
    pd.DataFrame(all_subject).to_csv(
        sd / "05_subject_ensemble_predictions.csv", index=False
    )
    pd.DataFrame(all_seed_auc_pc).to_csv(
        sd / "06_per_class_auroc_by_seed.csv", index=False
    )

    for k, df in tables.items():
        df.to_csv(sd / (names[k] + ".csv"), index=False)

    # Baseline sanity check.  Existing metrics must match exactly because Full
    # is reused rather than retrained.  AUROC is compared only if the updated
    # baseline summary already contains an AUROC column.
    checks = []
    baseline_main = base / "summary" / "01_main_subject_ensemble_metrics.csv"
    if baseline_main.exists():
        bm = pd.read_csv(baseline_main)
        for task in (2, 4):
            g = bm[
                (bm["task"] == task)
                & (bm["protocol"].astype(str).str.lower() == "loso")
            ]
            if len(g):
                b = g.iloc[0]
                tab = tables[(task, "channel_ablation")].iloc[0]
                row = {"task": task}
                for col in metric_cols:
                    if col in b.index:
                        row["baseline_" + col] = float(b[col])
                        row["sensor_full_" + col] = float(tab[col])
                        row["diff_" + col] = float(tab[col]) - float(b[col])
                checks.append(row)

    cdf = pd.DataFrame(checks)
    cdf.to_csv(sd / "00_baseline_match_check.csv", index=False)

    xlsx = sd / "sensor_loso_exact_summary_auroc.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        if not cdf.empty:
            cdf.to_excel(w, sheet_name="Baseline_Match_Check", index=False)

        for k, df in tables.items():
            df.to_excel(
                w,
                sheet_name=names[k].replace("Table_", "")[:31],
                index=False,
            )

        pd.DataFrame(all_ens).to_excel(
            w, sheet_name="Ensemble_Primary", index=False
        )
        pd.DataFrame(all_seed).to_excel(
            w, sheet_name="Metrics_By_Seed", index=False
        )
        pd.DataFrame(all_pc).to_excel(
            w, sheet_name="Per_Class_Ensemble", index=False
        )
        pd.DataFrame(all_seed_auc_pc).to_excel(
            w, sheet_name="PerClass_AUROC_BySeed", index=False
        )
        pd.DataFrame(all_cm).to_excel(
            w, sheet_name="Confusion_Matrix", index=False
        )
        pd.DataFrame(all_subject).to_excel(
            w, sheet_name="Subject_Ensemble", index=False
        )

    print(f"[DONE] {xlsx}")
    if not cdf.empty:
        print(
            "\n===== BASELINE MATCH CHECK "
            "(non-AUROC diffs should all be 0) ====="
        )
        print(cdf.to_string(index=False))

    for k, df in tables.items():
        print(f"\n===== {names[k]} =====")
        print(
            df[
                [
                    "setting",
                    "sensors",
                    "accuracy",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                    "micro_f1",
                    "auroc",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
