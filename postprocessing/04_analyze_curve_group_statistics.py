#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, ast, re
from pathlib import Path
import numpy as np
import pandas as pd

GROUP_ORDER = ["胸弯", "腰弯", "胸腰弯", "双弯"]


def norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).replace("\u3000", " ")).strip()


def parse_num(x):
    s = norm(x)
    if not s:
        return None
    if "单" in s: return 1
    if "双" in s: return 2
    if "三" in s: return 3
    m = re.search(r"([123])", s)
    return int(m.group(1)) if m else None


def parse_loc(x, missing_as_thoracic=False):
    s = norm(x)
    if (not s) or any(k in s for k in ["未注明", "未知", "不详"]):
        return "胸弯" if missing_as_thoracic else None
    if "胸腰" in s: return "胸腰弯"
    if "腰" in s and "胸" not in s: return "腰弯"
    if "胸" in s: return "胸弯"
    return None


def curve_group(curve_number, curve1, missing_as_thoracic=False):
    n = parse_num(curve_number)
    loc = parse_loc(curve1, missing_as_thoracic)
    if n == 2:
        return "双弯", ""
    if n == 1 and loc in {"胸弯", "腰弯", "胸腰弯"}:
        return loc, ""
    if n == 3:
        return None, "三弯"
    if n == 1 and loc is None:
        return None, "单弯但位置缺失/不明确"
    if n is None and loc is not None:
        return None, "有位置但弯数量缺失"
    return None, "弯型标注缺失/不明确"


def parse_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else np.nan


def subgroup_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    acc = float(np.mean(y_true == y_pred))

    # Curve subgroups are patients only. For Task-4 severity, average OVR
    # sensitivity/specificity over patient severity classes actually present.
    present = sorted(c for c in np.unique(y_true).tolist() if c in (1,2,3))
    sens, spec, details = [], [], []
    for c in present:
        pos = y_true == c
        neg = ~pos
        pred_pos = y_pred == c
        tp = int(np.sum(pos & pred_pos))
        fn = int(np.sum(pos & ~pred_pos))
        fp = int(np.sum(neg & pred_pos))
        tn = int(np.sum(neg & ~pred_pos))
        se = tp/(tp+fn) if tp+fn else np.nan
        sp = tn/(tn+fp) if tn+fp else np.nan
        sens.append(se); spec.append(sp)
        details.append((c,tp,fn,fp,tn,se,sp))
    sens_macro = float(np.nanmean(sens)) if sens else np.nan
    spec_macro = float(np.nanmean(spec)) if spec else np.nan
    return acc, sens_macro, spec_macro, present, details


def pct(x):
    return "" if not np.isfinite(x) else f"{100*x:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_csv", required=True,
                    help="summary/07_subject_ensemble_predictions.csv")
    ap.add_argument("--protocol", default="loso")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--missing_location_as_thoracic", action="store_true")
    args = ap.parse_args()

    src = Path(args.subject_csv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    req = {"task","protocol","subject_key","name","center","cobb_angle",
           "curve_number","curve1","y_true","y_pred","correct"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise RuntimeError(f"CSV缺少列: {miss}; 实际列={list(df.columns)}")

    # Use Task 4 LOSO subject-level 3-seed ensemble.
    df = df[(df.task.astype(int)==4) &
            (df.protocol.astype(str).str.lower()==args.protocol.lower())].copy()
    if df.empty:
        raise RuntimeError("没有找到 Task4 + 指定 protocol 的受试者结果")
    if df.subject_key.duplicated().any():
        raise RuntimeError("Subject ensemble 文件中同一 subject_key 出现重复行")

    groups, reasons = [], []
    for _, r in df.iterrows():
        g, why = curve_group(r.curve_number, r.curve1,
                             args.missing_location_as_thoracic)
        groups.append(g); reasons.append(why)
    df["弯曲位置"] = groups
    df["排除原因"] = reasons
    df["Cobb_deg"] = df.cobb_angle.map(parse_float)

    inc = df[df["弯曲位置"].isin(GROUP_ORDER)].copy()
    exc = df[~df["弯曲位置"].isin(GROUP_ORDER)].copy()

    rows, pc_rows = [], []
    for grp in GROUP_ORDER:
        g = inc[inc["弯曲位置"]==grp]
        if len(g)==0:
            rows.append({"弯曲位置":grp,"样本数量":0,"平均Cobb角":"",
                         "识别Accuracy":"","Sensitivity":"","Specificity":""})
            continue
        yt = g.y_true.astype(int).to_numpy()
        yp = g.y_pred.astype(int).to_numpy()
        acc,se,sp,present,details = subgroup_metrics(yt,yp)
        cobb = g.Cobb_deg.dropna().to_numpy(float)
        mc = float(np.mean(cobb)) if len(cobb) else np.nan
        rows.append({
            "弯曲位置":grp,
            "样本数量":int(len(g)),
            "平均Cobb角":"" if not np.isfinite(mc) else f"{mc:.2f}°",
            "识别Accuracy":pct(acc),
            "Sensitivity":pct(se),
            "Specificity":pct(sp),
            "Accuracy_numeric":acc,
            "Sensitivity_numeric":se,
            "Specificity_numeric":sp,
            "Mean_Cobb_deg":mc,
            "Cobb_SD_deg":float(np.std(cobb,ddof=1)) if len(cobb)>1 else np.nan,
            "Present_severity_classes":"|".join(map(str,present)),
            "Cobb_available_N":int(len(cobb)),
        })
        for c,tp,fn,fp,tn,cse,csp in details:
            pc_rows.append({"弯曲位置":grp,"class_id":c,"TP":tp,"FN":fn,
                            "FP":fp,"TN":tn,"sensitivity":cse,"specificity":csp})

    table = pd.DataFrame(rows)
    main_cols = ["弯曲位置","样本数量","平均Cobb角","识别Accuracy","Sensitivity","Specificity"]
    table[main_cols].to_csv(out/"Table_S13.csv", index=False, encoding="utf-8-sig")
    table.to_csv(out/"Table_S13_audit.csv", index=False, encoding="utf-8-sig")
    inc.to_csv(out/"Table_S13_subject_details.csv", index=False, encoding="utf-8-sig")
    exc.to_csv(out/"Table_S13_excluded_subjects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pc_rows).to_csv(out/"Table_S13_per_class_diagnostics.csv", index=False, encoding="utf-8-sig")

    xlsx = out/"Table_S13_curve_group_analysis.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        table[main_cols].to_excel(w, sheet_name="Table_S13", index=False)
        table.to_excel(w, sheet_name="Audit", index=False)
        inc.to_excel(w, sheet_name="Subject_Details", index=False)
        exc.to_excel(w, sheet_name="Excluded_Subjects", index=False)
        pd.DataFrame(pc_rows).to_excel(w, sheet_name="Per_Class", index=False)

    print("\n===== Table S13 =====")
    print(table[main_cols].to_string(index=False))
    print(f"\nIncluded subjects: {len(inc)}")
    print(f"Excluded/unassigned subjects: {len(exc)}")
    print(f"[DONE] {xlsx}")

if __name__ == "__main__":
    main()
