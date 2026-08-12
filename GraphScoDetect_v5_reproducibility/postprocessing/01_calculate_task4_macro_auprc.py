#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Task-4 subject-level AUPRC / Macro-AUPRC from EXISTING experiment results.

No retraining is required if summary/07_subject_ensemble_predictions.csv exists,
because that file already contains the final subject-level 3-seed ensemble
probabilities.

For each protocol:
    subject_id
    true_label
    softmax_normal
    softmax_mild
    softmax_moderate
    softmax_severe

Macro-AUPRC:
    average_precision_score(
        y_true_onehot,
        y_prob,
        average="macro"
    )

Also reports per-class one-vs-rest AUPRC, Micro-AUPRC, and Weighted-AUPRC.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

PROTOCOL_ORDER = ["gkf3", "gkf5", "gkf7", "gkf10", "loso", "loco"]
CLASS_IDS = [0, 1, 2, 3]
CLASS_NAMES = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}


def parse_prob(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr if arr.ndim == 1 else None

    s = str(x).strip()
    if not s:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            arr = np.asarray(parser(s), dtype=float)
            if arr.ndim == 1:
                return arr
        except Exception:
            pass
    return None


def first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_subject_id(row: pd.Series) -> str:
    for col in ["subject_id", "subject_key", "name"]:
        if col not in row.index:
            continue
        value = row[col]
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text:
            return text
    return ""


def prepare_task4_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    prob_col = first_existing(df, ["ensemble_prob", "prob", "mean_prob"])
    if prob_col is None:
        raise RuntimeError(
            "Cannot find ensemble_prob / prob / mean_prob in "
            "07_subject_ensemble_predictions.csv"
        )

    rows = []
    for _, r in df.iterrows():
        p = parse_prob(r[prob_col])
        if p is None:
            raise RuntimeError(f"Cannot parse probability for {get_subject_id(r)}")
        if len(p) != 4:
            raise RuntimeError(
                f"Task 4 probability must have 4 values, got {len(p)} "
                f"for {get_subject_id(r)}"
            )
        if not np.all(np.isfinite(p)):
            raise RuntimeError(f"NaN/Inf probability for {get_subject_id(r)}")

        y_true = int(r["y_true"])
        row = {
            "protocol": str(r["protocol"]).lower(),
            "subject_id": get_subject_id(r),
            "true_label": y_true,
            "true_label_name": CLASS_NAMES[y_true],
            "softmax_normal": float(p[0]),
            "softmax_mild": float(p[1]),
            "softmax_moderate": float(p[2]),
            "softmax_severe": float(p[3]),
        }

        for col in [
            "subject_key", "name", "center", "cobb_angle",
            "curve_number", "curve1", "y_pred", "n_seeds",
            "n_samples", "n_segments",
        ]:
            if col in r.index:
                row[col] = r[col]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_protocol_auprc(g: pd.DataFrame) -> dict:
    y_true = g["true_label"].astype(int).to_numpy()
    y_prob = g[
        ["softmax_normal", "softmax_mild", "softmax_moderate", "softmax_severe"]
    ].to_numpy(dtype=float)

    y_true_onehot = label_binarize(y_true, classes=CLASS_IDS)

    if y_true_onehot.shape != y_prob.shape:
        raise RuntimeError(
            f"Shape mismatch: y_true_onehot={y_true_onehot.shape}, "
            f"y_prob={y_prob.shape}"
        )

    macro_auprc = float(
        average_precision_score(y_true_onehot, y_prob, average="macro")
    )
    micro_auprc = float(
        average_precision_score(y_true_onehot, y_prob, average="micro")
    )
    weighted_auprc = float(
        average_precision_score(y_true_onehot, y_prob, average="weighted")
    )

    row = {
        "protocol": str(g["protocol"].iloc[0]),
        "n_subjects": int(len(g)),
        "macro_auprc": macro_auprc,
        "micro_auprc": micro_auprc,
        "weighted_auprc": weighted_auprc,
    }

    for c in CLASS_IDS:
        y_binary = y_true_onehot[:, c]
        support = int(y_binary.sum())
        row[f"{CLASS_NAMES[c]}_support"] = support
        if support == 0:
            row[f"auprc_{CLASS_NAMES[c]}"] = np.nan
        else:
            row[f"auprc_{CLASS_NAMES[c]}"] = float(
                average_precision_score(y_binary, y_prob[:, c])
            )

    row["all_four_classes_present"] = bool(
        all(row[f"{CLASS_NAMES[c]}_support"] > 0 for c in CLASS_IDS)
    )
    return row


def compute_random_baselines(g: pd.DataFrame) -> pd.DataFrame:
    n = len(g)
    rows = []
    for c in CLASS_IDS:
        support = int((g["true_label"].astype(int) == c).sum())
        prevalence = support / n if n else np.nan
        rows.append({
            "protocol": str(g["protocol"].iloc[0]),
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "support": support,
            "prevalence": prevalence,
            "random_auprc_baseline": prevalence,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protocols", nargs="+", default=PROTOCOL_ORDER)
    ap.add_argument("--task", type=int, default=4, choices=[4])
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = summary_dir / "07_subject_ensemble_predictions.csv"
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    required = {"task", "protocol", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    df = df[df["task"].astype(int).eq(args.task)].copy()
    df["protocol"] = df["protocol"].astype(str).str.lower()

    protocols = [str(p).lower() for p in args.protocols]
    df = df[df["protocol"].isin(protocols)].copy()
    if df.empty:
        raise RuntimeError(f"No Task-4 rows for protocols: {protocols}")

    prob_df = prepare_task4_probabilities(df)

    prob_dir = out_dir / "subject_probabilities"
    prob_dir.mkdir(exist_ok=True)

    metric_rows = []
    baseline_frames = []

    for protocol in protocols:
        g = prob_df[prob_df["protocol"].eq(protocol)].copy()
        if g.empty:
            print(f"[WARN] No Task-4 rows for {protocol}")
            continue

        g.to_csv(
            prob_dir / f"task4_{protocol}_subject_softmax.csv",
            index=False,
            encoding="utf-8-sig",
        )

        metric_rows.append(compute_protocol_auprc(g))
        baseline_frames.append(compute_random_baselines(g))

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        order_map = {p: i for i, p in enumerate(PROTOCOL_ORDER)}
        metrics["_order"] = metrics["protocol"].map(order_map)
        metrics = metrics.sort_values("_order").drop(columns="_order")

    baselines = (
        pd.concat(baseline_frames, ignore_index=True)
        if baseline_frames else pd.DataFrame()
    )

    metrics.to_csv(
        out_dir / "01_task4_auprc_by_protocol.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prob_df.to_csv(
        out_dir / "02_all_task4_subject_softmax.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baselines.to_csv(
        out_dir / "03_per_class_random_auprc_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )

    loso = metrics[metrics["protocol"].eq("loso")].copy()
    loso.to_csv(
        out_dir / "04_task4_loso_auprc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    xlsx = out_dir / "task4_auprc_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="AUPRC_By_Protocol", index=False)
        loso.to_excel(writer, sheet_name="LOSO_AUPRC", index=False)
        prob_df.to_excel(writer, sheet_name="All_Subject_Softmax", index=False)
        baselines.to_excel(writer, sheet_name="Random_Baseline", index=False)
        for protocol in protocols:
            g = prob_df[prob_df["protocol"].eq(protocol)]
            if not g.empty:
                g.to_excel(
                    writer,
                    sheet_name=f"{protocol}_softmax"[:31],
                    index=False,
                )

    print("\n===== TASK 4 SUBJECT-LEVEL AUPRC =====")
    show_cols = [
        "protocol", "n_subjects", "macro_auprc",
        "auprc_normal", "auprc_mild",
        "auprc_moderate", "auprc_severe",
        "micro_auprc", "weighted_auprc",
    ]
    print(metrics[show_cols].to_string(index=False))

    print("\n===== LOSO =====")
    if len(loso):
        r = loso.iloc[0]
        print(f"N = {int(r['n_subjects'])}")
        print(f"Macro-AUPRC = {r['macro_auprc']:.6f}")
        print(f"Normal AUPRC = {r['auprc_normal']:.6f}")
        print(f"Mild AUPRC = {r['auprc_mild']:.6f}")
        print(f"Moderate AUPRC = {r['auprc_moderate']:.6f}")
        print(f"Severe AUPRC = {r['auprc_severe']:.6f}")
    else:
        print("No LOSO result.")

    print(f"\n[DONE] {xlsx}")


if __name__ == "__main__":
    main()
