#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize label_v3 experiments at subject level.

Main evaluation procedure:
1. Average segment probabilities within each subject.
2. Average subject probabilities across random seeds.
3. Compute subject-level metrics from the seed-ensemble predictions.

In addition to pooled protocol results, LOCO is reported separately for every held-out
center. Centers configured as train-only (default: 青海) are forbidden from LOCO test
predictions and therefore do not appear as LOCO test-center rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


PROTOCOL_ORDER = ["gkf3", "gkf5", "gkf7", "gkf10", "loso", "loco"]
TASK_NAMES = {2: "binary", 4: "severity_4class", 6: "primary_location_3class"}
CLASS_NAMES = {
    2: ["normal", "scoliosis"],
    4: ["normal", "mild", "moderate", "severe"],
    6: ["thoracic", "thoracolumbar", "lumbar"],
}


def safe_mean(values: Sequence[Any]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: int,
    y_prob: np.ndarray | None = None,
):
    names = CLASS_NAMES[task]
    labels = list(range(len(names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    per_class = []
    for i, name in enumerate(names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if np.isfinite(sensitivity) and (precision + sensitivity)
            else 0.0
        )
        per_class.append({
            "class_id": i,
            "class_name": name,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "support": int(tp + fn),
            "precision": float(precision),
            "sensitivity": None if not np.isfinite(sensitivity) else float(sensitivity),
            "specificity": None if not np.isfinite(specificity) else float(specificity),
            "f1": float(f1),
        })

    sensitivity_macro = safe_mean([r["sensitivity"] for r in per_class])
    specificity_macro = safe_mean([r["specificity"] for r in per_class])
    sensitivity = per_class[1]["sensitivity"] if task == 2 else sensitivity_macro
    specificity = per_class[1]["specificity"] if task == 2 else specificity_macro
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Probability-based AUROC.
    # Binary: scoliosis (class 1) is positive.
    # Multiclass: macro average of valid one-vs-rest per-class AUROCs.
    auroc_per_class = []
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim != 2 or y_prob.shape != (len(y_true), len(names)):
            raise ValueError(
                f"Probability shape mismatch: expected {(len(y_true), len(names))}, got {y_prob.shape}"
            )
        for i in labels:
            binary_true = (y_true == i).astype(int)
            if len(np.unique(binary_true)) < 2:
                auc = np.nan
            else:
                auc = float(roc_auc_score(binary_true, y_prob[:, i]))
            auroc_per_class.append(auc)
            per_class[i]["auroc_ovr"] = None if not np.isfinite(auc) else float(auc)
    else:
        auroc_per_class = [np.nan] * len(names)
        for i in labels:
            per_class[i]["auroc_ovr"] = None

    if task == 2:
        auroc = auroc_per_class[1]
    else:
        valid_auc = [x for x in auroc_per_class if np.isfinite(x)]
        auroc = float(np.mean(valid_auc)) if valid_auc else np.nan

    present_ids = sorted(set(int(x) for x in y_true.tolist()))
    missing_ids = [i for i in labels if i not in present_ids]
    present_names = [names[i] for i in present_ids]
    missing_names = [names[i] for i in missing_ids]
    warning = ""
    if missing_names:
        warning = (
            "Test subset has no true samples for: " + ", ".join(missing_names) +
            ". Interpret balanced accuracy, macro metrics, and AUROC with per-class support."
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity) if sensitivity is not None and np.isfinite(sensitivity) else np.nan,
        "specificity": float(specificity) if specificity is not None and np.isfinite(specificity) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
        "auroc_macro_ovr": float(auroc) if np.isfinite(auroc) else np.nan,
        "auroc_valid_class_count": int(sum(np.isfinite(x) for x in auroc_per_class)),
        "sensitivity_macro_ovr": sensitivity_macro,
        "specificity_macro_ovr": specificity_macro,
        "present_true_classes": "|".join(present_names),
        "missing_true_classes": "|".join(missing_names),
        "metric_warning": warning,
    }
    return metrics, per_class, cm, cm_norm

def load_predictions(result_root: Path):
    sample_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    for path in sorted(result_root.glob("task*/**/predictions.json")):
        parts = path.relative_to(result_root).parts
        task = int(parts[0].replace("task", ""))
        protocol = parts[1]
        seed = int(parts[2].replace("seed_", ""))
        fold = parts[3].replace("fold_", "")
        obj = json.loads(path.read_text(encoding="utf-8"))
        top_test_center = str(obj.get("test_center") or "").strip()

        for r in obj.get("predictions", []):
            center = str(r.get("center") or "").strip()
            test_center = top_test_center or (center if protocol == "loco" else "")
            sample_rows.append({
                **r,
                "task": task,
                "protocol": protocol,
                "seed": seed,
                "fold": fold,
                "test_center": test_center,
            })

        for r in obj.get("subject_predictions", []):
            center = str(r.get("center") or "").strip()
            test_center = top_test_center or (center if protocol == "loco" else "")
            subject_rows.append({
                **r,
                "task": task,
                "protocol": protocol,
                "seed": seed,
                "fold": fold,
                "test_center": test_center,
            })

        metrics_path = path.with_name("metrics.json")
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_center = str(m.get("test_center") or top_test_center or "").strip()
            base = {
                "task": task,
                "protocol": protocol,
                "seed": seed,
                "fold": fold,
                "test_center": test_center,
                "loco_train_only_centers": "|".join(m.get("loco_train_only_centers") or []),
                "n_train_samples": m.get("n_train_samples"),
                "n_test_samples": m.get("n_test_samples"),
                "n_train_subjects": m.get("n_train_subjects"),
                "n_test_subjects": m.get("n_test_subjects"),
                "epochs": m.get("epochs"),
                "batch_size": m.get("batch_size"),
                "balanced_loss": m.get("balanced_loss"),
                "elapsed_seconds": m.get("elapsed_seconds"),
            }
            for level in ["sample_metrics", "subject_metrics"]:
                flat = {
                    k: v for k, v in (m.get(level) or {}).items()
                    if not isinstance(v, (list, dict))
                }
                run_rows.append({**base, "level": level.replace("_metrics", ""), **flat})

    if not subject_rows:
        raise RuntimeError(f"No predictions.json found under {result_root}")
    return pd.DataFrame(sample_rows), pd.DataFrame(subject_rows), pd.DataFrame(run_rows)


def parse_prob(x: Any) -> np.ndarray:
    if isinstance(x, str):
        return np.asarray(json.loads(x), dtype=float)
    return np.asarray(x, dtype=float)


def validate_out_of_fold(subject_df: pd.DataFrame, train_only_centers: Sequence[str]):
    duplicate = subject_df.groupby(["task", "protocol", "seed", "subject_key"]).size()
    bad = duplicate[duplicate != 1]
    if len(bad):
        raise RuntimeError(
            "A subject appears more than once within a task/protocol/seed. "
            f"Examples: {bad.head(10).to_dict()}"
        )

    forbidden = {str(x).strip() for x in train_only_centers if str(x).strip()}
    if forbidden:
        loco = subject_df[subject_df["protocol"] == "loco"].copy()
        tested = set(loco["test_center"].fillna("").astype(str).str.strip())
        leaked = forbidden & tested
        if leaked:
            raise RuntimeError(
                f"Train-only center(s) unexpectedly appear in LOCO test predictions: {sorted(leaked)}"
            )
        center_leak = loco[loco["center"].fillna("").astype(str).str.strip().isin(forbidden)]
        if len(center_leak):
            raise RuntimeError(
                "LOCO contains subjects from train-only centers in its test predictions. "
                f"Examples: {center_leak[['task','seed','fold','name','center']].head().to_dict('records')}"
            )


def metrics_by_seed(subject_df: pd.DataFrame):
    rows = []
    per_class_rows = []
    cm_rows = []
    for (task, protocol, seed), g in subject_df.groupby(["task", "protocol", "seed"], sort=False):
        task = int(task)
        yt = g["y_true"].astype(int).to_numpy()
        yp = g["y_pred"].astype(int).to_numpy()
        yprob = np.stack([parse_prob(x) for x in g["prob"]])
        metrics, per_class, cm, cm_norm = metric_bundle(yt, yp, task, yprob)
        rows.append({
            "task": task,
            "task_name": TASK_NAMES[task],
            "protocol": protocol,
            "seed": int(seed),
            "n_subjects": len(g),
            **metrics,
        })
        for r in per_class:
            per_class_rows.append({"task": task, "protocol": protocol, "seed": int(seed), **r})
        names = CLASS_NAMES[task]
        for i, true_name in enumerate(names):
            for j, pred_name in enumerate(names):
                cm_rows.append({
                    "task": task,
                    "protocol": protocol,
                    "seed": int(seed),
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(cm[i, j]),
                    "row_normalized": float(cm_norm[i, j]),
                })
    return pd.DataFrame(rows), pd.DataFrame(per_class_rows), pd.DataFrame(cm_rows)


def build_seed_ensemble(subject_df: pd.DataFrame):
    rows = []
    identity_cols = [
        "name", "center", "test_center", "cobb_angle", "curve_number", "curve1", "note", "n_samples"
    ]
    for (task, protocol, key), g in subject_df.groupby(["task", "protocol", "subject_key"], sort=False):
        truths = sorted(set(g["y_true"].astype(int).tolist()))
        if len(truths) != 1:
            raise RuntimeError(
                f"Inconsistent labels for task={task}, protocol={protocol}, subject={key}: {truths}"
            )
        probs = np.stack([parse_prob(x) for x in g["prob"]])
        p = probs.mean(axis=0)
        first = g.iloc[0]
        pred = int(p.argmax())
        row = {
            "task": int(task),
            "task_name": TASK_NAMES[int(task)],
            "protocol": protocol,
            "subject_key": key,
            "n_seeds": int(g["seed"].nunique()),
            "y_true": truths[0],
            "y_pred": pred,
            "correct": int(pred == truths[0]),
            "ensemble_prob": p.tolist(),
        }
        for col in identity_cols:
            row[col] = first.get(col, "")
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_tables(ensemble_df: pd.DataFrame):
    main_rows = []
    class_rows = []
    cm_count_rows = []
    cm_norm_rows = []
    for (task, protocol), g in ensemble_df.groupby(["task", "protocol"], sort=False):
        task = int(task)
        yt = g["y_true"].astype(int).to_numpy()
        yp = g["y_pred"].astype(int).to_numpy()
        yprob = np.stack([parse_prob(x) for x in g["ensemble_prob"]])
        metrics, per_class, cm, cm_norm = metric_bundle(yt, yp, task, yprob)
        main_rows.append({
            "task": task,
            "task_name": TASK_NAMES[task],
            "protocol": protocol,
            "n_subjects": len(g),
            "n_seeds_min": int(g["n_seeds"].min()),
            "n_seeds_max": int(g["n_seeds"].max()),
            **metrics,
        })
        for r in per_class:
            class_rows.append({
                "task": task,
                "task_name": TASK_NAMES[task],
                "protocol": protocol,
                **r,
            })
        names = CLASS_NAMES[task]
        for i, true_name in enumerate(names):
            count_row = {
                "task": task,
                "task_name": TASK_NAMES[task],
                "protocol": protocol,
                "true_class": true_name,
            }
            norm_row = dict(count_row)
            for j, pred_name in enumerate(names):
                count_row[f"pred_{pred_name}"] = int(cm[i, j])
                norm_row[f"pred_{pred_name}"] = float(cm_norm[i, j])
            cm_count_rows.append(count_row)
            cm_norm_rows.append(norm_row)

    main = pd.DataFrame(main_rows)
    main["protocol_order"] = main["protocol"].map({p: i for i, p in enumerate(PROTOCOL_ORDER)})
    main = main.sort_values(["task", "protocol_order"]).drop(columns="protocol_order")
    return main, pd.DataFrame(class_rows), pd.DataFrame(cm_count_rows), pd.DataFrame(cm_norm_rows)


def loco_center_tables(subject_df: pd.DataFrame, ensemble_df: pd.DataFrame):
    loco_seed = subject_df[subject_df["protocol"] == "loco"].copy()
    loco_ensemble = ensemble_df[ensemble_df["protocol"] == "loco"].copy()
    if loco_ensemble.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    # Ensure old-format result files still obtain a center key.
    loco_seed["test_center"] = loco_seed["test_center"].fillna("").astype(str).str.strip()
    loco_ensemble["test_center"] = loco_ensemble["test_center"].fillna("").astype(str).str.strip()
    loco_seed.loc[loco_seed["test_center"] == "", "test_center"] = loco_seed.loc[
        loco_seed["test_center"] == "", "center"
    ]
    loco_ensemble.loc[loco_ensemble["test_center"] == "", "test_center"] = loco_ensemble.loc[
        loco_ensemble["test_center"] == "", "center"
    ]

    center_seed_rows = []
    for (task, center, seed), g in loco_seed.groupby(["task", "test_center", "seed"], sort=False):
        task = int(task)
        yt = g["y_true"].astype(int).to_numpy()
        yp = g["y_pred"].astype(int).to_numpy()
        yprob = np.stack([parse_prob(x) for x in g["prob"]])
        metrics, _, _, _ = metric_bundle(yt, yp, task, yprob)
        center_seed_rows.append({
            "task": task,
            "task_name": TASK_NAMES[task],
            "test_center": center,
            "seed": int(seed),
            "n_subjects": len(g),
            **metrics,
        })

    center_rows = []
    class_rows = []
    cm_count_rows = []
    cm_norm_rows = []
    for (task, center), g in loco_ensemble.groupby(["task", "test_center"], sort=False):
        task = int(task)
        yt = g["y_true"].astype(int).to_numpy()
        yp = g["y_pred"].astype(int).to_numpy()
        yprob = np.stack([parse_prob(x) for x in g["ensemble_prob"]])
        metrics, per_class, cm, cm_norm = metric_bundle(yt, yp, task, yprob)
        center_rows.append({
            "task": task,
            "task_name": TASK_NAMES[task],
            "test_center": center,
            "n_subjects": len(g),
            "n_seeds_min": int(g["n_seeds"].min()),
            "n_seeds_max": int(g["n_seeds"].max()),
            **metrics,
        })
        for r in per_class:
            class_rows.append({
                "task": task,
                "task_name": TASK_NAMES[task],
                "test_center": center,
                **r,
            })
        names = CLASS_NAMES[task]
        for i, true_name in enumerate(names):
            count_row = {
                "task": task,
                "task_name": TASK_NAMES[task],
                "test_center": center,
                "true_class": true_name,
            }
            norm_row = dict(count_row)
            for j, pred_name in enumerate(names):
                count_row[f"pred_{pred_name}"] = int(cm[i, j])
                norm_row[f"pred_{pred_name}"] = float(cm_norm[i, j])
            cm_count_rows.append(count_row)
            cm_norm_rows.append(norm_row)

    center_df = pd.DataFrame(center_rows).sort_values(["task", "test_center"])
    center_seed_df = pd.DataFrame(center_seed_rows).sort_values(["task", "test_center", "seed"])
    return (
        center_df,
        center_seed_df,
        pd.DataFrame(class_rows),
        pd.DataFrame(cm_count_rows),
        pd.DataFrame(cm_norm_rows),
    )


def seed_mean_std(seed_df: pd.DataFrame):
    metric_cols = [
        "accuracy", "sensitivity", "specificity", "balanced_accuracy", "micro_f1",
        "macro_f1", "weighted_f1", "auroc", "sensitivity_macro_ovr", "specificity_macro_ovr",
    ]
    agg = seed_df.groupby(["task", "task_name", "protocol"])[metric_cols].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    agg.columns = [
        "_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c)
        for c in agg.columns
    ]
    n = seed_df.groupby(["task", "task_name", "protocol"])["seed"].nunique().reset_index(name="n_seeds")
    return agg.merge(n, on=["task", "task_name", "protocol"], how="left")


def completeness(subject_df: pd.DataFrame):
    rows = []
    for (task, protocol), g in subject_df.groupby(["task", "protocol"]):
        seed_counts = g.groupby("seed")["fold"].nunique()
        centers = sorted(set(g["test_center"].fillna("").astype(str)) - {""})
        rows.append({
            "task": int(task),
            "task_name": TASK_NAMES[int(task)],
            "protocol": protocol,
            "n_seeds": int(g["seed"].nunique()),
            "folds_min": int(seed_counts.min()),
            "folds_max": int(seed_counts.max()),
            "n_subject_predictions": len(g),
            "n_unique_subjects": int(g["subject_key"].nunique()),
            "test_centers": "|".join(centers),
            "complete_same_folds_across_seeds": bool(seed_counts.min() == seed_counts.max()),
        })
    return pd.DataFrame(rows)


def write_excel(path: Path, sheets: List[Tuple[str, pd.DataFrame]]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col_cells in ws.columns:
                max_len = 0
                for cell in col_cells[: min(ws.max_row, 300)]:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(value))
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 42)
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_xlsx", default="new_label_v3_selected_protocol_summary.xlsx")
    ap.add_argument("--loco_train_only_centers", nargs="*", default=["青海"])
    args = ap.parse_args()

    result_root = Path(args.result_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_df, subject_df, fold_metric_df = load_predictions(result_root)
    validate_out_of_fold(subject_df, args.loco_train_only_centers)
    seed_df, seed_class_df, seed_cm_df = metrics_by_seed(subject_df)
    ensemble_df = build_seed_ensemble(subject_df)
    main_df, ensemble_class_df, cm_count_df, cm_norm_df = ensemble_tables(ensemble_df)
    seed_summary_df = seed_mean_std(seed_df)
    completeness_df = completeness(subject_df)
    (
        loco_center_df,
        loco_center_seed_df,
        loco_center_class_df,
        loco_center_cm_df,
        loco_center_cm_norm_df,
    ) = loco_center_tables(subject_df, ensemble_df)

    outputs = {
        "01_main_subject_ensemble_metrics.csv": main_df,
        "02_metrics_by_seed.csv": seed_df,
        "03_seed_mean_std.csv": seed_summary_df,
        "04_per_class_subject_ensemble.csv": ensemble_class_df,
        "05_confusion_matrix_counts.csv": cm_count_df,
        "06_confusion_matrix_row_normalized.csv": cm_norm_df,
        "07_subject_ensemble_predictions.csv": ensemble_df,
        "08_subject_predictions_by_seed.csv": subject_df,
        "09_fold_metrics.csv": fold_metric_df,
        "10_run_completeness.csv": completeness_df,
        "11_sample_predictions.csv": sample_df,
        "12_per_class_by_seed.csv": seed_class_df,
        "13_confusion_matrix_by_seed_long.csv": seed_cm_df,
        "14_loco_per_center_ensemble_metrics.csv": loco_center_df,
        "15_loco_per_center_metrics_by_seed.csv": loco_center_seed_df,
        "16_loco_per_center_per_class.csv": loco_center_class_df,
        "17_loco_per_center_confusion_counts.csv": loco_center_cm_df,
        "18_loco_per_center_confusion_normalized.csv": loco_center_cm_norm_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")

    xlsx = out_dir / args.out_xlsx
    write_excel(xlsx, [
        ("Main_Subject_Ensemble", main_df),
        ("Metrics_By_Seed", seed_df),
        ("Seed_Mean_Std", seed_summary_df),
        ("Per_Class_Ensemble", ensemble_class_df),
        ("CM_Counts", cm_count_df),
        ("CM_Row_Normalized", cm_norm_df),
        ("Subject_Ensemble", ensemble_df),
        ("Subject_By_Seed", subject_df),
        ("Fold_Metrics", fold_metric_df),
        ("Run_Completeness", completeness_df),
        ("Sample_Predictions", sample_df),
        ("Per_Class_By_Seed", seed_class_df),
        ("CM_By_Seed_Long", seed_cm_df),
        ("LOCO_Center_Ensemble", loco_center_df),
        ("LOCO_Center_By_Seed", loco_center_seed_df),
        ("LOCO_Center_Per_Class", loco_center_class_df),
        ("LOCO_Center_CM_Count", loco_center_cm_df),
        ("LOCO_Center_CM_Norm", loco_center_cm_norm_df),
    ])

    print(f"[DONE] Summary workbook: {xlsx}")
    print("\n===== MAIN SUBJECT-LEVEL SEED ENSEMBLE =====")
    print(main_df.to_string(index=False))
    print("\n===== LOCO RESULTS BY HELD-OUT CENTER =====")
    print(loco_center_df.to_string(index=False) if len(loco_center_df) else "No LOCO results found.")
    print(f"\n[LOCO RULE] train-only centers: {args.loco_train_only_centers}; these centers are not tested.")


if __name__ == "__main__":
    main()
