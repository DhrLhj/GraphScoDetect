#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Task 2, Task 4, and Task 6 datasets from the label_v3 clinical sheet.

The standardized signal tensor is reused from an existing Task-4 dataset. Labels are read directly from the supplied Excel file after case/whitespace normalization; Cobb angle is used only for auditing and never overwrites the supplied label. The original source data are never
overwritten.

Outputs under OUT_DATA_ROOT:
  2/data_binary.npy, label_binary.npy, names_binary.npy, dataset_binary.json
  4/data_4class.npy, label_4class.npy, names_4class.npy, dataset_4class.json
  6/data_curve_type.npy, label_curve_type.npy, names_curve_type.npy,
    dataset_curve_type.json
  audits/*.csv and preparation_summary.json

Task definitions:
  Task 2: 0 normal/control, 1 scoliosis/patient
  Task 4: 0 normal/control, 1 mild, 2 moderate, 3 severe
  Task 6: 0 thoracic, 1 thoracolumbar, 2 lumbar, using explicit 弯1位置
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BINARY_MAP = {
    "normal/control": 0,
    "normal": 0,
    "control": 0,
    "scoliosis/patient": 1,
    "scoliosis": 1,
    "patient": 1,
}
FOUR_MAP = {
    "normal/control": 0,
    "normal": 0,
    "control": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}
LOCATION_MAP = {
    "胸弯": 0,
    "胸腰弯": 1,
    "腰弯": 2,
}
LABEL_NAMES = {
    2: ["normal", "scoliosis"],
    4: ["normal", "mild", "moderate", "severe"],
    6: ["thoracic", "thoracolumbar", "lumbar"],
}

CENTER_ALIASES = {
    "北京协和医院": "协和医院",
    "协和": "协和医院",
    "协和医院": "协和医院",
    "浙江大学医学院附属第一医院": "浙大一院",
    "浙江大学附属第一医院": "浙大一院",
    "浙大": "浙大一院",
    "浙大一院": "浙大一院",
    "潍坊市人民": "潍坊市人民医院",
    "潍坊人民医院": "潍坊市人民医院",
    "潍坊市人民医院": "潍坊市人民医院",
    "优联": "优联医院",
    "优联医院": "优联医院",
    "青海": "青海",
    "北科大": "北科大",
    "学校": "北科大",
}
NAME_ALIASES = {
}

CENTER_SUFFIXES = {
    "协和医院": ["协和医院", "协和"],
    "浙大一院": ["浙大一院", "浙大"],
    "潍坊市人民医院": ["潍坊市人民医院", "潍坊市人民", "潍坊"],
    "优联医院": ["优联医院", "优联"],
    "青海": ["青海"],
    "北科大": ["北科大", "学校"],
}


def is_missing(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return x is None


def clean_text(x: Any) -> str:
    if is_missing(x):
        return ""
    return str(x).replace("\u3000", " ").strip()


def norm_name(x: Any) -> str:
    return re.sub(r"\s+", "", clean_text(x))


def norm_center(x: Any) -> str:
    s = re.sub(r"\s+", "", clean_text(x))
    return CENTER_ALIASES.get(s, s)


def parse_float(x: Any) -> Optional[float]:
    if is_missing(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None


def normalize_label(raw: Any, mapping: Dict[str, int]) -> Optional[int]:
    s = clean_text(raw).lower()
    return mapping.get(s)


def normalize_location(raw: Any) -> Optional[int]:
    s = clean_text(raw)
    if s in LOCATION_MAP:
        return LOCATION_MAP[s]
    return None


def detect_note_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["Remark", "Remarks", "Note", "Notes", "备注", "说明", "Curve_summary"]
    for col in preferred:
        if col in df.columns:
            return col
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        return unnamed[-1]
    return None


def choose_sheet(excel: Path, requested: str) -> str:
    xls = pd.ExcelFile(excel)
    if requested and requested.lower() != "auto":
        if requested not in xls.sheet_names:
            raise ValueError(f"Sheet {requested!r} not found. Available: {xls.sheet_names}")
        return requested
    candidates = [s for s in xls.sheet_names if "Subject" in s or "明细" in s or "Clinical" in s]
    return candidates[0] if candidates else xls.sheet_names[0]


def load_clinical_records(excel: Path, sheet: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    df = pd.read_excel(excel, sheet_name=sheet)
    required = {
        "Name", "Center", "Cobb angle", "Binary label", "Four-class label",
        "Samples", "弯数量", "弯1位置", "弯2位置", "弯3位置",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Label sheet missing columns {sorted(missing)}. Available: {list(df.columns)}")

    note_col = detect_note_column(df)
    records: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        name = norm_name(row["Name"])
        if not name:
            continue
        center = norm_center(row["Center"])
        note = clean_text(row[note_col]) if note_col else ""
        records.append({
            "excel_row": int(row_index + 2),
            "name": name,
            "display_name": clean_text(row["Name"]),
            "center": center,
            "sex": clean_text(row.get("Sex")),
            "age": parse_float(row.get("Age")),
            "cobb_angle": parse_float(row["Cobb angle"]),
            "binary_raw": clean_text(row["Binary label"]),
            "binary_label": normalize_label(row["Binary label"], BINARY_MAP),
            "four_raw": clean_text(row["Four-class label"]),
            "four_label": normalize_label(row["Four-class label"], FOUR_MAP),
            "declared_samples": int(row["Samples"]) if not is_missing(row["Samples"]) else None,
            "curve_number": clean_text(row["弯数量"]),
            "curve1": clean_text(row["弯1位置"]),
            "curve2": clean_text(row["弯2位置"]),
            "curve3": clean_text(row["弯3位置"]),
            "location_label": normalize_location(row["弯1位置"]),
            "note": note,
        })
    return df, records, note_col or ""


def read_source(source_root: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    root = source_root / "4"
    data_path = root / "data_4class.npy"
    names_path = root / "names_4class.npy"
    json_path = root / "dataset_4class.json"
    for p in (data_path, names_path, json_path):
        if not p.exists():
            raise FileNotFoundError(f"Required source file not found: {p}")

    X = np.load(data_path)
    names = np.load(names_path, allow_pickle=True).astype(str)
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    samples = obj.get("samples")
    if not isinstance(samples, list) or len(samples) != len(X):
        samples = [{} for _ in range(len(X))]
    if len(names) != len(X):
        raise RuntimeError(f"Source length mismatch: X={len(X)}, names={len(names)}")
    return X, names, samples, obj


def build_record_indexes(records: Sequence[Dict[str, Any]]):
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_key[(rec["name"], rec["center"])].append(rec)
        by_name[rec["name"]].append(rec)
    return by_key, by_name


def resolve_record(
    source_name: str,
    source_center: str,
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    by_name: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    name = norm_name(source_name)
    name = NAME_ALIASES.get(name, name)
    center = norm_center(source_center)

    exact = by_key.get((name, center), []) if center else []
    if len(exact) == 1:
        return exact[0], "exact_name_center"
    if len(exact) > 1:
        return None, "ambiguous_exact_name_center"

    direct = by_name.get(name, [])
    if len(direct) == 1:
        return direct[0], "unique_name"
    if len(direct) > 1 and center:
        centered = [r for r in direct if r["center"] == center]
        if len(centered) == 1:
            return centered[0], "duplicate_name_center"

    # Legacy names sometimes append a center suffix, e.g. 刘玥彤协和.
    suffix_matches = []
    for base_name, candidates in by_name.items():
        if not name.startswith(base_name) or name == base_name:
            continue
        suffix = name[len(base_name):]
        for rec in candidates:
            if suffix in CENTER_SUFFIXES.get(rec["center"], []):
                suffix_matches.append(rec)
    if len(suffix_matches) == 1:
        return suffix_matches[0], "name_with_center_suffix"
    if len(suffix_matches) > 1:
        return None, "ambiguous_suffix_match"

    return None, "unmatched"


def sample_identity(sample: Dict[str, Any], fallback_name: str, index: int) -> Dict[str, str]:
    return {
        "source_name": clean_text(sample.get("name")) or clean_text(fallback_name),
        "source_center": norm_center(sample.get("center")),
        "source_subject_id": clean_text(sample.get("subject_id")) or norm_name(fallback_name),
        "sample_id": clean_text(sample.get("sample_id")) or str(index),
        "source_dataset": clean_text(sample.get("source_dataset")),
    }


def cobb_rule_25_moderate(cobb: Optional[float]) -> Optional[int]:
    """Audit rule A: <10 normal, [10,25) mild, [25,45) moderate, >=45 severe."""
    if cobb is None:
        return None
    if cobb < 10:
        return 0
    if cobb < 25:
        return 1
    if cobb < 45:
        return 2
    return 3


def cobb_rule_25_mild(cobb: Optional[float]) -> Optional[int]:
    """Audit rule B: <10 normal, [10,25] mild, (25,45] moderate, >45 severe."""
    if cobb is None:
        return None
    if cobb < 10:
        return 0
    if cobb <= 25:
        return 1
    if cobb <= 45:
        return 2
    return 3


def enrich_sample(sample: Dict[str, Any], rec: Dict[str, Any], subject_key: str) -> Dict[str, Any]:
    out = dict(sample)
    out["name"] = rec["display_name"]
    out["center"] = rec["center"]
    out["subject_key"] = subject_key
    out["label_v3_info"] = {
        "excel_row": rec["excel_row"],
        "binary_label": rec["binary_label"],
        "four_label": rec["four_label"],
        "location_label": rec["location_label"],
        "cobb_angle": rec["cobb_angle"],
        "curve_number": rec["curve_number"],
        "curve1": rec["curve1"],
        "curve2": rec["curve2"],
        "curve3": rec["curve3"],
        "note": rec["note"],
        "label_source": "uploaded_label_v3_excel",
    }
    info = dict(out.get("patient_info", {}) or {})
    info["cobb_label_angle"] = rec["cobb_angle"]
    info["remark"] = rec["note"]
    out["patient_info"] = info
    return out


def save_task(
    task: int,
    X: np.ndarray,
    names: np.ndarray,
    source_samples: List[Dict[str, Any]],
    source_obj: Dict[str, Any],
    selected: Sequence[int],
    labels: Sequence[int],
    matched_records: Sequence[Dict[str, Any]],
    out_root: Path,
    location_scope: str,
) -> Dict[str, Any]:
    selected_arr = np.asarray(selected, dtype=int)
    labels_arr = np.asarray(labels, dtype=np.int64)
    task_root = out_root / str(task)
    task_root.mkdir(parents=True, exist_ok=True)

    if task == 2:
        data_name, label_name, names_name, json_name = (
            "data_binary.npy", "label_binary.npy", "names_binary.npy", "dataset_binary.json"
        )
    elif task == 4:
        data_name, label_name, names_name, json_name = (
            "data_4class.npy", "label_4class.npy", "names_4class.npy", "dataset_4class.json"
        )
    elif task == 6:
        data_name, label_name, names_name, json_name = (
            "data_curve_type.npy", "label_curve_type.npy", "names_curve_type.npy", "dataset_curve_type.json"
        )
    else:
        raise ValueError(task)

    np.save(task_root / data_name, X[selected_arr].astype(np.float32, copy=False))
    np.save(task_root / label_name, labels_arr)

    selected_names = np.asarray([matched_records[i]["display_name"] for i in selected_arr], dtype=object)
    np.save(task_root / names_name, selected_names)
    np.save(task_root / "selected_source_indices.npy", selected_arr)

    selected_samples = []
    for source_idx in selected_arr:
        rec = matched_records[int(source_idx)]
        subject_key = f"{rec['name']}|{rec['center']}"
        selected_samples.append(enrich_sample(source_samples[int(source_idx)], rec, subject_key))

    out_obj = dict(source_obj)
    out_obj["samples"] = selected_samples
    out_obj["task_id"] = task
    out_obj["task_name"] = {
        2: "binary_scoliosis_classification_label_v3",
        4: "four_class_severity_classification_label_v3",
        6: "primary_curve_location_classification_label_v3",
    }[task]
    out_obj["label_names"] = {str(i): name for i, name in enumerate(LABEL_NAMES[task])}
    out_obj["source_task"] = 4
    out_obj["source_indices_file"] = "selected_source_indices.npy"
    out_obj["location_scope"] = location_scope if task == 6 else None
    with (task_root / json_name).open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    subjects = {s["subject_key"] for s in selected_samples}
    centers = Counter(s.get("center", "") for s in selected_samples)
    return {
        "task": task,
        "n_samples": int(len(selected_arr)),
        "n_subjects": int(len(subjects)),
        "label_counts_sample": {str(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
        "center_counts_sample": dict(centers),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_excel", required=True)
    ap.add_argument("--label_sheet", default="auto")
    ap.add_argument("--source_data_root", required=True,
                    help="QC-filtered data root containing 4/data_4class.npy and dataset_4class.json")
    ap.add_argument("--out_data_root", required=True)
    ap.add_argument("--location_scope", choices=["primary_all", "single_curve_only"], default="primary_all")
    ap.add_argument("--exclude_note_keywords", nargs="*", default=["不要数据"],
                    help="Subjects whose note contains any keyword are excluded from all training and testing tasks.")
    ap.add_argument("--exclude_all_postop", action="store_true",
                    help="Also exclude any row whose note contains 术后.")
    ap.add_argument("--allow_unmatched", action="store_true")
    ap.add_argument("--expected_source_samples", type=int, default=0, help="Optional expected source-sample count; 0 disables fixed-count warnings.")
    ap.add_argument("--copy_label_excel", action="store_true")
    args = ap.parse_args()

    label_excel = Path(args.label_excel).resolve()
    source_root = Path(args.source_data_root).resolve()
    out_root = Path(args.out_data_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    audit_root = out_root / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)

    sheet = choose_sheet(label_excel, args.label_sheet)
    _, clinical, note_col = load_clinical_records(label_excel, sheet)
    X, source_names, source_samples, source_obj = read_source(source_root)

    if args.expected_source_samples > 0 and len(X) != args.expected_source_samples:
        print(f"[WARN] Expected {args.expected_source_samples} source samples, found {len(X)}.")

    by_key, by_name = build_record_indexes(clinical)
    matched_records: List[Optional[Dict[str, Any]]] = [None] * len(X)
    audit_rows = []
    for i in range(len(X)):
        ident = sample_identity(source_samples[i], source_names[i], i)
        rec, method = resolve_record(
            ident["source_name"], ident["source_center"], by_key=by_key, by_name=by_name
        )
        matched_records[i] = rec
        audit_rows.append({
            "source_index": i,
            **ident,
            "match_method": method,
            "matched": int(rec is not None),
            "label_name": rec["display_name"] if rec else "",
            "label_center": rec["center"] if rec else "",
            "label_excel_row": rec["excel_row"] if rec else "",
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_root / "01_source_to_label_matching.csv", index=False, encoding="utf-8-sig")
    unmatched = audit_df[audit_df["matched"] == 0]
    if len(unmatched) and not args.allow_unmatched:
        raise RuntimeError(
            f"{len(unmatched)} source samples could not be matched to label.xlsx. "
            f"See {audit_root / '01_source_to_label_matching.csv'}"
        )

    # From this point, unmatched samples are excluded if --allow_unmatched was used.
    task_selected: Dict[int, List[int]] = {2: [], 4: [], 6: []}
    task_labels: Dict[int, List[int]] = {2: [], 4: [], 6: []}
    sample_manifest = []

    exclude_keywords = list(args.exclude_note_keywords)
    if args.exclude_all_postop and "术后" not in exclude_keywords:
        exclude_keywords.append("术后")

    for i, rec in enumerate(matched_records):
        if rec is None:
            continue
        exclusion_reasons = []
        matched_note_keyword = next((kw for kw in exclude_keywords if kw and kw in rec["note"]), "")
        if matched_note_keyword:
            exclusion_reasons.append(f"note_contains:{matched_note_keyword}")

        include2 = rec["binary_label"] is not None and not exclusion_reasons
        include4 = rec["four_label"] is not None and not exclusion_reasons
        include6 = (
            rec["binary_label"] == 1
            and rec["location_label"] is not None
            and not exclusion_reasons
            and (args.location_scope == "primary_all" or rec["curve_number"] == "单弯")
        )

        if include2:
            task_selected[2].append(i)
            task_labels[2].append(int(rec["binary_label"]))
        if include4:
            task_selected[4].append(i)
            task_labels[4].append(int(rec["four_label"]))
        if include6:
            task_selected[6].append(i)
            task_labels[6].append(int(rec["location_label"]))

        sample_manifest.append({
            "source_index": i,
            "name": rec["display_name"],
            "center": rec["center"],
            "subject_key": f"{rec['name']}|{rec['center']}",
            "cobb_angle": rec["cobb_angle"],
            "binary_raw": rec["binary_raw"],
            "binary_label": rec["binary_label"],
            "four_raw": rec["four_raw"],
            "four_label": rec["four_label"],
            "cobb_rule_25_moderate": cobb_rule_25_moderate(rec["cobb_angle"]),
            "agree_rule_25_moderate": int(
                rec["four_label"] == cobb_rule_25_moderate(rec["cobb_angle"])
            ) if rec["four_label"] is not None and cobb_rule_25_moderate(rec["cobb_angle"]) is not None else "",
            "cobb_rule_25_mild": cobb_rule_25_mild(rec["cobb_angle"]),
            "agree_rule_25_mild": int(
                rec["four_label"] == cobb_rule_25_mild(rec["cobb_angle"])
            ) if rec["four_label"] is not None and cobb_rule_25_mild(rec["cobb_angle"]) is not None else "",
            "curve_number": rec["curve_number"],
            "curve1": rec["curve1"],
            "curve2": rec["curve2"],
            "curve3": rec["curve3"],
            "location_label": rec["location_label"],
            "note": rec["note"],
            "include_task2": int(include2),
            "include_task4": int(include4),
            "include_task6": int(include6),
            "exclusion_reason": ";".join(exclusion_reasons),
        })

    manifest_df = pd.DataFrame(sample_manifest)
    manifest_df.to_csv(audit_root / "02_sample_label_manifest.csv", index=False, encoding="utf-8-sig")

    clinical_audit = pd.DataFrame(clinical)
    clinical_audit["cobb_rule_25_moderate"] = clinical_audit["cobb_angle"].map(cobb_rule_25_moderate)
    clinical_audit["agree_rule_25_moderate"] = (
        clinical_audit["four_label"] == clinical_audit["cobb_rule_25_moderate"]
    )
    clinical_audit["cobb_rule_25_mild"] = clinical_audit["cobb_angle"].map(cobb_rule_25_mild)
    clinical_audit["agree_rule_25_mild"] = (
        clinical_audit["four_label"] == clinical_audit["cobb_rule_25_mild"]
    )
    clinical_audit.to_csv(audit_root / "03_subject_label_audit.csv", index=False, encoding="utf-8-sig")

    excluded_subject_rows = []
    for rec in clinical:
        matched_keyword = next((kw for kw in exclude_keywords if kw and kw in rec["note"]), "")
        if matched_keyword:
            excluded_subject_rows.append({
                "excel_row": rec["excel_row"],
                "name": rec["display_name"],
                "center": rec["center"],
                "declared_samples": rec["declared_samples"],
                "binary_label": rec["binary_label"],
                "four_label": rec["four_label"],
                "curve1": rec["curve1"],
                "note": rec["note"],
                "matched_exclusion_keyword": matched_keyword,
            })
    pd.DataFrame(excluded_subject_rows).to_csv(
        audit_root / "04_excluded_subjects_do_not_use.csv", index=False, encoding="utf-8-sig"
    )

    matched_nonnull = [r if r is not None else {} for r in matched_records]
    summaries = []
    for task in (2, 4, 6):
        summaries.append(save_task(
            task=task,
            X=X,
            names=source_names,
            source_samples=source_samples,
            source_obj=source_obj,
            selected=task_selected[task],
            labels=task_labels[task],
            matched_records=matched_nonnull,
            out_root=out_root,
            location_scope=args.location_scope,
        ))

    included_clinical = [
        r for r in clinical
        if not any(kw and kw in r["note"] for kw in exclude_keywords)
    ]

    summary = {
        "label_excel": str(label_excel),
        "label_sheet": sheet,
        "note_column": note_col,
        "source_data_root": str(source_root),
        "out_data_root": str(out_root),
        "source_signal_shape": list(X.shape),
        "n_clinical_subject_rows": len(clinical),
        "n_source_samples": len(X),
        "exclude_note_keywords": exclude_keywords,
        "n_excluded_subject_rows": len(excluded_subject_rows),
        "excluded_declared_samples": int(sum((r.get("declared_samples") or 0) for r in excluded_subject_rows)),
        "location_scope": args.location_scope,
        "task_summaries": summaries,
        "clinical_subject_counts_before_exclusion": {
            "binary": dict(Counter(r["binary_label"] for r in clinical)),
            "four": dict(Counter(r["four_label"] for r in clinical)),
            "location": dict(Counter(r["location_label"] for r in clinical if r["binary_label"] == 1)),
        },
        "included_subject_counts_after_exclusion": {
            "binary": dict(Counter(r["binary_label"] for r in included_clinical)),
            "four": dict(Counter(r["four_label"] for r in included_clinical)),
            "location": dict(Counter(
                r["location_label"] for r in included_clinical
                if r["binary_label"] == 1 and r["location_label"] is not None
            )),
        },
    }
    with (out_root / "preparation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.copy_label_excel:
        shutil.copy2(label_excel, out_root / label_excel.name)

    print("\n[DONE] New-label datasets prepared")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[AUDIT] {audit_root}")


if __name__ == "__main__":
    main()
