#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate label_v3 and the QC signal root without modifying either input."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


BINARY = {
    "normal/control": 0, "normal": 0, "control": 0,
    "scoliosis/patient": 1, "scoliosis": 1, "patient": 1,
}
FOUR = {
    "normal/control": 0, "normal": 0, "control": 0,
    "mild": 1, "moderate": 2, "severe": 3,
}
LOCATION = {"胸弯": 0, "胸腰弯": 1, "腰弯": 2}


def clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\u3000", " ").strip()


def normalize(x: Any, mapping: dict[str, int]) -> Optional[int]:
    return mapping.get(clean(x).lower())


def find_sheet(path: Path, requested: str) -> str:
    xls = pd.ExcelFile(path)
    if requested != "auto":
        if requested not in xls.sheet_names:
            raise RuntimeError(f"Sheet {requested!r} not found; available={xls.sheet_names}")
        return requested
    candidates = [s for s in xls.sheet_names if "Subject" in s or "明细" in s or "Clinical" in s]
    return candidates[0] if candidates else xls.sheet_names[0]


def detect_note_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["备注", "说明", "Remark", "Remarks", "Note", "Notes", "Curve_summary"]
    for col in preferred:
        if col in df.columns:
            return col
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    return unnamed[-1] if unnamed else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_data_root")
    ap.add_argument("label_excel")
    ap.add_argument("--label_sheet", default="auto")
    ap.add_argument("--expected_samples", type=int, default=0, help="Optional expected source-sample count; 0 disables a fixed-count assertion.")
    ap.add_argument("--expected_subject_rows", type=int, default=0, help="Optional expected label-row count; 0 disables a fixed-count assertion.")
    ap.add_argument("--exclude_note_keywords", nargs="*", default=["不要数据"])
    args = ap.parse_args()

    source = Path(args.source_data_root).resolve()
    label = Path(args.label_excel).resolve()
    required_source = [
        source / "4" / "data_4class.npy",
        source / "4" / "names_4class.npy",
        source / "4" / "dataset_4class.json",
    ]
    for p in required_source:
        if not p.exists():
            raise FileNotFoundError(f"Missing source file: {p}")
    if not label.exists():
        raise FileNotFoundError(label)

    X = np.load(required_source[0], mmap_mode="r")
    names = np.load(required_source[1], allow_pickle=True)
    with required_source[2].open("r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = meta.get("samples", [])
    if len(X) != len(names) or len(samples) != len(X):
        raise RuntimeError(f"Source length mismatch: X={len(X)}, names={len(names)}, samples={len(samples)}")

    sheet = find_sheet(label, args.label_sheet)
    df = pd.read_excel(label, sheet_name=sheet)
    required_cols = [
        "Name", "Center", "Cobb angle", "Binary label", "Four-class label", "Samples",
        "弯数量", "弯1位置", "弯2位置", "弯3位置",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing label columns: {missing}; available={list(df.columns)}")

    binary = [normalize(x, BINARY) for x in df["Binary label"]]
    four = [normalize(x, FOUR) for x in df["Four-class label"]]
    loc = [LOCATION.get(clean(x)) for x in df["弯1位置"]]
    invalid_binary = df[[v is None for v in binary]][["Name", "Center", "Binary label"]]
    invalid_four = df[[v is None for v in four]][["Name", "Center", "Four-class label"]]
    declared_samples = pd.to_numeric(df["Samples"], errors="coerce")
    duplicated = df[df.duplicated(["Name", "Center"], keep=False)][["Name", "Center"]]

    note_col = detect_note_column(df)
    notes = df[note_col].fillna("").astype(str) if note_col else pd.Series([""] * len(df), index=df.index)
    exclude_mask = pd.Series(False, index=df.index)
    for kw in args.exclude_note_keywords:
        if kw:
            exclude_mask |= notes.str.contains(str(kw), regex=False, na=False)
    excluded = df.loc[exclude_mask, ["Name", "Center", "Samples"]].copy()
    included = df.loc[~exclude_mask].copy()

    included_binary = [normalize(x, BINARY) for x in included["Binary label"]]
    included_four = [normalize(x, FOUR) for x in included["Four-class label"]]
    included_loc = [
        LOCATION.get(clean(loc_raw))
        for loc_raw, binary_value in zip(included["弯1位置"], included_binary)
        if binary_value == 1
    ]
    included_samples = pd.to_numeric(included["Samples"], errors="coerce")

    print("===== SOURCE QC DATA =====")
    print(f"root: {source}")
    print(f"X.shape: {tuple(X.shape)}")
    print(f"names: {len(names)}, metadata samples: {len(samples)}")
    print(f"expected source sample count: {args.expected_samples if args.expected_samples > 0 else 'not fixed'}")
    print(f"source sample count match: {'not checked' if args.expected_samples <= 0 else len(X) == args.expected_samples}")

    print("\n===== LABEL V3 =====")
    print(f"file: {label}")
    print(f"sheet: {sheet}")
    print(f"note column: {note_col!r}")
    print(f"rows before exclusion: {len(df)} (expected {args.expected_subject_rows if args.expected_subject_rows > 0 else 'not fixed'})")
    print(f"declared Samples sum before exclusion: {int(declared_samples.sum()) if declared_samples.notna().all() else declared_samples.sum()}")
    print(f"excluded keywords: {args.exclude_note_keywords}")
    print(f"excluded subjects: {len(excluded)}")
    print(f"excluded declared samples: {int(pd.to_numeric(excluded['Samples'], errors='coerce').sum()) if len(excluded) else 0}")
    if len(excluded):
        print(excluded.to_string(index=False))
    print(f"included subjects: {len(included)}")
    print(f"included declared samples: {int(included_samples.sum()) if included_samples.notna().all() else included_samples.sum()}")
    print(f"duplicate Name+Center rows: {len(duplicated)}")
    print(f"invalid binary labels: {len(invalid_binary)}")
    print(f"invalid four-class labels: {len(invalid_four)}")
    print("included binary subject counts:", dict(Counter(v for v in included_binary if v is not None)))
    print("included four-class subject counts:", dict(Counter(v for v in included_four if v is not None)))
    print("included Task-6 patient primary-location counts:", dict(Counter(v for v in included_loc if v is not None)))
    print("included centers:", dict(Counter(clean(x) for x in included["Center"])))

    errors = []
    if args.expected_samples > 0 and len(X) != args.expected_samples:
        errors.append(f"source sample count is {len(X)}, expected {args.expected_samples}")
    if args.expected_subject_rows > 0 and len(df) != args.expected_subject_rows:
        errors.append(f"label row count is {len(df)}, expected {args.expected_subject_rows}")
    if declared_samples.isna().any() or int(declared_samples.sum()) != len(X):
        errors.append(f"sum(Samples)={declared_samples.sum()} does not match source samples={len(X)}")
    if len(duplicated):
        errors.append("duplicate Name+Center rows exist")
    if len(invalid_binary):
        errors.append("invalid binary labels exist")
    if len(invalid_four):
        errors.append("invalid four-class labels exist")
    if not note_col:
        errors.append("no note/remark column was detected; cannot enforce 不要数据 exclusion")

    if errors:
        print("\n[FAILED]")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    print("\n[PASS] Input structure is suitable for label_v3 task reconstruction.")
    print("[NOTE] Rows marked 不要数据 will be excluded from every task before split generation.")
    print("[NOTE] Four-class labels are used directly; Cobb angle is audit-only.")


if __name__ == "__main__":
    main()
