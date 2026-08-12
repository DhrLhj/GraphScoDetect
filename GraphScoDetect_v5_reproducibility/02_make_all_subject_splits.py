#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create subject-independent splits for Tasks 2, 4, and 6.

Supported protocols:
  gkf3, gkf5, gkf7, gkf10, loso, loco

The default experiment runner executes gkf3 -> gkf5 -> gkf7 -> gkf10 -> loco.
LOSO remains supported by this script and the training script, but is intentionally omitted
from the default sequence because of runtime.

For LOCO, centers passed through --loco_train_only_centers (default: 青海) are always
eligible for training and never used as held-out test centers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


PROTOCOL_ORDER = ["gkf3", "gkf5", "gkf7", "gkf10", "loso", "loco"]
PROTOCOL_K = {"gkf3": 3, "gkf5": 5, "gkf7": 7, "gkf10": 10}
TASK_FILES = {
    2: ("label_binary.npy", "dataset_binary.json"),
    4: ("label_4class.npy", "dataset_4class.json"),
    6: ("label_curve_type.npy", "dataset_curve_type.json"),
}


def clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def subject_key(sample: Dict[str, Any], index: int) -> str:
    key = clean(sample.get("subject_key"))
    if key:
        return key
    name, center = clean(sample.get("name")), clean(sample.get("center"))
    if name:
        return f"{name}|{center}"
    return clean(sample.get("subject_id")) or f"sample_{index}"


def load_task(data_root: Path, task: int):
    label_file, json_file = TASK_FILES[task]
    root = data_root / str(task)
    y = np.load(root / label_file).astype(int)
    with (root / json_file).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    samples = obj.get("samples")
    if not isinstance(samples, list) or len(samples) != len(y):
        raise RuntimeError(f"Task {task}: invalid sample metadata length")
    groups = np.asarray([subject_key(s or {}, i) for i, s in enumerate(samples)], dtype=object)
    centers = np.asarray([clean((s or {}).get("center")) for s in samples], dtype=object)
    return y, groups, centers


def build_subject_table(y: np.ndarray, groups: np.ndarray, centers: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sid in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == sid)
        labels = sorted(set(y[idx].tolist()))
        center_values = sorted(set(centers[idx].tolist()))
        if len(labels) != 1:
            raise RuntimeError(f"Subject {sid} has inconsistent labels: {labels}")
        if len(center_values) != 1:
            raise RuntimeError(f"Subject {sid} has inconsistent centers: {center_values}")
        if not center_values[0]:
            raise RuntimeError(f"Subject {sid} has an empty center, which prevents LOCO.")
        rows.append({
            "subject": sid,
            "label": int(labels[0]),
            "center": center_values[0],
            "sample_indices": idx.astype(int).tolist(),
            "n_samples": int(len(idx)),
        })
    return rows


def collect_indices(subjects: Sequence[str], by_subject: Dict[str, Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for sid in subjects:
        out.extend(by_subject[sid]["sample_indices"])
    return sorted(out)


def fold_record(
    fold_id: int,
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    by_subject: Dict[str, Dict[str, Any]],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    train_set, test_set = set(train_subjects), set(test_subjects)
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(f"Subject leakage in fold {fold_id}: {sorted(overlap)}")
    if not test_set:
        raise RuntimeError(f"Fold {fold_id} has an empty test subject set")
    train_idx = collect_indices(sorted(train_set), by_subject)
    test_idx = collect_indices(sorted(test_set), by_subject)
    rec: Dict[str, Any] = {
        "fold": int(fold_id),
        "train_indices": train_idx,
        "test_indices": test_idx,
        "train_subjects": sorted(train_set),
        "test_subjects": sorted(test_set),
        "n_train_samples": len(train_idx),
        "n_test_samples": len(test_idx),
        "n_train_subjects": len(train_set),
        "n_test_subjects": len(test_set),
        "train_subject_label_counts": {
            str(k): int(v) for k, v in Counter(by_subject[s]["label"] for s in train_set).items()
        },
        "test_subject_label_counts": {
            str(k): int(v) for k, v in Counter(by_subject[s]["label"] for s in test_set).items()
        },
        "train_center_counts": dict(Counter(by_subject[s]["center"] for s in train_set)),
        "test_center_counts": dict(Counter(by_subject[s]["center"] for s in test_set)),
    }
    if extra:
        rec.update(extra)
    return rec


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_stratified_kfold(table: List[Dict[str, Any]], n_splits: int, seed: int):
    subjects = np.asarray([r["subject"] for r in table], dtype=object)
    labels = np.asarray([r["label"] for r in table], dtype=int)
    counts = Counter(labels.tolist())
    if min(counts.values()) < n_splits:
        raise RuntimeError(
            f"Cannot make {n_splits}-fold subject-stratified CV: smallest class has "
            f"{min(counts.values())} subjects; counts={dict(counts)}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(subjects[tr].tolist(), subjects[te].tolist()) for tr, te in splitter.split(subjects, labels)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_split_root", required=True)
    ap.add_argument("--tasks", nargs="+", type=int, choices=[2, 4, 6], default=[2, 4, 6])
    ap.add_argument("--protocols", nargs="+", choices=PROTOCOL_ORDER, default=PROTOCOL_ORDER)
    ap.add_argument("--seed", type=int, default=42, help="Split seed shared by gkf3/5/7/10.")
    ap.add_argument(
        "--loco_train_only_centers", nargs="*", default=["青海"],
        help="Centers used in LOCO training but never as held-out test centers.",
    )
    args = ap.parse_args()

    requested = [p for p in PROTOCOL_ORDER if p in args.protocols]
    train_only_centers = {clean(x) for x in args.loco_train_only_centers if clean(x)}
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_split_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        y, groups, centers = load_task(data_root, task)
        table = build_subject_table(y, groups, centers)
        by_subject = {r["subject"]: r for r in table}
        all_subjects = [r["subject"] for r in table]
        task_out = out_root / str(task)
        task_out.mkdir(parents=True, exist_ok=True)

        center_counts = Counter(r["center"] for r in table)
        save_json({
            "task": task,
            "n_samples": int(len(y)),
            "n_subjects": int(len(table)),
            "subject_label_counts": {
                str(k): int(v) for k, v in Counter(r["label"] for r in table).items()
            },
            "subject_center_counts": dict(center_counts),
            "loco_train_only_centers_requested": sorted(train_only_centers),
            "subjects": table,
        }, task_out / "subject_table.json")

        for protocol in requested:
            if protocol in PROTOCOL_K:
                k = PROTOCOL_K[protocol]
                pairs = build_stratified_kfold(table, n_splits=k, seed=args.seed)
                folds = [fold_record(i, tr, te, by_subject) for i, (tr, te) in enumerate(pairs)]
                filename = f"groupkfold_{k}.json"
                payload = {
                    "task": task,
                    "protocol": protocol,
                    "split": f"subject_stratified_{k}_fold",
                    "seed": args.seed,
                    "folds": folds,
                }
            elif protocol == "loso":
                folds = []
                for i, sid in enumerate(sorted(all_subjects)):
                    train = [s for s in all_subjects if s != sid]
                    folds.append(fold_record(i, train, [sid], by_subject, {"test_subject": sid}))
                filename = "loso.json"
                payload = {
                    "task": task,
                    "protocol": protocol,
                    "split": "leave_one_subject_out",
                    "folds": folds,
                }
            elif protocol == "loco":
                all_centers = sorted(center_counts)
                test_centers = [c for c in all_centers if c not in train_only_centers]
                present_train_only = [c for c in all_centers if c in train_only_centers]
                if not test_centers:
                    raise RuntimeError(
                        f"Task {task}: no LOCO test centers remain after excluding train-only centers "
                        f"{sorted(train_only_centers)}"
                    )
                folds = []
                for i, test_center in enumerate(test_centers):
                    test = [r["subject"] for r in table if r["center"] == test_center]
                    train = [r["subject"] for r in table if r["center"] != test_center]
                    rec = fold_record(
                        i, train, test, by_subject,
                        {
                            "test_center": test_center,
                            "loco_train_only_centers": sorted(train_only_centers),
                            "present_train_only_centers": present_train_only,
                        },
                    )
                    tested_centers = set(rec["test_center_counts"])
                    forbidden = tested_centers & train_only_centers
                    if forbidden:
                        raise RuntimeError(
                            f"Task {task} LOCO fold {i}: train-only center leaked into test: {sorted(forbidden)}"
                        )
                    folds.append(rec)
                filename = "loco.json"
                payload = {
                    "task": task,
                    "protocol": protocol,
                    "split": "leave_one_center_out_with_train_only_centers",
                    "all_centers": all_centers,
                    "test_centers": test_centers,
                    "loco_train_only_centers": sorted(train_only_centers),
                    "present_train_only_centers": present_train_only,
                    "folds": folds,
                }
            else:
                raise AssertionError(protocol)

            save_json(payload, task_out / filename)
            print(f"[DONE] task={task} protocol={protocol} folds={len(folds)} -> {task_out / filename}")
            if protocol == "loco":
                print(
                    f"       test_centers={payload['test_centers']} "
                    f"train_only_centers={payload['loco_train_only_centers']}"
                )

        print(
            f"[TASK {task}] samples={len(y)}, subjects={len(table)}, "
            f"labels={dict(Counter(r['label'] for r in table))}, centers={dict(center_counts)}"
        )


if __name__ == "__main__":
    main()
