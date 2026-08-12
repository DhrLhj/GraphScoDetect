#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the revised GraphScoDetect model for Tasks 2, 4, and 6.

The script supports GKF3/GKF5/GKF7/GKF10/LOSO/LOCO and preserves the
subject-independent split/evaluation pipeline.  The revised model follows the
user-supplied GraphScoDetect implementation: segment embedding + learnable graph
message passing + BiLSTM temporal modeling, with encoder pretraining followed by
joint classification training.

A completed fold is skipped only when its saved run configuration matches the current
configuration; otherwise the script stops and asks for --overwrite or a new output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

try:
    from models import (
        GraphScoDetect,
        intra_subject_loss,
        supervised_contrastive_subject_loss,
    )
except ImportError as exc:
    raise ImportError(
        "The revised package requires GraphScoDetect from models.py. "
        "Please keep the user-supplied revised models.py in the repository root."
    ) from exc


PROTOCOL_ORDER = ["gkf3", "gkf5", "gkf7", "gkf10", "loso", "loco"]
SPLIT_FILES = {
    "gkf3": "groupkfold_3.json",
    "gkf5": "groupkfold_5.json",
    "gkf7": "groupkfold_7.json",
    "gkf10": "groupkfold_10.json",
    "loso": "loso.json",
    "loco": "loco.json",
}
TASK_FILES = {
    2: {"data": "data_binary.npy", "label": "label_binary.npy", "names": "names_binary.npy", "json": "dataset_binary.json", "n_classes": 2, "class_names": ["normal", "scoliosis"]},
    4: {"data": "data_4class.npy", "label": "label_4class.npy", "names": "names_4class.npy", "json": "dataset_4class.json", "n_classes": 4, "class_names": ["normal", "mild", "moderate", "severe"]},
    6: {"data": "data_curve_type.npy", "label": "label_curve_type.npy", "names": "names_curve_type.npy", "json": "dataset_curve_type.json", "n_classes": 3, "class_names": ["thoracic", "thoracolumbar", "lumbar"]},
}


_GRAPH_CACHE: Dict[Tuple[int, Tuple[int, ...], int, int], np.ndarray] = {}


def prepare_graph_array(
    X: np.ndarray,
    resample_len: int = 500,
    segment_len: int = 25,
) -> np.ndarray:
    """Convert [N,L,C] signals to revised-model input [N,T,C,S].

    This mirrors the preprocessing in the user-supplied model code:
      per-channel standardization -> resample to 500 -> 20 segments of length 25.
    If X is already [N,T,C,S], it is returned as float32 after validation.
    """
    X = np.asarray(X)
    if X.ndim == 4:
        if X.shape[-1] != segment_len:
            raise ValueError(
                f"4-D input must have segment_len={segment_len} on the last axis; got {X.shape}"
            )
        return X.astype(np.float32, copy=False)
    if X.ndim != 3:
        raise ValueError(f"Expected X=[N,L,C] or [N,T,C,S], got {X.shape}")
    if resample_len % segment_len != 0:
        raise ValueError("resample_len must be divisible by segment_len")

    key = (id(X), tuple(X.shape), int(resample_len), int(segment_len))
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    n, _, c = X.shape
    n_segments = resample_len // segment_len
    out = np.empty((n, n_segments, c, segment_len), dtype=np.float32)
    for i in range(n):
        x = np.asarray(X[i], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x = (x - mean) / std
        x = resample(x, resample_len, axis=0).astype(np.float32)
        x = x.reshape(n_segments, segment_len, c).transpose(0, 2, 1)
        out[i] = x
    _GRAPH_CACHE[key] = out
    return out


class GraphArrayDataset(Dataset):
    def __init__(
        self,
        X_graph: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        subject_ids: Sequence[str],
    ):
        self.X = torch.as_tensor(X_graph[indices], dtype=torch.float32)
        self.y = torch.as_tensor(y[indices], dtype=torch.long)
        self.subject_ids = [str(subject_ids[int(i)]) for i in indices]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.subject_ids[i]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_task(data_root: Path, task: int):
    cfg = TASK_FILES[task]
    root = data_root / str(task)
    X = np.load(root / cfg["data"])
    y = np.load(root / cfg["label"]).astype(int)
    names = np.load(root / cfg["names"], allow_pickle=True).astype(str)
    json_path = root / cfg["json"]
    label_path = root / cfg["label"]
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    samples = obj.get("samples")
    if not isinstance(samples, list) or not (len(X) == len(y) == len(names) == len(samples)):
        raise RuntimeError(f"Task {task}: inconsistent data/label/name/metadata lengths")
    selected_path = root / "selected_source_indices.npy"
    fp_parts = [file_sha256(label_path), file_sha256(json_path)]
    if selected_path.exists():
        fp_parts.append(file_sha256(selected_path))
    dataset_fingerprint = hashlib.sha256("|".join(fp_parts).encode("utf-8")).hexdigest()
    return X, y, names, samples, cfg, dataset_fingerprint


def load_folds(split_root: Path, task: int, protocol: str):
    path = split_root / str(task) / SPLIT_FILES[protocol]
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    folds = obj.get("folds")
    if not isinstance(folds, list):
        raise RuntimeError(f"Invalid split file: {path}")
    return folds, path, file_sha256(path)


def normalize_fold(fold: Dict[str, Any], fallback: int):
    fold_id = fold.get("fold", fallback)
    tr = fold.get("train_indices")
    te = fold.get("test_indices")
    if tr is None or te is None:
        raise KeyError(f"Fold missing train_indices/test_indices: {list(fold)}")
    return str(fold_id), np.asarray(tr, dtype=int), np.asarray(te, dtype=int)


def sample_meta(sample: Dict[str, Any], fallback_name: str, idx: int) -> Dict[str, Any]:
    info = sample.get("label_v3_info", sample.get("new_label_info", {})) or {}
    pinfo = sample.get("patient_info", {}) or {}
    name = clean(sample.get("name")) or clean(fallback_name)
    center = clean(sample.get("center"))
    key = clean(sample.get("subject_key")) or f"{name}|{center}"
    return {
        "sample_index": int(idx),
        "subject_key": key,
        "subject_id": clean(sample.get("subject_id")) or key,
        "name": name,
        "center": center,
        "source_dataset": clean(sample.get("source_dataset")),
        "sample_id": clean(sample.get("sample_id")) or str(idx),
        "cobb_angle": info.get("cobb_angle", pinfo.get("cobb_label_angle", "")),
        "curve_number": info.get("curve_number", ""),
        "curve1": info.get("curve1", ""),
        "curve2": info.get("curve2", ""),
        "curve3": info.get("curve3", ""),
        "note": info.get("note", pinfo.get("remark", "")),
    }


def per_class_stats(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int], names: Sequence[str]):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    total = int(cm.sum())
    rows = []
    for i, (label, name) in enumerate(zip(labels, names)):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) and not np.isnan(sensitivity) else 0.0
        rows.append({
            "label": int(label), "class_name": name, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "support": int(tp + fn), "precision": float(precision),
            "sensitivity": None if np.isnan(sensitivity) else float(sensitivity),
            "specificity": None if np.isnan(specificity) else float(specificity),
            "f1": float(f1),
        })
    return cm, rows


def safe_mean(values: Sequence[Any]) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: Sequence[str],
    y_prob: np.ndarray | None = None,
) -> Dict[str, Any]:
    labels = list(range(n_classes))
    cm, per_class = per_class_stats(y_true, y_pred, labels, class_names)
    sensitivity_macro = safe_mean([r["sensitivity"] for r in per_class])
    specificity_macro = safe_mean([r["specificity"] for r in per_class])
    if n_classes == 2:
        sensitivity = per_class[1]["sensitivity"]
        specificity = per_class[1]["specificity"]
    else:
        sensitivity = sensitivity_macro
        specificity = specificity_macro

    # AUROC is probability-based.  For binary classification class 1
    # (scoliosis) is the positive class.  For multiclass tasks, report the
    # macro mean of valid one-vs-rest class AUROCs.  A class with no positive
    # or no negative samples in the current test subset has undefined AUROC.
    auroc_per_class = []
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim != 2 or y_prob.shape != (len(y_true), n_classes):
            raise ValueError(
                f"y_prob must have shape {(len(y_true), n_classes)}, got {y_prob.shape}"
            )
        for c in labels:
            binary_true = (y_true == c).astype(int)
            if len(np.unique(binary_true)) < 2:
                auc = None
            else:
                auc = float(roc_auc_score(binary_true, y_prob[:, c]))
            auroc_per_class.append(auc)
            per_class[c]["auroc_ovr"] = auc
    else:
        auroc_per_class = [None] * n_classes
        for c in labels:
            per_class[c]["auroc_ovr"] = None

    if n_classes == 2:
        auroc = auroc_per_class[1]
    else:
        valid_auc = [x for x in auroc_per_class if x is not None and np.isfinite(x)]
        auroc = float(np.mean(valid_auc)) if valid_auc else None

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": None if sensitivity is None else float(sensitivity),
        "specificity": None if specificity is None else float(specificity),
        "sensitivity_macro_ovr": float(sensitivity_macro),
        "specificity_macro_ovr": float(specificity_macro),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc": None if auroc is None else float(auroc),
        "auroc_macro_ovr": None if auroc is None else float(auroc),
        "auroc_valid_class_count": int(sum(x is not None and np.isfinite(x) for x in auroc_per_class)),
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_normalized": cm_norm.astype(float).tolist(),
        "per_class": per_class,
        "n_predictions": int(len(y_true)),
        "true_label_counts": {str(k): int(v) for k, v in Counter(y_true.tolist()).items()},
        "pred_label_counts": {str(k): int(v) for k, v in Counter(y_pred.tolist()).items()},
    }


def class_weights(y_train: np.ndarray, n_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        raise RuntimeError(f"Training fold misses a class: {counts.tolist()}")
    weights = len(y_train) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def aggregate_subjects(rows: Sequence[Dict[str, Any]], n_classes: int, class_names: Sequence[str]):
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject_key"]].append(row)
    subject_rows = []
    for key, group in grouped.items():
        truths = sorted(set(int(r["y_true"]) for r in group))
        if len(truths) != 1:
            raise RuntimeError(f"Subject {key} has inconsistent test labels: {truths}")
        prob = np.asarray([r["prob"] for r in group], dtype=float).mean(axis=0)
        pred = int(prob.argmax())
        base = group[0]
        subject_rows.append({
            "subject_key": key, "name": base["name"], "center": base["center"],
            "y_true": truths[0], "y_pred": pred, "correct": int(pred == truths[0]),
            "prob": prob.tolist(), "n_samples": len(group),
            "cobb_angle": base.get("cobb_angle", ""), "curve_number": base.get("curve_number", ""),
            "curve1": base.get("curve1", ""), "note": base.get("note", ""),
        })
    yt = np.asarray([r["y_true"] for r in subject_rows], dtype=int)
    yp = np.asarray([r["y_pred"] for r in subject_rows], dtype=int)
    yprob = np.asarray([r["prob"] for r in subject_rows], dtype=float)
    return compute_metrics(yt, yp, n_classes, class_names, yprob), subject_rows


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]



def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=use_amp)


def train_one_fold(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    n_classes: int, device: str, epochs: int, batch_size: int, lr: float,
    weight_decay: float, dropout: float, seed: int, balanced_loss: bool,
    num_workers: int, amp: bool,
    subject_ids: Sequence[str] | None = None,
    pretrain_epochs: int = 20,
    joint_epochs: int | None = None,
    resample_len: int = 500,
    segment_len: int = 25,
    hidden_dim: int = 64,
    lstm_hidden: int = 128,
    lambda_inter: float = 0.5,
    gamma_class: float = 1.0,
    intra_margin: float = 1.0,
    temperature: float = 0.1,
):
    """Train one fold using the revised two-stage GraphScoDetect objective.

    `epochs` is kept for backward compatibility with the existing runner.  When
    `joint_epochs` is omitted, total epochs are split as 20% pretraining and
    80% joint training (100 -> 20 + 80, matching the supplied model).
    """
    set_seed(seed)
    if joint_epochs is None:
        pretrain_epochs = min(int(pretrain_epochs), int(epochs))
        joint_epochs = max(int(epochs) - int(pretrain_epochs), 0)
    total_epochs = int(pretrain_epochs) + int(joint_epochs)

    if subject_ids is None:
        subject_ids = [f"sample_{i}" for i in range(len(X))]
    if len(subject_ids) != len(X):
        raise ValueError(f"subject_ids length {len(subject_ids)} != X length {len(X)}")

    X_graph = prepare_graph_array(X, resample_len=resample_len, segment_len=segment_len)
    num_channels = int(X_graph.shape[2])
    model = GraphScoDetect(
        num_channels=num_channels,
        segment_len=segment_len,
        num_classes=n_classes,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
    ).to(device)

    # The revised source uses Adam, LR=1e-4, WD=1e-4.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weight = class_weights(y[train_idx], n_classes, device) if balanced_loss else None

    loader = DataLoader(
        GraphArrayDataset(X_graph, y, train_idx, subject_ids),
        batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    use_amp = bool(amp and device.startswith("cuda"))
    scaler = make_grad_scaler(use_amp)
    history = []

    # Stage 1: encoder pretraining.
    for epoch in range(1, int(pretrain_epochs) + 1):
        model.train()
        total_meter = []
        intra_meter = []
        inter_meter = []
        for xb, yb, sids in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                outputs = model(xb)
                l_intra = intra_subject_loss(
                    outputs["segment_repr"], yb, normal_label=0, margin=intra_margin
                )
                l_inter = supervised_contrastive_subject_loss(
                    outputs["segment_repr"], sids, temperature=temperature
                )
                loss = l_intra + lambda_inter * l_inter
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_meter.append(float(loss.item()))
            intra_meter.append(float(l_intra.detach().item()))
            inter_meter.append(float(l_inter.detach().item()))
        mean_loss = float(np.mean(total_meter)) if total_meter else float("nan")
        history.append({
            "epoch": epoch, "stage": "pretrain", "train_loss": mean_loss,
            "intra_loss": float(np.mean(intra_meter)) if intra_meter else float("nan"),
            "inter_loss": float(np.mean(inter_meter)) if inter_meter else float("nan"),
            "class_loss": float("nan"),
        })
        if epoch == 1 or epoch == pretrain_epochs or epoch % max(1, pretrain_epochs // 5) == 0:
            print(f"      [pretrain] epoch={epoch:03d}/{pretrain_epochs} loss={mean_loss:.6f}")

    # Stage 2: joint training.
    for j in range(1, int(joint_epochs) + 1):
        model.train()
        total_meter = []
        intra_meter = []
        inter_meter = []
        class_meter = []
        for xb, yb, sids in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                outputs = model(xb)
                l_intra = intra_subject_loss(
                    outputs["segment_repr"], yb, normal_label=0, margin=intra_margin
                )
                l_inter = supervised_contrastive_subject_loss(
                    outputs["segment_repr"], sids, temperature=temperature
                )
                l_class = nn.functional.cross_entropy(
                    outputs["logits"], yb, weight=class_weight
                )
                loss = l_intra + lambda_inter * l_inter + gamma_class * l_class
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_meter.append(float(loss.item()))
            intra_meter.append(float(l_intra.detach().item()))
            inter_meter.append(float(l_inter.detach().item()))
            class_meter.append(float(l_class.detach().item()))
        epoch_no = int(pretrain_epochs) + j
        mean_loss = float(np.mean(total_meter)) if total_meter else float("nan")
        history.append({
            "epoch": epoch_no, "stage": "joint", "train_loss": mean_loss,
            "intra_loss": float(np.mean(intra_meter)) if intra_meter else float("nan"),
            "inter_loss": float(np.mean(inter_meter)) if inter_meter else float("nan"),
            "class_loss": float(np.mean(class_meter)) if class_meter else float("nan"),
        })
        if j == 1 or j == joint_epochs or j % max(1, joint_epochs // 10) == 0:
            print(f"      [joint] epoch={j:03d}/{joint_epochs} loss={mean_loss:.6f}")

    test_loader = DataLoader(
        GraphArrayDataset(X_graph, y, test_idx, subject_ids),
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    model.eval()
    probs: List[List[float]] = []
    with torch.no_grad():
        for xb, _, _ in test_loader:
            with amp_autocast(use_amp):
                outputs = model(xb.to(device, non_blocking=True))
                logits = outputs["logits"]
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
    prob_arr = np.asarray(probs, dtype=float)
    pred = prob_arr.argmax(axis=1).astype(int)
    return model, pred, prob_arr, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--project_root", default="", help="Recorded for reproducibility; PYTHONPATH should already contain it.")
    ap.add_argument("--tasks", nargs="+", type=int, choices=[2, 4, 6], default=[2, 4, 6])
    ap.add_argument("--protocols", nargs="+", choices=PROTOCOL_ORDER, default=PROTOCOL_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--balanced_loss_tasks", nargs="*", type=int, default=[2, 4, 6])
    ap.add_argument("--epochs", type=int, default=100, help="Total epochs retained for compatibility; default 20 pretrain + 80 joint.")
    ap.add_argument("--pretrain_epochs", type=int, default=20)
    ap.add_argument("--joint_epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2, help="Recorded for compatibility; revised GraphScoDetect classifier uses dropout=0.2.")
    ap.add_argument("--resample_len", type=int, default=500)
    ap.add_argument("--segment_len", type=int, default=25)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lambda_inter", type=float, default=0.5)
    ap.add_argument("--gamma_class", type=float, default=1.0)
    ap.add_argument("--intra_margin", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--limit_folds", type=int, default=0)
    args = ap.parse_args()

    requested_protocols = [p for p in PROTOCOL_ORDER if p in args.protocols]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    data_root, split_root, out_root = map(lambda p: Path(p).resolve(), [args.data_root, args.split_root, args.out_root])
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] device={device}, model=GraphScoDetect, "
        f"pretrain={args.pretrain_epochs}, joint={args.joint_epochs}, "
        f"batch_size={args.batch_size}, protocols={requested_protocols}"
    )

    for protocol in requested_protocols:
        print(f"\n========== PROTOCOL {protocol} ==========")
        for task in args.tasks:
            X, y, names, samples, cfg, dataset_fingerprint = load_task(data_root, task)
            folds, split_path, split_fingerprint = load_folds(split_root, task, protocol)
            if args.limit_folds > 0:
                folds = folds[:args.limit_folds]
            for seed in args.seeds:
                for fold_pos, fold in enumerate(folds):
                    fold_id, train_idx, test_idx = normalize_fold(fold, fold_pos)
                    fold_dir = out_root / f"task{task}" / protocol / f"seed_{seed}" / f"fold_{fold_id}"
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    fold_metadata = {
                        key: fold.get(key)
                        for key in [
                            "test_center", "test_subject", "loco_train_only_centers",
                            "present_train_only_centers", "train_center_counts", "test_center_counts",
                            "train_subject_label_counts", "test_subject_label_counts",
                        ]
                        if key in fold
                    }
                    run_config = {
                        "task": task, "protocol": protocol, "seed": seed, "fold": fold_id,
                        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                        "weight_decay": args.weight_decay, "dropout": args.dropout,
                        "balanced_loss": task in set(args.balanced_loss_tasks),
                        "model": "GraphScoDetect", "n_classes": cfg["n_classes"],
                        "pretrain_epochs": args.pretrain_epochs, "joint_epochs": args.joint_epochs,
                        "resample_len": args.resample_len, "segment_len": args.segment_len,
                        "hidden_dim": args.hidden_dim, "lstm_hidden": args.lstm_hidden,
                        "lambda_inter": args.lambda_inter, "gamma_class": args.gamma_class,
                        "intra_margin": args.intra_margin, "temperature": args.temperature,
                        "split_file": str(split_path), "split_fingerprint": split_fingerprint,
                        "dataset_fingerprint": dataset_fingerprint, "project_root": str(args.project_root),
                        **fold_metadata,
                    }
                    run_config["config_hash"] = config_hash(run_config)
                    config_path = fold_dir / "run_config.json"
                    metrics_path = fold_dir / "metrics.json"
                    pred_path = fold_dir / "predictions.json"
                    if metrics_path.exists() and pred_path.exists() and config_path.exists() and not args.overwrite:
                        previous = json.loads(config_path.read_text(encoding="utf-8"))
                        if previous.get("config_hash") == run_config["config_hash"]:
                            print(f"[SKIP] task={task} protocol={protocol} seed={seed} fold={fold_id}")
                            continue
                        raise RuntimeError(
                            f"Existing result has a different configuration: {fold_dir}. "
                            "Use --overwrite or choose a new OUT_BASE."
                        )

                    np.save(fold_dir / "train.npy", train_idx)
                    np.save(fold_dir / "test.npy", test_idx)
                    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[RUN] task={task} protocol={protocol} seed={seed} fold={fold_id} train={len(train_idx)} test={len(test_idx)}")
                    start = time.time()
                    all_subject_ids = [
                        sample_meta(samples[i] or {}, names[i], i)["subject_key"]
                        for i in range(len(X))
                    ]
                    model, pred, prob, history = train_one_fold(
                        X, y, train_idx, test_idx, cfg["n_classes"], device,
                        args.epochs, args.batch_size, args.lr, args.weight_decay,
                        args.dropout, seed, task in set(args.balanced_loss_tasks),
                        args.num_workers, args.amp,
                        subject_ids=all_subject_ids,
                        pretrain_epochs=args.pretrain_epochs,
                        joint_epochs=args.joint_epochs,
                        resample_len=args.resample_len,
                        segment_len=args.segment_len,
                        hidden_dim=args.hidden_dim,
                        lstm_hidden=args.lstm_hidden,
                        lambda_inter=args.lambda_inter,
                        gamma_class=args.gamma_class,
                        intra_margin=args.intra_margin,
                        temperature=args.temperature,
                    )
                    y_true = y[test_idx]
                    rows = []
                    for j, source_idx in enumerate(test_idx):
                        meta = sample_meta(samples[int(source_idx)] or {}, names[int(source_idx)], int(source_idx))
                        rows.append({
                            **meta, "task": task, "protocol": protocol, "seed": seed, "fold": fold_id,
                            "y_true": int(y_true[j]), "y_pred": int(pred[j]),
                            "correct": int(y_true[j] == pred[j]), "prob": prob[j].tolist(),
                        })
                    sample_metrics = compute_metrics(y_true, pred, cfg["n_classes"], cfg["class_names"], prob)
                    subject_metrics, subject_rows = aggregate_subjects(rows, cfg["n_classes"], cfg["class_names"])
                    output = {
                        **run_config, "elapsed_seconds": float(time.time() - start),
                        "n_train_samples": int(len(train_idx)), "n_test_samples": int(len(test_idx)),
                        "n_train_subjects": int(len(set(sample_meta(samples[int(i)] or {}, names[int(i)], int(i))["subject_key"] for i in train_idx))),
                        "n_test_subjects": int(len(subject_rows)),
                        "train_label_counts": {str(k): int(v) for k, v in Counter(y[train_idx].tolist()).items()},
                        "test_label_counts": {str(k): int(v) for k, v in Counter(y[test_idx].tolist()).items()},
                        "sample_metrics": sample_metrics, "subject_metrics": subject_metrics,
                    }
                    metrics_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                    pred_path.write_text(json.dumps({
                        "task": task, "protocol": protocol, "seed": seed, "fold": fold_id,
                        "test_center": fold.get("test_center", ""),
                        "loco_train_only_centers": fold.get("loco_train_only_centers", []),
                        "predictions": rows, "subject_predictions": subject_rows,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    with (fold_dir / "training_history.csv").open("w", encoding="utf-8") as f:
                        f.write("epoch,stage,train_loss,intra_loss,inter_loss,class_loss\n")
                        for r in history:
                            f.write(
                                f"{r['epoch']},{r.get('stage','')},{r['train_loss']:.10f},"
                                f"{r.get('intra_loss', float('nan')):.10f},"
                                f"{r.get('inter_loss', float('nan')):.10f},"
                                f"{r.get('class_loss', float('nan')):.10f}\n"
                            )
                    if args.save_model:
                        torch.save(model.state_dict(), fold_dir / "model.pt")
                    auc_text = "NA" if subject_metrics.get("auroc") is None else f"{subject_metrics['auroc']:.4f}"
                    print(
                        f"[DONE] subject acc={subject_metrics['accuracy']:.4f} "
                        f"bacc={subject_metrics['balanced_accuracy']:.4f} "
                        f"micro-f1={subject_metrics['micro_f1']:.4f} auroc={auc_text}"
                    )

    print("\n[DONE] All requested protocols finished or were resumed successfully.")


if __name__ == "__main__":
    main()
