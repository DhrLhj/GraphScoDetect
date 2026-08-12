#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Task-4 subject embeddings from ONE representative LOSO-trained model.

Default model:
    Task 4 / LOSO / seed 42 / fold 0

Feature definition:
    Input to the final nn.Linear classification layer, with model.eval().
    For the revised GraphScoDetect architecture this is `temporal_repr`, the
    256-D pooled bidirectional-LSTM representation immediately before the classifier.

IMPORTANT:
    The SAME model is used to extract features for all subjects available in Task 4.
    Therefore every subject embedding lies in one common latent coordinate system.

    This is a qualitative representation visualization using a representative
    LOSO-trained model. It is NOT an all-subject out-of-fold evaluation.

No prediction probabilities are exported.

Main CSV:
    subject_id, center, label, label_name, feature_000, ..., feature_127
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


LABEL_NAMES = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}


class IndexDataset(Dataset):
    def __init__(self, X: np.ndarray, indices: Sequence[int]):
        self.X = torch.as_tensor(
            X[np.asarray(indices, dtype=int)],
            dtype=torch.float32,
        )
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.X[i], int(self.indices[i])


def clean(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def subject_key(sample: Dict[str, Any], fallback_name: str, index: int) -> str:
    key = clean(sample.get("subject_key"))
    if key:
        return key

    name = clean(sample.get("name")) or clean(fallback_name)
    center = clean(sample.get("center"))
    if name:
        return f"{name}|{center}"

    sid = clean(sample.get("subject_id"))
    return sid or f"sample_{index}"


def subject_id(sample: Dict[str, Any], fallback_name: str, index: int) -> str:
    sid = clean(sample.get("subject_id"))
    if sid:
        return sid

    name = clean(sample.get("name")) or clean(fallback_name)
    center = clean(sample.get("center"))
    return f"{name}|{center}" if name else f"sample_{index}"


def load_task4(experiment_root: Path):
    root = experiment_root / "data" / "4"

    X = np.load(root / "data_4class.npy").astype(np.float32)
    y = np.load(root / "label_4class.npy").astype(int)
    names = np.load(
        root / "names_4class.npy",
        allow_pickle=True,
    ).astype(str)

    with (root / "dataset_4class.json").open("r", encoding="utf-8") as f:
        obj = json.load(f)

    samples = obj.get("samples")
    if not isinstance(samples, list) or len(samples) != len(X):
        raise RuntimeError("Invalid Task-4 sample metadata.")

    if not (len(X) == len(y) == len(names)):
        raise RuntimeError("Task-4 X/y/names length mismatch.")

    if X.ndim != 3 or X.shape[-1] != 6:
        raise RuntimeError(
            f"Expected Task-4 X=[N,T,6], got {X.shape}"
        )

    return X, y, names, samples


def build_subject_metadata(
    y: np.ndarray,
    names: np.ndarray,
    samples: List[Dict[str, Any]],
):
    by_subject: Dict[str, Dict[str, Any]] = {}
    sample_subject = []

    for i, sample in enumerate(samples):
        sample = sample or {}

        key = subject_key(sample, names[i], i)
        sid = subject_id(sample, names[i], i)
        center = clean(sample.get("center"))

        sample_subject.append(key)

        if key not in by_subject:
            by_subject[key] = {
                "subject_key": key,
                "subject_id": sid,
                "center": center,
                "label": int(y[i]),
                "label_name": LABEL_NAMES[int(y[i])],
                "indices": [],
            }

        rec = by_subject[key]

        if rec["label"] != int(y[i]):
            raise RuntimeError(
                f"Inconsistent Task-4 labels for subject {key}."
            )

        if rec["center"] != center:
            raise RuntimeError(
                f"Inconsistent center for subject {key}."
            )

        rec["indices"].append(i)

    return by_subject, np.asarray(sample_subject, dtype=object)


def import_model(project_root: Path):
    sys.path.insert(0, str(project_root))
    from models import GraphScoDetect
    return GraphScoDetect


def prepare_graph_array(X: np.ndarray, resample_len: int = 500, segment_len: int = 25) -> np.ndarray:
    if X.ndim == 4:
        return X.astype(np.float32, copy=False)
    if X.ndim != 3:
        raise ValueError(f"Expected [N,L,C] or [N,T,C,S], got {X.shape}")
    if resample_len % segment_len != 0:
        raise ValueError("resample_len must be divisible by segment_len")
    from scipy.signal import resample
    n, _, c = X.shape
    ns = resample_len // segment_len
    out = np.empty((n, ns, c, segment_len), dtype=np.float32)
    for i in range(n):
        x = np.asarray(X[i], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x = (x - mean) / std
        x = resample(x, resample_len, axis=0).astype(np.float32)
        out[i] = x.reshape(ns, segment_len, c).transpose(0, 2, 1)
    return out


def load_model(model_cls, model_path: Path, run_config_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model weight not found: {model_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
    if int(cfg.get("task", -1)) != 4:
        raise RuntimeError(f"Selected checkpoint is not Task 4: {run_config_path}")
    if str(cfg.get("protocol", "")).lower() != "loso":
        raise RuntimeError(f"Selected checkpoint is not LOSO: {run_config_path}")
    model = model_cls(
        num_channels=6,
        segment_len=int(cfg.get("segment_len", 25)),
        num_classes=4,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        lstm_hidden=int(cfg.get("lstm_hidden", 128)),
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_sample_features(
    model: nn.Module,
    X: np.ndarray,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    resample_len: int = 500,
    segment_len: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    Xg = prepare_graph_array(X, resample_len=resample_len, segment_len=segment_len)
    class GraphIndexDataset(Dataset):
        def __init__(self, Xg, idx):
            self.X = torch.as_tensor(Xg[np.asarray(idx,dtype=int)], dtype=torch.float32)
            self.idx = np.asarray(idx,dtype=int)
        def __len__(self): return len(self.idx)
        def __getitem__(self,i): return self.X[i], int(self.idx[i])
    loader = DataLoader(
        GraphIndexDataset(Xg, indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feats=[]; source_indices=[]
    for xb, src_idx in loader:
        outputs = model(xb.to(device, non_blocking=True))
        feats.append(outputs["temporal_repr"].detach().float().cpu().numpy())
        source_indices.extend(src_idx.numpy().astype(int).tolist())
    return np.asarray(source_indices,dtype=int), np.concatenate(feats,axis=0)


def aggregate_subject_features(
    sample_indices: np.ndarray,
    sample_features: np.ndarray,
    sample_subject: np.ndarray,
):
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)

    for idx, feat in zip(
        sample_indices.tolist(),
        sample_features,
    ):
        sid = str(
            sample_subject[int(idx)]
        )
        grouped[sid].append(
            np.asarray(feat, dtype=np.float64)
        )

    return {
        sid: np.stack(arrs, axis=0).mean(axis=0)
        for sid, arrs in grouped.items()
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--experiment_root",
        required=True,
        help=(
            "Completed experiment root containing "
            "data/ and results/task4/loso/..."
        ),
    )
    ap.add_argument(
        "--project_root",
        required=True,
        help="Project root containing models.py.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Representative LOSO model seed.",
    )
    ap.add_argument(
        "--fold",
        default="0",
        help="Representative LOSO fold id.",
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    ap.add_argument(
        "--expected_subjects",
        type=int,
        default=0,
    )

    args = ap.parse_args()

    exp = Path(args.experiment_root).resolve()
    project = Path(args.project_root).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (
            "cpu"
            if args.device == "auto"
            else args.device
        )
    )

    print(f"[INFO] device={device}")
    print(
        f"[INFO] representative model: "
        f"Task4 / LOSO / seed={args.seed} / fold={args.fold}"
    )

    X, y, names, samples = load_task4(exp)

    by_subject, sample_subject = build_subject_metadata(
        y,
        names,
        samples,
    )

    if args.expected_subjects > 0 and len(by_subject) != args.expected_subjects:
        raise RuntimeError(
            f"Expected {args.expected_subjects} Task-4 subjects, got {len(by_subject)}."
        )

    model_dir = (
        exp
        / "results"
        / "task4"
        / "loso"
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )

    model_path = model_dir / "model.pt"
    run_config_path = model_dir / "run_config.json"

    Model = import_model(project)

    model, run_cfg = load_model(
        Model,
        model_path,
        run_config_path,
        device,
    )

    # Extract features from ALL Task-4 signal segments with the SAME model.
    all_indices = np.arange(
        len(X),
        dtype=int,
    )

    sample_idx, sample_feat = extract_sample_features(
        model, X, all_indices, device, args.batch_size, args.num_workers,
        resample_len=int(run_cfg.get("resample_len",500)),
        segment_len=int(run_cfg.get("segment_len",25)),
    )

    subject_feat = aggregate_subject_features(
        sample_idx,
        sample_feat,
        sample_subject,
    )

    feature_dim = len(
        next(iter(subject_feat.values()))
    )

    rows = []

    for sid in sorted(by_subject):
        rec = by_subject[sid]
        feat = subject_feat[sid]

        row = {
            "subject_id": rec["subject_id"],
            "center": rec["center"],
            "label": rec["label"],
            "label_name": rec["label_name"],
        }

        row.update({
            f"feature_{j:03d}": float(feat[j])
            for j in range(feature_dim)
        })

        rows.append(row)

    df = pd.DataFrame(rows)

    if args.expected_subjects > 0 and len(df) != args.expected_subjects:
        raise RuntimeError(
            f"Final CSV should have {args.expected_subjects} rows, got {len(df)}."
        )

    feature_csv = (
        out
        / f"task4_single_model_seed{args.seed}_fold{args.fold}_subject_features.csv"
    )
    df.to_csv(
        feature_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # Audit metadata only; no probabilities/logits.
    audit = []

    for sid in sorted(by_subject):
        rec = by_subject[sid]

        audit.append({
            "subject_key": sid,
            "subject_id": rec["subject_id"],
            "center": rec["center"],
            "label": rec["label"],
            "label_name": rec["label_name"],
            "n_signal_segments": len(rec["indices"]),
            "representative_seed": int(args.seed),
            "representative_fold": str(args.fold),
        })

    pd.DataFrame(audit).to_csv(
        out / "subject_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_info = {
        "task": 4,
        "protocol": "loso",
        "representative_seed": int(args.seed),
        "representative_fold": str(args.fold),
        "model_path": str(model_path),
        "run_config_path": str(run_config_path),
        "feature_definition": "GraphScoDetect temporal_repr: pooled 256-D BiLSTM representation before classifier",
        "feature_dim": int(feature_dim),
        "n_subjects": int(len(df)),
        "n_signal_segments": int(len(X)),
        "dropout": float(run_cfg.get("dropout", 0.1)),
        "note": (
            "The same representative LOSO-trained checkpoint was used to "
            "extract features for all subjects. This output is intended for "
            "qualitative latent-representation visualization, not OOF evaluation."
        ),
    }

    (
        out / "model_info.json"
    ).write_text(
        json.dumps(
            model_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== SINGLE-MODEL FEATURE EXPORT =====")
    print(f"subjects: {len(df)}")
    print(f"signal segments: {len(X)}")
    print(f"feature dim: {feature_dim}")
    print(f"model: {model_path}")
    print(f"output: {feature_csv}")
    print("prediction probabilities exported: NO")


if __name__ == "__main__":
    main()
