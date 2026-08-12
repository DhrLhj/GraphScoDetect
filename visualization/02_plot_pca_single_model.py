#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA visualization for Fig.4f using subject-level features extracted from
ONE representative LOSO-trained model.

Outputs:
    Fig4f_PCA_single_model.pdf
    Fig4f_PCA_single_model.png
    Fig4f_PCA_single_model.svg
    Fig4f_PCA_coordinates.csv
    Fig4f_PCA_variance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


LABEL_ORDER = [0, 1, 2, 3]

LABEL_NAMES = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}

# NPG / scientific-paper style palette.
COLORS = {
    0: "#3C5488",  # Normal: blue
    1: "#00A087",  # Mild: teal
    2: "#F39B7F",  # Moderate: salmon
    3: "#E64B35",  # Severe: red
}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--feature_csv",
        required=True,
    )
    ap.add_argument(
        "--out_dir",
        required=True,
    )
    ap.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Default False: PCA directly on learned embeddings. "
            "Use --standardize to z-score each feature dimension first."
        ),
    )
    ap.add_argument(
        "--expected_subjects",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--point_size",
        type=float,
        default=42,
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.82,
    )
    ap.add_argument(
        "--show_title",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = ap.parse_args()

    src = Path(args.feature_csv).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)

    feature_cols = sorted(
        [
            c for c in df.columns
            if c.startswith("feature_")
        ]
    )

    required = {
        "subject_id",
        "center",
        "label",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    if not feature_cols:
        raise RuntimeError(
            "No feature_* columns found."
        )

    if args.expected_subjects > 0 and len(df) != args.expected_subjects:
        raise RuntimeError(
            f"Expected {args.expected_subjects} subjects, got {len(df)}."
        )

    labels = df["label"].astype(int).to_numpy()

    unknown = sorted(
        set(labels.tolist()) - set(LABEL_ORDER)
    )

    if unknown:
        raise RuntimeError(
            f"Unexpected Task-4 labels: {unknown}"
        )

    X = df[feature_cols].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Feature matrix contains NaN/Inf."
        )

    if args.standardize:
        X_pca_input = StandardScaler().fit_transform(X)
        preprocessing = "z-score standardization + PCA"
    else:
        X_pca_input = X
        preprocessing = "PCA on raw learned embeddings"

    pca = PCA(
        n_components=2,
        svd_solver="full",
    )

    Z = pca.fit_transform(X_pca_input)

    evr = (
        pca.explained_variance_ratio_
        * 100.0
    )

    coord = df[
        ["subject_id", "center", "label"]
    ].copy()

    if "label_name" in df.columns:
        coord["label_name"] = df["label_name"]
    else:
        coord["label_name"] = coord[
            "label"
        ].map(LABEL_NAMES)

    coord["PC1"] = Z[:, 0]
    coord["PC2"] = Z[:, 1]

    coord.to_csv(
        out / "Fig4f_PCA_coordinates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        {
            "component": "PC1",
            "explained_variance_ratio": pca.explained_variance_ratio_[0],
            "explained_variance_percent": evr[0],
            "preprocessing": preprocessing,
        },
        {
            "component": "PC2",
            "explained_variance_ratio": pca.explained_variance_ratio_[1],
            "explained_variance_percent": evr[1],
            "preprocessing": preprocessing,
        },
    ]).to_csv(
        out / "Fig4f_PCA_variance.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(5.1, 4.3)
    )

    for label in LABEL_ORDER:
        mask = labels == label

        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=args.point_size,
            c=COLORS[label],
            alpha=args.alpha,
            label=(
                f"{LABEL_NAMES[label]} "
                f"(n={int(mask.sum())})"
            ),
            edgecolors="white",
            linewidths=0.45,
        )

    ax.set_xlabel(
        f"PC1 ({evr[0]:.1f}% variance)",
        fontsize=11,
    )

    ax.set_ylabel(
        f"PC2 ({evr[1]:.1f}% variance)",
        fontsize=11,
    )

    if args.show_title:
        ax.set_title(
            "PCA of subject-level representations",
            fontsize=11,
        )

    # Keep the light zero-reference lines used by the previous figure.
    ax.axhline(
        0,
        linewidth=0.55,
        alpha=0.18,
    )
    ax.axvline(
        0,
        linewidth=0.55,
        alpha=0.18,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=9,
        direction="out",
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="best",
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    ax.margins(0.08)

    fig.tight_layout()

    fig.savefig(
        out / "Fig4f_PCA_single_model.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        out / "Fig4f_PCA_single_model.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        out / "Fig4f_PCA_single_model.svg",
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n===== Fig.4f SINGLE-MODEL PCA =====")
    print(f"subjects: {len(df)}")
    print(f"feature_dim: {len(feature_cols)}")
    print(
        f"PC1 explained variance: {evr[0]:.2f}%"
    )
    print(
        f"PC2 explained variance: {evr[1]:.2f}%"
    )
    print(
        f"total PC1+PC2: {evr[0] + evr[1]:.2f}%"
    )
    print(f"preprocessing: {preprocessing}")

    print("class counts:")
    for label in LABEL_ORDER:
        print(
            f"  {LABEL_NAMES[label]}: "
            f"{int(np.sum(labels == label))}"
        )

    print(
        f"[DONE] "
        f"{out / 'Fig4f_PCA_single_model.png'}"
    )


if __name__ == "__main__":
    main()
