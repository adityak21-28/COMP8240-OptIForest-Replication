#!/usr/bin/env python3
"""Create report-ready plots from the current synthetic benchmark CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/synthetic_anomaly_benchmark_v1.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--plot-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    features = [name for name in df.columns if name.startswith("feature_")]
    if len(features) < 2:
        raise ValueError("CSV must contain at least two feature_* columns")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be binary (0=normal, 1=anomaly)")
    if df[features].isna().any().any():
        raise ValueError("Feature matrix has missing values")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X = df[features].to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD-based two-dimensional projection: deterministic and does not require
    # fitting another predictive model.
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    projection = X @ vt[:2].T
    rng = np.random.default_rng(args.seed)
    n_plot = min(args.plot_sample, len(df))
    idx = np.sort(rng.choice(len(df), size=n_plot, replace=False))
    subset = df.iloc[idx]
    projected = projection[idx]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    category_order = ["normal", "point", "contextual", "clustered"]
    for category in category_order:
        mask = subset["anomaly_type"].to_numpy() == category
        if mask.any():
            ax.scatter(projected[mask, 0], projected[mask, 1], s=7 if category == "normal" else 12,
                       alpha=0.35 if category == "normal" else 0.75, label=category)
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")
    ax.set_title("Synthetic anomaly benchmark v1 (sampled 2D projection)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "synthetic_pca.png", dpi=180)
    plt.close(fig)

    normal = df.loc[df["label"] == 0, features]
    corr = normal.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)), [f.replace("feature_", "f") for f in features], rotation=45, ha="right")
    ax.set_yticks(range(len(features)), [f.replace("feature_", "f") for f in features])
    ax.set_title("Feature correlations in the normal population")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(args.out_dir / "normal_feature_correlations.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'synthetic_pca.png'}")
    print(f"Wrote {args.out_dir / 'normal_feature_correlations.png'}")


if __name__ == "__main__":
    main()
