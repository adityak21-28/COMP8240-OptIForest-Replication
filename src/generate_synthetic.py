#!/usr/bin/env python3
"""Generate a reproducible labelled synthetic anomaly-detection benchmark.

The normal population contains correlated features. Three labelled anomaly
regimes are injected: point, contextual, and clustered. Labels are for
EVALUATION only; they must not be included in model features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_SEED = 8240


def _normal_population(n: int, rng: np.random.Generator, n_features: int) -> np.ndarray:
    """Generate a normal population with a few correlated feature pairs."""
    x = rng.normal(0.0, 1.0, size=(n, n_features))
    if n_features >= 2:
        x[:, 1] = 0.80 * x[:, 0] + rng.normal(0.0, 0.35, size=n)
    if n_features >= 3:
        x[:, 2] = -0.65 * x[:, 0] + rng.normal(0.0, 0.45, size=n)
    if n_features >= 4:
        x[:, 3] = 0.55 * x[:, 2] + rng.normal(0.0, 0.50, size=n)
    return x


def _allocate(total: int, count: int) -> List[int]:
    base, rem = divmod(total, count)
    return [base + (1 if i < rem else 0) for i in range(count)]


def generate_dataset(
    n_normal: int = 20_000,
    n_anomalies: int = 1_000,
    n_features: int = 10,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    if n_normal < 100 or n_anomalies < 30 or n_features < 4:
        raise ValueError("Use at least 100 normal rows, 30 anomaly rows, and 4 features.")

    rng = np.random.default_rng(seed)
    normal = _normal_population(n_normal, rng, n_features)
    normal_df = pd.DataFrame(normal, columns=[f"feature_{i+1}" for i in range(n_features)])
    normal_df["label"] = 0
    normal_df["anomaly_type"] = "normal"
    normal_df["difficulty"] = "none"

    regimes = ["point", "contextual", "clustered"]
    levels = ["low", "medium", "high"]
    regime_counts = _allocate(n_anomalies, len(regimes))
    chunks: List[pd.DataFrame] = []
    scales: Dict[str, Dict[str, float]] = {
        "point": {"low": 2.0, "medium": 2.8, "high": 3.6},
        "clustered": {"low": 1.4, "medium": 2.1, "high": 2.8},
    }

    for regime, regime_n in zip(regimes, regime_counts):
        level_counts = _allocate(regime_n, len(levels))
        for level, n_here in zip(levels, level_counts):
            if n_here == 0:
                continue
            x = _normal_population(n_here, rng, n_features)

            if regime == "point":
                # Shift 1-3 randomly selected features. Higher levels yield
                # larger shifts, while retaining variation in direction.
                magnitude = scales[regime][level]
                for row in range(n_here):
                    n_shift = int(rng.integers(1, min(3, n_features) + 1))
                    cols = rng.choice(n_features, size=n_shift, replace=False)
                    signs = rng.choice([-1.0, 1.0], size=n_shift)
                    x[row, cols] += signs * magnitude

            elif regime == "contextual":
                # Contextual anomalies break conditional relationships while
                # approximately preserving each feature's marginal distribution.
                # The normal generator defines x1 positively from x0, x2
                # negatively from x0, and x3 positively from x2. Because the
                # Gaussian noise is symmetric, reversing the sign of a
                # relationship preserves the theoretical marginal distribution.
                # Difficulty increases by reversing one, two, then three
                # relationships; this is a controlled conditional-structure test.
                x[:, 1] = -0.80 * x[:, 0] + rng.normal(0.0, 0.35, size=n_here)
                if level in {"medium", "high"} and n_features >= 3:
                    x[:, 2] = 0.65 * x[:, 0] + rng.normal(0.0, 0.45, size=n_here)
                if n_features >= 4:
                    # Recreate x3 from the current x2 so medium anomalies
                    # preserve this dependency, while high anomalies reverse it.
                    coefficient = -0.55 if level == "high" else 0.55
                    x[:, 3] = coefficient * x[:, 2] + rng.normal(0.0, 0.50, size=n_here)

            elif regime == "clustered":
                # Compact group. Difficulty controls its offset from normal.
                shift = scales[regime][level]
                cluster_center = np.array([shift, -shift, shift / 1.7])
                x[:, :3] = cluster_center + rng.normal(0.0, 0.28, size=(n_here, 3))

            chunk = pd.DataFrame(x, columns=[f"feature_{i+1}" for i in range(n_features)])
            chunk["label"] = 1
            chunk["anomaly_type"] = regime
            chunk["difficulty"] = level
            chunks.append(chunk)

    df = pd.concat([normal_df, *chunks], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "sample_id", np.arange(1, len(df) + 1))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=int, default=20_000)
    parser.add_argument("--anomalies", type=int, default=1_000)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_anomaly_benchmark_v1.csv"))
    args = parser.parse_args()

    df = generate_dataset(args.normal, args.anomalies, args.features, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    label_counts = df["label"].value_counts().sort_index().to_dict()
    type_counts = df["anomaly_type"].value_counts().sort_index().to_dict()
    difficulty_counts = df.loc[df.label == 1, "difficulty"].value_counts().sort_index().to_dict()
    feature_columns = [c for c in df.columns if c.startswith("feature_")]
    metadata = {
        "dataset_name": "synthetic_anomaly_benchmark_v1",
        "dataset_version": "1.0.0",
        "seed": args.seed,
        "rows": int(len(df)),
        "features": int(len(feature_columns)),
        "normal_rows": int((df.label == 0).sum()),
        "anomaly_rows": int((df.label == 1).sum()),
        "anomaly_rate": float(df.label.mean()),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "anomaly_type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "anomaly_difficulty_counts": {str(k): int(v) for k, v in difficulty_counts.items()},
        "feature_columns": feature_columns,
        "label_column": "label",
        "metadata_columns": ["sample_id", "anomaly_type", "difficulty"],
        "methodology": "Correlated Gaussian normal population plus controlled point shifts, contextual relationship reversals with preserved theoretical univariate Gaussian marginals, and compact shifted clusters.",
        "contextual_design": "Low/medium/high contextual anomalies reverse one/two/three feature dependencies; because the normal features are Gaussian with symmetric noise, sign reversals preserve the intended univariate Gaussian marginals while changing conditional relationships.",
        "data_license": "CC BY 4.0; original synthetic data generated by this project.",
    }
    meta_path = args.output.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote dataset: {args.output}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
