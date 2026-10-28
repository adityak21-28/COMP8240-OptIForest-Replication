#!/usr/bin/env python3
"""Export a labelled CSV to the original OptIForest repository's common format.

The upstream README demonstrates loading named datasets through demo.py. This
script writes a headerless numeric matrix with the binary label as the final
column into the upstream data/ folder. Verify the loader in demo.py before
running; the upstream repository is not copied into this project because its
repository page does not declare a licence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def export(csv_path: Path, repo_path: Path, dataset_name: str, label_col: str = "label") -> Path:
    if not dataset_name.replace("_", "").isalnum():
        raise ValueError("dataset name must contain only letters, numbers, and underscores")
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} is missing.")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        excluded = {label_col, "sample_id", "anomaly_type", "difficulty"}
        feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    matrix = df[feature_cols].copy()
    if not all(pd.api.types.is_numeric_dtype(matrix[c]) for c in feature_cols):
        raise ValueError("All model features must be numeric.")
    if df[label_col].isna().any() or not set(pd.Series(df[label_col]).unique()).issubset({0, 1}):
        raise ValueError("Labels must be complete and binary: 0=normal, 1=anomaly.")
    if set(pd.Series(df[label_col]).unique()) != {0, 1}:
        raise ValueError("Input must contain both normal (0) and anomaly (1) cases.")
    matrix[label_col] = df[label_col].astype(int)
    if matrix.isna().any().any():
        raise ValueError("Input contains missing values. Clean it before export.")
    if not all(pd.Series(matrix[col]).map(lambda value: pd.notna(value) and abs(float(value)) != float("inf")).all() for col in feature_cols):
        raise ValueError("Features contain infinite or non-finite values.")
    data_dir = repo_path / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Could not find upstream data directory: {data_dir}")
    dest = data_dir / f"{dataset_name}.csv"
    matrix.to_csv(dest, index=False, header=False)
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("data/synthetic_anomaly_benchmark_v1.csv"))
    p.add_argument("--repo", type=Path, required=True, help="Path to your locally cloned xiagll/OptIForest repository")
    p.add_argument("--name", default="synthetic_v1")
    p.add_argument("--label", default="label")
    a = p.parse_args()
    dest = export(a.csv, a.repo, a.name, a.label)
    print(f"Exported {a.csv} to {dest}")
    print("Inspect the upstream demo.py data-loading logic to confirm this headerless, label-last format is what it expects.")
    print(f"Then try: cd {a.repo} && python demo.py --dataset={a.name} --threshold=403 --branch=0")
    print("If demo.py uses a hard-coded dataset registry, add the new dataset name/path there without changing the detector code.")

if __name__ == "__main__":
    main()
