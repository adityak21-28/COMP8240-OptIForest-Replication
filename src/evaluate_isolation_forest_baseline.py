#!/usr/bin/env python3
"""Run a reproducible scikit-learn IsolationForest baseline.

This is a baseline/sanity check, NOT the OptIForest implementation. Keep its
results separate from any results produced by the authors' repository.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score


def metric_row(scope: str, y_true: np.ndarray, scores: np.ndarray, seed: int, n: int) -> dict:
    return {
        "model": "sklearn_IsolationForest_baseline",
        "scope": scope,
        "seed": seed,
        "n_samples": int(n),
        "n_anomalies": int(y_true.sum()),
        "anomaly_rate": float(y_true.mean()),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def evaluate(path: Path, label_col: str, n_runs: int, n_estimators: int, base_seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}.")
    meta_cols = {label_col, "sample_id", "anomaly_type", "difficulty"}
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in meta_cols]
    X_df = df[feature_cols]
    if X_df.isna().any().any() or not all(pd.api.types.is_numeric_dtype(X_df[c]) for c in feature_cols):
        raise ValueError("Features must be numeric and have no missing values.")
    X = X_df.to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Label column must contain both 0 (normal) and 1 (anomaly).")

    rows = []
    anomaly_types = sorted(set(df.get("anomaly_type", pd.Series([], dtype=str)).dropna()) - {"normal"})
    for run in range(n_runs):
        seed = base_seed + run
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=min(256, len(df)),
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X)
        # sklearn's lower score_samples means more anomalous; invert for AUC.
        scores = -model.score_samples(X)
        rows.append(metric_row("all_anomalies", y, scores, seed, len(y)))
        if "anomaly_type" in df.columns:
            for kind in anomaly_types:
                mask = (df.anomaly_type == "normal") | (df.anomaly_type == kind)
                yy = (df.loc[mask, "anomaly_type"].to_numpy() == kind).astype(int)
                rows.append(metric_row(kind, yy, scores[mask.to_numpy()], seed, int(mask.sum())))
        if "difficulty" in df.columns:
            for level in ("low", "medium", "high"):
                mask = (df.difficulty == "none") | (df.difficulty == level)
                yy = (df.loc[mask, "difficulty"].to_numpy() != "none").astype(int)
                rows.append(metric_row(f"difficulty_{level}", yy, scores[mask.to_numpy()], seed, int(mask.sum())))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("data/synthetic_anomaly_benchmark_v1.csv"))
    p.add_argument("--label", default="label")
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--seed", type=int, default=8240)
    p.add_argument("--output", type=Path, default=Path("results/isolation_forest_baseline_runs.csv"))
    a = p.parse_args()
    runs = evaluate(a.csv, a.label, a.runs, a.trees, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(a.output, index=False)
    keys = ["model", "scope"]
    summary = runs.groupby(keys).agg(
        n_runs=("seed", "nunique"),
        auc_roc_mean=("auc_roc", "mean"),
        auc_roc_sd=("auc_roc", "std"),
        ap_mean=("average_precision", "mean"),
        ap_sd=("average_precision", "std"),
        n_samples=("n_samples", "first"),
        n_anomalies=("n_anomalies", "first"),
    ).reset_index()
    summary_path = a.output.with_name(a.output.stem.replace("_runs", "_summary") + ".csv")
    summary.to_csv(summary_path, index=False)
    print("Baseline run results:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nRun-level results: {a.output}")
    print(f"Summary: {summary_path}")
    print("NOTE: These are scikit-learn IsolationForest baseline results, not OptIForest results.")

if __name__ == "__main__":
    main()
