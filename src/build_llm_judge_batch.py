#!/usr/bin/env python3
"""Prepare an LLM-judge batch without leaking the ground-truth label.

This script does not call an LLM. It prepares structured feature records for
review of explanation quality. Ground-truth labels and anomaly-type metadata
are intentionally omitted from the JSONL payload.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="CSV containing feature_* columns and an anomaly_score column")
    parser.add_argument("--score-col", default="anomaly_score")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/llm_judge_batch.jsonl"))
    args = parser.parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols or args.score_col not in df.columns:
        raise ValueError("Input CSV must contain feature_* columns and the requested score column")
    if df[feature_cols].isna().any().any() or not all(pd.api.types.is_numeric_dtype(df[c]) for c in feature_cols):
        raise ValueError("Feature columns must be numeric and have no missing values")
    ranked = df.sort_values(args.score_col, ascending=False).head(args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            payload = {
                "case_id": str(row.get("sample_id", rank)),
                "rank": rank,
                "anomaly_score": float(row[args.score_col]),
                "feature_values": {col: float(row[col]) for col in feature_cols},
                "judge_instruction": (
                    "Assess only whether a proposed natural-language explanation is supported by these feature values. "
                    "Do not infer a ground-truth label from the anomaly score alone. If no explanation is provided, "
                    "return a concise description of observable numerical deviations and state uncertainty."
                ),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Prepared {len(ranked)} label-free records at {args.output}")
    print("This script does not invoke an LLM. Record model/version, prompt, date and raw output if a judge is run.")

if __name__ == "__main__":
    main()
