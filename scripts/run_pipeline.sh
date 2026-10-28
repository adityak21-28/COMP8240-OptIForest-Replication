#!/usr/bin/env bash
set -euo pipefail
python src/generate_synthetic.py --output data/synthetic_anomaly_benchmark_v1.csv --seed 8240
python src/audit_dataset.py data/synthetic_anomaly_benchmark_v1.csv --output results/synthetic_dataset_audit.json
python src/plot_synthetic_dataset.py --csv data/synthetic_anomaly_benchmark_v1.csv --out-dir figures
python src/evaluate_isolation_forest_baseline.py --csv data/synthetic_anomaly_benchmark_v1.csv --runs 15 --trees 200 --output results/isolation_forest_baseline_runs.csv
python src/plot_results.py --summary results/isolation_forest_baseline_summary.csv --out-dir figures
python -m pytest -q
