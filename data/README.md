# Synthetic Anomaly Benchmark v1

This dataset is an original, programmatically generated benchmark for the COMP8240 OptIForest reproduction project. It contains no records copied from a real-person or third-party dataset.

## Version and schema

- Dataset version: `1.0.0`
- Generator: `src/generate_synthetic.py`
- Default random seed: `8240`
- Rows: `21,000`
- Numeric input features: `feature_1` through `feature_10`
- Normal observations: `20,000` (`label = 0`)
- Anomalies: `1,000` (`label = 1`)
- Overall anomaly prevalence: `1,000 / 21,000 = 4.7619%`

Columns:

| Column | Meaning | Use as model input? |
|---|---|---|
| `sample_id` | Stable row identifier in the frozen generated file | No |
| `feature_1` … `feature_10` | Numeric features | Yes |
| `label` | Ground truth: 0 normal, 1 anomaly | No; evaluation only |
| `anomaly_type` | `normal`, `point`, `contextual`, or `clustered` | No; evaluation subgroup only |
| `difficulty` | `none`, `low`, `medium`, or `high` | No; evaluation subgroup only |

## Generation design

The normal population is multivariate Gaussian with deliberately correlated feature pairs. The generator inserts three anomaly regimes:

1. **Point anomalies:** shift one to three features in individual observations, with the shift magnitude changing by difficulty.
2. **Contextual anomalies:** reverse one, two, or three conditional feature relationships at low, medium, and high difficulty. Because the features and noise are Gaussian and symmetric, the intended univariate feature marginals are preserved in distribution while the dependencies change.
3. **Clustered anomalies:** create a compact group displaced from the normal population in the first three dimensions. Difficulty controls displacement.

Anomalies are distributed nearly evenly across the three regimes and across difficulty levels. Because there are 1,000 anomalies, exact equality across three groups is impossible; the generator allocates 334, 333 and 333.

## Recreate the frozen CSV

From the project root, after installing `requirements.txt`:

```bash
python src/generate_synthetic.py --output data/synthetic_anomaly_benchmark_v1.csv --seed 8240
python src/audit_dataset.py data/synthetic_anomaly_benchmark_v1.csv --output results/synthetic_dataset_audit.json
python src/plot_synthetic_dataset.py --csv data/synthetic_anomaly_benchmark_v1.csv --out-dir figures
python -m pytest -q
```

The frozen CSV and JSON metadata are provided so a reviewer can inspect the exact data without first running the generator. The generator and seed allow it to be recreated.

## Caveats

This is a controlled synthetic benchmark, not a substitute for external real-world evaluation. Results depend on the specific generation rules and random seed. `label`, `anomaly_type`, `difficulty`, and `sample_id` must not be included in the detector's feature matrix. The dataset includes ground-truth labels by construction, so AUC-ROC and average precision can be computed directly.

## Licence

The original generated data may be shared under **CC BY 4.0** with attribution to Aditya Kamble and the project repository. The generator code is covered by the repository's MIT licence. Third-party software and datasets have separate terms.
