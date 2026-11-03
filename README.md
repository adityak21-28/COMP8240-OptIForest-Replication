# COMP8240 OptIForest — Replication, New Data and Analysis

Code, data and logs for *Reproducing OptIForest and Testing Its Generalisation to New Data* (Aditya Kamble, Macquarie University). The project reproduces the AD experiment of

> Xiang et al. 2023. *OptIForest: Optimal Isolation Forest for Anomaly Detection.* IJCAI-23, pp. 2379–2387. https://doi.org/10.24963/ijcai.2023/264

with the authors' code (https://github.com/xiagll/OptIForest — **linked, not copied**, because the upstream repository has no licence file), then tests it on new data.

## Results

| Dataset | Rows (anomalies) | Trees / runs | ROC-AUC | Average precision | Status |
|---|---|---|---|---|---|
| AD (original) | 3,279 (459) | 100 / 15 | 0.7758 ± 0.0059 | 0.4534 ± 0.0365 | Replication (paper: 0.774 ± 0.007; AUC-PR 0.438 ± 0.031) |
| Synthetic v1 | 21,000 (1,000) | 100 / 15 | 0.7828 ± 0.0044 | 0.3364 ± 0.0169 | Constructed data |
| Credit card, OpenML 1597 (sample) | 10,000 (17) | 10 / 5 | 0.8368 ± 0.0160 | 0.00534 ± 0.00057 | Reduced-scale run |
| Annthyroid, OpenML 40886 | 7,200 (534) | 100 / 15 | 0.5707 ± 0.0186 | 0.1287 ± 0.0133 | Exploratory (OpenML record "in preparation") |

Source: `results/experiment_summary.csv` and the logs below. The scikit-learn numbers in `results/isolation_forest_baseline_*.csv` are a **reference baseline, not OptIForest**.

## Layout

| Path | Contents |
|---|---|
| `data/` | Synthetic benchmark v1 (CSV, SHA-256, metadata), OpenML metadata for credit card and Annthyroid |
| `src/` | Generator, audit, baseline, OpenML downloaders, exporter to the upstream format, plots |
| `scripts/` | `run_pipeline.sh` (synthetic data + baseline + tests), `make_creditcard_diagnostic.sh` (reduced credit-card runner) |
| `results/` | OptIForest logs (stdout of `demo.py`), summary CSVs, audit JSON |
| `figures/`, `tests/`, `docs/` | Figures, six unit tests, LLM-judge protocol (prepared, not executed) |

Raw credit-card and Annthyroid CSVs are git-ignored; recreate them with the download scripts.

## 1. Synthetic data, baseline, tests (Python 3.10–3.12)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_pipeline.sh
```

Regenerates the dataset (seed 8240; checksum matches `data/synthetic_anomaly_benchmark_v1.sha256`), audits it, draws figures, runs the baseline (200 trees, 15 seeds) and the tests. `sample_id`, `label`, `anomaly_type` and `difficulty` are never model inputs.

## 2. Official OptIForest runs

Create a separate environment (the upstream pins do not install on current Python):

```bash
python -m venv .venv-official && source .venv-official/bin/activate
python -m pip install -r requirements-official.txt     # Python 3.12.14 / macOS was used
git clone https://github.com/xiagll/OptIForest external/OptIForest
```

Original AD replication — run from `external/OptIForest`:

```bash
python demo.py --dataset=ad --threshold=403 --branch=0 > ../../results/optiforest_ad_clean_run.log
```

New datasets — prepare, export to the upstream headerless-CSV format, then run `demo.py` with the same arguments:

```bash
python src/export_for_official_optiforest.py --csv data/synthetic_anomaly_benchmark_v1.csv --repo external/OptIForest --name synthetic_v1
python src/download_annthyroid_openml.py
python src/export_for_official_optiforest.py --csv data/annthyroid_openml_40886.csv --repo external/OptIForest --name annthyroid_openml_40886
python src/download_creditcard_openml.py        # 10,000-row stratified sample, seed 8240
python src/export_for_official_optiforest.py --csv data/creditcard_openml_1597.csv --repo external/OptIForest --name creditcard_openml_1597

cd external/OptIForest
python demo.py --dataset=synthetic_v1 --threshold=403 --branch=0 > ../../results/optiforest_synthetic_v1.log
python demo.py --dataset=annthyroid_openml_40886 --threshold=403 --branch=0 > ../../results/optiforest_annthyroid_openml_40886.log

# credit card: reduced runner (10 trees, 5 runs) derived from demo.py
bash ../../scripts/make_creditcard_diagnostic.sh .
python demo_creditcard_diagnostic.py --dataset=creditcard_openml_1597 --threshold=403 --branch=0 > ../../results/optiforest_creditcard_10trees_5runs.log
```

Notes: `demo.py` has no seed option, fits and scores on the same data, and its "PR score" is `sklearn.metrics.average_precision_score`. Printed standard deviations are population SDs (`np.std`). A 100-tree run on a 50,000-row credit-card sample was interrupted after more than 12 hours, which is why the reduced runner exists.

| Experiment | Log |
|---|---|
| AD replication | `results/optiforest_ad_clean_run.log` |
| Synthetic v1 | `results/optiforest_synthetic_v1.log` |
| Annthyroid 40886 | `results/optiforest_annthyroid_openml_40886.log` |
| Credit card, 10 trees, 5 runs | `results/optiforest_creditcard_10trees_5runs.log` |

## Licences

Code: MIT (`LICENSE`). Synthetic dataset: CC BY 4.0 (`DATA_LICENSE.md`), attribution to Aditya Kamble. Third-party code and data (upstream OptIForest, OpenML credit card and Annthyroid) keep their own terms and are not redistributed.
