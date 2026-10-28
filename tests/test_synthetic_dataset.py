from src.generate_synthetic import generate_dataset


def test_default_dataset_shape_and_labels():
    df = generate_dataset(seed=1234)
    assert len(df) == 21_000
    assert sum(c.startswith("feature_") for c in df.columns) == 10
    assert int((df.label == 0).sum()) == 20_000
    assert int((df.label == 1).sum()) == 1_000
    assert set(df.label.unique()) == {0, 1}
    assert set(df.anomaly_type.unique()) == {"normal", "point", "contextual", "clustered"}
    assert df.isna().sum().sum() == 0


def test_generation_is_deterministic():
    first = generate_dataset(n_normal=200, n_anomalies=60, seed=10)
    second = generate_dataset(n_normal=200, n_anomalies=60, seed=10)
    assert first.equals(second)


def test_audit_passes_for_generated_dataset(tmp_path):
    import pandas as pd
    from src.audit_dataset import audit

    df = generate_dataset(n_normal=300, n_anomalies=60, n_features=5, seed=4)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    report = audit(path)
    assert report["rows"] == 360
    assert report["numeric_feature_count"] == 5
    assert report["passed_basic_checks"] is True
    assert report["missing_by_column"]["label"] == 0

def test_export_excludes_metadata_from_model_features(tmp_path):
    import pandas as pd
    from src.export_for_official_optiforest import export

    repo = tmp_path / "upstream"
    (repo / "data").mkdir(parents=True)
    df = generate_dataset(n_normal=300, n_anomalies=60, n_features=5, seed=5)
    source = tmp_path / "source.csv"
    df.to_csv(source, index=False)
    destination = export(source, repo, "test_data")
    exported = pd.read_csv(destination, header=None)
    assert exported.shape == (360, 6)  # five features plus label
    assert exported.iloc[:, -1].isin([0, 1]).all()


def test_export_rejects_nonbinary_labels(tmp_path):
    import pandas as pd
    import pytest
    from src.export_for_official_optiforest import export

    repo = tmp_path / "upstream"
    (repo / "data").mkdir(parents=True)
    df = generate_dataset(n_normal=300, n_anomalies=60, n_features=5, seed=6)
    df.loc[df.index[0], "label"] = 2
    source = tmp_path / "invalid.csv"
    df.to_csv(source, index=False)
    with pytest.raises(ValueError, match="binary"):
        export(source, repo, "invalid_data")


def test_contextual_anomalies_reverse_feature_relationships():
    # With a fixed seed and sufficient rows, the contextual subsets should
    # display a sign-reversed relationship between feature_1 and feature_2.
    df = generate_dataset(n_normal=6000, n_anomalies=900, seed=2468)
    normal = df[df.anomaly_type == "normal"]
    contextual = df[df.anomaly_type == "contextual"]
    normal_corr = normal[["feature_1", "feature_2"]].corr().iloc[0, 1]
    contextual_corr = contextual[["feature_1", "feature_2"]].corr().iloc[0, 1]
    assert normal_corr > 0.60
    assert contextual_corr < -0.60
