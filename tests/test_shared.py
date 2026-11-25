import numpy as np
import pandas as pd

from shared.compare import compare
from shared.config import LABEL_TO_INDEX, LABELS
from shared.data_loader import (
    assign_splits, clean_features, load_splits, split_features_labels, split_parquet,
)
from shared.evaluator import save_results
from shared.tracer import build_alerts


def _sample_dataframe(row_count=100):
    return pd.DataFrame({
        "frame.time_epoch": np.arange(row_count)[::-1].astype(float),
        "wlan.sa": ["a"] * row_count, "wlan_radio.channel": [36] * row_count,
        "feature_one": range(row_count),
        "feature_two": [None if index % 2 else 1 for index in range(row_count)],
        "Label": ["Normal"] * (row_count - 10) + ["Deauth"] * 10,
    })


def test_split_removes_identifiers_and_keeps_metadata():
    features, labels, metadata = split_features_labels(_sample_dataframe())
    assert "wlan.sa" not in features and "frame.time_epoch" not in features and "Label" not in features
    assert "wlan.sa" in metadata and labels.iloc[-1] == LABEL_TO_INDEX["Deauth"]


def test_clean_flags_missing_and_aligns_columns():
    features, _, _ = split_features_labels(_sample_dataframe())
    schema = {"columns": ["feature_one", "feature_two"], "flag_columns": ["feature_two"]}
    cleaned = clean_features(features, schema)
    assert "feature_two__missing" in cleaned and (cleaned["feature_two"] >= -1).all()
    assert cleaned["feature_two"].isna().sum() == 0
    assert list(clean_features(features[["feature_one"]], schema).columns) == list(cleaned.columns)


def test_save_results_and_compare(tmp_path):
    labels = np.array([0, 0, 1, 1, 2])
    for directory, name in (("model_1_random_forest", "random_forest"),
                            ("model_2_lightgbm", "lightgbm"), ("model_3_mlp", "mlp")):
        metrics = save_results(tmp_path / "experiments" / directory / "results", name,
                               labels, np.array([0, 0, 1, 0, 2]), 1.0, 0.1)
        assert metrics["accuracy"] == 0.8 and 0 < metrics["macro_f1"] <= 1
    summary, per_attack = compare(tmp_path)
    assert list(summary["model"]) == ["random_forest", "lightgbm", "mlp"]
    assert per_attack.shape == (len(LABELS) - 1, 3)
    assert (tmp_path / "comparison" / "comparison_chart.png").exists()


def test_build_alerts_groups_incidents_and_flags_impersonation():
    deauth = LABEL_TO_INDEX["Deauth"]
    metadata = pd.DataFrame({
        "frame.time_epoch": [1.0, 1.2, 1.4, 9.0, 9.1, 2.0],
        "wlan.ta": ["ap", "ap", "ap", "ap", "ap", "x"],
        "wlan.sa": ["s"] * 6,
        "wlan.bssid": ["ap"] * 6,
        "wlan_radio.signal_dbm": [-50.0] * 6,
    })
    predicted = np.array([deauth, deauth, deauth, deauth, deauth, 0])
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.99])
    alerts = build_alerts(metadata, predicted, confidences, minimum_confidence=0.5, gap_seconds=1.0)
    assert [alert["frames"] for alert in alerts] == [3, 2]
    assert alerts[0]["impersonation"] and alerts[0]["source"]["column"] == "wlan.ta"


def _labelled_times():
    labels = np.array(["Normal"] * 100 + ["Deauth"] * 100 + ["SSH"] * 100, dtype=object)
    timestamps = np.concatenate([np.arange(100.0) + offset for offset in (0, 1000, 2000)])
    return labels, timestamps


def test_assign_splits_is_ordered_per_label_and_keeps_every_row():
    labels, timestamps = _labelled_times()
    split = assign_splits(labels, timestamps)
    assert (split >= 0).all()
    for label in ("Normal", "Deauth", "SSH"):
        mask = labels == label
        train, validation, test = (timestamps[mask & (split == index)] for index in (0, 1, 2))
        assert len(train) and len(validation) and len(test)
        assert train.max() < validation.min() and validation.max() < test.min()
        assert len(train) + len(validation) + len(test) == mask.sum()


def test_assign_splits_leaves_a_gap_when_asked():
    labels, timestamps = _labelled_times()
    split = assign_splits(labels, timestamps, gap_fraction=0.01)
    assert (split == -1).sum() == 6


def test_assign_splits_caps_train_rows_per_label():
    labels, timestamps = _labelled_times()
    split = assign_splits(labels, timestamps, train_cap_per_label=10)
    assert all(((labels == label) & (split == 0)).sum() == 10 for label in ("Normal", "Deauth", "SSH"))


def test_split_parquet_writes_three_files_with_every_label(tmp_path):
    labels, timestamps = _labelled_times()
    source = tmp_path / "all.parquet"
    pd.DataFrame({"frame.time_epoch": timestamps, "feature": range(300), "Label": labels}).to_parquet(source)
    counts = split_parquet(source, tmp_path / "out", batch_size=70)
    assert counts["train"] + counts["val"] + counts["test"] == 300
    for name in ("train", "val", "test"):
        assert set(pd.read_parquet(tmp_path / "out" / f"{name}.parquet")["Label"]) == {"Normal", "Deauth", "SSH"}


def test_load_splits_streams_float32_features_and_keeps_metadata(tmp_path):
    labels, timestamps = _labelled_times()
    frame = pd.DataFrame({
        "frame.time_epoch": timestamps, "wlan.ta": "ap", "wlan_radio.signal_dbm": -50.0,
        "feature_one": np.arange(300.0), "feature_two": [None if i % 2 else "1" for i in range(300)],
        "empty": [None] * 300, "Label": labels,
    })
    source = tmp_path / "all.parquet"
    frame.to_parquet(source)
    split_parquet(source, tmp_path / "out", batch_size=70)
    splits = load_splits(tmp_path / "out")
    assert splits["schema"] == {"columns": ["wlan_radio.signal_dbm", "feature_one", "feature_two"],
                                "flag_columns": ["feature_two"]}
    features = splits["features_train"]
    assert list(features.columns) == ["wlan_radio.signal_dbm", "feature_one", "feature_two", "feature_two__missing"]
    assert str(features["feature_one"].dtype) == "float32" and len(features) == len(splits["labels_train"])
    assert (features["feature_two"][features["feature_two__missing"] == 1] == -1).all()
    assert "wlan.ta" in splits["metadata_test"] and len(splits["metadata_test"]) == len(splits["features_test"])
