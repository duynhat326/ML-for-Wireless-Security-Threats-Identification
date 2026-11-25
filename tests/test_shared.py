import numpy as np
import pandas as pd

from shared.compare import compare
from shared.config import LABEL_TO_INDEX, LABELS
from shared.data_loader import (
    cap_normal, clean_features, fit_schema, split_features_labels, temporal_split,
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
    schema = fit_schema(features)
    cleaned = clean_features(features, schema)
    assert "feature_two__missing" in cleaned and (cleaned["feature_two"] >= -1).all()
    assert cleaned["feature_two"].isna().sum() == 0
    assert list(clean_features(features[["feature_one"]], schema).columns) == list(cleaned.columns)


def test_temporal_split_is_ordered_with_gap():
    train, validation, test = temporal_split(_sample_dataframe(100))
    assert train["frame.time_epoch"].max() < validation["frame.time_epoch"].min()
    assert validation["frame.time_epoch"].max() < test["frame.time_epoch"].min()
    assert len(train) + len(validation) + len(test) < 100


def test_cap_normal_only_reduces_normal():
    features, labels, _ = split_features_labels(_sample_dataframe())
    _, capped = cap_normal(features, labels, 20)
    assert (capped == LABEL_TO_INDEX["Deauth"]).sum() == 10
    assert (capped == LABEL_TO_INDEX["Normal"]).sum() == 20


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
