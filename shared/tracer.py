import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from shared.config import IMPERSONATION_LABELS, LABELS, NETWORK_LAYER_LABELS, NORMAL_INDEX, TIME_COLUMN
from shared.data_loader import clean_features, split_features_labels


def save_bundle(path, model, schema):
    joblib.dump({"model": model, "schema": schema}, path)


class ScikitLearnPredictor:
    def __init__(self, model, schema):
        self.model, self.schema = model, schema

    def predict_proba(self, features):
        probabilities = self.model.predict_proba(features)
        full_probabilities = np.zeros((len(features), len(LABELS)), dtype="float32")
        full_probabilities[:, np.asarray(self.model.classes_, dtype=int)] = probabilities
        return full_probabilities


def load_bundle(path):
    bundle = joblib.load(path)
    return ScikitLearnPredictor(bundle["model"], bundle["schema"])


def make_mlp(input_size, output_size, hidden_sizes=(256, 128), dropout=0.2):
    from torch import nn

    layers, width = [], input_size
    for hidden_size in hidden_sizes:
        layers += [nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden_size
    return nn.Sequential(*layers, nn.Linear(width, output_size))


class MlpPredictor:
    def __init__(self, path):
        import torch

        self.torch = torch
        bundle = torch.load(path, map_location="cpu")
        self.schema = bundle["schema"]
        self.mean = np.asarray(bundle["mean"], dtype="float32")
        self.scale = np.asarray(bundle["scale"], dtype="float32")
        self.network = make_mlp(bundle["input_size"], bundle["output_size"], tuple(bundle["hidden_sizes"]))
        self.network.load_state_dict(bundle["state_dict"])
        self.network.eval()

    def predict_proba(self, features):
        inputs = self.torch.tensor((features.to_numpy() - self.mean) / self.scale, dtype=self.torch.float32)
        with self.torch.no_grad():
            return self.torch.softmax(self.network(inputs), dim=1).numpy()


def load_model(path):
    return MlpPredictor(path) if str(path).endswith(".pt") else load_bundle(path)


def source_columns(label):
    if label in IMPERSONATION_LABELS:
        return ["wlan.ta", "wlan.sa"]
    if label in NETWORK_LAYER_LABELS:
        return ["ip.src", "wlan.sa"]
    return ["wlan.sa", "wlan.ta"]


def trace_source(frames, label):
    for column in source_columns(label):
        if column in frames and frames[column].notna().any():
            counts = frames[column].value_counts()
            return {"column": column, "value": str(counts.index[0]),
                    "share": round(float(counts.iloc[0]) / len(frames), 3)}
    return None


def _to_iso_time(timestamp):
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat(timespec="milliseconds")


def build_alerts(metadata, predicted_indices, confidences, minimum_confidence=0.5, gap_seconds=1.0):
    metadata = metadata.assign(
        predicted_index=predicted_indices, confidence=confidences,
        timestamp=pd.to_numeric(metadata[TIME_COLUMN], errors="coerce"))
    attacks = metadata[(metadata.predicted_index != NORMAL_INDEX) & (metadata.confidence >= minimum_confidence)]
    attacks = attacks.dropna(subset=["timestamp"]).sort_values("timestamp")
    attacks = attacks.assign(incident=attacks.groupby("predicted_index")["timestamp"].transform(
        lambda times: (times.diff() > gap_seconds).cumsum()))
    alerts = []
    for (class_index, _), frames in attacks.groupby(["predicted_index", "incident"]):
        label = LABELS[int(class_index)]
        has_bssid = "wlan.bssid" in frames and frames["wlan.bssid"].notna().any()
        alerts.append({
            "time_start": _to_iso_time(frames.timestamp.min()),
            "time_end": _to_iso_time(frames.timestamp.max()),
            "attack": label,
            "confidence": round(float(frames.confidence.mean()), 3),
            "frames": int(len(frames)),
            "source": trace_source(frames, label),
            "impersonation": label in IMPERSONATION_LABELS,
            "bssid": str(frames["wlan.bssid"].mode().iloc[0]) if has_bssid else None,
            "signal_strength_dbm_median": float(
                pd.to_numeric(frames["wlan_radio.signal_dbm"], errors="coerce").median())
            if "wlan_radio.signal_dbm" in frames else None,
        })
    return sorted(alerts, key=lambda alert: alert["time_start"])


def detect_and_trace(dataframe, model, minimum_confidence=0.5, gap_seconds=1.0):
    features, _, metadata = split_features_labels(dataframe)
    probabilities = model.predict_proba(clean_features(features, model.schema))
    return build_alerts(metadata, probabilities.argmax(axis=1), probabilities.max(axis=1),
                        minimum_confidence, gap_seconds)


def write_alerts(alerts, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(alerts, file, ensure_ascii=False, indent=2)
