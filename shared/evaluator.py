import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from shared.config import LABELS


def measure_latency_milliseconds(predict, features, sample_size=1000):
    sample = features.iloc[:sample_size] if hasattr(features, "iloc") else features[:sample_size]
    start = time.perf_counter()
    predict(sample)
    return (time.perf_counter() - start) * 1000 / max(len(sample), 1)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_figure(figure, path):
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(true_labels, predicted_labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_indices = list(range(len(LABELS)))
    counts = confusion_matrix(true_labels, predicted_labels, labels=label_indices).astype(float)
    normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    figure, plot_axes = plt.subplots(figsize=(9, 8))
    image = plot_axes.imshow(normalized, cmap="Oranges", vmin=0, vmax=1)
    plot_axes.set_xticks(label_indices, LABELS, rotation=60, ha="right")
    plot_axes.set_yticks(label_indices, LABELS)
    plot_axes.set_xlabel("Predicted")
    plot_axes.set_ylabel("True")
    plot_axes.set_title("Confusion matrix (row-normalized)")
    figure.colorbar(image, ax=plot_axes)
    _save_figure(figure, path)


def plot_feature_importance(names, values, path, top=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values)
    indices = np.argsort(values)[-top:]
    figure, plot_axes = plt.subplots(figsize=(8, 6))
    plot_axes.barh(np.asarray(names)[indices], values[indices], color="#cc785c")
    plot_axes.set_title(f"Top {top} feature importances")
    _save_figure(figure, path)


def plot_training_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(10, 4))
    loss_axes.plot(history["loss"], color="#cc785c")
    loss_axes.set_title("Train loss")
    loss_axes.set_xlabel("Epoch")
    accuracy_axes.plot(history["val_accuracy"], color="#5db8a6")
    accuracy_axes.set_title("Validation accuracy")
    accuracy_axes.set_xlabel("Epoch")
    _save_figure(figure, path)


def save_results(output_directory, model_name, true_labels, predicted_labels,
                 train_time_seconds, latency_milliseconds, feature_importance=None):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    true_labels, predicted_labels = np.asarray(true_labels), np.asarray(predicted_labels)
    label_indices = list(range(len(LABELS)))
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=np.unique(true_labels),
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, predicted_labels, labels=label_indices,
                                      average="weighted", zero_division=0)),
        "train_time_seconds": float(train_time_seconds),
        "predict_latency_milliseconds": float(latency_milliseconds),
    }
    save_json(metrics, output / "metrics.json")
    report = classification_report(true_labels, predicted_labels, labels=label_indices,
                                   target_names=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.loc[LABELS + ["macro avg", "weighted avg"]].to_csv(
        output / "per_class_report.csv", index_label="class")
    plot_confusion_matrix(true_labels, predicted_labels, output / "confusion_matrix.png")
    if feature_importance is not None:
        plot_feature_importance(*feature_importance, output / "feature_importance.png")
    return metrics
