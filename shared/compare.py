import json
from pathlib import Path

import pandas as pd

from shared.config import LABELS

MODEL_DIRECTORIES = ("model_1_random_forest", "model_2_lightgbm", "model_3_mlp")


def compare(root=None):
    root = Path(root) if root else Path.cwd()
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows, f1_by_model = [], {}
    for directory in MODEL_DIRECTORIES:
        results_directory = root / "experiments" / directory / "results"
        metrics = json.loads((results_directory / "metrics.json").read_text(encoding="utf-8"))
        rows.append(metrics)
        report = pd.read_csv(results_directory / "per_class_report.csv", index_col="class")
        f1_by_model[metrics["model"]] = report.loc[LABELS[1:], "f1-score"]
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(output / "summary_table.csv", index=False)
    per_attack = pd.DataFrame(f1_by_model).round(4)
    per_attack.index.name = "attack"
    per_attack.to_csv(output / "f1_per_attack_comparison.csv")
    _chart(summary, output / "comparison_chart.png")
    return summary, per_attack


def _chart(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
              ("train_time_seconds", "Train time (seconds)"),
              ("predict_latency_milliseconds", "Latency (milliseconds per frame)")]
    figure, all_axes = plt.subplots(1, len(panels), figsize=(15, 4))
    for plot_axes, (column, title) in zip(all_axes, panels):
        plot_axes.bar(summary["model"], summary[column], color=["#cc785c", "#5db8a6", "#e8a55a"][:len(summary)])
        plot_axes.set_title(title)
        plot_axes.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    print(compare()[0].to_string(index=False))
