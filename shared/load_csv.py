import re
from pathlib import Path

import pandas as pd


def _sort_key(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return (path.parent.name, int(match.group(1)) if match else -1)


def convert(source, output):
    source, output = Path(source), Path(output)
    files = sorted(source.rglob("*.csv"), key=_sort_key)
    if not files:
        raise FileNotFoundError(f"không có file .csv nào dưới {source}")
    file_rows, column_stats = [], {}
    for path in files:
        dataframe = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
        if "Label" not in dataframe:
            raise ValueError(f"{path}: thiếu cột Label")
        destination = output / path.parent.name / (path.stem + ".parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(destination, compression="zstd", index=False)

        timestamps = pd.to_numeric(dataframe["frame.time_epoch"], errors="coerce")
        file_rows.append({"folder": path.parent.name, "file": path.name, "rows": len(dataframe),
                          "time_min": timestamps.min(), "time_max": timestamps.max(),
                          "time_sorted": bool(timestamps.is_monotonic_increasing),
                          "labels": dataframe["Label"].value_counts().to_dict()})
        for column in dataframe.columns:
            stats = column_stats.setdefault(column, {"rows": 0, "nulls": 0, "dtypes": set()})
            stats["rows"] += len(dataframe)
            stats["nulls"] += int(dataframe[column].isna().sum())
            stats["dtypes"].add(str(dataframe[column].dtype))

    pd.DataFrame(file_rows).to_csv(output / "data_card_files.csv", index=False)
    pd.DataFrame([{"column": column, "null_ratio": stats["nulls"] / stats["rows"],
                   "dtypes": "|".join(sorted(stats["dtypes"]))} for column, stats in column_stats.items()]
                 ).to_csv(output / "data_card_columns.csv", index=False)
