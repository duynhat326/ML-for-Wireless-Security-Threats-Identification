from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import (
    DATA_DIRECTORY, DROP_COLUMNS, GROUP_COLUMN, LABEL_COLUMN, LABEL_TO_INDEX,
    METADATA_COLUMNS, SEED, TIME_COLUMN,
)

BATCH_SIZE = 100_000


def encode_labels(series):
    labels = series.map(LABEL_TO_INDEX)
    if labels.isna().any():
        raise ValueError(f"nhãn lạ: {sorted(series[labels.isna()].unique())}")
    return labels.astype("int64")


def split_features_labels(dataframe):
    metadata = dataframe[[column for column in METADATA_COLUMNS if column in dataframe.columns]].copy()
    labels = encode_labels(dataframe[LABEL_COLUMN]) if LABEL_COLUMN in dataframe.columns else None
    removed = [LABEL_COLUMN, GROUP_COLUMN, *DROP_COLUMNS]
    features = dataframe.drop(columns=[column for column in removed if column in dataframe.columns])
    return features, labels, metadata


def clean_features(features, schema):
    numeric = features.reindex(columns=schema["columns"]).apply(pd.to_numeric, errors="coerce")
    flags = numeric[schema["flag_columns"]].isna().astype("int8").add_suffix("__missing")
    return pd.concat([numeric.fillna(-1).astype("float32"), flags], axis=1)


def assign_splits(labels, timestamps, train_fraction=0.70, validation_fraction=0.15,
                  gap_fraction=0.0, train_cap_per_label=None, seed=SEED):
    split = np.full(len(labels), -1, dtype="int8")
    generator = np.random.default_rng(seed)
    for label in pd.unique(labels):
        positions = np.flatnonzero(labels == label)
        ordered = positions[np.argsort(timestamps[positions], kind="stable")]
        count = len(ordered)
        gap_rows = int(count * gap_fraction)
        train_end = int(count * train_fraction)
        validation_end = int(count * (train_fraction + validation_fraction))
        train_rows = ordered[:train_end]
        if train_cap_per_label and len(train_rows) > train_cap_per_label:
            train_rows = generator.choice(train_rows, train_cap_per_label, replace=False)
        split[train_rows] = 0
        split[ordered[train_end + gap_rows:validation_end]] = 1
        split[ordered[validation_end + gap_rows:]] = 2
    split[np.isnan(timestamps)] = -1
    return split


def split_parquet(source, output_directory, train_cap_per_label=None, gap_fraction=0.0, batch_size=200_000):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    key = parquet.read(columns=[LABEL_COLUMN, TIME_COLUMN]).to_pandas()
    timestamps = pd.to_numeric(key[TIME_COLUMN], errors="coerce").to_numpy(dtype="float64")
    split = assign_splits(key[LABEL_COLUMN].to_numpy(), timestamps, gap_fraction=gap_fraction,
                          train_cap_per_label=train_cap_per_label)
    del key, timestamps
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "test")
    writers, offset = {}, 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        part = split[offset:offset + len(table)]
        offset += len(table)
        for index, name in enumerate(names):
            rows = np.flatnonzero(part == index)
            if len(rows) == 0:
                continue
            if name not in writers:
                writers[name] = pq.ParquetWriter(output_directory / f"{name}.parquet", table.schema,
                                                 compression="zstd")
            writers[name].write_table(table.take(pa.array(rows)))
    for writer in writers.values():
        writer.close()
    return {name: int((split == index).sum()) for index, name in enumerate(names)}


def _feature_columns(path):
    import pyarrow.parquet as pq

    removed = {LABEL_COLUMN, GROUP_COLUMN, *DROP_COLUMNS}
    return [name for name in pq.ParquetFile(path).schema_arrow.names if name not in removed]


def _numeric_batches(path, columns, batch_size=BATCH_SIZE):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas().apply(pd.to_numeric, errors="coerce")


def fit_schema(path):
    columns = _feature_columns(path)
    present = pd.Series(0, index=columns, dtype="int64")
    missing = pd.Series(0, index=columns, dtype="int64")
    for numeric in _numeric_batches(path, columns):
        present += numeric.notna().sum()
        missing += numeric.isna().sum()
    kept = [column for column in columns if present[column] > 0]
    return {"columns": kept, "flag_columns": [column for column in kept if missing[column] > 0]}


def select_rows_with_cap(labels, cap, seed=SEED):
    labels = np.asarray(labels)
    generator = np.random.default_rng(seed)
    keep = np.ones(len(labels), dtype=bool)
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        if len(positions) > cap:
            keep[positions] = False
            keep[generator.choice(positions, cap, replace=False)] = True
    return keep


def load_feature_matrix(path, schema, keep=None):
    import pyarrow.parquet as pq

    columns, flag_columns = schema["columns"], schema["flag_columns"]
    names = columns + [column + "__missing" for column in flag_columns]
    total_rows = pq.ParquetFile(path).metadata.num_rows
    data = np.empty((total_rows if keep is None else int(keep.sum()), len(names)), dtype="float32")
    start = filled = 0
    for numeric in _numeric_batches(path, columns):
        stop = start + len(numeric)
        part = numeric if keep is None else numeric[keep[start:stop]]
        count = len(part)
        data[filled:filled + count, :len(columns)] = part.fillna(-1).to_numpy(dtype="float32")
        data[filled:filled + count, len(columns):] = part[flag_columns].isna().to_numpy(dtype="float32")
        filled += count
        start = stop
    return data, names


def fit_standardizer(array, chunk_rows=500_000):
    total = np.zeros(array.shape[1])
    squares = np.zeros(array.shape[1])
    for start in range(0, len(array), chunk_rows):
        block = array[start:start + chunk_rows].astype("float64")
        total += block.sum(axis=0)
        squares += (block ** 2).sum(axis=0)
    mean = total / len(array)
    scale = np.sqrt(np.maximum(squares / len(array) - mean ** 2, 0.0))
    return mean.astype("float32"), np.where(scale < 1e-6, 1.0, scale).astype("float32")


def standardize_in_place(array, mean, scale, chunk_rows=500_000):
    for start in range(0, len(array), chunk_rows):
        block = array[start:start + chunk_rows]
        block -= mean
        block /= scale


def load_labels_and_metadata(path, with_metadata=False):
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    wanted = [LABEL_COLUMN] + ([c for c in METADATA_COLUMNS if c in names] if with_metadata else [])
    frame = pq.read_table(path, columns=wanted).to_pandas()
    labels = encode_labels(frame.pop(LABEL_COLUMN))
    return labels, frame


def load_splits(data_directory=DATA_DIRECTORY, as_matrix=False, train_cap_per_label=None):
    paths = {name: Path(data_directory) / f"{name}.parquet" for name in ("train", "val", "test")}
    schema = fit_schema(paths["train"])
    labels_train = load_labels_and_metadata(paths["train"])[0]
    keep = select_rows_with_cap(labels_train.to_numpy(), train_cap_per_label) if train_cap_per_label else None
    if keep is not None:
        labels_train = labels_train[keep].reset_index(drop=True)

    def features(name, keep_rows=None):
        data, names = load_feature_matrix(paths[name], schema, keep_rows)
        return data if as_matrix else pd.DataFrame(data, columns=names, copy=False)

    labels_test, metadata_test = load_labels_and_metadata(paths["test"], with_metadata=True)
    return {
        "schema": schema,
        "features_train": features("train", keep),
        "labels_train": labels_train,
        "features_val": features("val"),
        "labels_val": load_labels_and_metadata(paths["val"])[0],
        "features_test": features("test"),
        "labels_test": labels_test,
        "metadata_test": metadata_test,
    }
