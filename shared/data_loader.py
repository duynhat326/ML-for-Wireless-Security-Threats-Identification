from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import (
    DATA_DIRECTORY, DROP_COLUMNS, GROUP_COLUMN, LABEL_COLUMN, LABEL_TO_INDEX,
    METADATA_COLUMNS, NORMAL_INDEX, SEED, TIME_COLUMN,
)


def split_features_labels(dataframe):
    metadata = dataframe[[column for column in METADATA_COLUMNS if column in dataframe.columns]].copy()
    labels = None
    if LABEL_COLUMN in dataframe.columns:
        labels = dataframe[LABEL_COLUMN].map(LABEL_TO_INDEX)
        if labels.isna().any():
            unknown = sorted(dataframe.loc[labels.isna(), LABEL_COLUMN].unique())
            raise ValueError(f"nhãn lạ: {unknown}")
        labels = labels.astype("int64")
    removed = [LABEL_COLUMN, GROUP_COLUMN, *DROP_COLUMNS]
    features = dataframe.drop(columns=[column for column in removed if column in dataframe.columns])
    return features, labels, metadata


def fit_schema(features):
    numeric = features.apply(pd.to_numeric, errors="coerce")
    columns = [column for column in numeric.columns if numeric[column].notna().any()]
    flag_columns = [column for column in columns if numeric[column].isna().any()]
    return {"columns": columns, "flag_columns": flag_columns}


def clean_features(features, schema):
    numeric = features.reindex(columns=schema["columns"]).apply(pd.to_numeric, errors="coerce")
    flags = numeric[schema["flag_columns"]].isna().astype("int8").add_suffix("__missing")
    return pd.concat([numeric.fillna(-1).astype("float32"), flags], axis=1)


def temporal_split(dataframe, train_fraction=0.70, validation_fraction=0.15, gap_fraction=0.01):
    dataframe = dataframe.reset_index(drop=True)
    timestamps = pd.to_numeric(dataframe[TIME_COLUMN], errors="coerce").dropna().sort_values(kind="stable")
    dataframe = dataframe.loc[timestamps.index]
    row_count = len(dataframe)
    gap_rows = int(row_count * gap_fraction)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    return (dataframe.iloc[:train_end],
            dataframe.iloc[train_end + gap_rows:validation_end],
            dataframe.iloc[validation_end + gap_rows:])


def split_all(dataframe):
    if GROUP_COLUMN in dataframe.columns:
        groups = [group for _, group in dataframe.groupby(GROUP_COLUMN, sort=False)]
    else:
        groups = [dataframe]
    parts = [temporal_split(group) for group in groups]
    return tuple(pd.concat(part, ignore_index=True) for part in zip(*parts))


def cap_normal(features, labels, cap, seed=SEED):
    normal = np.flatnonzero(labels.to_numpy() == NORMAL_INDEX)
    attacks = np.flatnonzero(labels.to_numpy() != NORMAL_INDEX)
    if len(normal) > cap:
        normal = np.random.default_rng(seed).choice(normal, cap, replace=False)
    indices = np.sort(np.concatenate([attacks, normal]))
    return features.iloc[indices], labels.iloc[indices]


def load_splits(data_directory=DATA_DIRECTORY, normal_cap=None):
    raw = {name: pd.read_parquet(Path(data_directory) / f"{name}.parquet")
           for name in ("train", "val", "test")}
    features_train, labels_train, _ = split_features_labels(raw["train"])
    schema = fit_schema(features_train)
    if normal_cap:
        features_train, labels_train = cap_normal(features_train, labels_train, normal_cap)
    features_val, labels_val, _ = split_features_labels(raw["val"])
    features_test, labels_test, metadata_test = split_features_labels(raw["test"])
    return {
        "schema": schema,
        "features_train": clean_features(features_train, schema), "labels_train": labels_train,
        "features_val": clean_features(features_val, schema), "labels_val": labels_val,
        "features_test": clean_features(features_test, schema), "labels_test": labels_test,
        "metadata_test": metadata_test,
    }
