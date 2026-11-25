import pandas as pd
import pytest

from shared.load_csv import convert


def _write_csv(path, times, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame.time_epoch": times, "wlan.sa": ["a"] * len(times),
                  "Label": labels}).to_csv(path, index=False)


def test_convert_orders_by_number_and_writes_card(tmp_path):
    _write_csv(tmp_path / "in/1.Deauth/Deauth_10.csv", [3.0, 4.0], ["Normal", "Deauth"])
    _write_csv(tmp_path / "in/1.Deauth/Deauth_2.csv", [1.0, 2.0], ["Normal", "Normal"])
    convert(tmp_path / "in", tmp_path / "out")

    card = pd.read_csv(tmp_path / "out/data_card_files.csv")
    assert list(card.file) == ["Deauth_2.csv", "Deauth_10.csv"]
    assert card.rows.sum() == 4 and card.time_sorted.all()
    assert len(pd.read_parquet(tmp_path / "out/1.Deauth/Deauth_10.parquet")) == 2


def test_missing_label_raises(tmp_path):
    (tmp_path / "in").mkdir()
    pd.DataFrame({"frame.time_epoch": [1.0]}).to_csv(tmp_path / "in/x_0.csv", index=False)
    with pytest.raises(ValueError):
        convert(tmp_path / "in", tmp_path / "out")


def test_no_csv_raises(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(FileNotFoundError):
        convert(tmp_path / "in", tmp_path / "out")
