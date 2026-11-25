import os
from pathlib import Path

SEED = 42

LABELS = [
    "Normal", "Deauth", "Disas", "(Re)Assoc", "RogueAP", "Krack", "Kr00k",
    "SSH", "Botnet", "Malware", "SQL_Injection", "SSDP", "Evil_Twin",
    "Website_spoofing",
]
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABELS)}
NORMAL_INDEX = LABEL_TO_INDEX["Normal"]

LABEL_COLUMN = "Label"
TIME_COLUMN = "frame.time_epoch"
GROUP_COLUMN = "source_file"

SOURCE_METADATA_COLUMNS = [
    "wlan.sa", "wlan.ta", "wlan.ra", "wlan.da", "wlan.bssid", "wlan.ssid",
    "ip.src", "ip.dst",
]
DROP_COLUMNS = SOURCE_METADATA_COLUMNS + [
    TIME_COLUMN, "frame.number", "wlan_radio.channel", "wlan_radio.frequency",
]
METADATA_COLUMNS = SOURCE_METADATA_COLUMNS + [
    TIME_COLUMN, "wlan_radio.channel", "wlan_radio.signal_dbm",
]

IMPERSONATION_LABELS = {"Deauth", "Disas", "Kr00k", "RogueAP"}
NETWORK_LAYER_LABELS = {"SSH", "Botnet", "Malware", "SQL_Injection", "SSDP", "Website_spoofing"}

KAGGLE_INPUT = Path("/kaggle/input")
AWID3_CLEAN_PATH = Path("/kaggle/input/datasets/solefit/awid3-clean-dataset/awid3_clean.parquet")


def find_input_file(filename):
    if KAGGLE_INPUT.exists():
        found = next(KAGGLE_INPUT.rglob(filename), None)
        if found:
            return found
    return KAGGLE_INPUT / filename


DATA_DIRECTORY = Path(os.environ["AWID3_DATA"]) if "AWID3_DATA" in os.environ else find_input_file("train.parquet").parent
REPO_ROOT = Path(os.environ.get(
    "REPO_ROOT", "/kaggle/working/ML-for-Wireless-Security-Threats-Identification"))
