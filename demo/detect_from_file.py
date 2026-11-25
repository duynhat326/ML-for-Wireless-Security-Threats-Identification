import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.tracer import detect_and_trace, load_model


def read_packets(path):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    alerts = detect_and_trace(read_packets(arguments.packets), load_model(arguments.model),
                              arguments.min_confidence, arguments.gap_seconds)
    for alert in alerts:
        source = alert["source"]
        origin = f'{source["column"]}={source["value"]} ({source["share"]:.0%})' if source else "unknown"
        note = " [impersonated access point]" if alert["impersonation"] else ""
        print(f'{alert["time_start"]}  {alert["attack"]:<16} confidence={alert["confidence"]:.2f}  '
              f'frames={alert["frames"]:<7} source: {origin}{note}')
    if not alerts:
        print("No attack detected.")
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
