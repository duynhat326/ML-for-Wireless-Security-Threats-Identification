import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.tracer import detect_and_trace, load_model

st.set_page_config(page_title="AWID3 IDS", layout="wide")


@st.cache_resource
def get_model(path):
    return load_model(path)


@st.cache_data
def read_packets(upload):
    if upload.name.endswith(".parquet"):
        return pd.read_parquet(upload)
    return pd.read_csv(upload, encoding="ISO-8859-1", low_memory=False)


st.title("Wireless attack detection and tracing")

with st.sidebar:
    model_path = st.text_input(
        "Model file (.pkl or .pt)", "experiments/model_2_lightgbm/saved_model/model.pkl")
    minimum_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)
    gap_seconds = st.slider("Incident gap (seconds)", 0.1, 10.0, 1.0, 0.1)
    upload = st.file_uploader("Packets (CSV or Parquet)", type=["csv", "parquet"])

if upload is None:
    st.info("Upload an AWID3 CSV or Parquet file to start.")
    st.stop()

if not Path(model_path).is_file():
    st.error(f"Model file not found: {model_path}")
    st.stop()

dataframe = read_packets(upload)
alerts = detect_and_trace(dataframe, get_model(model_path), minimum_confidence, gap_seconds)

frames_column, incidents_column, attack_frames_column = st.columns(3)
frames_column.metric("Frames", f"{len(dataframe):,}")
incidents_column.metric("Incidents", len(alerts))
attack_frames_column.metric("Attack frames", f"{sum(alert['frames'] for alert in alerts):,}")

if not alerts:
    st.success("No attack detected.")
    st.stop()

table = pd.json_normalize(alerts, sep="_")
st.subheader("Incidents")
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Frames by attack type")
st.bar_chart(table.groupby("attack")["frames"].sum())

if table["impersonation"].any():
    st.warning(
        "For Deauth, Disas, Kr00k and RogueAP the reported MAC belongs to the "
        "impersonated access point, not the attacker."
    )

st.download_button(
    "Download alerts (JSON)",
    json.dumps(alerts, ensure_ascii=False, indent=2),
    file_name="alerts.json",
    mime="application/json",
)
