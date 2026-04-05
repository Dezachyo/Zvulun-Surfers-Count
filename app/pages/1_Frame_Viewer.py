import os
import sys
from pathlib import Path
import sqlite3
import json
import pandas as pd
import streamlit as st

# Allow imports from app/
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from utils import get_latest_raw_annotated_pair, find_closest_file_by_time

st.set_page_config(page_title="Frame Viewer", page_icon="🖼️", layout="wide")
st.title("Frame Viewer")

if os.getenv("S3_BUCKET"):
    st.info("Frame Viewer is only available when running locally.")
    st.stop()

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "app_runs" / "surfer_data.db"
VISUALS_DIR = ROOT / "data" / "app_runs" / "visuals"


@st.cache_data(ttl=3)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp", conn)
    conn.close()
    df["bbox_list"] = df["bbox_list"].apply(json.loads)
    return df


df = load_data()
df["time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(config.TZ)
df["window_ind"] = (df["frame_ind"] == 1).cumsum()
if config.YOLO_WARMUP_FRAMES:
    df = df[df["frame_ind"] > config.YOLO_WARMUP_FRAMES]

win_stats = (
    df.groupby("window_ind")
    .agg(
        t_start=("time", "min"),
        mean_count=("surfer_count", "mean"),
        median_count=("surfer_count", "median"),
        max_count=("surfer_count", "max"),
        std_count=("surfer_count", "std"),
        n_frames=("frame_ind", "count"),
    )
)

image_files = sorted(VISUALS_DIR.glob("*.jpg"))

# ── Latest raw / annotated pair ──────────────────────────────────────────────
raw, annotated = get_latest_raw_annotated_pair(VISUALS_DIR)
col_raw, col_ann = st.columns(2)
with col_raw:
    st.image(raw) if raw else st.info("No raw image yet.")
with col_ann:
    st.image(annotated) if annotated else st.info("No annotated image yet.")

st.divider()

# ── Window selector → closest frame ──────────────────────────────────────────
selected_date = st.date_input(
    "Filter date",
    min_value=df["time"].min().date(),
    max_value=df["time"].max().date(),
    value=win_stats["t_start"].max().date(),
)

day = pd.Timestamp(selected_date).tz_localize(config.TZ)
win_stats_plot = win_stats[
    (win_stats["t_start"] >= day) &
    (win_stats["t_start"] < day + pd.Timedelta(days=1))
].reset_index()

col_table, col_frame = st.columns([1, 1])

with col_table:
    event = st.dataframe(
        win_stats_plot,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True,
        column_config={
            "mean_count": st.column_config.ProgressColumn(
                "mean_count",
                help="Mean surfers for window",
                format="",
                min_value=0,
                max_value=50,
            )
        },
    )

with col_frame:
    selected_row = win_stats_plot.iloc[-1]
    if event.selection.rows:
        selected_row = win_stats_plot.loc[event.selection.rows[0]]
        st.write("Selected time:", selected_row["t_start"])

    target_dt = selected_row["t_start"]
    closest = find_closest_file_by_time(image_files, target_dt, tz="Asia/Jerusalem")
    if closest:
        st.image(closest)
    else:
        st.info("No matching frame found.")
