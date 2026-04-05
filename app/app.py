import os
import streamlit as st

# Bridge Streamlit Cloud secrets to env vars before importing utils
# (utils reads S3_BUCKET at import time to set CLOUD_MODE)
try:
    for _k in ["S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]:
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass  # no secrets.toml — running locally with .env

import pandas as pd
import sqlite3
import json
from pathlib import Path
import altair as alt
from datetime import date
from datetime import datetime
import config
from utils import CLOUD_MODE, get_db_path, ensure_dbs_from_s3, \
    filepath_to_dt, load_weather_data, get_best_forecasts, \
    enrich_win_stats, get_live_wx, build_surfer_wx_heatmap



# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = get_db_path("surfer_data")
WEATHER_DB_PATH = get_db_path("weather_data")
VISUALS_DIR = ROOT / "data" / "app_runs" / "visuals"


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Surfers Count Dashboard",
    page_icon="🏄",
    layout="wide"
)

st.title("Surfers Count Dashboard")

# -------------------------
# Load SQLite safely
# -------------------------
@st.cache_data(ttl=3)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp", conn)
    conn.close()

    # Parse JSON bbox lists
    df["bbox_list"] = df["bbox_list"].apply(json.loads)
    return df

@st.cache_data(ttl=3)
def load_weather():
    """Loads raw weather forecasts for display."""
    conn = sqlite3.connect(WEATHER_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM forecasts ORDER BY scrape_timestamp", conn)
    conn.close()
    return df

@st.cache_data(ttl=3)
def load_weather_processed():
    """Loads and enriches weather forecasts for joining onto win_stats."""
    return load_weather_data(WEATHER_DB_PATH, config.TZ)


# -------------------------
# Main Data Load
# -------------------------

ensure_dbs_from_s3()
df = load_data()
df_weather = load_weather()

#Prepare df
df["time"] = pd.to_datetime(df["timestamp"], utc=True)\
                      .dt.tz_convert(config.TZ)
df['window_ind'] = (df['frame_ind'] == 1).cumsum()
if config.YOLO_WARMUP_FRAMES: #TODO Put in Config file 
    df = df[df['frame_ind']> config.YOLO_WARMUP_FRAMES]

win_stats = (
    df.groupby('window_ind')
      .agg(
          t_start=('time', 'min'),
          mean_count=('surfer_count', 'mean'),
          median_count=('surfer_count', 'median'),
          max_count = ('surfer_count', 'max'),
          std_count=('surfer_count', 'std'),
          cv_count=('surfer_count', lambda x: x.std() / x.mean()),
          n_frames=('frame_ind', 'count')
      )
)

_best_forecasts = get_best_forecasts(load_weather_processed())
win_stats = enrich_win_stats(win_stats, _best_forecasts, config.TZ)

latest_window_stats = win_stats.iloc[-1]

_current_wx, _prev_wx = get_live_wx(_best_forecasts, config.TZ)

hourly_summary = (
    win_stats.groupby(win_stats['t_start'].dt.hour)['mean_count']
    .mean()
    .reset_index()
)

hourly_summary.columns = ['hour', 'mean_count']


if df.empty:
    st.warning("Waiting for inference to start…")
    st.stop()

# -------------------------
# Define parameters for realtime plot
# -------------------------

# find the latest window time (Full Timestamp)
last_inference_window = win_stats["t_start"].max()
day_realtime = last_inference_window.date() 


# # Live or not logic
is_live = (pd.Timestamp.now(config.TZ) - last_inference_window) < pd.Timedelta(minutes= config.delta_is_live)

if is_live:
    # Get the current time in your local timezone
    now = pd.Timestamp.now(config.TZ)

    # Filter for rows that match TODAY and the CURRENT HOUR
    mask = (
        (win_stats['t_start'].dt.date == now.date()) & 
        (win_stats['t_start'].dt.hour == now.hour)
    )

    # Calculate the mean for just those detections
    current_val = win_stats.loc[mask, config.COUNT_STAT_REALTIME_COL].mean()

    # Fill with 0 if no detections have happened yet this hour to avoid errors
    current_val = 0 if pd.isna(current_val) else current_val

    # Create the dataframe for the red bar
    now_df = pd.DataFrame({'hour': [now.hour], 'current_count': [current_val]})
        
# -------------------------
# Live Row
# -------------------------


col_live, col_2 = st.columns([2, 1])
with col_live:
    # We use a 'pusher' column at the end to keep the first three tight
    # Ratios: [Indicator, Text, Button, Spacer]
    h_ind, h_text, h_btn, h_spacer = st.columns([0.6, 2.5, 0.4, 3], gap="small")

    with h_ind:
        color = "green" if is_live else "red"
        label = "LIVE" if is_live else "OFFLINE"
        # Using a standard bold span instead of H3 to keep it small
        st.markdown(f"**:{color}[● {label}]**") 

    with h_text:
        if is_live:
            status_str = f'Current Status ({last_inference_window.strftime("%H:%M")})'
        else:
            status_str = f'Last Collected Day ({last_inference_window.date()})'
        
        # Using markdown with a specific size or st.caption for smaller font
        st.markdown(f"<p style='font-size:14px; margin:0; padding:0;'>{status_str}</p>", unsafe_allow_html=True)

    with h_btn:
        # small button
        if st.button("🔄", help="Refresh Data"):
            st.cache_data.clear()
            st.rerun()
# -------------------------
# Metrics helpers
# -------------------------

_LABEL_COLOR = "#808080"

_METRIC_COLORS = {
    "wind_speed":   _LABEL_COLOR,
    "wind_dir":     _LABEL_COLOR,
    "swell_height": _LABEL_COLOR,
    "swell_period": _LABEL_COLOR,
}

def _delta_html(delta_num, fmt, unit):
    color  = "#09ab3b" if delta_num >= 0 else "#ff2b2b"
    prefix = "▲" if delta_num >= 0 else "▼"
    return f'<p style="margin:0;font-size:13px;color:{color};">{prefix} {delta_num:{fmt}} {unit}</p>'

def _metric_html(label, value_str, delta_num=None, fmt=".0f", unit="", label_color="#31333F"):
    d = _delta_html(delta_num, fmt, unit) if delta_num is not None and not pd.isna(delta_num) else ""
    return (
        f'<div>'
        f'<p style="margin:0;font-size:14px;color:{label_color};font-weight:600;">{label}</p>'
        f'<p style="margin:0;font-size:28px;font-weight:700;color:#31333F;">{value_str}</p>'
        f'{d}</div>'
    )

def _wind_dir_html(degrees, delta_deg=None):
    arrow = (
        f'<span style="display:inline-block;transform:rotate({degrees+180:.0f}deg);'
        f'font-size:28px;font-weight:700;line-height:1;">↑</span>'
    )
    d = _delta_html(delta_deg, ".0f", "°") if delta_deg is not None and not pd.isna(delta_deg) else ""
    return (
        f'<div>'
        f'<p style="margin:0;font-size:14px;color:{_LABEL_COLOR};font-weight:600;">Wind Direction</p>'
        f'{arrow}{d}</div>'
    )

# -------------------------
# Metrics
# -------------------------


with st.container(border=True):
    if not is_live:
        st.markdown("<p style='color:#808080;'>Waiting for live data...</p>", unsafe_allow_html=True)
    else:
        n = len(config.LIVE_METRICS)
        col_count, *wx_cols, pusher = st.columns([1] + [1] * n + [5], gap="small")

        with col_count:
            count_val = latest_window_stats[config.COUNT_STAT_REALTIME_COL]
            prev_count = win_stats.iloc[-2][config.COUNT_STAT_REALTIME_COL] if len(win_stats) > 1 else None
            delta = (count_val - prev_count) if prev_count is not None else None
            st.markdown(_metric_html("Surfers Now", f"{round(count_val)}", delta, ".1f", "", _LABEL_COLOR), unsafe_allow_html=True)

        for col, m in zip(wx_cols, config.LIVE_METRICS):
            with col:
                color = _METRIC_COLORS.get(m["col"], "#31333F")
                if _current_wx is not None and not pd.isna(_current_wx[m["col"]]):
                    val = _current_wx[m["col"]]
                    delta = (val - _prev_wx[m["col"]]) if _prev_wx is not None else None
                    if m["col"] == "wind_dir":
                        st.markdown(_wind_dir_html(val), unsafe_allow_html=True)
                    else:
                        st.markdown(_metric_html(m["label"], f"{val:{m['fmt']}} {m['unit']}", delta, m["fmt"], m["unit"], color), unsafe_allow_html=True)
                else:
                    st.markdown(_metric_html(m["label"], "N/A", label_color=color), unsafe_allow_html=True)
        




# -------------------------
# REALTIME Chart
# -------------------------

 
st.write(f"Latest inference data: {last_inference_window.strftime('%Y-%m-%d %H:%M:%S')}")

col_realtime, col_hourly_trend = st.columns([3,3])
height = 350

smooth_on = True
smooth_minutes = config.SMOOTH_MINUTES

with col_realtime:
    with st.container(border=True):
        st.markdown("##### Daily Trend (Realtime View)")
        if not is_live:
            st.markdown("<p style='color:#808080;margin-top:8px;'>Waiting for live data...</p>", unsafe_allow_html=True)
        else:
            day = pd.Timestamp(day_realtime).tz_localize(config.TZ)

            win_stats_plot = win_stats[
                (win_stats["t_start"] >= day) &
                (win_stats["t_start"] < day + pd.Timedelta(days=1))
            ]

            # ---- time-based smoothing (VIEW ONLY) ----
            if smooth_on:
                win_stats_plot[f"{config.COUNT_STAT_REALTIME_COL}_smooth"] = (
                    win_stats_plot
                    .sort_values("t_start")
                    .rolling(
                        window=f"{smooth_minutes}min",
                        on="t_start",
                        min_periods=1
                    )[config.COUNT_STAT_REALTIME_COL]
                    .mean()
                )
                y_plot = f"{config.COUNT_STAT_REALTIME_COL}_smooth"
            else:
                y_plot = config.COUNT_STAT_REALTIME_COL

            if config.ROUND_TO_INT:
                win_stats_plot = win_stats_plot.copy()
                win_stats_plot[y_plot] = win_stats_plot[y_plot].round(0)

            start_window = day + pd.Timedelta(hours=4)
            end_window = day + pd.Timedelta(hours=20)

            x_scale = alt.Scale(
                domain=[start_window, end_window],
                nice=False,
                clamp=True
            )

            chart = (
                alt.Chart(win_stats_plot)
                .mark_area(
                    color="lightblue",
                    interpolate="step-after",
                    line=True,
                    opacity=0.8
                )
                .encode(
                    x=alt.X(
                        "t_start:T",
                        scale=x_scale,
                        title="Time"
                    ),
                    y=alt.Y(y_plot, type="quantitative", title="count"),
                    tooltip=[
                        alt.Tooltip("t_start:T", title="Time", format="%Y-%m-%d %H:%M:%S"),
                        alt.Tooltip(y_plot, type="quantitative", title="count", format=".2f"),
                    ]
                )
                .properties(height=height)
            )

            st.altair_chart(chart, use_container_width=True)

with col_hourly_trend:
    with st.container(border=True):
        st.markdown("##### Hourly Trend")

        # 3. Base Chart: Historical Average (Faded Blue)
        base = alt.Chart(hourly_summary).mark_bar(
            color="#0068c9", 
            opacity=0.4, # Keeps the "historical" feel
            cornerRadiusTopLeft=2,
            cornerRadiusTopRight=2
        ).encode(
            x=alt.X(
                'hour:O', 
                title='hour', 
                scale=alt.Scale(domain=list(range(5, 21))),
                axis=alt.Axis(labelAngle=0, grid=False) # Horizontal ticks
            ),
            y=alt.Y('mean_count:Q',
                    title='Avg Surfers',
                    scale=alt.Scale(domain=(0, hourly_summary['mean_count'].max() * 2)),
                    axis=alt.Axis(gridColor='#f0f0f0')),
            tooltip=[
                alt.Tooltip('hour:O', title='Hour'),
                alt.Tooltip('mean_count:Q', title='Historical Avg', format='.1f')
            ]
        )
        if is_live:
            # 4. Overlay: Live Indicator (Same Width Red Bar)
            now_bar = alt.Chart(now_df).mark_bar(
                color="#FF4B4B", 
                opacity=0.6,
                stroke="white", # Adds a border to separate it from the background
                strokeWidth=0,
                cornerRadiusTopLeft=10,
                cornerRadiusTopRight=10
            ).encode(
                x='hour:O',
                y='current_count:Q',
                tooltip=[alt.Tooltip('current_count:Q', title='Live Surfer Count')]
            )

        # 5. Combine
        chart = (base + now_bar if is_live else base).properties(
            height=height
        ).configure_view(
            strokeOpacity=0
        )
        st.altair_chart(chart, use_container_width=True)

        


# -------------------------
# Offline chart + Heatmap (side by side)
# -------------------------

col_offline, col_heatmap = st.columns([1, 1])

# Dates sorted by number of windows descending
_date_window_counts = (
    win_stats.groupby(win_stats["t_start"].dt.date)
    .size()
    .sort_values(ascending=False)
)
_available_dates = _date_window_counts.index.tolist()

with col_offline:
    with st.container(border=True):
        st.markdown("##### Daily View")

        selected_date = st.selectbox(
            "Pick a date",
            options=_available_dates,
            format_func=lambda d: f"{d}  ({_date_window_counts[d]} windows)",
            index=0,
        )

        y_col = config.COUNT_STAT_REALTIME_COL
        smooth_on = True
        smooth_minutes = config.SMOOTH_MINUTES

        day = pd.Timestamp(selected_date).tz_localize(config.TZ)
        win_stats_plot = win_stats[
            (win_stats["t_start"] >= day) &
            (win_stats["t_start"] < day + pd.Timedelta(days=1))
        ]

        if smooth_on:
            win_stats_plot[f"{y_col}_smooth"] = (
                win_stats_plot
                .sort_values("t_start")
                .rolling(window=f"{smooth_minutes}min", on="t_start", min_periods=1)[y_col]
                .mean()
            )
            y_plot = f"{y_col}_smooth"
        else:
            y_plot = y_col

        if config.ROUND_TO_INT:
            win_stats_plot = win_stats_plot.copy()
            win_stats_plot[y_plot] = win_stats_plot[y_plot].round(0)

        x_domain = [day, day + pd.Timedelta(days=1)]
        chart = (
            alt.Chart(win_stats_plot)
            .mark_area(color="lightblue", interpolate="step-after", line=True, opacity=0.8)
            .encode(
                x=alt.X("t_start:T", scale=alt.Scale(domain=x_domain, nice=False), title="Time"),
                y=alt.Y(y_plot, type="quantitative", title="count"),
                tooltip=[
                    alt.Tooltip("t_start:T", title="Time", format="%Y-%m-%d %H:%M:%S"),
                    alt.Tooltip(y_plot, type="quantitative", title="count", format=".2f"),
                ]
            )
        )
        st.altair_chart(chart, use_container_width=True)

with col_heatmap:
    with st.container(border=True):
        st.markdown("##### Mean Surfer Count  |  Wave Height × Wind Speed")

        hm_date_range = st.date_input(
            "Date range",
            value=(win_stats["t_start"].min().date(), win_stats["t_start"].max().date()),
            min_value=win_stats["t_start"].min().date(),
            max_value=win_stats["t_start"].max().date(),
            key="hm_date_range",
        )

        if len(hm_date_range) == 2:
            start_dt = pd.Timestamp(hm_date_range[0]).tz_localize(config.TZ)
            end_dt   = pd.Timestamp(hm_date_range[1]).tz_localize(config.TZ) + pd.Timedelta(days=1)
            ws_hm = win_stats[(win_stats["t_start"] >= start_dt) & (win_stats["t_start"] < end_dt)]
        else:
            ws_hm = win_stats

        hm_agg, h_domain, w_domain = build_surfer_wx_heatmap(ws_hm)

        if hm_agg.empty:
            st.info("Not enough weather-matched windows to build heatmap yet.")
        else:
            h_marginal = hm_agg.groupby("h_mid")["mean_surfers"].mean().reset_index()
            w_marginal = hm_agg.groupby("w_mid")["mean_surfers"].mean().reset_index()

            _rect = (
                alt.Chart(hm_agg).mark_rect().encode(
                    x=alt.X("h_mid:O", title="Swell Height (m)", sort=h_domain),
                    y=alt.Y("w_mid:O", title="Wind Speed (kn)",  sort=w_domain),
                    color=alt.Color("mean_surfers:Q", scale=alt.Scale(scheme="blues"), title="Mean Surfers"),
                    tooltip=[
                        alt.Tooltip("h_mid:Q",        title="Swell Height", format=".2f"),
                        alt.Tooltip("w_mid:Q",        title="Wind Speed",   format=".1f"),
                        alt.Tooltip("mean_surfers:Q", title="Mean Surfers", format=".1f"),
                    ],
                )
            )
            _text = (
                alt.Chart(hm_agg).mark_text(fontSize=11).encode(
                    x=alt.X("h_mid:O", sort=h_domain),
                    y=alt.Y("w_mid:O", sort=w_domain),
                    text=alt.Text("mean_surfers:Q", format=".1f"),
                    color=alt.value("#333333"),
                )
            )
            top_bar = (
                alt.Chart(h_marginal)
                .mark_bar(color="#0068c9", opacity=0.6, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("h_mid:O", title="", sort=h_domain, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("mean_surfers:Q", title="Avg Surfers", axis=alt.Axis(tickCount=3)),
                )
                .properties(width=300, height=80)
            )
            right_bar = (
                alt.Chart(w_marginal)
                .mark_bar(color="#0068c9", opacity=0.6, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    y=alt.Y("w_mid:O", title="", sort=w_domain, axis=alt.Axis(labels=False, ticks=False)),
                    x=alt.X("mean_surfers:Q", title="Avg Surfers", axis=alt.Axis(tickCount=3)),
                )
                .properties(width=80, height=200)
            )
            main_hm = (_rect + _text).properties(width=300, height=200)

            heatmap_chart = alt.vconcat(
                top_bar,
                alt.hconcat(main_hm, right_bar, spacing=5),
                spacing=5,
            )
            st.altair_chart(heatmap_chart, use_container_width=False)
