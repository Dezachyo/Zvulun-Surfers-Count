import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import utils.config as config
from app.utils import load_weather_data, get_best_forecasts, enrich_win_stats


alt.renderers.enable('html') # or 'notebook' if in classic Jupyter
# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "app_runs" / "surfer_data.db"
WG_DB_PATH = ROOT / "data" / "app_runs" / "weather_data.db"
VISUALS_DIR = ROOT / "data" / "app_runs" / "visuals"

# -------------------------
# Load SQLite safely
# -------------------------

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp", conn)
    conn.close()

    # Parse JSON bbox lists
    df["bbox_list"] = df["bbox_list"].apply(json.loads)
    return df


# -------------------------
# Main Data Load
# -------------------------
df = load_data()
weather_df = load_weather_data(WG_DB_PATH, config.TZ)


# -------------------------
# Time-series
# -------------------------




orig_df=  df.copy()



df["time"] = pd.to_datetime(df["timestamp"], utc=True)\
                      .dt.tz_convert(config.TZ)


df['window_ind'] = (df['frame_ind'] == 1).cumsum()

#or
#df = df.set_index(['window_id', 'frame_id']).sort_index()

win_stats = (
    df.groupby('window_ind')
      .agg(
          t_start=('time', 'min'),
          mean_count=('surfer_count', 'mean'),
          median_count=('surfer_count', 'median'),
          std_count=('surfer_count', 'std'),
          cv_count=('surfer_count', lambda x: x.std() / x.mean()),
          n_frames=('frame_ind', 'count')
      )
)

df_combined = df.merge(
    win_stats[['t_start']],
    left_on='window_ind',
    right_index=True
)

day = pd.Timestamp('2026-01-03', tz='Asia/Jerusalem')

df_plot = df_combined[
    (df_combined['time'] >= day) &
    (df_combined['time'] < day + pd.Timedelta(days=1))
]
win_stats_plot = win_stats[
    (win_stats['t_start'] >= day) &
    (win_stats['t_start'] < day + pd.Timedelta(days=1))
]

#%%
# ordered window indices (already done)
window_order = win_stats_plot.sort_values('t_start').index

# create time labels
time_labels = (
    win_stats_plot
    .loc[window_order, 't_start']
    .dt.strftime('%H:%M')
)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 5))
sns.boxplot(
    data=df_plot,
    x='window_ind',
    y='surfer_count',
    order=window_order,
    showfliers=False,
    color='lightgray'
)

sns.stripplot(
    data=df_plot,
    x='window_ind',
    y='surfer_count',
    order=window_order,
    color='black',
    size=2,
    alpha=0.4,
    jitter=0.25
)

plt.xticks(
    ticks=range(len(window_order)),
    labels=time_labels,
    rotation=90
)

# mean
plt.plot(
    range(len(window_order)),
    win_stats_plot.loc[window_order, 'mean_count'],
    color='red',
    marker='o',
    label='Mean'
)

# median
plt.plot(
    range(len(window_order)),
    win_stats_plot.loc[window_order, 'median_count'],
    color='blue',
    marker='o',
    label='Median'
)


# %%


# 2. Aggregate
# We group by the hour and take the mean
hourly_summary = (
    win_stats.groupby(win_stats['t_start'].dt.hour)['mean_count']
    .mean()
    .reset_index()
)


hourly_summary.columns = ['hour', 'mean_count']
import altair as alt
import pandas as pd
import datetime

# 1. Force the HTML renderer for Jupyter
alt.renderers.enable('html')

# 2. Prepare "Now" Data (Current hour at Zvulun Beach)
current_hour = 12
current_count = 10  # Your live count
now_df = pd.DataFrame({'hour': [current_hour], 'current_count': [current_count]})

# 3. Base Chart: Historical Average (Faded Blue)
base = alt.Chart(hourly_summary).mark_bar(
    color="#0068c9", 
    opacity=0.4, # Keeps the "historical" feel
    cornerRadiusTopLeft=2,
    cornerRadiusTopRight=2
).encode(
    x=alt.X(
        'hour:O', 
        title='Hour of Day', 
        scale=alt.Scale(domain=list(range(5, 21))),
        axis=alt.Axis(labelAngle=0, grid=False) # Horizontal ticks
    ),
    y=alt.Y('mean_count:Q', title='Avg Surfers', axis=alt.Axis(gridColor='#f0f0f0')),
    tooltip=[
        alt.Tooltip('hour:O', title='Hour'),
        alt.Tooltip('mean_count:Q', title='Historical Avg', format='.1f')
    ]
)

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
chart = (base + now_bar).properties(
    width=800,
    height=200
).configure_view(
    strokeOpacity=0
)

chart.display()
# %%

# -------------------------
# Weather Data Overview
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Weather Data Overview", fontsize=15)

# 1. Scraping timeline (unique scrape times)
ax = axes[0, 0]
scrape_times = weather_df["scrape_dt"].drop_duplicates().sort_values()
ax.scatter(scrape_times, [1] * len(scrape_times), s=10, alpha=0.5)
ax.set_title("Scraping Timeline")
ax.set_xlabel("Scrape Time")
ax.set_yticks([])
ax.tick_params(axis="x", rotation=45)

# 2. Distribution of wave heights
ax = axes[0, 1]
sns.kdeplot(weather_df["swell_height"].dropna(), ax=ax, fill=True, color="steelblue")
ax.set_title("Swell Height Distribution")
ax.set_xlabel("Swell Height (m)")
ax.set_ylabel("Density")

# 3. Distribution of wind speed
ax = axes[0, 2]
sns.kdeplot(weather_df["wind_speed"].dropna(), ax=ax, fill=True, color="steelblue")
ax.set_title("Wind Speed Distribution")
ax.set_xlabel("Wind Speed (knots)")
ax.set_ylabel("Density")

# 4. Swell height vs period (scatter + regression)
ax = axes[1, 0]
ax.scatter(weather_df["swell_height"], weather_df["swell_period"], alpha=0.3, s=15)
m, b = np.polyfit(weather_df["swell_height"].dropna(), weather_df["swell_period"].dropna(), 1)
x_range = np.linspace(weather_df["swell_height"].min(), weather_df["swell_height"].max(), 100)
ax.plot(x_range, m * x_range + b, color="red", linewidth=1.5)
ax.set_title("Swell Height vs Period")
ax.set_xlabel("Swell Height (m)")
ax.set_ylabel("Swell Period (s)")

# 5. Heatmap: x=swell height bins, y=wind speed bins, color=mean swell period
ax = axes[1, 1]
wdf = weather_df[["swell_height", "wind_speed", "swell_period"]].dropna().copy()
wdf["h_bin"] = pd.cut(wdf["swell_height"], bins=6)
wdf["w_bin"] = pd.cut(wdf["wind_speed"], bins=6)
pivot = wdf.groupby(["w_bin", "h_bin"], observed=True)["swell_period"].mean().unstack()
sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5,
            cbar_kws={"label": "Avg Period (s)"})
ax.set_title("Avg Swell Period\n(Wind vs Wave Height)")
ax.set_xlabel("Swell Height (m)")
ax.set_ylabel("Wind Speed (knots)")

# 6. Delta forecast distribution (in hours)
ax = axes[1, 2]
delta_hours = weather_df["delta_forecast"].dt.total_seconds() / 3600
sns.kdeplot(delta_hours.dropna(), ax=ax, fill=True, color="mediumseagreen")
ax.set_title("Delta Forecast Distribution")
ax.set_xlabel("Hours Ahead of Scrape")
ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
# %%

# -------------------------
# Heatmap with marginal distributions
# -------------------------
wdf = weather_df[["swell_height", "wind_speed", "swell_period"]].dropna().copy()
n_bins = 6
h_edges = np.linspace(wdf["swell_height"].min(), wdf["swell_height"].max(), n_bins + 1)
w_edges = np.linspace(wdf["wind_speed"].min(),   wdf["wind_speed"].max(),   n_bins + 1)
h_mids  = [f"{(h_edges[i] + h_edges[i+1]) / 2:.1f}" for i in range(n_bins)]
w_mids  = [f"{(w_edges[i] + w_edges[i+1]) / 2:.1f}" for i in range(n_bins)]

wdf["h_bin"] = pd.cut(wdf["swell_height"], bins=h_edges, include_lowest=True)
wdf["w_bin"] = pd.cut(wdf["wind_speed"],   bins=w_edges, include_lowest=True)
pivot    = wdf.groupby(["w_bin", "h_bin"], observed=True)["swell_period"].mean().unstack()
h_counts = wdf["h_bin"].value_counts().sort_index()
w_counts = wdf["w_bin"].value_counts().sort_index()
n_cols, n_rows = pivot.shape[1], pivot.shape[0]

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[3, 0.15, 0.6],
    height_ratios=[1, 3],
    hspace=0.05, wspace=0.05
)
ax_main  = fig.add_subplot(gs[1, 0])
ax_top   = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[1, 2])
ax_cbar  = fig.add_subplot(gs[1, 1])

# Main heatmap — use midpoint labels instead of interval notation
sns.heatmap(pivot, ax=ax_main, cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.5, cbar=True, cbar_ax=ax_cbar,
            xticklabels=h_mids, yticklabels=w_mids,
            cbar_kws={"label": "Avg Swell Period (s)"})
ax_main.set_xlabel("Swell Height (m)")
ax_main.set_ylabel("Wind Speed (knots)")
ax_main.tick_params(axis="x", rotation=0)
ax_main.tick_params(axis="y", rotation=0)

# Top marginal — swell height distribution
ax_top.bar([i + 0.5 for i in range(n_cols)], h_counts.values,
           width=0.8, color="steelblue", alpha=0.7)
ax_top.set_xlim(0, n_cols)
ax_top.set_ylabel("Count")
ax_top.tick_params(labelbottom=False)
ax_top.set_title("Avg Swell Period  |  Wave Height × Wind Speed")

# Right marginal — wind speed distribution (horizontal)
ax_right.barh([i + 0.5 for i in range(n_rows)], w_counts.values,
              height=0.8, color="steelblue", alpha=0.7)
ax_right.set_ylim(0, n_rows)
ax_right.set_xlabel("Count")
ax_right.tick_params(labelleft=False)

plt.show()
# %%

# -------------------------
# Join windows with weather
# -------------------------

win_stats = enrich_win_stats(win_stats, get_best_forecasts(weather_df), config.TZ)

print(f"Windows total:        {len(win_stats)}")
print(f"Windows with weather: {win_stats['wind_speed'].notna().sum()}")
print(win_stats.head())
# %%

# -------------------------
# Correlation: Weather vs Surfer Counts
# Working dataframe: windows that have both surfer data and weather data
# -------------------------
df_wx = (
    win_stats[win_stats["t_start"].dt.hour.between(6, 18)]
    [["mean_count", "swell_height", "wind_speed", "wind_dir"]]
    .dropna()
    .copy()
)

# %%
# 1. Swell Height vs Surfer Count
# Jitter on both axes reveals density hidden by discrete WindGuru steps (0.25 m)
scatter_h = (
    alt.Chart(df_wx)
    .mark_circle(opacity=0.3, color="steelblue", size=40)
    .encode(
        x=alt.X("swell_height:Q", title="Swell Height (m)").scale(zero=False),
        y=alt.Y("mean_count:Q",   title="Mean Surfer Count"),
        tooltip=[
            alt.Tooltip("swell_height:Q", title="Swell Height", format=".2f"),
            alt.Tooltip("mean_count:Q",   title="Mean Count",   format=".1f"),
        ],
    )
    .transform_calculate(
        swell_height_j="datum.swell_height + (random() - 0.5) * 0.1",
        mean_count_j="datum.mean_count   + (random() - 0.5) * 0.3",
    )
    .encode(
        x=alt.X("swell_height_j:Q", title="Swell Height (m)").scale(zero=False),
        y=alt.Y("mean_count_j:Q",   title="Mean Surfer Count"),
    )
)
loess_h = (
    alt.Chart(df_wx)
    .transform_loess("swell_height", "mean_count", bandwidth=0.4)
    .mark_line(color="red", strokeWidth=2)
    .encode(
        x=alt.X("swell_height:Q"),
        y=alt.Y("mean_count:Q"),
    )
)
(scatter_h + loess_h).properties(
    title="Swell Height vs Surfer Count",
    width=600, height=350,
).display()

# %%
# 2. Wind Speed vs Surfer Count
# Jitter reveals density — wind_speed is integer-valued from scraper
scatter_w = (
    alt.Chart(df_wx)
    .mark_circle(opacity=0.3, color="#0068c9", size=40)
    .transform_calculate(
        wind_speed_j="datum.wind_speed + (random() - 0.5) * 0.5",
        mean_count_j="datum.mean_count + (random() - 0.5) * 0.3",
    )
    .encode(
        x=alt.X("wind_speed_j:Q", title="Wind Speed (kn)").scale(zero=False),
        y=alt.Y("mean_count_j:Q", title="Mean Surfer Count"),
        tooltip=[
            alt.Tooltip("wind_speed:Q", title="Wind Speed", format=".1f"),
            alt.Tooltip("mean_count:Q", title="Mean Count", format=".1f"),
        ],
    )
)
loess_w = (
    alt.Chart(df_wx)
    .transform_loess("wind_speed", "mean_count", bandwidth=0.4)
    .mark_line(color="red", strokeWidth=2)
    .encode(
        x=alt.X("wind_speed:Q"),
        y=alt.Y("mean_count:Q"),
    )
)
(scatter_w + loess_w).properties(
    title="Wind Speed vs Surfer Count",
    width=600, height=350,
).display()

# %%
# 3. Wind Direction vs Surfer Count — Polar bar chart
N_SECTORS  = 16
sector_deg = 360 / N_SECTORS          # 22.5 — keep as float throughout
bin_centers = np.arange(0, 360, sector_deg)  # [0, 22.5, 45, ..., 337.5]

df_wx["dir_bin"] = (
    ((df_wx["wind_dir"] + sector_deg / 2) % 360) // sector_deg * sector_deg
)

polar_df = (
    df_wx.groupby("dir_bin", observed=True)["mean_count"]
    .mean()
    .reindex(bin_centers, fill_value=0)
    .reset_index()
)
polar_df.columns = ["dir_deg", "mean_count"]

theta   = np.deg2rad(polar_df["dir_deg"])   # exactly N_SECTORS values
radii   = polar_df["mean_count"].values
width   = np.deg2rad(sector_deg) * 0.9
compass = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
           "S","SSW","SW","WSW","W","WNW","NW","NNW"]

fig_polar, ax_polar = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)

bars = ax_polar.bar(theta, radii, width=width, bottom=0,
                    color="steelblue", alpha=0.7, edgecolor="white")

norm = plt.Normalize(radii.min(), radii.max())
cmap = plt.cm.YlOrRd
for bar, r in zip(bars, radii):
    bar.set_facecolor(cmap(norm(r)))

# Use the same theta array for ticks — count always matches labels
ax_polar.set_xticks(theta)
ax_polar.set_xticklabels(compass, fontsize=8)
ax_polar.set_title("Mean Surfer Count by Wind Direction", pad=20)
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax_polar, shrink=0.6, label="Mean Surfer Count")
plt.tight_layout()
plt.show()

# %%
# -------------------------
# Helper functions — binning
# -------------------------

def make_bin_edges_fixed(series, step):
    """Edges aligned to step boundaries, covering the full range."""
    lo = np.floor(series.min() / step) * step
    return np.arange(lo, series.max() + step, step)

def make_bin_edges_equal(series, n_bins):
    """n equal-width bins via linspace."""
    return np.linspace(series.min(), series.max(), n_bins + 1)

def build_surfer_heatmap(win_stats, h_edges, w_edges):
    """
    Given pre-computed edges, return (pivot, h_mids, w_mids, h_counts, w_counts).
    pivot rows = wind bins (low→high), columns = swell-height bins.
    """
    df = win_stats[["mean_count", "swell_height", "wind_speed"]].dropna().copy()
    df["h_bin"] = pd.cut(df["swell_height"], bins=h_edges, include_lowest=True)
    df["w_bin"] = pd.cut(df["wind_speed"],   bins=w_edges, include_lowest=True)

    pivot = (
        df.groupby(["w_bin", "h_bin"], observed=False)["mean_count"]
        .mean()
        .unstack()
        .iloc[::-1]   # low wind at bottom
    )
    h_counts = df.groupby("h_bin", observed=False)["mean_count"].count().reindex(pivot.columns)
    w_counts = df.groupby("w_bin", observed=False)["mean_count"].count().reindex(pivot.index)

    n_cols, n_rows = pivot.shape[1], pivot.shape[0]
    h_mids = [f"{(h_edges[i] + h_edges[i+1]) / 2:.1f}" for i in range(n_cols)]
    w_mids = [f"{(w_edges[i] + w_edges[i+1]) / 2:.1f}" for i in range(len(w_edges) - 2, -1, -1)]

    return pivot, h_mids, w_mids, h_counts, w_counts

# %%
# -------------------------
# 4. Altair heatmap — app-style (blues, marginal bars)
# -------------------------
h_edges = make_bin_edges_fixed(win_stats["swell_height"].dropna(), step=0.2)
w_edges = make_bin_edges_fixed(win_stats["wind_speed"].dropna(),   step=2)

df_hm = win_stats[["mean_count", "swell_height", "wind_speed"]].dropna().copy()
df_hm["h_bin"] = pd.cut(df_hm["swell_height"], bins=h_edges, include_lowest=True)
df_hm["w_bin"] = pd.cut(df_hm["wind_speed"],   bins=w_edges, include_lowest=True)
df_hm["h_mid"] = df_hm["h_bin"].apply(lambda x: round((x.left + x.right) / 2, 2) if pd.notna(x) else np.nan)
df_hm["w_mid"] = df_hm["w_bin"].apply(lambda x: round((x.left + x.right) / 2, 2) if pd.notna(x) else np.nan)
df_hm = df_hm.dropna(subset=["h_mid", "w_mid"]).drop(columns=["h_bin", "w_bin"])

hm_agg = (
    df_hm
    .groupby(["h_mid", "w_mid"], observed=True)["mean_count"]
    .mean()
    .reset_index()
    .rename(columns={"mean_count": "mean_surfers"})
)

h_domain = sorted(hm_agg["h_mid"].unique())
w_domain = sorted(hm_agg["w_mid"].unique(), reverse=True)   # high wind at top

# Main heatmap
_rect = (
    alt.Chart(hm_agg)
    .mark_rect()
    .encode(
        x=alt.X("h_mid:O", title="Swell Height (m)", sort=h_domain),
        y=alt.Y("w_mid:O", title="Wind Speed (kn)",  sort=w_domain),
        color=alt.Color(
            "mean_surfers:Q",
            scale=alt.Scale(scheme="blues"),
            title="Mean Surfers",
        ),
        tooltip=[
            alt.Tooltip("h_mid:Q", title="Swell Height", format=".2f"),
            alt.Tooltip("w_mid:Q", title="Wind Speed",   format=".1f"),
            alt.Tooltip("mean_surfers:Q", title="Mean Surfers", format=".1f"),
        ],
    )
)

_text = (
    alt.Chart(hm_agg)
    .mark_text(fontSize=11)
    .encode(
        x=alt.X("h_mid:O", sort=h_domain),
        y=alt.Y("w_mid:O", sort=w_domain),
        text=alt.Text("mean_surfers:Q", format=".1f"),
        color=alt.value("#333333"),
    )
)

main_hm = (_rect + _text).properties(width=360, height=240)

# Marginal aggregates — mean surfer count per bin, aligned to heatmap axes
h_marginal = hm_agg.groupby("h_mid")["mean_surfers"].mean().reset_index()
w_marginal = hm_agg.groupby("w_mid")["mean_surfers"].mean().reset_index()

top_bar = (
    alt.Chart(h_marginal)
    .mark_bar(color="#0068c9", opacity=0.6, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("h_mid:O", title="", sort=h_domain, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("mean_surfers:Q", title="Avg Surfers", axis=alt.Axis(tickCount=3)),
    )
    .properties(width=360, height=80)
)

right_bar = (
    alt.Chart(w_marginal)
    .mark_bar(color="#0068c9", opacity=0.6, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
    .encode(
        y=alt.Y("w_mid:O", title="", sort=w_domain, axis=alt.Axis(labels=False, ticks=False)),
        x=alt.X("mean_surfers:Q", title="Avg Surfers", axis=alt.Axis(tickCount=3)),
    )
    .properties(width=80, height=240)
)

(
    alt.vconcat(
        top_bar,
        alt.hconcat(main_hm, right_bar, spacing=5),
        spacing=5,
    )
    .properties(title="Mean Surfer Count  |  Wave Height × Wind Speed")
    .configure_view(strokeOpacity=0)
    .configure_axis(grid=False)
    .configure_title(fontSize=14)
).display()
# %%

# -------------------------
# 5. Wind direction by hour of day
# -------------------------
def circular_mean_deg(s):
    """Circular mean to handle 0/360 wrap-around."""
    rad = np.radians(s.dropna())
    return (np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)

wx_hourly = (
    win_stats[win_stats["t_start"].dt.hour.between(6, 21)]
    .assign(hour=lambda df: df["t_start"].dt.hour)
    [["hour", "wind_dir", "mean_count"]]
    .dropna()
    .groupby("hour", as_index=False)
    .agg(
        mean_dir=("wind_dir", circular_mean_deg),
        mean_surfers=("mean_count", "mean"),
        n=("mean_count", "count"),
    )
)

compass_ref = pd.DataFrame({
    "dir_deg": [0, 45, 90, 135, 180, 225, 270, 315],
    "label":   ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
})

_rules = (
    alt.Chart(compass_ref)
    .mark_rule(strokeDash=[3, 3], color="#dddddd")
    .encode(y=alt.Y("dir_deg:Q", scale=alt.Scale(domain=[0, 360])))
)

_line = (
    alt.Chart(wx_hourly)
    .mark_line(color="#0068c9", strokeWidth=2)
    .encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y(
            "mean_dir:Q",
            title="Mean Wind Direction",
            scale=alt.Scale(domain=[0, 360]),
            axis=alt.Axis(
                values=[0, 45, 90, 135, 180, 225, 270, 315],
                labelExpr="{'0':'N','45':'NE','90':'E','135':'SE','180':'S','225':'SW','270':'W','315':'NW'}[datum.value]",
            ),
        ),
    )
)

_dots = (
    alt.Chart(wx_hourly)
    .mark_circle(size=80)
    .encode(
        x=alt.X("hour:O"),
        y=alt.Y("mean_dir:Q"),
        size=alt.Size("mean_surfers:Q", title="Avg Surfers"),
        tooltip=[
            alt.Tooltip("hour:O",         title="Hour"),
            alt.Tooltip("mean_dir:Q",     title="Mean Dir (°)", format=".0f"),
            alt.Tooltip("mean_surfers:Q", title="Avg Surfers",  format=".1f"),
            alt.Tooltip("n:Q",            title="Windows"),
        ],
    )
)

(_rules + _line + _dots).properties(
    title="Mean Wind Direction by Hour of Day",
    width=500,
    height=300,
).configure_view(strokeOpacity=0).configure_axis(grid=False).configure_title(fontSize=14).display()
# %%
