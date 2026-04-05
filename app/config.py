
TZ = "Asia/Jerusalem"

YOLO_WARMUP_FRAMES = 40

COUNT_STAT_REALTIME_COL = 'mean_count'
ROUND_TO_INT = True       # round surfer counts to nearest integer in all chart displays
SMOOTH_MINUTES = 20  # smoothing window in minutes applied to surfer count charts

delta_is_live = 10 # minutes - how recent should the data be to be considered "live" for the realtime view

# Live weather metric cards — each entry drives one st.metric column.
# col: column name in _best_forecasts
# label: display name
# unit: appended to the value and delta strings
# fmt: Python format spec for the number
LIVE_METRICS = [
    {"col": "wind_speed",   "label": "Wind Speed",  "unit": "kn", "fmt": ".0f"},
    {"col": "wind_dir",     "label": "Wind Direction",    "unit": "°",  "fmt": ".0f"},
    {"col": "swell_height", "label": "Wave Height", "unit": "m",  "fmt": ".1f"},
    {"col": "swell_period", "label": "Period",      "unit": "s",  "fmt": ".0f"},
]