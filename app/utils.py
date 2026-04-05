
import os
import subprocess
import tarfile
import time
import urllib.request
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

_WEATHER_COLS = ["forecast_dt", "wind_const_speed", "gust_speed", "wind_speed",
                 "wind_dir", "swell_height", "swell_period", "swell_dir"]

# Cloud mode: S3_BUCKET env var is set by Streamlit secrets bridge in app.py
CLOUD_MODE = bool(os.getenv("S3_BUCKET"))
_ROOT = Path(__file__).resolve().parents[1]
_LITESTREAM_BIN = "/tmp/litestream"
_LITESTREAM_VERSION = "0.5.10"
_DB_REFRESH_SECS = 180  # re-download from S3 at most once every 3 minutes


def get_db_path(name: str) -> Path:
    """Return path to a SQLite DB: /tmp/ in cloud mode, local data/ in dev mode."""
    if CLOUD_MODE:
        return Path(f"/tmp/{name}.db")
    return _ROOT / "data" / "app_runs" / f"{name}.db"


def _install_litestream():
    """Download litestream binary to /tmp if not already present."""
    if os.path.exists(_LITESTREAM_BIN):
        return
    url = (
        f"https://github.com/benbjohnson/litestream/releases/download/"
        f"v{_LITESTREAM_VERSION}/litestream-{_LITESTREAM_VERSION}-linux-x86_64.tar.gz"
    )
    tmp_tar = "/tmp/litestream.tar.gz"
    urllib.request.urlretrieve(url, tmp_tar)
    with tarfile.open(tmp_tar) as tf:
        tf.extract("litestream", "/tmp")
    os.chmod(_LITESTREAM_BIN, 0o755)


def ensure_dbs_from_s3():
    """Restore/refresh DBs from S3 using litestream. No-op in local dev mode.

    Uses a file-mtime check so the actual restore runs at most once every
    _DB_REFRESH_SECS seconds, even though this function is called on every
    Streamlit rerun.
    """
    if not CLOUD_MODE:
        return
    _install_litestream()
    bucket = os.environ["S3_BUCKET"]
    for name in ["surfer_data", "weather_data"]:
        out_path = Path(f"/tmp/{name}.db")
        if out_path.exists() and (time.time() - out_path.stat().st_mtime) < _DB_REFRESH_SECS:
            continue  # still fresh
        tmp_path = Path(f"/tmp/{name}.db.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        subprocess.run(
            [_LITESTREAM_BIN, "restore", "-o", str(tmp_path), f"s3://{bucket}/{name}.db"],
            check=True,
        )
        os.replace(tmp_path, out_path)  # atomic swap


def load_weather_data(db_path, tz):
    """Load weather forecasts and enrich with forecast_dt, scrape_dt, delta_forecast."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM forecasts WHERE scrape_timestamp IS NOT NULL AND scrape_timestamp != 'None'",
        conn
    )
    conn.close()

    def build_forecast_dt(row):
        scrape_ts = pd.Timestamp(row["scrape_timestamp"])
        day  = int(row["number_day"])
        hour = int(row["hour"])
        year, month = scrape_ts.year, scrape_ts.month
        if scrape_ts.day - day > 15:
            month += 1
            if month > 12:
                month, year = 1, year + 1
        try:
            dt = pd.Timestamp(year=year, month=month, day=day, hour=hour)
            return dt.tz_localize(tz)
        except Exception:
            return pd.NaT

    df["forecast_dt"]  = df.apply(build_forecast_dt, axis=1)
    df["scrape_dt"]    = pd.to_datetime(df["scrape_timestamp"], utc=True).dt.tz_convert(tz).dt.floor("min")
    df["delta_forecast"] = df["forecast_dt"] - df["scrape_dt"]
    return df


def get_best_forecasts(weather_df):
    """Deduplicate weather: one row per forecast slot, keeping the freshest scrape."""
    return (
        weather_df[weather_df["delta_forecast"] > pd.Timedelta(0)]
        .sort_values("delta_forecast")
        .drop_duplicates(subset=["forecast_dt"])
        [_WEATHER_COLS]
    )


def enrich_win_stats(win_stats, best_forecasts, tz):
    """Left-join weather columns onto win_stats, keyed on floored hour."""
    return (
        win_stats.assign(
            forecast_dt=win_stats["t_start"].dt.floor("h").dt.tz_convert(tz)
        )
        .merge(best_forecasts, on="forecast_dt", how="left")
        .drop(columns=["forecast_dt"])
    )


def build_surfer_wx_heatmap(win_stats, h_step=0.2, w_step=2):
    """Bin win_stats by swell height × wind speed; return (hm_agg, h_domain, w_domain) for Altair."""
    def _edges(series, step):
        lo = np.floor(series.min() / step) * step
        return np.arange(lo, series.max() + step, step)

    h_edges = _edges(win_stats["swell_height"].dropna(), h_step)
    w_edges = _edges(win_stats["wind_speed"].dropna(),   w_step)

    df = win_stats[["mean_count", "swell_height", "wind_speed"]].dropna().copy()
    df["h_bin"] = pd.cut(df["swell_height"], bins=h_edges, include_lowest=True)
    df["w_bin"] = pd.cut(df["wind_speed"],   bins=w_edges, include_lowest=True)
    df["h_mid"] = df["h_bin"].apply(lambda x: round((x.left + x.right) / 2, 2) if pd.notna(x) else float("nan"))
    df["w_mid"] = df["w_bin"].apply(lambda x: round((x.left + x.right) / 2, 2) if pd.notna(x) else float("nan"))
    df = df.dropna(subset=["h_mid", "w_mid"])

    hm_agg = (
        df.groupby(["h_mid", "w_mid"], observed=True)["mean_count"]
        .mean()
        .reset_index()
        .rename(columns={"mean_count": "mean_surfers"})
    )
    h_domain = sorted(hm_agg["h_mid"].unique())
    w_domain = sorted(hm_agg["w_mid"].unique(), reverse=True)
    return hm_agg, h_domain, w_domain


def get_live_wx(best_forecasts, tz):
    """Return (current_wx, prev_wx) rows from best_forecasts for live metric display.

    current_wx: forecast slot whose forecast_dt is closest to now.
    prev_wx:    the slot immediately before it on the same calendar day, or None.
    """
    if best_forecasts.empty:
        return None, None

    now = pd.Timestamp.now(tz)
    time_diffs = (best_forecasts["forecast_dt"] - now).abs()
    current_wx = best_forecasts.loc[time_diffs.idxmin()]
    current_dt = current_wx["forecast_dt"]

    earlier = best_forecasts[
        (best_forecasts["forecast_dt"] < current_dt) &
        (best_forecasts["forecast_dt"].dt.date == current_dt.date())
    ]
    prev_wx = earlier.iloc[-1] if not earlier.empty else None
    return current_wx, prev_wx


def get_latest_raw_annotated_pair(visuals_dir: Path):
    files = sorted(visuals_dir.glob("*.jpg"))

    for f in reversed(files):
        if f.name.endswith("_raw.jpg"):
            annotated = visuals_dir / f.name.replace("_raw.jpg", "_annotated.jpg")
            if annotated.exists():
                return f, annotated

    raise FileNotFoundError("No matching raw/annotated pair found")

def filepath_to_dt(path: Path, tz: str) -> pd.Timestamp:
    """
    Extract datetime from filename and localize to timezone.

    Parameters
    ----------
    path : Path
        Image file path with name YYYYMMDD_HHMMSS_*.jpg
    tz : str
        Timezone string (e.g. 'Asia/Jerusalem')

    Returns
    -------
    pd.Timestamp (tz-aware)
    """
    date_str, time_str = path.name.split("_", 2)[:2]

    return (
        pd.to_datetime(
            date_str + time_str,
            format="%Y%m%d%H%M%S"
        )
        .tz_localize(tz)
    )

def find_closest_file_by_time(
    files: list[Path],
    target_dt: pd.Timestamp,
    tz: str
) -> Path:
    """
    Find file whose timestamp (from filename) is closest to target_dt.

    Parameters
    ----------
    files : list[Path]
        List of image file paths
    target_dt : pd.Timestamp
        Target datetime (tz-aware)
    tz : str
        Timezone string

    Returns
    -------
    Path
        Closest file
    """
    if target_dt.tzinfo is None:
        raise ValueError("target_dt must be timezone-aware")

    best_file = None
    best_delta = None

    for f in files:
        try:
            file_dt = filepath_to_dt(f, tz)
        except Exception:
            continue  # skip badly formatted files

        delta = abs((file_dt - target_dt).total_seconds())

        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_file = f

    if best_file is None:
        raise FileNotFoundError("No valid timestamped files found")

    return best_file
