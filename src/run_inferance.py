import sys
from pathlib import Path
import pandas as pd
import cv2
from ultralytics import YOLO
import time as _time
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from sahi import AutoDetectionModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.tracking import UltralyticsDetector, SahiDetector
from utils.wgscraper import ScraperWg
from utils.inferance import run_sampling_winow, save_window_outputs
import utils.config as config

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
MODEL_DIR = ROOT / "models" #production model
MODEL_PATH = MODEL_DIR / "best.pt"
TRACKER_YAML_PATH = ROOT/'utils'/'bytetrack.yaml'
# Where data will be stored
SAVE_RUN_DIR = ROOT / "data" / "app_runs"
SAVE_VISUAL_DIR = SAVE_RUN_DIR / "visuals"

SAVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
SAVE_VISUAL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = SAVE_RUN_DIR / "surfer_data.db"
WEATHER_DB_PATH = SAVE_RUN_DIR / "weather_data.db"

# -------------- SQLite Setup---------------- #TODO move to separate module utis.db.py
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn_weather = sqlite3.connect(WEATHER_DB_PATH, check_same_thread=False)

cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    timestamp TEXT,
    frame_ind INTEGER,
    surfer_count INTEGER,
    avg_confidence REAL,
    bbox_list TEXT,
    active_trackers_id TEXT
)
""")

# Initialize Weather Table
cursor_w = conn_weather.cursor()
cursor_w.execute("""
CREATE TABLE IF NOT EXISTS forecasts (
    day TEXT, number_day INTEGER, hour INTEGER, 
    wind_const_speed INTEGER, gust_speed INTEGER, 
    swell_height INTEGER, swell_period INTEGER, 
    wind_dir INTEGER, swell_dir INTEGER,
    wind_speed INTEGER, arrow_wind_dir INTEGER, 
    arrow_swell_dir INTEGER, scrape_timestamp TEXT
)
""")

conn.commit()
conn_weather.commit()

# ---------------- Load Model ------------------
#model = YOLO(MODEL_DIR / "yolov8s.pt")
model = YOLO(MODEL_PATH)

detector = UltralyticsDetector(
    model=model,
    tracker_yaml=TRACKER_YAML_PATH,
    conf=0.1,
    iou=0.2,
)

# Sahi 

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH.as_posix(),
    confidence_threshold=0.6,
    device="cpu",
)


detector_sahi = SahiDetector(
    sahi_model=detection_model,
    run_every=30,
)

class_names = model.names
surfer_class_id = [cls_id for cls_id, name in class_names.items() if name.lower() == "surfer"]

# ---------------- Load Video ------------------
cap = cv2.VideoCapture(config.WEBCAMS['zvulun']['stream_url_sd'], cv2.CAP_FFMPEG)

save_data = config.SAVE_DATA
track_expired_frames = config.TRACK_EXPIRED_FRAMES
duration = config.SAMPLING_DURATION_SEC
sampling_interval_sec = config.SAMPLING_INTERVAL_MIN * 60

try:
    while True:
        cycle_start = _time.time()
        window_start_ts = datetime.now(ISRAEL_TZ)
        print(f"\n=== Sampling window started at {window_start_ts} ===")

        df, frames, stop = run_sampling_winow(
            cap=cap,
            duration=duration,
            detector=detector,
            track_id=surfer_class_id,
            track_expired_frames=track_expired_frames,
            is_live_stream= True
        )
        if stop:
            print("Stop requested by user (ESC)")
            break
        # ---------------- Save results ----------------
        window_end_ts = datetime.now(ISRAEL_TZ)

        if save_data:
            save_window_outputs(
                df=df,
                frames=frames,
                conn=conn,
                save_visual_dir=SAVE_VISUAL_DIR,
                log_path=SAVE_RUN_DIR / "sampling_windows.log",
                window_start_ts=window_start_ts,
                window_end_ts=window_end_ts,
                stop=stop,
            )
                    # 3. Sync Weather (only if stream was alive — empty df means no frames captured)
            if not df.empty:
                sync_ts = df['timestamp'].iloc[0]
                scraper = ScraperWg(config.WEBCAMS['zvulun']['windguru_url'])
                weather_sql_df = scraper.get_synced_weather(10, sync_ts)
                if weather_sql_df is not None:
                    weather_sql_df.to_sql("forecasts", conn_weather, if_exists="append", index=False)
                    conn_weather.commit()
                scraper.close()

        
        # ---------------- Sleep until next cycle ----------------
        elapsed = _time.time() - cycle_start
        sleep_time = max(0, sampling_interval_sec - elapsed)

        print(
            f"Window finished in {elapsed:.1f}s. "
            f"Sleeping {sleep_time:.1f}s until next window."
        )

        _time.sleep(sleep_time)
 
finally:
    cap.release()
    conn.close()
    cv2.destroyAllWindows()