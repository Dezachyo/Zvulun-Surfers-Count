import sys
from pathlib import Path
import cv2
import numpy as np
from time import time
import pandas as pd
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from zoneinfo import ZoneInfo
from utils.preprocessing import mask_top_fraction, bottom_exclude_mask_from_water
from utils.viz import draw_counter,draw_active_trackers , annotate_frame
from utils.tracking import box_area_filter_idx, DetectionResult
from utils.config import STREAM_URL, TZ

# --------------------------
# Helpers
# --------------------------
def frame_results_to_row(
    det,                 # DetectionResult
    frame_ind,
    active_trackers,
):
    """
    det: DetectionResult (after filtering to surfers only)
    active_trackers: iterable of active track IDs

    Returns a dict with Python-native types only.
    """

    # List of [x1, y1, x2, y2]
    bbox_list = (
        det.boxes_xyxy.tolist()
        if det.boxes_xyxy is not None and len(det.boxes_xyxy) > 0
        else []
    )

    avg_confidence = (
        float(np.mean(det.scores))
        if det.scores is not None and len(det.scores) > 0
        else 0.0
    )

    surfer_count = (
    len(active_trackers)
    if det.track_ids is not None
    else len(det.boxes_xyxy)
)
    
    return {
        "timestamp": datetime.now(ZoneInfo(TZ)).isoformat(),
        "frame_ind": frame_ind,
        "surfer_count": surfer_count,
        "avg_confidence": avg_confidence,
        "bbox_list": bbox_list,
        "active_trackers_id": list(map(int, active_trackers)),
    }
def prepare_df_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame with Python-native objects into a SQL-safe DataFrame.

    - Lists / dicts -> JSON strings
    - Datetimes -> ISO strings (if any)
    - Scalars left untouched
    """

    df = df.copy()

    for col in df.columns:
        # Skip empty columns
        if df[col].isna().all():
            continue

        sample = df[col].dropna().iloc[0]

        # JSON-serializable structures
        if isinstance(sample, (list, dict)):
            df[col] = df[col].apply(json.dumps)

        # Datetime objects
        elif isinstance(sample, (pd.Timestamp, datetime)):
            df[col] = df[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

        # Everything else (int, float, str) → leave as is
        else:
            pass

    return df

def save_window_outputs(
    df,
    frames,
    conn,
    save_visual_dir,
    log_path,
    window_start_ts,
    window_end_ts,
    stop,
):
    # Save detections to DB
    if not df.empty:
        df_sql = prepare_df_for_sql(df)
        df_sql.to_sql("detections", conn, if_exists="append", index=False)
        conn.commit()

    # Save representative frames
    if frames:
        i = min(10, len(frames) - 1)
        f = frames[i]
        ts = window_end_ts.strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(str(save_visual_dir / f"{ts}_raw.jpg"), f["raw"])
        if "annotated" in f:
            cv2.imwrite(
                str(save_visual_dir / f"{ts}_annotated.jpg"),
                f["annotated"],
            )

    # Log window summary
    log_sampling_window(
        log_path=log_path,
        start_time=window_start_ts,
        end_time=window_end_ts,
        n_frames=len(frames),
        n_rows=len(df),
        avg_surfer_count=df["surfer_count"].mean() if not df.empty else 0.0,
        status="interrupted" if stop else "ok",
    )


def log_sampling_window(
    log_path,
    start_time,
    end_time,
    n_frames,
    n_rows,
    avg_surfer_count,
    status,
):
    """
    Append a single sampling-window entry to a text log.

    Parameters
    ----------
    log_path : Path or str
        Path to the log file.

    start_time : datetime
        Local datetime when the window started.

    end_time : datetime
        Local datetime when the window ended.

    n_frames : int
        Number of frames processed in the window.

    n_rows : int
        Number of detection rows collected.

    avg_surfer_count : float
        Mean surfer count across frames.

    status : str
        Window status (e.g. 'ok', 'interrupted', 'error').
    """

    duration_sec = (end_time - start_time).total_seconds()

    line = (
        f"{start_time.isoformat()} | "
        f"{end_time.isoformat()} | "
        f"{duration_sec:.2f} | "
        f"frames={n_frames} | "
        f"rows={n_rows} | "
        f"avg_count={avg_surfer_count:.2f} | "
        f"{status}\n"
    )

    with open(log_path, "a") as f:
        f.write(line)


def run_sampling_winow(cap, duration, detector, track_id, *, track_expired_frames,is_live_stream: bool = False):
    """
        Run a single time-limited sampling window over a video stream, performing
        object detection, tracking, and per-frame data collection.

        This function:
        - Reads frames continuously from an open OpenCV VideoCapture
        - Applies preprocessing (masking)
        - Runs YOLO detection + tracking
        - Filters detections to a target class (e.g. surfers)
        - Maintains frame-based track persistence
        - Collects per-frame metadata and optional annotated frames
        - Allows user interruption via ESC key

        The function is designed to be called repeatedly by an outer scheduler
        (e.g. periodic sampling every N minutes).

        Parameters
        ----------
        cap : cv2.VideoCapture
            An *already opened* OpenCV VideoCapture object connected to a live stream
            or video file. The capture is reused across sampling windows.

        duration : float
            Length of the sampling window in **seconds** (wall-clock time).

        model : ultralytics.YOLO
            A loaded YOLO model instance used for detection and tracking.
            The model is expected to support `model.track()`.

        track_id : list[int]
            List of class IDs to retain (e.g. surfer class IDs).
            All other detections are discarded.

        conf : float, optional (default=0.1)
            Confidence threshold passed to YOLO detection.

        iou : float, optional (default=0.2)
            IoU threshold passed to the YOLO tracker.

        tracker_ymal : str or Path
            Path to the tracker configuration YAML file (e.g. ByteTrack config).

        track_expired_frames : int
            Number of frames after which a track is considered inactive if not seen.
            Tracking persistence is frame-based (not time-based).

        Returns
        -------
        df : pandas.DataFrame
            Per-frame metadata collected during the sampling window.
            Each row corresponds to one processed frame and includes:
            - timestamp
            - frame index
            - surfer count
            - bounding boxes
            - active tracker IDs
            - confidence statistics

        frames : list[dict]
            List of frame dictionaries collected during the window.
            Each element contains:
            - "raw": original unmasked frame (np.ndarray)
            - "annotated": annotated frame (np.ndarray), if detections exist

        stop_requested : bool
            Indicates whether the user requested termination by pressing ESC.
            - True  → stop the entire application
            - False → window ended normally

        Notes
        -----
        - The function does **not** write to disk or a database.
        Persistence is handled by the caller.
        - ESC handling is implemented via OpenCV (`cv2.waitKey`) and is the
        *primary* stop mechanism
        - Tracking is frame-based to avoid dependence on real-time clock drift
        when processing live streams.
        - On stream read failure, the function attempts to reconnect and
        reset tracking state.

        """      
    track_last_seen_frame = {}
    frame_count = 0
    window_start = time()
    
    data = []
    frames = []
    
    while time() - window_start < duration:
        ret, frame = cap.read()
                
        if not ret:
            if is_live_stream:
                print("⚠️ Stream read failed — reconnecting")
                frame_count = 0
                cap.release()
                cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

                track_last_seen_frame.clear()
                if hasattr(detector, "model") and hasattr(detector.model, "tracker"):
                    detector.model.tracker.reset()
                continue
            else:
                # EOF for video file → end window cleanly
                print("🎬 End of video reached")
                break
        
        frame_count += 1
        # ================= Prepro =======================
        frame_masked = mask_top_fraction(frame,frac=0.15)
        # Maybe add bottom mask()
                
        # ================= Detection =====================
        try:
            #SAHI Detector has frame_idx
            det = detector.detect(frame_masked, frame_count)
        except TypeError:
            # UltralyticsDetector has no frame_idx
            det = detector.detect(frame_masked)

        # ================= Filtering =====================
        # Area-based filter
        keep_idx = box_area_filter_idx(
            det.boxes_xyxy,
            frame_masked.shape,
            max_area_frac=0.001,
        )

        det = DetectionResult(
            boxes_xyxy=det.boxes_xyxy[keep_idx],
            class_ids=det.class_ids[keep_idx],
            scores=det.scores[keep_idx],
            track_ids=None if det.track_ids is None else det.track_ids[keep_idx],
        )

        # Class filter (e.g., surfers only)
        surfer_idx = np.isin(det.class_ids, track_id)

        det = DetectionResult(
            boxes_xyxy=det.boxes_xyxy[surfer_idx],
            class_ids=det.class_ids[surfer_idx],
            scores=det.scores[surfer_idx],
            track_ids=None if det.track_ids is None else det.track_ids[surfer_idx],
        )

        # ================= Tracking logic =================
        # EXACTLY the same logic you had before
        if det.track_ids is not None:
            for tid in det.track_ids:
                track_last_seen_frame[tid] = frame_count

        active_tracks = {
            tid for tid, f in track_last_seen_frame.items()
            if frame_count - f < track_expired_frames
        }

        # prune expired tracks (same as before)
        track_last_seen_frame = {
            tid: f for tid, f in track_last_seen_frame.items()
            if frame_count - f < track_expired_frames
        }

        # ================= Counting ======================
        if det.track_ids is not None:
            # Ultralytics: count active trackers
            count = len(active_tracks)
        else:
            # SAHI: count detections in this frame
            count = len(det.boxes_xyxy)

        # ================= Data collection ===============
        row = frame_results_to_row(
            det=det,
            frame_ind=frame_count,
            active_trackers=active_tracks,
        )
        data.append(row)

        # ================= Visualization =================
        if len(det.boxes_xyxy) > 0:
            annotated_for_save = annotate_frame(
                frame=frame, 
                det=det,
                color=(0, 255, 0),
                thickness=2,  
                active_tracks=active_tracks if det.track_ids is not None else None,
                count=count,
            )
            frame_dict = {"raw": frame, "annotated": annotated_for_save}
        else:
            frame_dict = {"raw": frame}

        frames.append(frame_dict)

        raw_with_count = draw_counter(frame_masked, count)

        if "annotated" in frame_dict:
            annotated = frame_dict["annotated"]
            if raw_with_count.shape != annotated.shape:
                annotated = cv2.resize(
                    annotated,
                    (raw_with_count.shape[1], raw_with_count.shape[0]),
                )
            combined = np.hstack([raw_with_count, annotated])
        else:
            combined = raw_with_count

        cv2.imshow("Surfer Monitoring", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            df = pd.DataFrame(data)
            return df, frames, True

    # ================= Normal window end =================
    df = pd.DataFrame(data)
    return df, frames, False
