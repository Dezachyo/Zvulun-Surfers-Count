import cv2
import os
import time
import datetime
from pathlib import Path
import sys
from utils.config import STREAM_URL


def collect_frames(
    save_dir,          # no default argument here!
    mode="time",
    rate=1,
    max_frames=None
):
    """
    Collect frames from stream with configurable rate.
    Paths are always resolved relative to project root.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    frame_counter = 0
    saved_counter = 0
    last_save_time = time.time()

    print(f"Saving frames to: {save_dir}")
    print(f"Capture mode = {mode}, rate = {rate}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream dropped — reconnecting...")
            cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
            continue

        now = time.time()

        # ------------------------------
        # Save every N seconds
        # ------------------------------
        if mode == "time":
            if now - last_save_time >= rate:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = save_dir / f"{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"[SAVED] {filename}")
                last_save_time = now
                saved_counter += 1

        # ------------------------------
        # Save every N frames
        # ------------------------------
        elif mode == "frames":
            if frame_counter % rate == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = save_dir / f"{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"[SAVED] {filename}")
                saved_counter += 1

        frame_counter += 1

        if max_frames and saved_counter >= max_frames:
            print("Reached max_frames. Stopping.")
            break

        cv2.imshow("Frame Collector", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

