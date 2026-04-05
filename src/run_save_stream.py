import cv2
import time
from pathlib import Path
import datetime
import sys
# --------------------------------------------------------
# Setup project root 
# --------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parents[1]  
sys.path.append(str(ROOT_DIR)) 

from utils.config import STREAM_URL
 
is_hd = STREAM_URL.split('Zvulun_')[1].startswith('1080')
   
DATA_DIR = ROOT_DIR / "data" 
save_dir = DATA_DIR/ 'stream_videos'
if is_hd:
    save_dir = DATA_DIR/ 'stream_videos_1080p'
save_dir.mkdir(parents=True, exist_ok=True) 

# Output file path
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = Path(save_dir/ f"{timestamp}_recorded.mp4")

def record_stream(stream_url=STREAM_URL, output_file=output_path, fps=25):
    """
    Record an HLS video stream to disk using OpenCV.
    """

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Could not open stream.")
        return

    # Get stream width/height or fallback to 1280×720
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    print(f"Recording at resolution: {w}x{h}")

    # Define video codec + output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # or "avc1"
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))

    print(f"▶ Recording started → {output_file}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠ Stream dropped — reconnecting...")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            time.sleep(0.5)
            continue

        writer.write(frame)   # Save frame to file

        # Optional preview window:
        cv2.imshow("Recording...", frame)
        if cv2.waitKey(1) == 27:  # ESC to stop
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("⏹ Recording stopped.")


if __name__ == "__main__":
    record_stream()
