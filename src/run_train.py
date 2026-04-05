import sys
from pathlib import Path
from ultralytics import YOLO
import torch


# --------------------------------------------------------
# Locate project root directory
# --------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parents[1]              # one level up from /src
DATA_DIR = ROOT / "data" / "Dabush"  # your dataset
YAML_PATH = DATA_DIR / "data.yaml"       # roboflow's YAML
P2_CFG_PATH = ROOT / "models" / "yolov8s-p2.yaml"
WEIGHTS_PATH = ROOT / "models" / "yolov8s.pt"


print(f"Project root: {ROOT}")
print(f"Dataset YAML: {YAML_PATH}")


def train():
    # 1. Verify GPU status
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {device}")

    # 2. Build the P2 architecture from YAML and transfer weights
    # This construction method ensures the P2 magnifying head is active.
    model = YOLO(str(P2_CFG_PATH)).load(str(WEIGHTS_PATH))

    # 3. Launch training with specialized SOD parameters
    model.train(
        data=str(YAML_PATH),
        epochs=100,           # More epochs to help 112 images converge
        imgsz=640,            # Standard training resolution
        batch=16,
        device=device,        # Explicitly set GPU
        amp=True,             # Saves VRAM on GPU runs
        
        # --- Augmentations optimized for small datasets ---
        mosaic=0.4,           # Reduced noise to prevent overfitting
        scale=0.5,            # Forces model to learn distant silhouettes
        hsv_h=0.015,          # Hue shifts for shifting water color
        hsv_s=0.7,            # Saturation for different lighting
        hsv_v=0.4,            # Value/Brightness for sun glare
        fliplr=0.5,           # Mirroring to double your 112 images
        patience=50,          # Don't stop during early training noise
        
        name="Dabush_P2_GPU_SOD_v2",
        project=str(ROOT / "runs" / "detect")
    )

if __name__ == "__main__":
    train()