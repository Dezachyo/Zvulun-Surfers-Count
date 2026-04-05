# SurfersCount

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)
![SQLite](https://img.shields.io/badge/Storage-SQLite-lightgrey)
![AWS S3](https://img.shields.io/badge/Cloud-AWS_S3-orange)

A personal project combining two interests: data and surfing. A YOLO model watches the Zvulun beach webcam 24/7, counts surfers every 5 minutes, and pairs every count with a live WindGuru weather forecast — building a dataset that quantifies when surfers show up and what conditions they prefer.



**[Live Demo →](https://zvulun-surfers-count-ad465jlrzcuedhqkh2xkrw.streamlit.app/)**

---

## How It Works

```
Live HLS Webcam Stream (Dabush Beach, Herzliya)
            │
            ▼
  Frame Sampling  (every 20 s)
            │
  Frame Masking   (top 15% removed — sky)
            │
  YOLOv8s-P2 + ByteTrack
            │
            ├──► surfer_data.db   (SQLite — surfer counts, UTC timestamps)
            │
  WindGuru Scraper (every 5 min, Selenium + headless Chrome)
            │
            └──► weather_data.db  (SQLite — wind, swell height, period, direction)
                        │
                        ▼
         Streamlit Dashboard  ──  Offline Analysis Scripts
```

---

## Model Development

This was the hardest part. Surfers in a beach webcam are **small, distant blobs** viewed from a fixed elevated angle. Generic people-detectors fail immediately — "surfer" isn't even a COCO class.

### The naive attempt — public datasets

I started with surfing datasets from Roboflow: clean, well-labeled, close-up surf photography. The model hit mAP@50 of **0.94–0.97**. Then I pointed it at the actual webcam feed and it barely detected anything. High scores, wrong application.

### Switching to real frames

I labeled frames directly from the Dabush webcam — the exact viewpoint and lighting the model would run on in production. mAP@50 dropped to **~0.55**. That's the honest number: small objects, water glare, overcast days, early morning shadows. Swapping the backbone from YOLOv8n → YOLOv8s changed nothing. The architecture wasn't the bottleneck.

### Two things that actually helped

**Masking the background.** The top 15% of the frame is timestamp/sky/deep sea — all noise. Masking it out before training gave a meaningful improvement in localisation quality (mAP@50-95 +0.04).

**Adding a P2 detection head.** The standard YOLOv8 architecture's smallest detection stride is 8 pixels. The P2 variant adds a stride-4 head specifically for small objects. For surfers at this distance, that's the right tool. Combined with 100 epochs on GPU, this pushed mAP@50 to **0.691** on real webcam frames — the best result across all experiments.


### Filtering and counting at inference time

Two post-detection steps clean up the results before anything gets counted:

**Size filter.** Any bounding box larger than 0.1% of the frame area is discarded. At this camera distance a real surfer is a small object — anything bigger is a boat, buoy, or other clutter.

**Tracker memory window.** The surfer count isn't just the number of boxes visible in the current frame. ByteTrack assigns each detection a persistent ID, and the system keeps a `track_last_seen_frame` dictionary recording the last frame each ID appeared. The active count is the number of unique IDs seen within the last N frames. This means a surfer who ducks under a wave or is briefly occluded stays counted — the count reflects who is in the water, not just who the model can see right now.

### Final model at a glance

| Parameter | Value |
|---|---|
| Architecture | YOLOv8s + P2 head |
| Pretrained weights | YOLOv8s (COCO) |
| Training data | Labeled frames from Dabush webcam |
| Epochs | 100 |
| Batch / image size | 16 / 640×640 |
| mAP@50 (val) | 0.691 |
| Precision / Recall | 0.688 / 0.669 |
| Augmentations | Mosaic, random erasing, HSV jitter, RandAugment |

---

## Dashboard

The Streamlit app (`app/app.py`) is the main way to consume the data:

- **Live metrics** — current surfer count, wind speed, swell height, swell period
- **Daily trend** — realtime area chart of surfer activity throughout the day
- **Hourly historical average** — what a typical hour looks like, with today's live data overlaid
- **Swell × Wind heatmap** — mean surfer count across swell height and wind speed combinations, filterable by date range
- **Image viewer** — browse annotated frames, linked to the detection table by timestamp

Note: Live mertics and Daily trend are off when main code (run_inferance.py) is not running -no incoming data. 

<img width="1559" height="759" alt="Dashboard live" src="https://github.com/user-attachments/assets/5caaa624-070f-4986-82aa-12b44e6a9f96" />


**[Open the live dashboard →](https://surferscount-jmilv5du45qzjz6ktwqf8p.streamlit.app/)**

---

## Tech Stack

| Layer | Tools |
|---|---|
| Detection | Ultralytics YOLOv8, ByteTrack, SAHI |
| Weather | Selenium + headless Chrome → WindGuru |
| Storage | SQLite |
| Dashboard | Streamlit, Altair, Plotly |
| Analysis | pandas, numpy, matplotlib, seaborn, scipy |
| Deployment | Streamlit Community Cloud, Litestream, AWS S3 |

---

## Project Structure

```
SurfersCount/
├── app/                        # Streamlit dashboard
│   ├── app.py                  # Main dashboard page
│   ├── utils.py                # Shared helpers (weather join, heatmap builder, cloud DB restore)
│   ├── config.py               # Dashboard settings (timezone, smoothing, live threshold)
│   └── pages/
│       └── 1_Frame_Viewer.py   # Browse annotated frames by timestamp
├── src/                        # Detection & analysis scripts
│   ├── run_inferance.py        # Production inference loop (runs 24/7)
│   ├── run_data_analysis.py    # Offline analysis & visualisations
│   ├── run_train.py            # YOLO model training
│   └── run_save_stream.py      # Stream capture utility (data collection)
├── utils/                      # Shared library
│   ├── config.py               # Stream URL, WindGuru URL, timezone, beach locations
│   ├── inferance.py            # Sampling window logic + tracker memory window
│   ├── tracking.py             # YOLO / SAHI detector wrappers
│   ├── wgscraper.py            # Selenium WindGuru scraper
│   ├── preprocessing.py        # Frame masking (top 15% removal)
│   ├── viz.py                  # Frame annotation helpers
│   ├── frame_capture.py        # HLS stream frame collection
│   └── bytetrack.yaml          # ByteTrack tracker config
├── requirements.txt            # Full ML environment (CPU; see comments for GPU/CUDA)
├── requirements_app.txt        # Slim cloud deployment deps (no torch/ultralytics)
├── litestream.yml              # SQLite → S3 replication config
└── .env.example                # Required environment variables template
```

---

## Setup

### Prerequisites
- Python 3.10+
- Chrome + ChromeDriver (for WindGuru scraping via Selenium)
- AWS S3 bucket + Litestream (for cloud DB sync — optional for local use)

### Local development

```bash
# 1. Clone
git clone https://github.com/Dezachyo/SurfersCount.git
cd SurfersCount

# 2. Install (CPU)
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env        # fill in your AWS keys (optional for local-only use)

# 4. Set your webcam stream + WindGuru URL
#    Edit utils/config.py — WEBCAMS and WINDGURU_URL

# 5. Run the detection loop
python src/run_inferance.py

# 6. Launch the dashboard
streamlit run app/app.py
```

### GPU / CUDA

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Cloud deployment (Streamlit Community Cloud)

1. Fork this repo and connect it to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Set the secrets from `.env.example` in the Streamlit Cloud dashboard
3. On startup, `app/utils.py` restores the SQLite DBs from S3 via Litestream automatically
4. Run `litestream replicate -config litestream.yml` locally alongside `run_inferance.py` to keep S3 in sync

Streamlit Cloud uses `requirements_app.txt` (slim, no torch/ultralytics) — the heavy ML stack only runs on your local machine.
