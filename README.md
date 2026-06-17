# Two-Camera CCTV Suspicious Activity Detection

Python prototype for an academic and home-security CCTV project. It reads CCTV/RTSP/webcam/video input, detects people, tracks them across frames, analyzes suspicious motion patterns, logs events, saves snapshots, and sends phone alerts through Telegram or email.

## What This Prototype Detects

- Unauthorized intrusion into a configured restricted zone
- Running behavior based on tracked movement speed
- Possible chase behavior when two people move fast near each other
- Possible fall using bounding-box shape and downward motion
- Possible fight using close distance plus high motion intensity
- Home safety hazards are available in `README_HOME_SAFETY.md`.

This project detects suspicious activity indicators, not confirmed crimes.

## Architecture

```text
CCTV / RTSP / Webcam / Video File
        ↓
OpenCV Frame Reader
        ↓
YOLO Person Detection
        ↓
Centroid Tracking
        ↓
Behavior Rules
        ↓
Event Logger + Snapshot Saver
        ↓
Telegram / Email Alert
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your Telegram bot token and chat ID.

### GPU Note

If `torch.cuda.is_available()` returns `False`, your environment has CPU-only PyTorch. For an RTX 3050, install the CUDA build that matches your driver from the official PyTorch selector. A typical Windows command is:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Camera Sources

Edit `config/config.yaml`.

For a webcam:

```yaml
source: 0
```

For a video file:

```yaml
source: samples/test_video.mp4
```

For a CCTV RTSP stream:

```yaml
source: rtsp://username:password@camera_ip:554/stream
```

RTSP is a live video URL exposed by many CCTV cameras or NVR/DVR recorders. If you do not know your camera brand, first test the stream in VLC or ONVIF Device Manager.

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

Press `q` in the video window to stop.

For headless logging/alerts:

```powershell
.\.venv\Scripts\python.exe main.py --no-display
```

## Restricted Zone

To detect intrusion, add polygon points in `config/config.yaml`:

```yaml
restricted_zone: [[100, 120], [500, 120], [520, 400], [80, 400]]
```

The person's center point entering this polygon triggers an intrusion event.

## Important First-Version Notes

Enabled cameras run concurrently using one worker thread per camera. For the first demo, keep both streams at a modest resolution such as 640x360 or 640x480 and use `yolov8n.pt` on your RTX 3050 laptop GPU.

## Academic Framing

Recommended title:

**Two-Camera CCTV-Based Suspicious Activity Detection and Mobile Alert System Using Deep Learning**

Recommended framing:

**Suspicious activity detection**, not crime detection.

## Suggested Report Sections

- Problem statement
- Objectives
- System architecture
- Dataset or sample video sources
- Methodology
- Event detection rules
- Implementation details
- Results and screenshots
- Limitations
- Future scope

## Limitations

- Accuracy depends on camera angle, lighting, and occlusion.
- Running is not always suspicious.
- Fall detection may confuse lying down or sitting with falling.
- Fight detection is approximate in this prototype.
- The system should be used for alerts and human verification, not automatic legal conclusions.
