# Home Safety CCTV Analysis Add-On

Runs on CCTV/video only. Detects practical home hazards and outputs label, confidence, bbox, recommendation, and severe alert flag.

## Files

```text
home_safety_main.py
config/home_safety.yaml
src/home_safety/detector.py
src/home_safety/recommender.py
src/home_safety/recurring.py
src/home_safety/logger.py
src/home_safety/runner.py
src/home_safety/visualizer.py
```

## Install

```powershell
cd "C:\Users\Aaditya BY\OneDrive\Desktop\Desktop\Project POI"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configure

Edit `config/home_safety.yaml`.

Use webcam:

```yaml
source: 0
```

Use CCTV RTSP:

```yaml
source: rtsp://user:password@camera_ip:554/stream
```

Use video file:

```yaml
source: samples/Falling1.mp4
```

Set walking paths:

```yaml
walking_paths:
  - [[40, 260], [620, 260], [620, 470], [40, 470]]
```

## Run

With display:

```powershell
.\.venv\Scripts\python.exe home_safety_main.py --config config/home_safety.yaml
```

Headless/offline:

```powershell
.\.venv\Scripts\python.exe home_safety_main.py --config config/home_safety.yaml --no-display --offline
```

## Outputs

```text
events/home_safety/hazards.csv
events/home_safety/hazards.jsonl
events/home_safety/snapshots/
events/home_safety/recurring_zones.json
```

## Hazard Methods

- `obstacle_in_path`: YOLO object detection + walking path polygon.
- `fall_risk_person_on_floor`: person box horizontal aspect ratio. Placeholder heuristic, not medical-grade.
- `wet_floor_or_spill`: bright low-saturation reflection/spill heuristic. Placeholder heuristic.
- `sharp_corner_edge_zone`: corner-density heuristic from CCTV frame. Placeholder heuristic.
- `recurring_hazard_zone`: repeated hazards in the same grid cell over time.

## Alerts

Set in `config/home_safety.yaml`:

```yaml
alerts:
  telegram_enabled: true
  email_enabled: false
```

Add credentials in `.env` using `.env.example`.
