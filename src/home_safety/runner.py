from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from src.alert_manager import AlertManager
from src.camera_stream import CameraStream
from src.config_loader import load_config
from src.home_safety.detector import HomeSafetyDetector
from src.home_safety.logger import HazardLogger
from src.home_safety.recurring import RecurringZoneTracker
from src.home_safety.visualizer import draw


def run_home_safety(config_path: str, no_display: bool, offline: bool) -> None:
    cfg = load_config(config_path)
    enabled = [c for c in cfg["cameras"] if c.get("enabled", True)]
    logger = HazardLogger(cfg["logging"]["csv_path"], cfg["logging"]["jsonl_path"])
    alerts = AlertManager(cfg["alerts"].get("telegram_enabled", False), cfg["alerts"].get("email_enabled", False))
    recurring = RecurringZoneTracker(cfg["logging"]["recurring_path"], int(cfg["hazards"].get("recurring_alert_count", 5)))

    with ThreadPoolExecutor(max_workers=len(enabled)) as ex:
        futures = [
            ex.submit(_run_camera, cam, cfg, logger, alerts, recurring, no_display, offline)
            for cam in enabled
        ]
        for f in as_completed(futures):
            f.result()


def _run_camera(camera: dict, cfg: dict, logger: HazardLogger, alerts: AlertManager, recurring: RecurringZoneTracker, no_display: bool, offline: bool) -> None:
    detector = HomeSafetyDetector(camera, cfg["model"], cfg["hazards"])
    stream = CameraStream(camera["id"], camera.get("name", camera["id"]), camera["source"], int(cfg["model"].get("frame_skip", 1)))
    snapshot_dir = Path(cfg["logging"]["snapshot_dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Home safety started: {camera.get('name', camera['id'])}")
    for vf in stream.frames():
        frame = vf.image
        hazards = detector.detect(frame, vf.timestamp)
        extra = []
        for hazard in hazards:
            recurring_hazard = recurring.update(hazard)
            if recurring_hazard:
                extra.append(recurring_hazard)
        hazards.extend(extra)

        draw(frame, vf.camera_name, hazards, camera.get("walking_paths", []))
        for hazard in hazards:
            logger.write(hazard)
            snapshot = _save(snapshot_dir, hazard, frame) if cfg["alerts"].get("save_snapshots", True) else None
            print(f"{hazard.camera_id} {hazard.label} {hazard.confidence}: {hazard.recommendation}")
            if hazard.alert:
                _send_alert(alerts, hazard, snapshot)

        if not no_display:
            cv2.imshow(f"home_safety_{camera['id']}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if offline and not isinstance(camera["source"], int):
            continue
    if not no_display:
        cv2.destroyWindow(f"home_safety_{camera['id']}")


def _save(snapshot_dir: Path, hazard, frame) -> str:
    path = snapshot_dir / f"{hazard.camera_id}_{hazard.label}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def _send_alert(alerts: AlertManager, hazard, snapshot: str | None) -> None:
    class EventAdapter:
        alert_title = "Home Safety Hazard Alert"
        camera_id = hazard.camera_id
        camera_name = hazard.camera_name
        event_type = hazard.label
        track_id = None
        confidence = hazard.confidence
        timestamp = hazard.timestamp
        reason = hazard.recommendation

    try:
        alerts.send(EventAdapter(), snapshot)
    except Exception as exc:
        print(f"Home safety alert failed: {exc}")
