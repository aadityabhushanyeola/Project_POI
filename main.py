from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from src.alert_manager import AlertManager
from src.behavior_analyzer import BehaviorAnalyzer
from src.camera_stream import CameraStream
from src.config_loader import load_config
from src.event_logger import EventLogger
from src.person_detector import PersonDetector
from src.tracker import CentroidTracker
from src.visualizer import draw_header, draw_tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-camera CCTV suspicious activity detection prototype")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV display windows")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    event_config = config["events"]
    tracking_config = config["tracking"]
    alert_config = config["alerts"]
    logging_config = config["logging"]

    alert_manager = AlertManager(
        telegram_enabled=bool(alert_config.get("telegram_enabled", False)),
        email_enabled=bool(alert_config.get("email_enabled", False)),
    )
    event_logger = EventLogger(logging_config["csv_path"])

    enabled_cameras = [camera for camera in config["cameras"] if camera.get("enabled", True)]
    if not enabled_cameras:
        raise RuntimeError("No enabled cameras in config/config.yaml")

    with ThreadPoolExecutor(max_workers=len(enabled_cameras)) as executor:
        futures = [
            executor.submit(
                run_camera,
                camera_config=camera_config,
                model_config=model_config,
                tracking_config=tracking_config,
                event_config=event_config,
                alert_config=alert_config,
                logging_config=logging_config,
                alert_manager=alert_manager,
                event_logger=event_logger,
                no_display=args.no_display,
                frame_skip=int(model_config.get("frame_skip", 1)),
            )
            for camera_config in enabled_cameras
        ]

        for future in as_completed(futures):
            future.result()


def run_camera(
    camera_config: dict,
    model_config: dict,
    tracking_config: dict,
    event_config: dict,
    alert_config: dict,
    logging_config: dict,
    alert_manager: AlertManager,
    event_logger: EventLogger,
    no_display: bool,
    frame_skip: int,
) -> None:
    detector = PersonDetector(
        weights=model_config["weights"],
        confidence=float(model_config["confidence"]),
        image_size=int(model_config["image_size"]),
        device=str(model_config.get("device", "auto")),
    )
    stream = CameraStream(
        camera_id=camera_config["id"],
        camera_name=camera_config.get("name", camera_config["id"]),
        source=camera_config["source"],
        frame_skip=frame_skip,
    )
    tracker = CentroidTracker(
        max_lost_frames=int(tracking_config["max_lost_frames"]),
        match_distance_px=float(tracking_config["match_distance_px"]),
        min_confirmed_frames=int(tracking_config["min_confirmed_frames"]),
    )
    analyzer = BehaviorAnalyzer(camera_config, event_config)
    snapshot_dir = Path(logging_config["snapshot_dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting {camera_config.get('name', camera_config['id'])}: {camera_config['source']}")

    for video_frame in stream.frames():
        frame = video_frame.image
        detections = detector.detect(frame)
        tracks = tracker.update(detections, video_frame.timestamp)
        events = analyzer.analyze(tracks, frame, video_frame.timestamp)

        analyzer.draw_zones(frame)
        draw_tracks(frame, tracks)
        draw_header(frame, video_frame.camera_name)

        for event in events:
            snapshot_path = None
            if alert_config.get("save_snapshots", True):
                snapshot_path = _save_snapshot(snapshot_dir, event.event_type, video_frame.camera_id, frame)

            event_logger.write(event, snapshot_path)
            print(f"[{time.strftime('%H:%M:%S')}] {event.camera_id} {event.event_type}: {event.reason}")

            try:
                alert_manager.send(event, snapshot_path)
            except Exception as exc:
                print(f"Alert failed: {exc}")

        if not no_display:
            cv2.imshow(video_frame.camera_id, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if not no_display:
        cv2.destroyWindow(camera_config["id"])


def _save_snapshot(snapshot_dir: Path, event_type: str, camera_id: str, frame) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = snapshot_dir / f"{camera_id}_{event_type}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


if __name__ == "__main__":
    main()
