from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

from src.behavior_analyzer import SuspiciousEvent


class EventLogger:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "camera_id", "camera_name", "event_type", "track_id", "confidence", "reason", "snapshot_path"])

    def write(self, event: SuspiciousEvent, snapshot_path: str | None) -> None:
        with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp)),
                        event.camera_id,
                        event.camera_name,
                        event.event_type,
                        event.track_id if event.track_id is not None else "",
                        f"{event.confidence:.2f}",
                        event.reason,
                        snapshot_path or "",
                    ]
                )
