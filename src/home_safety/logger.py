from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

from src.home_safety.types import Hazard


class HazardLogger:
    def __init__(self, csv_path: str, jsonl_path: str):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.lock = threading.Lock()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["time", "camera_id", "label", "confidence", "bbox", "alert", "recommendation", "source"])

    def write(self, hazard: Hazard) -> None:
        data = hazard.to_dict()
        with self.lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hazard.timestamp)),
                    hazard.camera_id,
                    hazard.label,
                    hazard.confidence,
                    list(hazard.bbox),
                    hazard.alert,
                    hazard.recommendation,
                    hazard.source,
                ])
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
