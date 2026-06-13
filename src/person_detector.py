from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

cache_dir = Path(".cache").resolve()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))

from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float


class PersonDetector:
    def __init__(self, weights: str, confidence: float, image_size: int, device: str = "auto"):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.image_size = image_size
        self.device = None if device == "auto" else device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results: list[Any] = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id != 0:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection((x1, y1, x2, y2), confidence))

        return detections
