from __future__ import annotations

import os
from pathlib import Path

cache_dir = Path(".cache").resolve()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))

import cv2
import numpy as np
from ultralytics import YOLO

from src.home_safety.recommender import recommend
from src.home_safety.types import Hazard


class HomeSafetyDetector:
    def __init__(self, camera: dict, model_cfg: dict, hazard_cfg: dict):
        self.camera = camera
        self.hazard_cfg = hazard_cfg
        self.model = YOLO(model_cfg["weights"])
        self.conf = float(model_cfg["confidence"])
        self.imgsz = int(model_cfg["image_size"])
        self.device = None if model_cfg.get("device", "auto") == "auto" else model_cfg.get("device")
        self.obstacle_classes = set(hazard_cfg.get("obstacle_classes", []))

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Hazard]:
        hazards: list[Hazard] = []
        hazards.extend(self._detect_obstacles_and_fall_risk(frame, timestamp))
        hazards.extend(self._detect_wet_floor(frame, timestamp))
        hazards.extend(self._detect_sharp_edges(frame, timestamp))
        return hazards

    def _detect_obstacles_and_fall_risk(self, frame: np.ndarray, ts: float) -> list[Hazard]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        hazards: list[Hazard] = []
        names = result.names
        if result.boxes is None:
            return hazards

        for box in result.boxes:
            cls = int(box.cls[0].item())
            name = names.get(cls, str(cls))
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = (x1, y1, x2, y2)

            if name in self.obstacle_classes and _inside_any_path(bbox, self.camera.get("walking_paths", [])):
                hazards.append(self._hazard("obstacle_in_path", conf, bbox, ts, "yolo"))

            if name == "person" and _looks_like_fallen_person(bbox):
                hazards.append(self._hazard("fall_risk_person_on_floor", min(0.85, conf + 0.1), bbox, ts, "yolo_aspect_ratio"))

        return hazards

    def _detect_wet_floor(self, frame: np.ndarray, ts: float) -> list[Hazard]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = cv2.inRange(hsv, (0, 0, 145), (179, 80, 255))
        mask = cv2.bitwise_and(mask, cv2.inRange(v, 150, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hazards: list[Hazard] = []
        min_area = int(self.hazard_cfg.get("wet_floor_min_area", 900))
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, hgt = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + hgt)
            if self.camera.get("walking_paths") and not _inside_any_path(bbox, self.camera["walking_paths"]):
                continue
            conf = min(0.82, 0.45 + area / 8000)
            hazards.append(self._hazard("wet_floor_or_spill", conf, bbox, ts, "color_reflection"))
        return hazards

    def _detect_sharp_edges(self, frame: np.ndarray, ts: float) -> list[Hazard]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=160, qualityLevel=0.03, minDistance=14)
        if corners is None:
            return []

        points = np.int32(corners.reshape(-1, 2))
        zones = self.camera.get("child_zones") or self.camera.get("walking_paths") or []
        if not zones:
            zones = [[[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]]]

        hazards: list[Hazard] = []
        min_count = int(self.hazard_cfg.get("sharp_corner_min_corners", 18))
        for zone in zones:
            pts = np.array(zone, np.int32)
            inside = [p for p in points if cv2.pointPolygonTest(pts, (float(p[0]), float(p[1])), False) >= 0]
            if len(inside) < min_count:
                continue
            arr = np.array(inside)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            conf = min(0.78, 0.35 + len(inside) / 100)
            hazards.append(self._hazard("sharp_corner_edge_zone", conf, bbox, ts, "corner_density"))
        return hazards[:2]

    def _hazard(self, label: str, confidence: float, bbox: tuple[int, int, int, int], ts: float, source: str) -> Hazard:
        severe = confidence >= float(self.hazard_cfg.get("severe_confidence", 0.72))
        return Hazard(
            camera_id=self.camera["id"],
            camera_name=self.camera.get("name", self.camera["id"]),
            label=label,
            confidence=round(float(confidence), 3),
            bbox=bbox,
            recommendation=recommend(label, severe),
            alert=severe,
            timestamp=ts,
            source=source,
        )


def _inside_any_path(bbox: tuple[int, int, int, int], paths: list) -> bool:
    if not paths:
        return True
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(cv2.pointPolygonTest(np.array(path, np.int32), (float(cx), float(cy)), False) >= 0 for path in paths)


def _looks_like_fallen_person(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h > 1.15
