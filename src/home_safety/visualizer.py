from __future__ import annotations

import cv2
import numpy as np

from src.home_safety.types import Hazard


def draw(frame: np.ndarray, camera_name: str, hazards: list[Hazard], paths: list) -> None:
    for path in paths or []:
        pts = np.array(path, np.int32)
        cv2.polylines(frame, [pts], True, (255, 180, 0), 2)

    for hazard in hazards:
        x1, y1, x2, y2 = hazard.bbox
        color = (0, 0, 255) if hazard.alert else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{hazard.label} {hazard.confidence:.2f}", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, f"Home Safety: {camera_name}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
