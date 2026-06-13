from __future__ import annotations

import cv2
import numpy as np

from src.tracker import Track


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        points = [(int(x), int(y)) for _, x, y in track.history[-12:]]
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, (255, 200, 0), 2)


def draw_header(frame: np.ndarray, camera_name: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, camera_name, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
