from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.person_detector import Detection


@dataclass
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    lost_frames: int = 0
    history: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.xyxy
        return max(1.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.xyxy
        return max(1.0, y2 - y1)


class CentroidTracker:
    """Small dependency-free tracker suitable for a prototype and report explanation."""

    def __init__(self, max_lost_frames: int, match_distance_px: float, min_confirmed_frames: int):
        self.max_lost_frames = max_lost_frames
        self.match_distance_px = match_distance_px
        self.min_confirmed_frames = min_confirmed_frames
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_indexes = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            tx, ty = track.center
            for detection_index, detection in enumerate(detections):
                dx, dy = _center(detection.xyxy)
                distance = math.hypot(tx - dx, ty - dy)
                if distance <= self.match_distance_px:
                    candidate_pairs.append((distance, track_id, detection_index))

        for _, track_id, detection_index in sorted(candidate_pairs, key=lambda item: item[0]):
            if track_id in unmatched_track_ids and detection_index in unmatched_detection_indexes:
                matches.append((track_id, detection_index))
                unmatched_track_ids.remove(track_id)
                unmatched_detection_indexes.remove(detection_index)

        for track_id, detection_index in matches:
            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.xyxy = detection.xyxy
            track.confidence = detection.confidence
            track.last_seen = timestamp
            track.hits += 1
            track.lost_frames = 0
            cx, cy = track.center
            track.history.append((timestamp, cx, cy))
            track.history = track.history[-30:]

        for track_id in unmatched_track_ids:
            self._tracks[track_id].lost_frames += 1

        for detection_index in unmatched_detection_indexes:
            detection = detections[detection_index]
            track = Track(
                track_id=self._next_id,
                xyxy=detection.xyxy,
                confidence=detection.confidence,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            cx, cy = track.center
            track.history.append((timestamp, cx, cy))
            self._tracks[self._next_id] = track
            self._next_id += 1

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.lost_frames <= self.max_lost_frames
        }

        return [
            track
            for track in self._tracks.values()
            if track.hits >= self.min_confirmed_frames and track.lost_frames == 0
        ]


def _center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2
