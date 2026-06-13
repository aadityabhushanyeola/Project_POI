from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.tracker import Track


@dataclass
class SuspiciousEvent:
    camera_id: str
    camera_name: str
    event_type: str
    track_id: int | None
    confidence: float
    timestamp: float
    reason: str


class BehaviorAnalyzer:
    def __init__(self, camera_config: dict, event_config: dict):
        self.camera_id = camera_config["id"]
        self.camera_name = camera_config.get("name", self.camera_id)
        self.restricted_zone = camera_config.get("restricted_zone") or []
        self.event_config = event_config
        self._last_event_at: dict[str, float] = {}
        self._fall_candidates: dict[int, float] = {}
        self._running_candidates: dict[int, float] = {}
        self._fight_started_at: dict[tuple[int, int], float] = {}

    def analyze(self, tracks: list[Track], frame: np.ndarray, timestamp: float) -> list[SuspiciousEvent]:
        events: list[SuspiciousEvent] = []
        events.extend(self._detect_intrusion(tracks, timestamp))
        events.extend(self._detect_running(tracks, timestamp))
        events.extend(self._detect_fall(tracks, timestamp))
        events.extend(self._detect_chase_and_fight(tracks, timestamp))
        return [event for event in events if self._not_in_cooldown(event, timestamp)]

    def draw_zones(self, frame: np.ndarray) -> None:
        if len(self.restricted_zone) >= 3:
            points = np.array(self.restricted_zone, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color=(0, 0, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def _detect_intrusion(self, tracks: list[Track], timestamp: float) -> list[SuspiciousEvent]:
        if len(self.restricted_zone) < 3:
            return []

        polygon = np.array(self.restricted_zone, dtype=np.int32)
        events = []
        for track in tracks:
            cx, cy = track.center
            inside = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0
            if inside:
                events.append(self._event("intrusion", track.track_id, 0.85, timestamp, "Person entered restricted zone"))
        return events

    def _detect_running(self, tracks: list[Track], timestamp: float) -> list[SuspiciousEvent]:
        threshold = float(self.event_config["running_speed_px_per_sec"])
        min_duration = float(self.event_config["running_min_duration_seconds"])
        events = []

        for track in tracks:
            speed = _speed_px_per_sec(track)
            if speed >= threshold:
                self._running_candidates.setdefault(track.track_id, timestamp)
                if timestamp - self._running_candidates[track.track_id] >= min_duration:
                    events.append(self._event("running", track.track_id, 0.7, timestamp, f"High movement speed: {speed:.1f}px/s"))
            else:
                self._running_candidates.pop(track.track_id, None)

        return events

    def _detect_fall(self, tracks: list[Track], timestamp: float) -> list[SuspiciousEvent]:
        ratio_threshold = float(self.event_config["fall_aspect_ratio_threshold"])
        drop_threshold = float(self.event_config["fall_center_drop_px"])
        min_duration = float(self.event_config["fall_min_duration_seconds"])
        events = []

        for track in tracks:
            aspect_ratio = track.width / track.height
            center_drop = _center_drop(track)
            looks_like_fall = aspect_ratio >= ratio_threshold and center_drop >= drop_threshold

            if looks_like_fall:
                self._fall_candidates.setdefault(track.track_id, timestamp)
                if timestamp - self._fall_candidates[track.track_id] >= min_duration:
                    reason = f"Body box became horizontal and dropped {center_drop:.1f}px"
                    events.append(self._event("possible_fall", track.track_id, 0.72, timestamp, reason))
            else:
                self._fall_candidates.pop(track.track_id, None)

        return events

    def _detect_chase_and_fight(self, tracks: list[Track], timestamp: float) -> list[SuspiciousEvent]:
        events = []
        chase_distance = float(self.event_config["chase_distance_px"])
        fight_distance = float(self.event_config["fight_distance_px"])
        fight_motion = float(self.event_config["fight_motion_px_per_sec"])
        fight_duration = float(self.event_config["fight_min_duration_seconds"])

        for i, first in enumerate(tracks):
            for second in tracks[i + 1 :]:
                distance = _distance(first, second)
                first_speed = _speed_px_per_sec(first)
                second_speed = _speed_px_per_sec(second)

                if distance <= chase_distance and first_speed > fight_motion and second_speed > fight_motion:
                    events.append(self._event("possible_chase", None, 0.65, timestamp, "Two people moving fast near each other"))

                pair_key = tuple(sorted((first.track_id, second.track_id)))
                if distance <= fight_distance and first_speed > fight_motion and second_speed > fight_motion:
                    self._fight_started_at.setdefault(pair_key, timestamp)
                    if timestamp - self._fight_started_at[pair_key] >= fight_duration:
                        events.append(self._event("possible_fight", None, 0.68, timestamp, "Close interaction with high motion intensity"))
                else:
                    self._fight_started_at.pop(pair_key, None)

        return events

    def _event(self, event_type: str, track_id: int | None, confidence: float, timestamp: float, reason: str) -> SuspiciousEvent:
        return SuspiciousEvent(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            event_type=event_type,
            track_id=track_id,
            confidence=confidence,
            timestamp=timestamp,
            reason=reason,
        )

    def _not_in_cooldown(self, event: SuspiciousEvent, timestamp: float) -> bool:
        cooldown = float(self.event_config["cooldown_seconds"])
        key = f"{event.camera_id}:{event.event_type}:{event.track_id}"
        last_seen = self._last_event_at.get(key, 0.0)
        if timestamp - last_seen < cooldown:
            return False

        self._last_event_at[key] = timestamp
        return True


def _speed_px_per_sec(track: Track) -> float:
    if len(track.history) < 2:
        return 0.0

    start = track.history[0]
    end = track.history[-1]
    elapsed = max(0.001, end[0] - start[0])
    return float(np.hypot(end[1] - start[1], end[2] - start[2]) / elapsed)


def _center_drop(track: Track) -> float:
    if len(track.history) < 3:
        return 0.0

    early_y = track.history[0][2]
    latest_y = track.history[-1][2]
    return max(0.0, latest_y - early_y)


def _distance(first: Track, second: Track) -> float:
    fx, fy = first.center
    sx, sy = second.center
    return float(np.hypot(fx - sx, fy - sy))
