from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class VideoFrame:
    camera_id: str
    camera_name: str
    frame_index: int
    timestamp: float
    image: object


class CameraStream:
    def __init__(self, camera_id: str, camera_name: str, source: str | int, frame_skip: int = 1):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.frame_skip = max(1, int(frame_skip))

    def frames(self) -> Iterator[VideoFrame]:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source for {self.camera_id}: {self.source}")

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if frame_index % self.frame_skip != 0:
                    continue

                yield VideoFrame(
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    frame_index=frame_index,
                    timestamp=time.time(),
                    image=frame,
                )
        finally:
            capture.release()
